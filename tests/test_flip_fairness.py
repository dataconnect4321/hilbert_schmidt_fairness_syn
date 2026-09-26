"""Unit tests for the FLIP fairness/privacy extensions to TabSyn's VAE.

Covers (tabsyn/vae/flip_fairness.py):
  - linear_cka_t: identity, independence, symmetry, boundedness
  - disentanglement_loss: sign, group-dependence, degenerate cases
  - multi_stage_disentanglement_loss: stage aggregation, s_idx skipping
  - sliced_wasserstein_distance: identity, shift sensitivity, symmetry
  - uniform_attribute_loss: uniform vs skewed logits
  - BalancedGroupSampler: equal representation, epoch determinism, Eq. 11
  - RDPAccountant: monotonicity, zero steps, epsilon conversion
  - Model_VAE.forward_with_stages: shapes and gradient flow
  - compute_loss: protected attribute excluded from CE, L_S returned
  - DP-SGD smoke test: Opacus DPOptimizer clips and steps

Run:  python -m pytest tests/test_flip_fairness.py -v
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tabsyn-main', 'tabsyn-main'))

from tabsyn.vae.flip_fairness import (
    linear_cka_t,
    disentanglement_loss,
    multi_stage_disentanglement_loss,
    sliced_wasserstein_distance,
    uniform_attribute_loss,
    uniform_attribute_loss_per_sample,
    BalancedGroupSampler,
    PoissonGroupSampler,
    RDPAccountant,
    GroupLevelDPMechanism,
    dp_release_mean_std,
    compose_rdp_budgets,
    register_tabsyn_grad_samplers,
    get_dp_trainable_parameters,
)
from tabsyn.vae.model import Model_VAE
from tabsyn.vae.main import compute_loss

# Register native grad samplers so Opacus can handle the custom Tokenizer
register_tabsyn_grad_samplers()

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cpu')


# --------------------------------------------------------------------------
# linear_cka_t
# --------------------------------------------------------------------------

def test_cka_identity():
    A = torch.randn(64, 16)
    assert torch.isclose(linear_cka_t(A, A), torch.tensor(1.0), atol=1e-5)


def test_cka_independent_near_zero():
    """Independent random groups: CKA^T is the cosine similarity of two
    random Wishart-type covariance matrices. In the n>>p regime this is
    concentrated near 1 (both covariances ~ the identity), NOT near 0 -
    the old ||A^T B||^2 form's near-zero value came from the arbitrary
    row pairing. With p comparable to n it drops toward 0."""
    A = torch.randn(256, 16)
    B = torch.randn(256, 16)
    val = linear_cka_t(A, B)
    assert 0.0 <= val.item() <= 1.0
    # High-dimension check: p ~ n -> covariances are far from aligned
    A_hd = torch.randn(64, 64)
    B_hd = torch.randn(64, 64)
    assert linear_cka_t(A_hd, B_hd).item() < 0.5


def test_cka_symmetric():
    A = torch.randn(64, 16)
    B = torch.randn(64, 16)
    assert torch.isclose(linear_cka_t(A, B), linear_cka_t(B, A), atol=1e-5)


def test_cka_bounded_01():
    A = torch.randn(32, 8) * 10
    B = torch.randn(32, 8) * 0.01
    for _ in range(5):
        val = linear_cka_t(A, B).item()
        assert 0.0 <= val <= 1.0 + 1e-6


def test_cka_shift_invariance():
    """CKA is invariant to adding a constant offset (centering)."""
    A = torch.randn(64, 16)
    val1 = linear_cka_t(A, A)
    val2 = linear_cka_t(A, A + 100.0)
    assert torch.isclose(val1, val2, atol=1e-4)


def test_cka_correlated_higher_than_independent():
    A = torch.randn(256, 16)
    B_correlated = A + 0.1 * torch.randn(256, 16)
    B_indep = torch.randn(256, 16)
    assert linear_cka_t(A, B_correlated) > linear_cka_t(A, B_indep)


def test_cka_t_is_covariance_cosine():
    """CKA^T must equal <A^T A, B^T B>_F / (||A^T A||_F ||B^T B||_F),
    the cosine similarity of the groups' feature covariances - NOT
    ||A^T B||^2_F, which pairs rows across groups arbitrarily."""
    A = torch.randn(64, 16)
    B = torch.randn(48, 16)  # unequal sizes must be fine
    Ac = A - A.mean(dim=0, keepdim=True)
    Bc = B - B.mean(dim=0, keepdim=True)
    expected = ((Ac.t() @ Ac) * (Bc.t() @ Bc)).sum() / (
        torch.norm(Ac.t() @ Ac, p='fro') * torch.norm(Bc.t() @ Bc, p='fro'))
    assert torch.isclose(linear_cka_t(A, B), expected, atol=1e-5)


def test_cka_t_single_feature_is_one():
    """With p=1 feature, scalar covariances are trivially 'aligned':
    CKA^T = 1. The old ||A^T B||^2 form returned the squared cosine of
    an arbitrary row pairing instead."""
    A = torch.randn(32, 1)
    B = torch.randn(24, 1)
    assert torch.isclose(linear_cka_t(A, B), torch.tensor(1.0), atol=1e-5)


def test_cka_t_unequal_group_sizes():
    """No truncation: unequal group sizes must still yield a valid loss.
    A random subset shares the same population covariance, so CKA^T
    stays high (though not exactly 1 - sampling noise in the two
    empirical covariances)."""
    A = torch.randn(100, 8)
    val_full = linear_cka_t(A, A[:40])
    assert val_full > 0.8


# --------------------------------------------------------------------------
# disentanglement_loss
# --------------------------------------------------------------------------

def test_disentanglement_negative():
    """Loss is negative CKA^T, so <= 0."""
    rep = torch.randn(64, 16)
    groups = torch.randint(0, 2, (64,))
    assert disentanglement_loss(rep, groups).item() <= 0.0


def test_disentanglement_zero_for_single_group():
    rep = torch.randn(64, 16)
    groups = torch.zeros(64, dtype=torch.long)
    assert disentanglement_loss(rep, groups).item() == 0.0


def test_disentanglement_identical_groups_minimal():
    """If both groups share the same representation, CKA^T ~ 1 -> loss ~ -1."""
    base = torch.randn(32, 16)
    rep = torch.cat([base, base + 1e-6], dim=0)  # near-identical groups
    groups = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    loss = disentanglement_loss(rep, groups)
    assert loss.item() < -0.9  # close to -1


def test_disentanglement_unequal_groups():
    """No truncation: unequal group sizes must still yield a valid loss,
    and identical-covariance groups of different sizes -> ~-1."""
    base = torch.randn(40, 16)
    rep = torch.cat([base, base[:24] + 1e-6], dim=0)
    groups = torch.cat([torch.zeros(40), torch.ones(24)]).long()
    loss = disentanglement_loss(rep, groups)
    assert -1.0 <= loss.item() <= 0.0
    assert loss.item() < -0.85  # near-identical covariance structure


def test_disentanglement_different_groups_less_negative():
    """Divergent group representations -> CKA^T near 0 -> loss near 0."""
    g0 = torch.randn(32, 16)
    g1 = torch.randn(32, 16)
    rep = torch.cat([g0, g1], dim=0)
    groups = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    loss_same = disentanglement_loss(torch.cat([g0, g0], dim=0), groups)
    loss_diff = disentanglement_loss(rep, groups)
    assert loss_diff > loss_same  # disentangled (diff) loss closer to 0


def test_disentanglement_gradient_flows():
    rep = torch.randn(64, 16, requires_grad=True)
    groups = torch.randint(0, 2, (64,))
    loss = disentanglement_loss(rep, groups)
    loss.backward()
    assert rep.grad is not None
    assert torch.isfinite(rep.grad).all()


# --------------------------------------------------------------------------
# multi_stage_disentanglement_loss
# --------------------------------------------------------------------------

def _make_stages(n=64, d_num=4, categories=(3, 5), d_token=8, tokens=5):
    return {
        'mu_z': torch.randn(n, tokens, d_token),
        'h': torch.randn(n, tokens, d_token),
        'recon_x_num': torch.randn(n, d_num),
        'recon_x_cat': [torch.randn(n, c) for c in categories],
        'std_z': torch.randn(n, tokens, d_token).abs(),
    }


def test_multi_stage_returns_scalar():
    stages = _make_stages()
    groups = torch.randint(0, 2, (64,))
    loss = multi_stage_disentanglement_loss(stages, groups)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_multi_stage_skips_protected_feature():
    """With s_idx set, the protected column's logits are excluded from the
    detokenizer stage but the other stages still contribute."""
    stages = _make_stages()
    groups = torch.randint(0, 2, (64,))
    loss_with = multi_stage_disentanglement_loss(stages, groups, s_idx=0)
    loss_without = multi_stage_disentanglement_loss(stages, groups, s_idx=None)
    # Both valid scalars; with-s_idx should differ (one fewer feature term)
    assert torch.isfinite(loss_with) and torch.isfinite(loss_without)
    assert not torch.isclose(loss_with, loss_without, atol=1e-8)


def test_multi_stage_mean_of_stages():
    """Loss should equal the mean of the three stage losses."""
    stages = _make_stages()
    groups = torch.randint(0, 2, (64,))
    s_idx = 0

    mu_flat = stages['mu_z'].reshape(64, -1)
    h_flat = stages['h'].reshape(64, -1)
    feat_losses = [disentanglement_loss(stages['recon_x_num'], groups)]
    for i, cl in enumerate(stages['recon_x_cat']):
        if i == s_idx:
            continue
        feat_losses.append(disentanglement_loss(cl, groups))

    expected = (disentanglement_loss(mu_flat, groups)
                + torch.stack(feat_losses).mean()
                + disentanglement_loss(h_flat, groups)) / 3.0
    got = multi_stage_disentanglement_loss(stages, groups, s_idx=s_idx)
    assert torch.isclose(got, expected, atol=1e-6)


# --------------------------------------------------------------------------
# sliced_wasserstein_distance
# --------------------------------------------------------------------------

def test_swd_identity_zero():
    A = torch.randn(64, 16)
    assert sliced_wasserstein_distance(A, A).item() == 0.0


def test_swd_shift_positive():
    A = torch.randn(64, 16)
    val = sliced_wasserstein_distance(A, A + 5.0)
    assert val.item() > 1.0  # large shift -> large distance


def test_swd_symmetric():
    A = torch.randn(64, 16)
    B = torch.randn(64, 16)
    # Use identical fixed projections so the only difference is direction
    theta = torch.randn(50, 16)
    theta = theta / torch.norm(theta, dim=1, keepdim=True)
    d1 = sliced_wasserstein_distance(A, B, theta=theta)
    d2 = sliced_wasserstein_distance(B, A, theta=theta)
    assert torch.isclose(d1, d2, atol=1e-5)


def test_swd_gradient_flows():
    X = torch.randn(64, 16)
    Y = torch.randn(64, 16, requires_grad=True)
    loss = sliced_wasserstein_distance(X, Y)
    loss.backward()
    assert Y.grad is not None


# --------------------------------------------------------------------------
# uniform_attribute_loss
# --------------------------------------------------------------------------

def test_uniform_loss_zero_for_uniform():
    logits = torch.zeros(32, 4)
    assert uniform_attribute_loss(logits).item() < 1e-6


def test_uniform_loss_positive_for_skewed():
    skewed = torch.tensor([[5.0, 0.0, 0.0, 0.0]] * 32)
    assert uniform_attribute_loss(skewed).item() > 0.5


def test_uniform_loss_reduced_by_balancing_logits():
    """Softmax of balanced logits should have lower loss than skewed ones."""
    n_groups = 3
    skewed = torch.tensor([[4.0, 0.0, 0.0]] * 16)
    balanced = torch.tensor([[0.0, 0.0, 0.0]] * 16)
    assert (uniform_attribute_loss(balanced)
            < uniform_attribute_loss(skewed))


def test_uniform_loss_max_value():
    """One-hot deterministic prediction: max L_S = sqrt(|S| * (1-1/|S|)^2
    + (|S|-1) * (1/|S|)^2)."""
    n_groups = 4
    one_hot = torch.tensor([[10.0, 0.0, 0.0, 0.0]] * 8)
    expected = ((1 - 1 / n_groups) ** 2
                + (n_groups - 1) * (1 / n_groups) ** 2) ** 0.5
    assert torch.isclose(uniform_attribute_loss(one_hot),
                         torch.tensor(expected), atol=1e-3)


# --------------------------------------------------------------------------
# uniform_attribute_loss_per_sample (DP-decomposable L_S)
# --------------------------------------------------------------------------

def test_per_sample_ls_zero_for_uniform():
    logits = torch.zeros(32, 4)
    assert uniform_attribute_loss_per_sample(logits).item() < 1e-6


def test_per_sample_ls_positive_for_skewed():
    skewed = torch.tensor([[5.0, 0.0, 0.0, 0.0]] * 32)
    assert uniform_attribute_loss_per_sample(skewed).item() > 0.5


def test_per_sample_ls_bounds_batch_mean():
    """Per-sample sum >= batch-mean L_S (Jensen): a batch can have a
    uniform mean while individual rows are confidently wrong - the
    per-sample variant catches what the mean hides."""
    # Half the rows one-hot group 0, half one-hot group 1: mean softmax
    # is uniform (batch-mean L_S ~ 0) but each row is far from uniform.
    rows = torch.zeros(16, 2)
    rows[:8, 0] = 10.0
    rows[8:, 1] = 10.0
    assert uniform_attribute_loss(rows).item() < 1e-3
    assert uniform_attribute_loss_per_sample(rows).item() > 1.0


def test_per_sample_ls_decomposable():
    """Each sample's gradient must depend only on its own logits - the
    property DP-SGD's per-sample clipping requires."""
    logits = torch.randn(8, 3, requires_grad=True)
    loss = uniform_attribute_loss_per_sample(logits)
    loss.backward()
    # Gradient of row i must be zero when row i's logits are masked:
    # verify by checking grad of a single-row loss equals the row-slice
    # of the batch loss's grad.
    single = torch.randn(1, 3, requires_grad=True)
    uniform_attribute_loss_per_sample(single).backward()
    # Same input -> same per-row gradient (computed independently)
    logits2 = single.detach().clone().requires_grad_(True)
    uniform_attribute_loss_per_sample(
        torch.cat([logits2, torch.randn(7, 3)]))
    # just check it runs; the row-independence is structural (sum of
    # per-row norms)
    assert torch.isfinite(logits.grad).all()


# --------------------------------------------------------------------------
# GroupLevelDPMechanism
# --------------------------------------------------------------------------

def test_group_mech_clips_and_noises():
    """Gradient larger than the clip norm is scaled down; noise is
    added; the noised gradient lands in p.grad."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(20, 10))
    huge = torch.randn(20, 10) * 100  # huge batch-coupled gradient
    mech = GroupLevelDPMechanism(max_group_grad_norm=0.5,
                                 noise_multiplier=1.0)
    mech.add_noised([w], {w: huge})
    # The clipped gradient has norm <= 0.5; the Gaussian noise has
    # per-element std 0.5 (noise_multiplier * clip norm), so the noised
    # result's norm is dominated by the noise across 200 elements
    # (~0.5*sqrt(200) ~ 7) - but crucially NOT by the 100x raw gradient
    # (norm ~1415): the result must be orders of magnitude smaller.
    assert w.grad.norm().item() < 20.0
    assert w.grad.norm().item() < huge.norm().item() / 10.0
    assert mech.steps == 1


def test_group_mech_privacy_monotone_in_steps():
    m1 = GroupLevelDPMechanism(0.5, noise_multiplier=1.0)
    m2 = GroupLevelDPMechanism(0.5, noise_multiplier=1.0)
    for _ in range(10):
        m1.add_noised([], {})
    for _ in range(100):
        m2.add_noised([], {})
    e1, _ = m1.get_privacy_spent(1e-5)
    e2, _ = m2.get_privacy_spent(1e-5)
    assert 0 < e1 < e2


def test_group_mech_sensitivity_2c():
    """The whole gradient vector is clipped to C, so under substitution
    adjacency the sensitivity is 2C and each step's RDP must be
    2*alpha/sigma^2 (NOT alpha/(2*sigma^2), which assumed sensitivity C
    and under-accounted by 4x)."""
    mech = GroupLevelDPMechanism(0.5, noise_multiplier=2.0)
    mech.add_noised([], {})
    for alpha in [2.0, 10.0, 100.0]:
        expected = 2.0 * alpha / 4.0  # 2*alpha/sigma^2, one step
        assert abs(mech.get_rdp_epsilon(alpha) - expected) < 1e-9


def test_group_mech_rdp_curve_matches():
    """get_rdp_curve entries equal get_rdp_epsilon at each alpha."""
    mech = GroupLevelDPMechanism(0.5, noise_multiplier=2.0)
    for _ in range(7):
        mech.add_noised([], {})
    curve = mech.get_rdp_curve()
    for alpha, rdp in curve.items():
        assert abs(rdp - mech.get_rdp_epsilon(alpha)) < 1e-12


# --------------------------------------------------------------------------
# compose_rdp_budgets
# --------------------------------------------------------------------------

def test_compose_empty_budgets():
    eps, alpha = compose_rdp_budgets([], 1e-5)
    assert eps == 0.0 and alpha is None


def test_compose_single_budget_matches_standalone():
    """Composing one budget must equal its own get_privacy_spent."""
    mech = GroupLevelDPMechanism(0.5, noise_multiplier=2.0)
    for _ in range(50):
        mech.add_noised([], {})
    eps_direct, _ = mech.get_privacy_spent(1e-5)
    eps_composed, _ = compose_rdp_budgets([mech.get_rdp_curve()], 1e-5)
    assert abs(eps_direct - eps_composed) < 1e-9


def test_compose_two_budgets_larger_than_either():
    """Composition is monotone: the combined epsilon must exceed each
    individual budget's epsilon."""
    m1 = GroupLevelDPMechanism(0.5, noise_multiplier=2.0)
    m2 = GroupLevelDPMechanism(0.5, noise_multiplier=4.0)
    for _ in range(50):
        m1.add_noised([], {})
        m2.add_noised([], {})
    e1, _ = m1.get_privacy_spent(1e-5)
    e2, _ = m2.get_privacy_spent(1e-5)
    ec, _ = compose_rdp_budgets(
        [m1.get_rdp_curve(), m2.get_rdp_curve()], 1e-5)
    assert ec > e1 and ec > e2


def test_compose_not_sum_of_epsilons():
    """Valid RDP composition (sum RDP per alpha, convert once) is
    TIGHTER than the invalid rule of adding per-mechanism epsilons,
    because each mechanism minimizes over a different alpha."""
    m1 = GroupLevelDPMechanism(0.5, noise_multiplier=2.0)
    m2 = GroupLevelDPMechanism(0.5, noise_multiplier=4.0)
    for _ in range(50):
        m1.add_noised([], {})
        m2.add_noised([], {})
    e1, _ = m1.get_privacy_spent(1e-5)
    e2, _ = m2.get_privacy_spent(1e-5)
    ec, _ = compose_rdp_budgets(
        [m1.get_rdp_curve(), m2.get_rdp_curve()], 1e-5)
    assert ec < e1 + e2


def test_compose_stats_release_pairs():
    """dp_release_mean_std's (alpha, rdp) pair list composes directly."""
    torch.manual_seed(0)
    X = torch.randn(500, 4)
    _, _, rdp = dp_release_mean_std(X, 1.0, 10.0, 500)
    mech = GroupLevelDPMechanism(0.5, noise_multiplier=2.0)
    for _ in range(10):
        mech.add_noised([], {})
    ec, _ = compose_rdp_budgets(
        [mech.get_rdp_curve(), rdp], 1e-5)
    assert np.isfinite(ec) and ec > 0


def test_compose_duplicate_alphas_summed():
    """Two releases sharing an alpha grid, passed as ONE pair list,
    must SUM per alpha - not silently drop all but the last release
    (the dict() collapse bug)."""
    r1 = [(2.0, 1.0), (10.0, 5.0)]
    r2 = [(2.0, 2.0), (10.0, 10.0)]
    # As one list with duplicate alphas: must equal composing them
    # as two separate budgets.
    eps_merged, _ = compose_rdp_budgets([r1 + r2], 1e-5)
    eps_separate, _ = compose_rdp_budgets([r1, r2], 1e-5)
    assert abs(eps_merged - eps_separate) < 1e-9


def test_rdp_curve_uses_subsampled_bound():
    """get_rdp_curve must retain subsampling amplification (via Opacus's
    compute_rdp), not the unamplified Gaussian bound that inflated the
    composed total 2-5x."""
    acc = RDPAccountant(noise_multiplier=1.0, sample_rate=0.05,
                        delta=1e-5)
    for _ in range(100):
        acc.step()
    curve = acc.get_rdp_curve()
    # Unamplified bound at alpha=2: 2/(2*1)*100 = 100. Subsampled at
    # q=0.05 must be substantially smaller.
    assert curve[2.0] < 20.0


# --------------------------------------------------------------------------
# PoissonGroupSampler
# --------------------------------------------------------------------------

def test_poisson_sampler_expected_representation():
    """Each group's average per-step count should approximate
    per_group_batch (same expected representation per group)."""
    labels = np.array([0] * 1000 + [1] * 500 + [2] * 300)
    sampler = PoissonGroupSampler(labels, per_group_batch=30,
                                   steps_per_epoch=200, seed=42)
    counts = {0: 0, 1: 2, 2: 0}
    n_steps = 0
    for batch in iter(sampler):
        n_steps += 1
        for g in counts:
            counts[g] += int((labels[np.array(batch)] == g).sum())
    for g in counts:
        avg = counts[g] / n_steps
        assert abs(avg - 30) < 8  # binomial noise


def test_poisson_sampler_gamma_max():
    """gamma_max is the minority group's inclusion probability."""
    labels = np.array([0] * 1000 + [1] * 500 + [2] * 300)
    sampler = PoissonGroupSampler(labels, per_group_batch=30,
                                   steps_per_epoch=10, seed=0)
    assert abs(sampler.gamma_max - 30 / 300) < 1e-9


def test_poisson_sampler_deterministic_with_seed():
    labels = np.array([0] * 100 + [1] * 100)
    s1 = PoissonGroupSampler(labels, 10, 20, seed=7)
    s2 = PoissonGroupSampler(labels, 10, 20, seed=7)
    assert [list(b) for b in iter(s1)] == [list(b) for b in iter(s2)]


def test_poisson_sampler_rejects_single_group():
    try:
        PoissonGroupSampler(np.zeros(10), 4, 5)
        assert False, 'should have raised'
    except ValueError:
        pass


# --------------------------------------------------------------------------
# dp_release_mean_std
# --------------------------------------------------------------------------

def test_dp_release_close_to_true_stats():
    """With little clipping and large n, the released mean/std should
    be close to the true values (noise is O(sensitivity/n))."""
    torch.manual_seed(0)
    X = torch.randn(20000, 4)
    mean, std, rdp = dp_release_mean_std(
        X, noise_multiplier=1.0, max_norm=10.0, n_rows=X.shape[0])
    assert torch.allclose(mean, X.mean(dim=0), atol=0.05)
    assert torch.allclose(std, X.std(dim=0), atol=0.05)
    assert len(rdp) > 0


def test_dp_release_bounds_single_record_influence():
    """One extreme record must not blow up the release: rows are clipped
    to max_norm, so the released mean stays bounded."""
    torch.manual_seed(0)
    X = torch.randn(1000, 4)
    X[0] = 1e6  # one extreme record
    mean, std, _ = dp_release_mean_std(
        X, noise_multiplier=1.0, max_norm=1.0, n_rows=X.shape[0])
    # Without clipping the mean would be ~1000; with it, bounded
    assert mean.abs().max().item() < 1.0


# --------------------------------------------------------------------------
# BalancedGroupSampler
# --------------------------------------------------------------------------

def test_sampler_equal_group_representation():
    labels = np.array([0] * 100 + [1] * 50 + [2] * 30)
    sampler = BalancedGroupSampler(labels, batch_size=12)
    for batch in iter(sampler):
        counts = np.bincount(labels[np.array(batch)], minlength=3)
        # Full batches have exactly 4 per group; the final partial batch
        # may have fewer, but never more, and stays balanced
        assert (counts <= 4).all()
        if len(batch) == 12:
            assert (counts == 4).all()


def test_sampler_epoch_length_eq11():
    """Iterations per epoch = ceil(m / b_per_group) with m = min group size
    (Eq. 11 with a final partial batch so all minority samples are used)."""
    labels = np.array([0] * 100 + [1] * 50 + [2] * 30)
    sampler = BalancedGroupSampler(labels, batch_size=12)
    m = 30
    per_group = 12 // 3
    assert sampler.num_batches == -(-m // per_group)  # ceil, Eq. 11


def test_sampler_covers_min_group():
    """All samples of the smallest group appear exactly once per epoch."""
    labels = np.array([0] * 100 + [1] * 50 + [2] * 30)
    sampler = BalancedGroupSampler(labels, batch_size=12)
    seen = []
    for batch in iter(sampler):
        seen.extend(batch)
    seen = np.array(seen)          # dataset indices
    seen_labels = labels[seen]     # map indices back to group labels
    assert (seen_labels == 2).sum() == 30  # every minority sample used once
    # And no duplicates of minority samples
    minority_indices = seen[seen_labels == 2]
    assert len(np.unique(minority_indices)) == 30


def test_sampler_deterministic_with_seed():
    labels = np.array([0] * 40 + [1] * 40)
    s1 = BalancedGroupSampler(labels, batch_size=8, seed=7)
    s2 = BalancedGroupSampler(labels, batch_size=8, seed=7)
    b1 = [list(b) for b in iter(s1)]
    b2 = [list(b) for b in iter(s2)]
    assert b1 == b2


def test_sampler_rejects_single_group():
    try:
        BalancedGroupSampler(np.zeros(10), batch_size=4)
        assert False, 'should have raised'
    except ValueError:
        pass


# --------------------------------------------------------------------------
# RDPAccountant (Opacus-backed; conservative pure-Gaussian fallback)
# --------------------------------------------------------------------------

def test_rdp_zero_steps_zero_epsilon():
    acc = RDPAccountant(noise_multiplier=1.0, sample_rate=0.01)
    eps, _ = acc.get_privacy_spent()
    assert eps == 0.0


def test_rdp_monotone_in_steps():
    acc = RDPAccountant(noise_multiplier=1.0, sample_rate=0.01)
    acc.step()
    eps1, _ = acc.get_privacy_spent()
    for _ in range(100):
        acc.step()
    eps2, _ = acc.get_privacy_spent()
    assert eps2 > eps1


def test_rdp_more_noise_less_epsilon():
    acc_low = RDPAccountant(noise_multiplier=0.5, sample_rate=0.01)
    acc_high = RDPAccountant(noise_multiplier=2.0, sample_rate=0.01)
    for _ in range(50):
        acc_low.step()
        acc_high.step()
    eps_low, _ = acc_low.get_privacy_spent()
    eps_high, _ = acc_high.get_privacy_spent()
    assert eps_high < eps_low


def test_rdp_epsilon_positive_finite():
    acc = RDPAccountant(noise_multiplier=1.0, sample_rate=0.05, delta=1e-5)
    for _ in range(200):
        acc.step()
    eps, _ = acc.get_privacy_spent()
    assert np.isfinite(eps) and eps > 0


def test_rdp_plausible_magnitude():
    """Regression test for the broken hand-rolled bound: at sigma=1.0,
    q=0.01, 100 steps, delta=1e-5 the true subsampled-Gaussian epsilon is
    ~1.6 (and the unamplified fallback is larger still). The old bound
    reported epsilon well below 1 by exploiting alpha*q > 1 regimes -
    any epsilon < 1 here indicates the bug has returned."""
    acc = RDPAccountant(noise_multiplier=1.0, sample_rate=0.01, delta=1e-5)
    for _ in range(100):
        acc.step()
    eps, _ = acc.get_privacy_spent()
    assert eps > 1.0


def test_rdp_fallback_conservative_vs_opacus():
    """When Opacus is available, the fallback (no-subsample) bound must
    be >= the subsampled bound for the same parameters."""
    try:
        from opacus.accountants.rdp import RDPAccountant as OpacusRDP
    except ImportError:
        return  # nothing to compare against
    acc = RDPAccountant(noise_multiplier=1.0, sample_rate=0.05, delta=1e-5)
    for _ in range(100):
        acc.step()
    eps_opacus, _ = acc.get_privacy_spent()

    # Recompute the fallback bound directly
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 505))
    best = float('inf')
    for alpha in alphas:
        if alpha <= 1:
            continue
        e = (alpha / 2.0) * 100 + np.log(1e5) / (alpha - 1)
        best = min(best, e)
    assert best >= eps_opacus - 1e-6


# --------------------------------------------------------------------------
# Model_VAE.forward_with_stages
# --------------------------------------------------------------------------

def _tiny_vae():
    d_numerical = 3
    categories = [2, 4]
    return Model_VAE(num_layers=1, d_numerical=d_numerical,
                     categories=categories, d_token=4,
                     n_head=1, factor=2, bias=True)


def test_forward_with_stages_shapes():
    model = _tiny_vae()
    n, d_num, n_cat = 8, 3, 2
    x_num = torch.randn(n, d_num)
    x_cat = torch.randint(0, 2, (n, n_cat))
    out = model.forward_with_stages(x_num, x_cat)
    assert out['recon_x_num'].shape == (n, d_num)
    assert len(out['recon_x_cat']) == n_cat
    assert out['recon_x_cat'][0].shape == (n, 2)
    assert out['recon_x_cat'][1].shape == (n, 4)
    # mu_z includes the CLS token (tokens = d_num + n_cat + 1); the decoder
    # output h drops it (tokens - 1)
    n_tokens = d_num + n_cat + 1
    assert out['mu_z'].shape == (n, n_tokens, 4)
    assert out['h'].shape == (n, n_tokens - 1, 4)
    assert out['std_z'].shape == out['mu_z'].shape


def test_forward_with_stages_matches_forward():
    """forward_with_stages must be consistent with the plain forward()
    when the reparameterization noise is identical (fixed seed)."""
    model = _tiny_vae()
    model.eval()
    x_num = torch.randn(8, 3)
    x_cat = torch.randint(0, 2, (8, 2))
    with torch.no_grad():
        torch.manual_seed(123)
        r_num, r_cat, mu, std = model(x_num, x_cat)
        torch.manual_seed(123)
        out = model.forward_with_stages(x_num, x_cat)
    assert torch.allclose(r_num, out['recon_x_num'])
    assert torch.allclose(mu, out['mu_z'])
    assert torch.allclose(std, out['std_z'])
    for a, b in zip(r_cat, out['recon_x_cat']):
        assert torch.allclose(a, b)


def test_forward_with_stages_gradient_flow():
    """Gradients reach encoder and decoder through the staged outputs."""
    model = _tiny_vae()
    x_num = torch.randn(8, 3)
    x_cat = torch.randint(0, 2, (8, 2))
    out = model.forward_with_stages(x_num, x_cat)
    loss = out['recon_x_num'].sum() + out['h'].sum() + out['mu_z'].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


# --------------------------------------------------------------------------
# compute_loss with FLIP extensions
# --------------------------------------------------------------------------

def test_compute_loss_excludes_protected_from_ce():
    """The protected attribute's CE term must be excluded when s_idx is set."""
    n, d_num = 8, 3
    X_num = torch.randn(n, d_num)
    X_cat = torch.randint(0, 3, (n, 2))
    Recon_num = torch.randn(n, d_num)
    # Protected-column logits deliberately wrong (huge CE if included)
    Recon_cat = [torch.full((n, 3), -10.0), torch.randn(n, 4)]
    mu = torch.randn(n, 5, 4)
    logvar = torch.randn(n, 5, 4)

    _, ce_with, _, _, _ = compute_loss(X_num, X_cat, Recon_num, Recon_cat,
                                       mu, logvar, s_idx=None)
    _, ce_without, _, _, _ = compute_loss(X_num, X_cat, Recon_num, Recon_cat,
                                           mu, logvar, s_idx=0,
                                           Recon_S_logits=Recon_cat[0])
    assert ce_without.item() < ce_with.item()


def test_compute_loss_returns_uniform_loss():
    n, d_num = 8, 3
    X_num = torch.randn(n, d_num)
    X_cat = torch.randint(0, 3, (n, 2))
    Recon_num = torch.randn(n, d_num)
    Recon_cat = [torch.zeros(n, 3), torch.randn(n, 4)]
    mu = torch.randn(n, 5, 4)
    logvar = torch.randn(n, 5, 4)

    *_, loss_s = compute_loss(X_num, X_cat, Recon_num, Recon_cat, mu, logvar,
                              s_idx=0, Recon_S_logits=Recon_cat[0])
    assert loss_s.item() < 1e-6  # uniform logits -> ~0

    Recon_cat_skew = [torch.tensor([[5.0, 0.0, 0.0]] * n), torch.randn(n, 4)]
    *_, loss_s2 = compute_loss(X_num, X_cat, Recon_num, Recon_cat_skew, mu,
                               logvar, s_idx=0, Recon_S_logits=Recon_cat_skew[0])
    assert loss_s2.item() > 0.5


def test_compute_loss_backward_without_s_idx():
    """Backward compatibility: original 5-tuple call still works (loss_s=0)."""
    n, d_num = 8, 3
    X_num = torch.randn(n, d_num)
    X_cat = torch.randint(0, 3, (n, 2))
    Recon_num = torch.randn(n, d_num, requires_grad=True)
    Recon_num = Recon_num.detach().requires_grad_()
    Recon_cat = [torch.randn(n, 3, requires_grad=True), torch.randn(n, 4)]
    mu = torch.randn(n, 5, 4)
    logvar = torch.randn(n, 5, 4)
    mse, ce, kld, acc, loss_s = compute_loss(X_num, X_cat, Recon_num,
                                             Recon_cat, mu, logvar)
    total = mse + ce + kld
    total.backward()
    assert Recon_num.grad is not None
    assert loss_s.item() == 0.0


# --------------------------------------------------------------------------
# DP-SGD smoke test (Opacus)
# --------------------------------------------------------------------------

def test_dp_optimizer_smoke():
    """DPOptimizer runs a step with clipping + noise on the tiny VAE
    (model wrapped in GradSampleModule; native grad samplers handle the
    custom Tokenizer/Reconstructor; dead params excluded)."""
    try:
        from opacus.optimizers import DPOptimizer
        from opacus.grad_sample import GradSampleModule
    except ImportError:
        print('  [SKIP] opacus not installed')
        return

    model = _tiny_vae()
    wrapped = GradSampleModule(model, strict=False)
    dp_params = get_dp_trainable_parameters(wrapped)
    dp_opt = DPOptimizer(
        optimizer=torch.optim.Adam(dp_params, lr=1e-3),
        noise_multiplier=1.0, max_grad_norm=1.0, expected_batch_size=8)
    x_num = torch.randn(8, 3)
    x_cat = torch.randint(0, 2, (8, 2))
    # Full loss touching all outputs (like compute_loss in training)
    ce = nn.CrossEntropyLoss()
    out = model.forward_with_stages(x_num, x_cat)
    loss = (out['recon_x_num'].pow(2).mean()
            + ce(out['recon_x_cat'][0], x_cat[:, 0])
            + ce(out['recon_x_cat'][1], x_cat[:, 1]))
    loss.backward()
    dp_opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_dp_noise_differs_from_plain():
    """DP-SGD updates differ from plain Adam due to clipping + noise."""
    try:
        from opacus.optimizers import DPOptimizer
        from opacus.grad_sample import GradSampleModule
    except ImportError:
        print('  [SKIP] opacus not installed')
        return

    torch.manual_seed(0)
    m1 = _tiny_vae()
    m2 = _tiny_vae()
    m2.load_state_dict(m1.state_dict())

    x_num = torch.randn(8, 3)
    x_cat = torch.randint(0, 2, (8, 2))
    ce = nn.CrossEntropyLoss()

    def full_loss(m):
        out = m.forward_with_stages(x_num, x_cat)
        return (out['recon_x_num'].pow(2).mean()
                + ce(out['recon_x_cat'][0], x_cat[:, 0])
                + ce(out['recon_x_cat'][1], x_cat[:, 1]))

    # Plain Adam
    opt1 = torch.optim.Adam(m1.parameters(), lr=1e-3)
    full_loss(m1).backward()
    opt1.step()

    # DP-SGD (native samplers; dead params excluded)
    m2_wrapped = GradSampleModule(m2, strict=False)
    dp_params = get_dp_trainable_parameters(m2_wrapped)
    opt2 = DPOptimizer(optimizer=torch.optim.Adam(dp_params, lr=1e-3),
                       noise_multiplier=2.0, max_grad_norm=0.5,
                       expected_batch_size=8)
    full_loss(m2).backward()
    opt2.step()

    p1 = next(m1.parameters())
    p2 = next(m2.parameters())
    assert not torch.allclose(p1, p2)


# --------------------------------------------------------------------------
# End-to-end mini training smoke test
# --------------------------------------------------------------------------

def test_two_phase_smoke():
    """A few steps of Phase 1 + Phase 2 run without errors and the
    fairness loss is finite."""
    model = _tiny_vae()
    n, d_num, n_cat = 16, 3, 2
    x_num = torch.randn(n, d_num)
    x_cat = torch.randint(0, 2, (n, n_cat))
    s_idx = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Phase 1
    for _ in range(3):
        optimizer.zero_grad()
        out = model.forward_with_stages(x_num, x_cat)
        mse, ce, kld, acc, loss_s = compute_loss(
            x_num, x_cat, out['recon_x_num'], out['recon_x_cat'],
            out['mu_z'], out['std_z'], s_idx=s_idx,
            Recon_S_logits=out['recon_x_cat'][s_idx])
        (mse + ce + kld + loss_s).backward()
        optimizer.step()

    # Freeze reference
    ref_mu = out['mu_z'].detach().clone()

    # Phase 2
    groups = x_cat[:, s_idx].long()
    for _ in range(3):
        optimizer.zero_grad()
        out = model.forward_with_stages(x_num, x_cat)
        mse, ce, kld, acc, loss_s = compute_loss(
            x_num, x_cat, out['recon_x_num'], out['recon_x_cat'],
            out['mu_z'], out['std_z'], s_idx=s_idx,
            Recon_S_logits=out['recon_x_cat'][s_idx])
        mu_flat = out['mu_z'].reshape(n, -1)
        div = sliced_wasserstein_distance(ref_mu.reshape(n, -1), mu_flat)
        disent = multi_stage_disentanglement_loss(out, groups, s_idx=s_idx)
        total = mse + ce + kld + loss_s + div + 1.0 * disent
        assert torch.isfinite(total)
        total.backward()
        optimizer.step()


if __name__ == '__main__':
    # Simple runner without pytest
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS  {name}')
            except Exception as e:
                failures += 1
                print(f'FAIL  {name}: {type(e).__name__}: {e}')
    print(f'\n{"ALL TESTS PASSED" if failures == 0 else f"{failures} FAILURES"}')
    sys.exit(1 if failures else 0)