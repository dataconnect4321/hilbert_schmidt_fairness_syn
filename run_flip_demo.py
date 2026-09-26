"""Run FLIP (fair + private TabSyn VAE) on a sample of bmo_nationwide.csv
and evaluate the fidelity / privacy / fairness trade-off.

Pipeline:
  1. Load a subsample of the preprocessed HMDA data (from preprocess.py).
  2. Train the FLIP-modified TabSyn VAE:
       Phase 1 (quality):    beta-VAE loss + uniform attribute loss L_S
       Phase 2 (fairness):   + SWD divergence penalty + lambda * CKA^T
                             disentanglement at latent/detokenizer/decoder
     with balanced group sampling and optional DP-SGD (Renyi DP).
  3. Generate synthetic records from the learned latent space.
  4. Score the result:
       Fidelity:  per-column Wasserstein-1 (numeric) / TV distance (cat),
                  correlation-matrix preservation.
       Privacy:   DCR ratio - nearest-neighbor distance from synthetic to
                  real vs. real (holdout) to real. ~1.0 is healthy.
       Fairness:  demographic parity difference/ratio of the synthetic
                 origination rate across protected groups, compared to
                 the real data's disparity.

The TRADEOFF_PARAMS block below is the knob panel: each parameter moves
the fidelity/privacy/fairness triangle in a documented direction.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'tabsyn-main', 'tabsyn-main'))

from tabsyn.vae.model import Model_VAE
from tabsyn.vae.flip_fairness import (
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
from tabsyn.vae.main import compute_loss

# ==========================================================================
# TRADEOFF PARAMETERS - the fidelity / privacy / fairness knob panel
# ==========================================================================
TRADEOFF_PARAMS = {
    # ---- FAIRNESS knobs -------------------------------------------------
    # Weight of the CKA^T disentanglement term (Phase 2). Higher lambda ->
    # stronger bias removal in the latent space, but pushing too far
    # degrades reconstruction fidelity (paper Fig. 3: fairness-quality
    # trade-off). 0 disables explicit disentanglement.
    'LAMBDA_FAIR': 2.0,

    # Epochs of quality-only training before disentanglement starts.
    # More Phase-1 epochs -> better representation first, gentler
    # fairness intervention afterwards.
    'PHASE1_EPOCHS': 100,
    'PHASE2_EPOCHS': 60,

    # ---- PRIVACY knobs --------------------------------------------------
    # DP-SGD Gaussian noise multiplier (sigma). Higher sigma -> stronger
    # (epsilon, delta) privacy guarantee, but noisier gradients -> lower
    # fidelity, and per the paper, noise also weakens the fairness
    # intervention's effectiveness (Sec. 6.3).
    'NOISE_MULTIPLIER': 0.8,

    # Per-sample gradient L2 clipping norm. Smaller -> tighter privacy
    # accounting per step but slower/harder optimization.
    'MAX_GRAD_NORM': 1.0,

    # Target delta for the (epsilon, delta)-DP guarantee.
    'DP_DELTA': 1e-5,

    # Master switch for DP-SGD. Turning this off isolates the fairness
    # intervention's effect without privacy noise.
    'USE_DP': True,

    # ---- FIDELITY knobs -------------------------------------------------
    # KL-divergence weight (beta-VAE). Lower beta -> better reconstruction
    # (fidelity) at the cost of a less regular latent space.
    'BETA': 1e-3,

    # ---- STABILITY knobs -------------------------------------------------
    # Cap on the SWD divergence penalty. Without this, the penalty grows
    # unboundedly as the Phase-2 encoder drifts from the frozen reference
    # (compounded by DP-noise gradient perturbations), which can explode
    # the loss and destabilize long runs.
    'MAX_DIV_PENALTY': 10.0,
    # Group-level DP mechanism knobs (Phase-2 batch-coupled fairness
    # terms). The SWD anchor and CKA^T disentanglement are batch-level
    # functionals: one record changes every other record's gradient, so
    # record-level DP-SGD cannot cover them. Instead their gradient is
    # clipped to MAX_GROUP_GRAD_NORM (bounding sensitivity to any one
    # protected group) and noised at GROUP_NOISE_MULTIPLIER, giving a
    # GROUP-level (epsilon, delta) guarantee, accounted separately from
    # the record-level DP-SGD budget and reported alongside it.
    'MAX_GROUP_GRAD_NORM': 0.5,
    'GROUP_NOISE_MULTIPLIER': 3.0,

    # DP release of generation-time statistics (latent mean/std and the
    # numeric clamp bounds). Computing these directly on training data
    # would be a fresh non-private computation on raw records (min/max
    # are especially sensitive). Each release is a Gaussian mechanism
    # with row contributions clipped to STATS_MAX_NORM and noise
    # STATS_NOISE_MULTIPLIER, spent once (not per step) into the budget.
    'STATS_MAX_NORM': 10.0,
    'STATS_NOISE_MULTIPLIER': 1.0,
    # Multiple of released std for the DP range proxy (mean +/- k*std);
    # order statistics (true min/max) cannot be released privately with
    # bounded sensitivity.
    'STATS_RANGE_K': 4.0,
    # Learning-rate scale applied when Phase 2 begins. Smaller Phase-2
    # steps keep the fairness intervention from destroying the Phase-1
    # representation (the paper's 'controlled bias mitigation').
    'PHASE2_LR_SCALE': 0.2,

    # Adam's epsilon. DP-SGD noise gives every parameter a nonzero
    # gradient variance even when the true signal is ~0; Adam's default
    # eps=1e-8 lets 1/sqrt(v) blow up on that noise. A larger eps damps
    # this and is the main fix for exploding loss under DP-SGD.
    'ADAM_EPS': 1e-4,

    # Hard cap on the aggregated (post-clip, post-noise) gradient norm
    # right before the optimizer step - a safety net beyond Opacus's
    # per-sample clipping, which only bounds each sample's contribution,
    # not the noisy sum actually applied to the weights.
    'MAX_OPTIMIZER_GRAD_NORM': 5.0,

    # L2 weight decay. DP-SGD injects Gaussian noise into every gradient
    # step regardless of the true signal, which makes weights undergo an
    # unbounded random walk over many epochs. Weight decay adds a
    # restoring force pulling weights back toward zero, which is the
    # standard fix for this drift (gradient clipping alone does not help,
    # since large logits saturate softmax and keep the CE gradient small
    # even while the loss *value* explodes).
    'WEIGHT_DECAY': 1e-2,

    # Clamp applied to reconstructed categorical logits before the
    # cross-entropy loss, as a hard backstop against the same failure
    # mode: even one drifted weight can produce a huge logit whose CE
    # loss value is enormous despite a small, saturated gradient.
    'LOGIT_CLAMP': 15.0,

    # Clamp applied to the reconstructed numeric features before the MSE
    # loss. Unlike CE, squared error does NOT saturate - its gradient
    # grows linearly with the error - so a drifted weight here keeps
    # feeding a larger and larger gradient back in. This closes that loop.
    'NUM_RECON_CLAMP': 8.0,

    # ---- Data / compute knobs -------------------------------------------
    'SAMPLE_SIZE': 12000,      # rows drawn from the preprocessed CSV
    'BATCH_SIZE': 1024,        # split evenly across protected groups
    # Sampler: 'poisson' (default) makes each record's inclusion
    # independent per step, which is exactly what Opacus's subsampled-
    # Gaussian accounting assumes. 'balanced' uses the shuffle-based
    # BalancedGroupSampler (fixed-size group-balanced batches, every
    # minority sample once per epoch) - Chua et al. (ICML 2024) showed
    # shuffle-based DP-SGD can leak more than Poisson accounting
    # reports, so the Poisson accounting is NOT formally valid for it.
    'SAMPLER': 'poisson',
    # Poisson steps per epoch (only used when SAMPLER='poisson').
    # Match the balanced sampler's epoch length: ceil(m / per_group).
    'POISSON_STEPS_PER_EPOCH': None,  # None -> ceil(m / per_group)
    'D_TOKEN': 8,
    'NUM_LAYERS': 2,
    'LR': 1e-3,
    'SEED': 42,

    # 'cuda' if available, else 'cpu' (set in main())
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
}

SENSITIVE_COL = 'derived_race'  # protected attribute - most biased in this
                               # data (White 0.61 vs minority ~0.32-0.43
                               # origination rates; DP ratio ~0.53)
TARGET_COL = 'target'          # 1 = originated, 0 = denied (from preprocess.py)

# Use the three largest race groups so balanced batches have enough
# samples per group (Joint / Not Available / tiny groups excluded)
RACE_GROUPS = ['White', 'Black or African American', 'Asian']

NUMERIC_COLS = ['loan_amount', 'income', 'loan_to_value_ratio',
                'property_value', 'tract_population']
CATEGORICAL_COLS = [SENSITIVE_COL, TARGET_COL, 'conforming_loan_limit',
                    'derived_loan_product_type']

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'Data', 'bmo_nationwide_preprocessed.csv')


# ==========================================================================
# Data preparation
# ==========================================================================
def load_sample(params):
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Keep the protected groups with enough samples for balanced batches
    df = df[df[SENSITIVE_COL].isin(RACE_GROUPS)]

    # SAMPLE_SIZE=None -> use every row of the filtered groups
    if params['SAMPLE_SIZE'] is not None:
        df = df.sample(n=min(params['SAMPLE_SIZE'], len(df)),
                       random_state=params['SEED'])
    df = df.reset_index(drop=True)

    # Numeric: coerce, impute median, standardize
    num = df[NUMERIC_COLS].apply(pd.to_numeric, errors='coerce')
    num = num.fillna(num.median())
    num = num.fillna(0)
    num_means = num.mean()
    num_stds = num.std().replace(0, 1)
    num = ((num - num_means) / num_stds).values.astype(np.float32)

    # Categorical: label-encode (fill missing with a placeholder category
    # first - pd.factorize maps NaN to -1 which breaks CE loss targets)
    cat_encoders = {}
    cat_cols = []
    for c in CATEGORICAL_COLS:
        vals = df[c].astype(str).fillna('Missing').replace('nan', 'Missing')
        codes, uniques = pd.factorize(vals)
        cat_encoders[c] = uniques
        cat_cols.append(codes.astype(np.int64))
    cat = np.stack(cat_cols, axis=1)

    categories = [len(u) for u in cat_encoders.values()]
    s_idx = CATEGORICAL_COLS.index(SENSITIVE_COL)
    t_idx = CATEGORICAL_COLS.index(TARGET_COL)

    return (torch.tensor(num), torch.tensor(cat), categories,
            s_idx, t_idx, cat_encoders, df, num_means, num_stds)


# ==========================================================================
# FLIP training (two-phase, balanced sampling, optional DP-SGD)
# ==========================================================================
def train_flip(X_num, X_cat, categories, s_idx, params, device):
    torch.manual_seed(params['SEED'])
    n, d_numerical = X_num.shape

    model = Model_VAE(params['NUM_LAYERS'], d_numerical, categories,
                      params['D_TOKEN'], n_head=1, factor=8,
                      bias=True).to(device)

    groups = X_cat[:, s_idx].long().numpy()
    per_group = params['BATCH_SIZE'] // len(np.unique(groups))
    if params.get('SAMPLER', 'poisson') == 'poisson':
        # Per-group Poisson sampling: each record included independently
        # with per-group probability q_g = per_group / m_g. This makes the
        # Poisson sampling assumption of Opacus's RDP bound exactly true
        # (the shuffle-based sampler violates it; Chua et al. ICML 2024).
        m = int(np.bincount(groups).min())
        steps_per_epoch = params.get('POISSON_STEPS_PER_EPOCH') or \
            int(np.ceil(m / per_group))
        sampler = PoissonGroupSampler(groups, per_group,
                                       steps_per_epoch,
                                       seed=params['SEED'])
        print(f'Poisson sampling: {steps_per_epoch} steps/epoch, '
              f'gamma_max={sampler.gamma_max:.4f}')
    else:
        sampler = BalancedGroupSampler(groups, params['BATCH_SIZE'],
                                       seed=params['SEED'])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_num, X_cat),
        batch_sampler=sampler)

    accountant = None
    group_mech = None
    if params['USE_DP']:
        from opacus.optimizers import DPOptimizer
        from opacus.grad_sample import GradSampleModule
        register_tabsyn_grad_samplers()
        model = GradSampleModule(model, strict=False)
        dp_params = get_dp_trainable_parameters(model)
        optimizer = DPOptimizer(
            optimizer=torch.optim.Adam(dp_params, lr=params['LR'],
                                       eps=params['ADAM_EPS'],
                                       weight_decay=params['WEIGHT_DECAY']),
            noise_multiplier=params['NOISE_MULTIPLIER'],
            max_grad_norm=params['MAX_GRAD_NORM'],
            expected_batch_size=params['BATCH_SIZE'])
        # Sampling rate for the accountant: gamma_max, the largest
        # per-record inclusion probability across groups. Under the
        # Poisson sampler this is exactly the minority group's q; under
        # the balanced sampler it is the conservative per-step rate
        # (every minority sample appears once per epoch). Either way it
        # is the rate that must be accounted, not batch/N.
        if isinstance(sampler, PoissonGroupSampler):
            gamma_max = sampler.gamma_max
        else:
            m_minority = int(np.bincount(groups).min())
            gamma_max = per_group / m_minority
        accountant = RDPAccountant(
            noise_multiplier=params['NOISE_MULTIPLIER'],
            sample_rate=gamma_max,
            delta=params['DP_DELTA'])
        # Gaussian mechanism for the batch-coupled fairness terms
        # (SWD anchor + CKA^T): their gradients cannot be decomposed per
        # sample, so record-level DP-SGD does not cover them. Clipping
        # bounds the sensitivity (2C under substitution); the noise
        # gives a record-level guarantee. Under the Poisson sampler the
        # subsampling amplification applies to this mechanism too, so
        # gamma_max is passed for the amplified accounting (the largest
        # remaining win: the fairness term drops several-fold).
        group_mech = GroupLevelDPMechanism(
            max_group_grad_norm=params['MAX_GROUP_GRAD_NORM'],
            noise_multiplier=params['GROUP_NOISE_MULTIPLIER'],
            sample_rate=gamma_max if params.get('SAMPLER', 'poisson') == 'poisson' else None)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['LR'],
                                     eps=params['ADAM_EPS'],
                                     weight_decay=params['WEIGHT_DECAY'])

    raw = model._module if params['USE_DP'] else model
    beta = params['BETA']
    total_epochs = params['PHASE1_EPOCHS'] + params['PHASE2_EPOCHS']
    reference_encoder = None

    for epoch in range(total_epochs):
        phase = 1 if epoch < params['PHASE1_EPOCHS'] else 2

        # Phase-2 LR drop: shrink the step size once disentanglement
        # starts so fairness updates perturb the quality representation
        # gently instead of destroying it.
        if phase == 2 and epoch == params['PHASE1_EPOCHS']:
            new_lr = params['LR'] * params['PHASE2_LR_SCALE']
            for g in optimizer.param_groups:
                g['lr'] = new_lr
            print(f'--- Phase 2: LR dropped to {new_lr:.2e} ---')

        epoch_loss = 0.0
        n_batches = 0

        for bnum, bcat in loader:
            bnum, bcat = bnum.to(device), bcat.to(device)
            model.train()
            optimizer.zero_grad()

            stages = raw.forward_with_stages(bnum, bcat)
            # Clamp reconstructions before the loss: a single drifted
            # weight can send a logit/numeric output to a huge value.
            # CE saturates on huge logits (small gradient despite a huge
            # loss value), but MSE does not, so both need a backstop -
            # see WEIGHT_DECAY / NUM_RECON_CLAMP notes above.
            clamp = params['LOGIT_CLAMP']
            recon_cat_clamped = [torch.clamp(c, -clamp, clamp)
                                 for c in stages['recon_x_cat']]
            recon_num_clamped = torch.clamp(
                stages['recon_x_num'], -params['NUM_RECON_CLAMP'],
                params['NUM_RECON_CLAMP'])
            mse, ce, kld, acc, loss_s = compute_loss(
                bnum, bcat, recon_num_clamped, recon_cat_clamped,
                stages['mu_z'], stages['std_z'], s_idx=s_idx,
                Recon_S_logits=recon_cat_clamped[s_idx])

            # ---- Pass 1: per-sample-decomposable loss (record-level DP) --
            # mse + ce + kld + L_S are all sums/means of per-sample terms
            # (L_S uses the per-sample variant), so Opacus's per-sample
            # clipping bounds each record's influence on this gradient.
            loss_private = mse + ce + beta * kld + loss_s
            loss_private.backward(retain_graph=phase == 2)

            # ---- Pass 2: batch-coupled fairness terms (group-level DP) --
            # The SWD anchor and CKA^T disentanglement mix samples across
            # the batch; their gradient cannot be attributed per sample.
            # Under DP they are handled by the GroupLevelDPMechanism
            # (clip to a fixed norm, add group-calibrated noise) instead
            # of Opacus, giving a group-level rather than record-level
            # guarantee for these terms.
            fair_grads = None
            if phase == 2:
                mu_flat = stages['mu_z'].reshape(bnum.shape[0], -1)
                # Divergence penalty vs. the frozen Phase-1 encoder:
                # at the phase boundary, snapshot the encoder weights into
                # a separate frozen module (no in-place weight swapping).
                if reference_encoder is None:
                    from tabsyn.vae.model import Encoder_model
                    reference_encoder = Encoder_model(
                        params['NUM_LAYERS'],
                        model.Reconstructor.d_numerical,
                        categories, params['D_TOKEN'],
                        n_head=1, factor=8).to(device)
                    reference_encoder.Tokenizer.load_state_dict(
                        raw.VAE.Tokenizer.state_dict())
                    reference_encoder.VAE_Encoder.load_state_dict(
                        raw.VAE.encoder_mu.state_dict())
                    reference_encoder.eval()
                    for p_ in reference_encoder.parameters():
                        p_.requires_grad_(False)
                with torch.no_grad():
                    ref_mu = reference_encoder(bnum, bcat)
                ref_mu_flat = ref_mu.reshape(bnum.shape[0], -1)
                disent = multi_stage_disentanglement_loss(
                    stages, bcat[:, s_idx].long(), s_idx=s_idx)
                # Clamp the divergence penalty: without a cap the SWD term
                # grows unboundedly as the encoder drifts from the frozen
                # reference under DP noise, exploding the loss.
                div = sliced_wasserstein_distance(ref_mu_flat, mu_flat)
                div = torch.clamp(div, max=params['MAX_DIV_PENALTY'])
                loss_fair = div + params['LAMBDA_FAIR'] * disent

                if group_mech is not None:
                    # Capture the fairness gradient WITHOUT Opacus's hooks:
                    # they would try to pop activation buffers already
                    # consumed by pass 1 (IndexError) and misattribute the
                    # batch-coupled gradient per sample. disable_hooks() -
                    # NOT remove_hooks(), which is permanent in this Opacus
                    # version - makes the capture hooks return early while
                    # staying registered; enable_hooks() restores them for
                    # the next batch's pass 1. The captured gradient is
                    # clipped+noised by the group-level mechanism and added
                    # to the aggregate after Opacus's pre_step() below.
                    model.disable_hooks()
                    try:
                        fair_grads = torch.autograd.grad(
                            loss_fair, dp_params, allow_unused=True)
                    finally:
                        model.enable_hooks()
                    fair_grads = {
                        p_: g for p_, g in zip(dp_params, fair_grads)
                        if g is not None}
                else:
                    # Non-DP: accumulate into the same graph
                    loss_fair.backward()

            # Safety-net clip on the aggregated update: Opacus's
            # max_grad_norm only bounds each per-sample gradient before
            # summation/noising, not the final noisy gradient Adam sees.
            if params['USE_DP']:
                if optimizer.pre_step():
                    # Add the group-level noised fairness gradient to the
                    # record-level aggregate so one step applies both.
                    if fair_grads:
                        group_mech.add_noised(dp_params, fair_grads)
                    torch.nn.utils.clip_grad_norm_(
                        dp_params, params['MAX_OPTIMIZER_GRAD_NORM'])
                    optimizer.original_optimizer.step()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params['MAX_OPTIMIZER_GRAD_NORM'])
                optimizer.step()
            if accountant is not None:
                accountant.step()
            epoch_loss += (loss_private + (loss_fair if phase == 2 else 0)).item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == total_epochs - 1:
            msg = f'epoch {epoch:3d} [phase {phase}] loss={epoch_loss/max(1,n_batches):.4f}'
            if accountant is not None:
                eps, alpha = accountant.get_privacy_spent()
                msg += f'  eps_record={eps:.2f}'
            if group_mech is not None and group_mech.steps > 0:
                geps, _ = group_mech.get_privacy_spent(params['DP_DELTA'])
                msg += f'  eps_group={geps:.2f}'
            print(msg)

    return raw, accountant, group_mech


# ==========================================================================
# Generation
# ==========================================================================
@torch.no_grad()
def generate(model, n_gen, device, params, X_num=None, X_cat=None,
             stats_release=None):
    """Sample latents from a per-dimension Gaussian fit to the encoded
    training latents, then decode. (The full TabSyn pipeline trains a
    diffusion prior on latents; a Gaussian stand-in keeps this demo
    self-contained.)

    The Gaussian is fit to DP-RELEASED latent statistics (per-dim
    mean/std of the encoded training data), NOT to a standard normal:
    with a small beta the KL term barely regularizes the latent space,
    so the actual latents are far from N(0, I). Sampling standard-normal
    latents feeds the decoder out-of-distribution inputs, which makes it
    extrapolate and produce extreme (e.g. negative after de-normalizing)
    numeric values.

    PRIVACY: the empirical latent mean/std are computed on the training
    data, which is a fresh computation on raw records - NOT post-
    processing of the private model. They are therefore released through
    the Gaussian mechanism (dp_release_mean_std): each row's latent is
    clipped to STATS_MAX_NORM, the sums are noised at
    STATS_NOISE_MULTIPLIER, and the release's RDP is RETURNED so the
    caller composes it into the total budget (compose_rdp_budgets).
    Pass the pre-computed release via stats_release=(mean, std) to avoid
    recomputing it.

    Returns (syn_num, syn_cat, stats_budgets) where stats_budgets is a
    LIST of (alpha, rdp) pair lists - one per statistics release (latent
    mean/std, range proxy) - for composition. Each release is a separate
    budget: merging them into one pair list would create duplicate alpha
    keys that compose_rdp_budgets's dict conversion would silently
    collapse (keeping only the last release per alpha).
    """
    model.eval()
    # Probe latent shape
    dummy_num = torch.zeros(2, model.Reconstructor.d_numerical, device=device)
    dummy_cat = torch.zeros(2, len(model.Reconstructor.cat_recons),
                           dtype=torch.long, device=device)
    probe = model.forward_with_stages(dummy_num, dummy_cat)
    tokens, d = probe['mu_z'].shape[1], probe['mu_z'].shape[2]

    stats_budgets = []
    if X_num is not None and X_cat is not None:
        if stats_release is not None:
            lat_mean, lat_std = stats_release
        else:
            # DP release of the empirical latent statistics
            enc = model.forward_with_stages(X_num.to(device), X_cat.to(device))
            mu_z = enc['mu_z'].reshape(-1, tokens * d)
            lat_mean, lat_std, rdp1 = dp_release_mean_std(
                mu_z, params['STATS_NOISE_MULTIPLIER'],
                params['STATS_MAX_NORM'], mu_z.shape[0])
            stats_budgets.append(rdp1)
            lat_mean = lat_mean.reshape(1, tokens, d)
            lat_std = lat_std.reshape(1, tokens, d)
        mu = lat_mean + lat_std * torch.randn(
            n_gen, tokens, d, device=device)
    else:
        # Fallback: standard normal (kept for API compatibility)
        mu = torch.randn(n_gen, tokens, d, device=device)

    h = model.VAE.decoder(mu[:, 1:])
    recon_num, recon_cat = model.Reconstructor(h)

    syn_num = recon_num.cpu().numpy()
    syn_cat = np.stack([c.argmax(dim=-1).cpu().numpy() for c in recon_cat],
                       axis=1)

    # Backstop: clip synthetic numerics to a DP-RELEASED range proxy.
    # The original implementation clipped to the real data's per-column
    # min/max - a fresh non-private computation on raw records, and
    # min/max are especially sensitive (one extreme record moves them
    # arbitrarily). Order statistics cannot be released with bounded
    # sensitivity, so instead we release mean +/- k*std under the same
    # Gaussian mechanism (a standard DP range surrogate) and clip to
    # that. k = STATS_RANGE_K (default 4) covers ~all of a Gaussian's mass.
    if X_num is not None:
        Xn = X_num.to(device)
        r_mean, r_std, rdp2 = dp_release_mean_std(
            Xn, params['STATS_NOISE_MULTIPLIER'],
            params['STATS_MAX_NORM'], Xn.shape[0])
        stats_budgets.append(rdp2)
        k = params['STATS_RANGE_K']
        lo = (r_mean - k * r_std).cpu().numpy()
        hi = (r_mean + k * r_std).cpu().numpy()
        syn_num = np.clip(syn_num, lo[None, :], hi[None, :])

    # Domain-knowledge clamp: loan amounts, incomes, property values
    # and LTV ratios are non-negative by definition. The mean +/- k*std
    # proxy above can still dip below zero for skewed columns (the
    # committed report showed an income of -31.15); clamping at 0 uses
    # only PUBLIC knowledge about the domain, so it costs no privacy.
    syn_num = np.clip(syn_num, 0.0, None)

    return syn_num, syn_cat, stats_budgets


# ==========================================================================
# Metrics
# ==========================================================================
def fidelity_metrics(real_num, syn_num, real_cat, syn_cat, cat_encoders):
    """Column-wise distributional fidelity."""
    from scipy.stats import wasserstein_distance
    # Numeric: Wasserstein-1 normalized by real std (dimensionless)
    w1 = []
    for j in range(real_num.shape[1]):
        std = real_num[:, j].std() + 1e-8
        w1.append(wasserstein_distance(real_num[:, j], syn_num[:, j]) / std)
    # Categorical: total variation distance of frequency distributions
    tv = []
    for j, c in enumerate(CATEGORICAL_COLS):
        k = len(cat_encoders[c])
        p = np.bincount(real_cat[:, j], minlength=k) / len(real_cat)
        q = np.bincount(syn_cat[:, j], minlength=k) / len(syn_cat)
        tv.append(0.5 * np.abs(p - q).sum())
    # Correlation preservation (numeric columns)
    cr = np.corrcoef(real_num.T)
    cs = np.corrcoef(syn_num.T)
    corr_delta = float(np.abs(cr - cs).mean())
    return {'num_w1_mean': float(np.mean(w1)),
            'cat_tv_mean': float(np.mean(tv)),
            'corr_delta': corr_delta}


def privacy_metrics(real_num, syn_num, params):
    """DCR ratio: min dist(synthetic -> real) / min dist(real holdout ->
    real train). Values near >= 1.0 mean synthetic records are no closer
    to real records than real records are to each other."""
    rng = np.random.default_rng(params['SEED'])
    n = len(real_num)
    hold = rng.choice(n, size=min(500, n), replace=False)
    mask = np.ones(n, dtype=bool); mask[hold] = False
    train_idx = np.where(mask)[0]

    rt = torch.tensor(real_num[train_idx]).float()
    rh = torch.tensor(real_num[hold]).float()
    sy = torch.tensor(syn_num).float()

    d_sr = torch.cdist(sy, rt).min(dim=1).values  # synthetic -> real
    d_rr = torch.cdist(rh, rt).min(dim=1).values  # real holdout -> real train
    ratio = float(d_sr.mean() / (d_rr.mean() + 1e-8))
    return {'dcr_ratio': ratio,
            'syn_to_real_min': float(d_sr.mean()),
            'real_to_real_min': float(d_rr.mean())}


def fairness_metrics(real_cat, syn_cat, t_idx, s_idx, cat_encoders,
                     real_num=None, syn_num=None):
    """Demographic parity of the synthetic outcome across protected
    groups, compared with the real data's disparity.

    Also computes two probes against gaming of the headline DP metric
    (a generator that merely shuffles the protected column scores
    perfectly on DP difference while still encoding group through
    other features):

      - attribute_probe: AUC of a classifier predicting the protected
        attribute from the OTHER synthetic features. ~0.5 = the
        synthetic features carry no group signal; >0.5 means group is
        still encoded in income/geography/etc.
      - downstream_fairness: train a classifier on SYNTHETIC data to
        predict the target, measure its demographic parity on REAL
        data broken out by real group. This tests whether the synthetic
        data transmits the real disparity to downstream models.
    """
    def dp(cat):
        rates = {}
        # Look up the factorize CODE for the label '1' (= originated)
        # instead of assuming code 1: pd.factorize assigns codes in
        # first-appearance order, so '0'->0/'1'->1 only if '0' appears
        # first in the data. Hardcoding == 1 silently measures the
        # DENIAL rate when the ordering flips, reversing every group
        # comparison in the report.
        try:
            t_pos = list(cat_encoders[TARGET_COL]).index('1')
        except ValueError:
            t_pos = 1  # fallback: assume binary 0/1 encoding
        for g in range(len(cat_encoders[SENSITIVE_COL])):
            m = cat[:, s_idx] == g
            if m.sum() == 0:
                continue
            rates[g] = float((cat[m, t_idx] == t_pos).mean())
        vals = list(rates.values())
        return rates, max(vals) - min(vals), min(vals) / (max(vals) + 1e-8)

    real_rates, real_dp_diff, real_dp_ratio = dp(real_cat)
    syn_rates, syn_dp_diff, syn_dp_ratio = dp(syn_cat)
    out = {
        'real_rates': {str(cat_encoders[SENSITIVE_COL][k]): v
                       for k, v in real_rates.items()},
        'syn_rates': {str(cat_encoders[SENSITIVE_COL][k]): v
                      for k, v in syn_rates.items()},
        'real_dp_diff': real_dp_diff, 'syn_dp_diff': syn_dp_diff,
        'real_dp_ratio': real_dp_ratio, 'syn_dp_ratio': syn_dp_ratio,
        'fairness_gain': real_dp_diff - syn_dp_diff,
    }

    # ---- Anti-gaming probes -------------------------------------------
    if real_num is not None and syn_num is not None:
        out['attribute_probe_auc'] = _attribute_probe_auc(
            syn_num, syn_cat, s_idx, cat_encoders)
        out['downstream_fairness'] = _downstream_fairness(
            real_num, real_cat, syn_num, syn_cat, t_idx, s_idx,
            cat_encoders)
    return out


def _attribute_probe_auc(syn_num, syn_cat, s_idx, cat_encoders):
    """Can the protected attribute be predicted from the OTHER synthetic
    features? AUC per group (one-vs-rest), averaged. ~0.5 = no signal.

    A generator can trivially make the synthetic protected column
    uniform (L_S does exactly that) while income, geography and loan
    type still predict it. This probe catches that.
    """
    n_groups = len(cat_encoders[SENSITIVE_COL])
    # Feature matrix: numerics + all categorical columns except S
    feats = [syn_num]
    for j in range(syn_cat.shape[1]):
        if j == s_idx:
            continue
        k = syn_cat[:, j].max() + 1
        onehot = np.eye(k)[syn_cat[:, j]]
        feats.append(onehot)
    X = np.concatenate(feats, axis=1)
    y = syn_cat[:, s_idx]

    aucs = []
    for g in range(n_groups):
        y_bin = (y == g).astype(int)
        if y_bin.sum() < 10 or (1 - y_bin).sum() < 10:
            continue
        aucs.append(_fast_auc(X, y_bin))
    return float(np.mean(aucs)) if aucs else float('nan')


def _fast_auc(X, y, n_iter=30):
    """AUC of a logistic-regression-style linear probe via ranked random
    projections (no sklearn dependency): AUC of the projection score.

    Uses a small ridge-then-rank approximation: fit ridge regression to
    y, score, and compute AUC by rank statistics. Fast and dependency-
    free; sufficient as a probe (not a classifier benchmark).
    """
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    # Ridge via normal equations with regularization for stability
    lam = 1e-2 * n
    XtX = Xc.T @ Xc + lam * np.eye(d)
    Xty = Xc.T @ y
    w = np.linalg.solve(XtX, Xty)
    scores = Xc @ w
    order = np.argsort(scores)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _downstream_fairness(real_num, real_cat, syn_num, syn_cat, t_idx,
                         s_idx, cat_encoders):
    """Train a target-predictor on SYNTHETIC data, evaluate its
    demographic parity on REAL data by real group.

    This is the check that matters in practice: if a bank trains on the
    synthetic data, does the resulting model treat real groups
    equitably? A synthetic dataset whose own DP difference is ~0 can
    still produce a disparate model if group-correlated features
    survived generation.
    """
    def feats(cat, num):
        parts = [num]
        for j in range(cat.shape[1]):
            if j == s_idx:
                continue  # the protected attribute is NOT a model feature
            k = int(cat[:, j].max()) + 1
            parts.append(np.eye(k)[cat[:, j]])
        return np.concatenate(parts, axis=1)

    X_syn = feats(syn_cat, syn_num)
    y_syn = syn_cat[:, t_idx]
    X_real = feats(real_cat, real_num)
    y_real = real_cat[:, t_idx]

    # Ridge classifier (dependency-free, same _fast_auc machinery).
    # Map the target to +/-1 by the CODE for label '1' (not a hardcoded
    # 1 - see the t_pos note in fairness_metrics), fit, and threshold
    # the score at 0.
    try:
        t_pos = list(cat_encoders[TARGET_COL]).index('1')
    except ValueError:
        t_pos = 1
    y_pm = np.where(y_syn == t_pos, 1.0, -1.0)
    Xc = X_syn - X_syn.mean(axis=0, keepdims=True)
    d = Xc.shape[1]
    lam = 1e-2 * len(Xc)
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ y_pm)
    b = y_pm.mean() - X_syn.mean(axis=0) @ w
    scores = X_real @ w + b
    pred = (scores > 0).astype(int)

    rates = {}
    for g in range(len(cat_encoders[SENSITIVE_COL])):
        m = real_cat[:, s_idx] == g
        if m.sum() == 0:
            continue
        rates[str(cat_encoders[SENSITIVE_COL][g])] = float(pred[m].mean())
    vals = list(rates.values())
    if len(vals) < 2:
        return {'rates': rates, 'dp_diff': 0.0, 'dp_ratio': 1.0}
    dp_diff = max(vals) - min(vals)
    dp_ratio = min(vals) / (max(vals) + 1e-8)
    return {'rates': rates, 'dp_diff': dp_diff, 'dp_ratio': dp_ratio}


# ==========================================================================
# Synthetic data export
# ==========================================================================
SYNTHETIC_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'Data', 'flip_synthetic_bmo_nationwide.csv')


def decode_synthetic_data(syn_num, syn_cat, num_means, num_stds, cat_encoders):
    """Reverse the normalization/label-encoding from load_sample() so the
    synthetic array is human-readable in the same units/labels as the
    source CSV."""
    num_denorm = syn_num * num_stds.values[None, :] + num_means.values[None, :]
    df = pd.DataFrame(num_denorm, columns=NUMERIC_COLS)
    for j, col in enumerate(CATEGORICAL_COLS):
        uniques = cat_encoders[col]
        codes = np.clip(syn_cat[:, j], 0, len(uniques) - 1)
        df[col] = uniques[codes]
    return df


def save_synthetic_data(syn_num, syn_cat, num_means, num_stds, cat_encoders):
    df = decode_synthetic_data(syn_num, syn_cat, num_means, num_stds, cat_encoders)
    os.makedirs(os.path.dirname(SYNTHETIC_DATA_PATH), exist_ok=True)
    df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    print(f'Synthetic data ({len(df)} rows) saved to {SYNTHETIC_DATA_PATH}')
    return df


# ==========================================================================
# HTML report
# ==========================================================================
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'flip_report.html')


def _verdict(value, good, moderate, higher_is_better):
    """Return (label, css_class) for a metric against good/moderate cutoffs."""
    if higher_is_better:
        if value >= good:
            return 'Good', 'ok'
        if value >= moderate:
            return 'Moderate', 'warn'
        return 'Poor', 'bad'
    else:
        if value <= good:
            return 'Good', 'ok'
        if value <= moderate:
            return 'Moderate', 'warn'
        return 'Poor', 'bad'


def generate_html_report(params, fid, priv, fair, eps, encoders, synthetic_df,
                        report_path=None, eps_group=None):
    """Generate the HTML report. report_path defaults to the demo's
    REPORT_PATH; other pipelines (e.g. run_flip_full_pipeline.py) pass
    their own so the reports coexist."""
    fid_w1_v, fid_w1_c = _verdict(fid['num_w1_mean'], 0.2, 0.4, higher_is_better=False)
    fid_tv_v, fid_tv_c = _verdict(fid['cat_tv_mean'], 0.15, 0.3, higher_is_better=False)
    fid_cr_v, fid_cr_c = _verdict(fid['corr_delta'], 0.05, 0.15, higher_is_better=False)
    priv_v, priv_c = _verdict(priv['dcr_ratio'], 1.0, 0.5, higher_is_better=True)
    fair_v, fair_c = _verdict(fair['syn_dp_ratio'], 0.8, 0.6, higher_is_better=True)

    eps_html = (f'<span class="card-value">{eps:.2f}</span>'
               if eps is not None else '<span class="card-value">n/a (no DP)</span>')
    eps_note = (f'A lower epsilon is a stronger guarantee; {eps:.2f} at '
               f'delta={params["DP_DELTA"]:.0e} is a typical research-grade budget.'
               if eps is not None else
               'DP-SGD was disabled for this run (USE_DP=False), so there is no '
               'formal privacy guarantee on this model.')
    if eps_group is not None:
        eps_note += (f' The record-level epsilon covers the per-sample losses '
                     f'(MSE/CE/KLD/L_S); the batch-coupled fairness terms '
                     f'(SWD anchor, CKA^T) are covered by a separate '
                     f'GROUP-level guarantee of epsilon={eps_group:.2f} at the '
                     f'same delta, because their gradients cannot be '
                     f'decomposed per sample.')

    real_rows = ''.join(f'<tr><td>{g}</td><td>{r:.4f}</td><td>{fair["syn_rates"].get(g, float("nan")):.4f}</td></tr>'
                        for g, r in fair['real_rates'].items())

    probe_cards = ''
    if 'attribute_probe_auc' in fair and not math.isnan(fair['attribute_probe_auc']):
        auc = fair['attribute_probe_auc']
        auc_v, auc_c = _verdict(0.55, 0.65, 0.7, higher_is_better=False) \
            if auc <= 0.7 else ('High group signal', 'bad')
        probe_cards += (f'<div class="card {auc_c}-card"><span class="card-label">'
                        f'Attribute probe AUC</span><span class="card-value">'
                        f'{auc:.4f} ({auc_v})</span></div>')
    if 'downstream_fairness' in fair:
        dwn = fair['downstream_fairness']
        dwn_v, dwn_c = _verdict(dwn['dp_ratio'], 0.8, 0.6, higher_is_better=True)
        probe_cards += (f'<div class="card {dwn_c}-card"><span class="card-label">'
                        f'Downstream DP ratio (real eval)</span><span class="card-value">'
                        f'{dwn["dp_ratio"]:.4f} ({dwn_v})</span></div>')

    param_rows = ''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in params.items())

    sample = synthetic_df.head(10)
    sample_header = ''.join(f'<th>{c}</th>' for c in sample.columns)
    sample_rows = ''.join(
        '<tr>' + ''.join(f'<td>{v:.2f}</td>' if isinstance(v, float) else f'<td>{v}</td>'
                         for v in row) + '</tr>'
        for row in sample.itertuples(index=False))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FLIP Training Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f4f6f8; color: #22303c; }}
  header {{ background: #1d3557; color: #fff; padding: 30px 40px; }}
  header h1 {{ margin: 0 0 6px 0; }}
  header p {{ margin: 0; opacity: 0.85; }}
  main {{ max-width: 1000px; margin: 0 auto; padding: 30px 20px; }}
  section {{ background: #fff; border-radius: 10px; padding: 25px; margin-bottom: 25px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  h2 {{ margin-top: 0; color: #1d3557; border-bottom: 2px solid #e0e4e8; padding-bottom: 8px; }}
  .cards {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 15px; }}
  .card {{ background: #f0f4f8; border-radius: 8px; padding: 12px 18px; min-width: 150px; }}
  .card-label {{ display: block; font-size: 0.75em; text-transform: uppercase; color: #6b7a8c; }}
  .card-value {{ display: block; font-size: 1.15em; font-weight: 600; }}
  .ok-card {{ background: #e6f4ea; }} .warn-card {{ background: #fff4e0; }} .bad-card {{ background: #fdecea; }}
  .explanation {{ background: #f0f4f8; border-radius: 8px; padding: 15px 20px; margin-bottom: 15px; }}
  .explanation p {{ margin: 4px 0; line-height: 1.55; font-size: 0.92em; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92em; margin-top: 10px; }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #e8ecf0; }}
  th {{ background: #f0f4f8; }}
  code {{ background: #e2e8ee; padding: 1px 5px; border-radius: 4px; font-size: 0.9em; }}
</style>
</head>
<body>
<header>
  <h1>FLIP Training Report</h1>
  <p>Fair + Private TabSyn VAE on bmo_nationwide.csv &middot; protected attribute: {SENSITIVE_COL}</p>
</header>
<main>

  <section>
    <h2>Fidelity - how close is the synthetic data to the real data?</h2>
    <div class="explanation">
      <p><strong>Numeric Wasserstein-1</strong> measures the average "cost" of
      moving the synthetic numeric distribution onto the real one, normalized by
      each column's standard deviation. 0 = identical distributions.</p>
      <p><strong>Categorical total variation (TV)</strong> is the fraction of
      categorical records that would need to change category for the synthetic
      frequency distribution to exactly match the real one. 0 = identical.</p>
      <p><strong>Correlation-matrix delta</strong> is the average absolute
      difference between the real and synthetic numeric correlation matrices -
      it checks whether relationships between columns (not just each column on
      its own) survived generation.</p>
    </div>
    <div class="cards">
      <div class="card {fid_w1_c}-card"><span class="card-label">Numeric W1</span><span class="card-value">{fid['num_w1_mean']:.4f} ({fid_w1_v})</span></div>
      <div class="card {fid_tv_c}-card"><span class="card-label">Categorical TV</span><span class="card-value">{fid['cat_tv_mean']:.4f} ({fid_tv_v})</span></div>
      <div class="card {fid_cr_c}-card"><span class="card-label">Correlation delta</span><span class="card-value">{fid['corr_delta']:.4f} ({fid_cr_v})</span></div>
    </div>
  </section>

  <section>
    <h2>Privacy - could synthetic records leak real training data?</h2>
    <div class="explanation">
      <p><strong>DCR ratio</strong> (distance-to-closest-record) compares how
      close synthetic records sit to the real training data versus how close a
      held-out real record sits to that same training data. A ratio near or
      above 1.0 means synthetic records are no closer to individual training
      records than other real records naturally are - i.e. no evidence of
      memorization. A ratio well below 0.5 would suggest the model copied
      training examples rather than generalizing.</p>
      <p><strong>DP guarantee (epsilon, delta)</strong> is the formal
      differential-privacy bound from DP-SGD training, tracked by a Renyi-DP
      accountant across every training step. It holds regardless of what any
      attacker does with the trained model - unlike DCR, which is only an
      empirical check on this one generated batch.</p>
    </div>
    <div class="cards">
      <div class="card {priv_c}-card"><span class="card-label">DCR ratio</span><span class="card-value">{priv['dcr_ratio']:.4f} ({priv_v})</span></div>
      <div class="card"><span class="card-label">DP guarantee</span>{eps_html}</div>
    </div>
    <p style="font-size:0.9em; color:#555;">{eps_note}</p>
  </section>

  <section>
    <h2>Fairness - does the synthetic data reduce group bias?</h2>
    <div class="explanation">
      <p>Each protected group's <strong>loan origination rate</strong> is shown
      below for the real data and the FLIP-generated synthetic data. The
      <strong>demographic parity (DP) difference</strong> is the gap between the
      highest- and lowest-rate group; the <strong>DP ratio</strong> is the
      lowest rate divided by the highest. The widely used four-fifths rule
      treats a ratio below 0.80 as evidence of disparate impact.</p>
      <p>FLIP's disentanglement objective (CKA-based, Sec. 4.2-4.3 of the
      paper) pushes the VAE's latent and reconstructed representations toward
      statistical independence from the protected attribute during Phase 2
      training, so the synthetic data's group gap should shrink relative to
      the real data's.</p>
    </div>
    <table>
      <thead><tr><th>Group</th><th>Real rate</th><th>Synthetic rate</th></tr></thead>
      <tbody>{real_rows}</tbody>
    </table>
    <div class="cards" style="margin-top:15px;">
      <div class="card"><span class="card-label">DP difference (real &rarr; syn)</span><span class="card-value">{fair['real_dp_diff']:.4f} &rarr; {fair['syn_dp_diff']:.4f}</span></div>
      <div class="card {fair_c}-card"><span class="card-label">DP ratio (real &rarr; syn)</span><span class="card-value">{fair['real_dp_ratio']:.4f} &rarr; {fair['syn_dp_ratio']:.4f} ({fair_v})</span></div>
      <div class="card"><span class="card-label">Fairness gain</span><span class="card-value">{fair['fairness_gain']:+.4f}</span></div>
    </div>
    <div class="explanation" style="margin-top:15px;">
      <p><strong>Anti-gaming probes.</strong> The DP metrics above can be
      gamed by a generator that merely randomizes the protected column
      while other features still encode it. Two additional checks:</p>
      <p><strong>Attribute probe AUC</strong>: how well the protected
      attribute can be predicted from the OTHER synthetic features
      (income, geography, loan type). 0.5 = no group signal; higher
      means group information survives disentanglement.</p>
      <p><strong>Downstream fairness</strong>: a model trained on the
      synthetic data to predict the target (without seeing the protected
      attribute) is evaluated on the real data, broken out by real group.
      This measures whether the synthetic data transmits disparity to
      downstream users - the fairness outcome that actually matters.</p>
    </div>
    <div class="cards">
      {probe_cards}
    </div>
  </section>

  <section>
    <h2>Synthetic data sample</h2>
    <p style="font-size:0.9em; color:#555;">First 10 rows of the
    {len(synthetic_df):,}-row synthetic dataset generated by decoding random
    latents through the trained FLIP VAE (numeric columns denormalized,
    categorical columns decoded back to their original labels). The full
    dataset was saved to <code>Data/flip_synthetic_bmo_nationwide.csv</code>.</p>
    <div style="overflow-x:auto;">
    <table>
      <thead><tr>{sample_header}</tr></thead>
      <tbody>{sample_rows}</tbody>
    </table>
    </div>
  </section>

  <section>
    <h2>Training configuration</h2>
    <p style="font-size:0.9em; color:#555;">The fidelity/privacy/fairness
    trade-off knobs used for this run (see <code>TRADEOFF_PARAMS</code> in
    <code>run_flip_demo.py</code> for what each one controls).</p>
    <table>
      <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>{param_rows}</tbody>
    </table>
  </section>

</main>
</body>
</html>"""

    out_path = report_path if report_path is not None else REPORT_PATH
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\nHTML report written to {out_path}')


# ==========================================================================
# Main
# ==========================================================================
def main():
    params = TRADEOFF_PARAMS
    device = torch.device(params['DEVICE'])
    if device.type == 'cuda':
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('Using CPU (GPU not available)')
    print('=' * 70)
    print('FLIP demo on bmo_nationwide.csv - fidelity/privacy/fairness')
    print('=' * 70)
    print(f'\nTradeoff parameters:')
    for k, v in params.items():
        print(f'  {k:<18} = {v}')

    X_num, X_cat, categories, s_idx, t_idx, encoders, df, num_means, num_stds = load_sample(params)
    print(f'\nSample: {len(X_num)} rows, {X_num.shape[1]} numeric, '
          f'{X_cat.shape[1]} categorical (categories={categories})')
    print(f'Protected attribute: {SENSITIVE_COL} '
          f'{dict(zip(range(len(encoders[SENSITIVE_COL])), encoders[SENSITIVE_COL]))}')

    print('\nTraining FLIP VAE...')
    model, accountant, group_mech = train_flip(X_num, X_cat, categories,
                                               s_idx, params, device)

    print('\nGenerating synthetic data...')
    syn_num, syn_cat, stats_budgets = generate(model, len(X_num), device,
                                               params, X_num=X_num,
                                               X_cat=X_cat)
    synthetic_df = save_synthetic_data(syn_num, syn_cat, num_means, num_stds, encoders)

    fid = fidelity_metrics(X_num.numpy(), syn_num, X_cat.numpy(), syn_cat,
                           encoders)
    priv = privacy_metrics(X_num.numpy(), syn_num, params)
    fair = fairness_metrics(X_cat.numpy(), syn_cat, t_idx, s_idx, encoders,
                           real_num=X_num.numpy(), syn_num=syn_num)

    eps = None
    eps_group = None
    eps_total = None
    if accountant is not None:
        eps, alpha = accountant.get_privacy_spent()
    if group_mech is not None and group_mech.steps > 0:
        eps_group, _ = group_mech.get_privacy_spent(params['DP_DELTA'])
    # Compose ALL budgets into the true end-to-end guarantee: DP-SGD
    # (record-level) + fairness-gradient mechanism (record-level,
    # substitution adjacency) + the statistics releases. Valid RDP
    # composition sums the divergence at each alpha and converts once -
    # adding per-mechanism epsilons is NOT a valid rule.
    if accountant is not None:
        budgets = [accountant.get_rdp_curve()]
        if group_mech is not None and group_mech.steps > 0:
            budgets.append(group_mech.get_rdp_curve())
        budgets.extend(stats_budgets)  # each release is its own budget
        eps_total, _ = compose_rdp_budgets(budgets, params['DP_DELTA'])

    print('\n' + '=' * 70)
    print('RESULTS')
    print('=' * 70)
    print(f'\n--- FIDELITY (lower w1/TV/corr-delta = better) ---')
    print(f'  Numeric Wasserstein-1 (normalized): {fid["num_w1_mean"]:.4f}')
    print(f'  Categorical TV distance:            {fid["cat_tv_mean"]:.4f}')
    print(f'  Correlation-matrix delta:            {fid["corr_delta"]:.4f}')
    print(f'\n--- PRIVACY ---')
    print(f'  DCR ratio (syn->real / real->real):  {priv["dcr_ratio"]:.4f}')
    print(f'    (>= 1.0 healthy; < 0.5 = memorization risk)')
    if eps is not None:
        print(f'  DP guarantee (record-level, per-sample losses): '
              f'(epsilon={eps:.2f}, delta={params["DP_DELTA"]})')
        if eps_group is not None:
            print(f'  DP guarantee (fairness-gradient mechanism):     '
                  f'(epsilon={eps_group:.2f}, delta={params["DP_DELTA"]})')
        if eps_total is not None:
            print(f'  TOTAL end-to-end epsilon (composed):           '
                  f'(epsilon={eps_total:.2f}, delta={params["DP_DELTA"]})')
    else:
        print(f'  DP guarantee: disabled (USE_DP=False)')
    print(f'\n--- FAIRNESS (origination rate by group) ---')
    print(f'  Real:      {fair["real_rates"]}')
    print(f'  Synthetic: {fair["syn_rates"]}')
    print(f'  DP difference  real={fair["real_dp_diff"]:.4f} -> '
          f'syn={fair["syn_dp_diff"]:.4f}  '
          f'(gain {fair["fairness_gain"]:+.4f})')
    print(f'  DP ratio       real={fair["real_dp_ratio"]:.4f} -> '
          f'syn={fair["syn_dp_ratio"]:.4f}')
    if 'attribute_probe_auc' in fair:
        print(f'  Attribute probe AUC (0.5 = no group signal in other '
              f'features): {fair["attribute_probe_auc"]:.4f}')
    if 'downstream_fairness' in fair:
        df_ = fair['downstream_fairness']
        print(f'  Downstream model (trained on synthetic, evaluated on real): '
              f'DP diff={df_["dp_diff"]:.4f}, DP ratio={df_["dp_ratio"]:.4f}')
        print(f'    per-group prediction rates: {df_["rates"]}')

    generate_html_report(params, fid, priv, fair, eps, encoders, synthetic_df,
                         eps_group=eps_group)

    print('\n' + '=' * 70)
    print('TRADEOFF GUIDE - how to move each metric')
    print('=' * 70)
    print(f'''
  Want MORE FAIRNESS?      raise LAMBDA_FAIR (e.g. 2.0-5.0), or extend
                          PHASE2_EPOCHS. Cost: fidelity drops as the
                          latent is forced to ignore the protected attr.
  Want MORE PRIVACY?       raise NOISE_MULTIPLIER (e.g. 1.5-3.0) or lower
                          MAX_GRAD_NORM. Cost: fidelity drops, and per
                          the paper (Sec. 6.3) privacy noise also blunts
                          the fairness intervention - expect to raise
                          LAMBDA_FAIR to compensate.
  Want MORE FIDELITY?      lower NOISE_MULTIPLIER, lower LAMBDA_FAIR,
                          raise PHASE1_EPOCHS, lower BETA. Cost: weaker
                          privacy guarantee and less bias mitigation.

  The three goals compete: FLIP's two-phase design lets you lock in a
  quality representation first (Phase 1) and then dial in exactly as
  much fairness (LAMBDA_FAIR) and privacy (NOISE_MULTIPLIER) as your
  epsilon/fidelity budget allows.
''')


if __name__ == '__main__':
    main()