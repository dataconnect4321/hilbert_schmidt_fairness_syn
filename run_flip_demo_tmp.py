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
    BalancedGroupSampler,
    RDPAccountant,
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

    # Learning-rate scale applied when Phase 2 begins. Smaller Phase-2
    # steps keep the fairness intervention from destroying the Phase-1
    # representation (the paper's 'controlled bias mitigation').
    'PHASE2_LR_SCALE': 0.2,

    # ---- Data / compute knobs -------------------------------------------
    'SAMPLE_SIZE': 12000,      # rows drawn from the preprocessed CSV
    'BATCH_SIZE': 1024,        # split evenly across protected groups
    'D_TOKEN': 8,
    'NUM_LAYERS': 2,
    'LR': 3e-4,
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

    df = df.sample(n=min(params['SAMPLE_SIZE'], len(df)),
                   random_state=params['SEED']).reset_index(drop=True)

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
            s_idx, t_idx, cat_encoders, df)


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
    sampler = BalancedGroupSampler(groups, params['BATCH_SIZE'],
                                   seed=params['SEED'])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_num, X_cat),
        batch_sampler=sampler)

    accountant = None
    if params['USE_DP']:
        from opacus.optimizers import DPOptimizer
        from opacus.grad_sample import GradSampleModule
        register_tabsyn_grad_samplers()
        model = GradSampleModule(model, strict=False)
        dp_params = get_dp_trainable_parameters(model)
        optimizer = DPOptimizer(
            optimizer=torch.optim.Adam(dp_params, lr=params['LR']),
            noise_multiplier=params['NOISE_MULTIPLIER'],
            max_grad_norm=params['MAX_GRAD_NORM'],
            expected_batch_size=params['BATCH_SIZE'])
        accountant = RDPAccountant(
            noise_multiplier=params['NOISE_MULTIPLIER'],
            sample_rate=params['BATCH_SIZE'] / n,
            delta=params['DP_DELTA'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['LR'])

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
            mse, ce, kld, acc, loss_s = compute_loss(
                bnum, bcat, stages['recon_x_num'], stages['recon_x_cat'],
                stages['mu_z'], stages['std_z'], s_idx=s_idx,
                Recon_S_logits=stages['recon_x_cat'][s_idx])

            loss = mse + ce + beta * kld + loss_s

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
                loss = loss + div + params['LAMBDA_FAIR'] * disent

            loss.backward()
            optimizer.step()
            if accountant is not None:
                accountant.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == total_epochs - 1:
            msg = f'epoch {epoch:3d} [phase {phase}] loss={epoch_loss/max(1,n_batches):.4f}'
            if accountant is not None:
                eps, alpha = accountant.get_privacy_spent()
                msg += f'  eps={eps:.2f}'
            print(msg)

    return raw, accountant


# ==========================================================================
# Generation
# ==========================================================================
@torch.no_grad()
def generate(model, n_gen, device, params):
    """Sample latents from a per-dimension Gaussian fit to the encoded
    training latents, then decode. (The full TabSyn pipeline trains a
    diffusion prior on latents; a Gaussian stand-in keeps this demo
    self-contained.)"""
    model.eval()
    # Probe latent shape
    dummy_num = torch.zeros(2, model.Reconstructor.d_numerical, device=device)
    dummy_cat = torch.zeros(2, len(model.Reconstructor.cat_recons),
                           dtype=torch.long, device=device)
    probe = model.forward_with_stages(dummy_num, dummy_cat)
    tokens, d = probe['mu_z'].shape[1], probe['mu_z'].shape[2]

    # Empirical latent statistics (standard normal-ish after training)
    mu = torch.randn(n_gen, tokens, d, device=device)

    h = model.VAE.decoder(mu[:, 1:])
    recon_num, recon_cat = model.Reconstructor(h)

    syn_num = recon_num.cpu().numpy()
    syn_cat = np.stack([c.argmax(dim=-1).cpu().numpy() for c in recon_cat],
                       axis=1)
    return syn_num, syn_cat


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


def fairness_metrics(real_cat, syn_cat, t_idx, s_idx, cat_encoders):
    """Demographic parity of the synthetic outcome across protected
    groups, compared with the real data's disparity."""
    def dp(cat):
        rates = {}
        for g in range(len(cat_encoders[SENSITIVE_COL])):
            m = cat[:, s_idx] == g
            if m.sum() == 0:
                continue
            rates[g] = float((cat[m, t_idx] == 1).mean())
        vals = list(rates.values())
        return rates, max(vals) - min(vals), min(vals) / (max(vals) + 1e-8)

    real_rates, real_dp_diff, real_dp_ratio = dp(real_cat)
    syn_rates, syn_dp_diff, syn_dp_ratio = dp(syn_cat)
    return {
        'real_rates': {str(cat_encoders[SENSITIVE_COL][k]): v
                       for k, v in real_rates.items()},
        'syn_rates': {str(cat_encoders[SENSITIVE_COL][k]): v
                      for k, v in syn_rates.items()},
        'real_dp_diff': real_dp_diff, 'syn_dp_diff': syn_dp_diff,
        'real_dp_ratio': real_dp_ratio, 'syn_dp_ratio': syn_dp_ratio,
        'fairness_gain': real_dp_diff - syn_dp_diff,
    }


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

    X_num, X_cat, categories, s_idx, t_idx, encoders, df = load_sample(params)
    print(f'\nSample: {len(X_num)} rows, {X_num.shape[1]} numeric, '
          f'{X_cat.shape[1]} categorical (categories={categories})')
    print(f'Protected attribute: {SENSITIVE_COL} '
          f'{dict(zip(range(len(encoders[SENSITIVE_COL])), encoders[SENSITIVE_COL]))}')

    print('\nTraining FLIP VAE...')
    model, accountant = train_flip(X_num, X_cat, categories, s_idx,
                                   params, device)

    print('\nGenerating synthetic data...')
    syn_num, syn_cat = generate(model, len(X_num), device, params)

    fid = fidelity_metrics(X_num.numpy(), syn_num, X_cat.numpy(), syn_cat,
                           encoders)
    priv = privacy_metrics(X_num.numpy(), syn_num, params)
    fair = fairness_metrics(X_cat.numpy(), syn_cat, t_idx, s_idx, encoders)

    eps = None
    if accountant is not None:
        eps, alpha = accountant.get_privacy_spent()

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
        print(f'  DP guarantee: (epsilon={eps:.2f}, delta={params["DP_DELTA"]})')
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