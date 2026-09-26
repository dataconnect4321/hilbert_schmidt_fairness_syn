"""Run the FULL FLIP pipeline (VAE + latent diffusion) on a sample of
bmo_nationwide.csv and evaluate fidelity / privacy / fairness.

This is the complete two-stage TabSyn pipeline with FLIP fairness/privacy
training, unlike run_flip_demo.py which uses a Gaussian stand-in for the
latent prior:

  Stage 1 (FLIP VAE):   train the TabSyn VAE with the FLIP objectives
                        (two-phase: quality, then disentanglement),
                        balanced group sampling, and optional DP-SGD.
  Stage 2 (diffusion):  train TabSyn's MLPDiffusion denoiser (EDM
                        parameterization) on the VAE's latents, then
                        SAMPLE new latents from the diffusion prior -
                        this is TabSyn's real generation mechanism.
  Decode + evaluate:    decode sampled latents to records, score
                        fidelity/privacy/fairness, and emit the HTML
                        report + synthetic CSV (same outputs as the
                        demo).

Reuses the data loading, FLIP training, metrics, and HTML report from
run_flip_demo.py so both pipelines stay directly comparable.
"""

import os
import sys
import copy
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tabsyn-main', 'tabsyn-main'))

# Reuse everything from the demo script
from run_flip_demo import (
    TRADEOFF_PARAMS, SENSITIVE_COL, TARGET_COL, RACE_GROUPS,
    NUMERIC_COLS, CATEGORICAL_COLS, DATA_PATH,
    load_sample, train_flip, fidelity_metrics, privacy_metrics,
    fairness_metrics, decode_synthetic_data, generate_html_report,
)

from tabsyn.model import MLPDiffusion, Model
from tabsyn.diffusion_utils import sample as edm_sample
from tabsyn.vae.flip_fairness import (dp_release_mean_std, RDPAccountant,
                                      compose_rdp_budgets)

# ==========================================================================
# DIFFUSION PARAMETERS - Stage 2 knobs
# ==========================================================================
DIFFUSION_PARAMS = {
    # Training epochs for the latent diffusion denoiser. TabSyn's default
    # pipeline uses 10000+; on a 12k-row sample a few thousand suffices.
    'EPOCHS': 3000,

    # Batch size for diffusion training. MUST be much smaller under
    # DP-SGD than under plain Adam: Opacus stores a per-sample gradient
    # copy of every parameter, so memory scales as batch_size *
    # n_params * 4 bytes. With DIM_T=1024 the largest layer is
    # Linear(1024, 2048) (~2.1M params): batch 4096 needs ~34 GB of
    # grad_sample alone (OOM on an 8 GB GPU); batch 256 needs ~2 GB.
    # Lower batch also LOWERS the sampling rate q, improving the
    # privacy budget - the only cost is more steps per epoch.
    'BATCH_SIZE': 256,

    'LR': 1e-3,

    # Hidden width of the MLPDiffusion denoiser (TabSyn default: 1024).
    'DIM_T': 1024,

    # Number of Euler/Heun function evaluations when sampling latents
    # from the trained diffusion prior (TabSyn default: 50).
    'SAMPLE_STEPS': 50,

    # How many synthetic rows to generate (defaults to the training-set
    # size, matching the demo).
    'N_GEN': None,  # None -> len(training data)

    # ---- DP-SGD (Stage 2) ---------------------------------------------
    # Train the latent diffusion prior with DP-SGD so the FULL pipeline
    # carries an end-to-end guarantee. The latents are outputs of the
    # DP-trained VAE encoder (post-processing of a private model), so
    # DP-SGD on them composes with Stage 1's budget into a guarantee
    # w.r.t. the raw training data.
    #
    # The EDM loss is per-sample decomposable (each row draws its own
    # sigma and noise; the loss is a per-row squared error), and
    # MLPDiffusion is standard Linear+SiLU (no BatchNorm), so Opacus's
    # per-sample clipping covers every parameter of this stage.
    'USE_DP': True,          # falls back to plain Adam when False

    # DP-SGD noise multiplier for Stage 2. With BATCH_SIZE=256 on a
    # 12k-row sample the sampling rate q ~ 0.02, which gives a
    # meaningful epsilon at moderate noise; raise NOISE_MULTIPLIER
    # and/or lower EPOCHS to tighten the budget further.
    'NOISE_MULTIPLIER': 1.0,

    # Per-sample gradient L2 clipping norm for Stage 2.
    'MAX_GRAD_NORM': 1.0,

    # Target delta for Stage 2's (epsilon, delta)-DP guarantee.
    'DP_DELTA': 1e-5,
}

SYNTHETIC_DATA_PATH = os.path.join(HERE, 'Data',
                                   'flip_full_synthetic_bmo_nationwide.csv')
REPORT_PATH = os.path.join(HERE, 'flip_full_report.html')

# Full-dataset mode: train on ALL rows of the three usable race groups
# (~24.7k rows instead of the 12k SAMPLE_SIZE default) and write to
# distinct output paths so the sample-size comparison is preserved.
SYNTHETIC_DATA_PATH_ALL = os.path.join(
    HERE, 'Data', 'flip_full_synthetic_bmo_nationwide_all.csv')
REPORT_PATH_ALL = os.path.join(HERE, 'flip_full_report_all.html')


# ==========================================================================
# Stage 2: latent diffusion (TabSyn's real generation mechanism)
# ==========================================================================
def train_diffusion(latents, params, device):
    """Train TabSyn's MLPDiffusion denoiser (EDM preconditioning) on the
    VAE latents.

    Follows tabsyn/main.py's normalization convention: latents are shifted
    by their mean and halved before training, and sampled latents are
    doubled and un-shifted (sample() in tabsyn/sample.py). The mean shift
    is a function of the DP-trained encoder's outputs (post-processing of
    a private model), so it is covered by Stage 1's guarantee.

    PRIVACY: when params['USE_DP'] is True the denoiser is trained with
    DP-SGD (Opacus GradSampleModule + DPOptimizer) and its own RDP
    accountant. The EDM loss is per-sample decomposable and MLPDiffusion
    is standard Linear+SiLU, so per-sample clipping covers the whole
    stage - no group-level mechanism is needed here (unlike Stage 1's
    batch-coupled fairness terms). The Stage-2 epsilon composes with
    Stage 1's budget into the pipeline's end-to-end guarantee.

    Returns (model, latent_mean, accountant) where accountant is None
    when DP is disabled.
    """
    z = torch.tensor(latents, dtype=torch.float32)
    mean = z.mean(dim=0, keepdim=True)
    z_norm = (z - mean) / 2

    in_dim = z_norm.shape[1]
    denoise_fn = MLPDiffusion(in_dim, params['DIM_T']).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)

    accountant = None
    if params.get('USE_DP'):
        from opacus.optimizers import DPOptimizer
        from opacus.grad_sample import GradSampleModule

        # MLPDiffusion is standard Linear+SiLU: Opacus's native grad
        # samplers handle every parameter (strict=False tolerates the
        # non-trainable PositionalEmbedding/FourierEmbedding submodules).
        wrapped = GradSampleModule(model, strict=False)
        dp_params = [p for p in wrapped.parameters() if p.requires_grad]
        optimizer = DPOptimizer(
            optimizer=torch.optim.Adam(dp_params, lr=params['LR'],
                                       weight_decay=0),
            noise_multiplier=params['NOISE_MULTIPLIER'],
            max_grad_norm=params['MAX_GRAD_NORM'],
            expected_batch_size=params['BATCH_SIZE'])
        # Plain shuffled batches: each sample appears exactly once per
        # epoch, so its per-step inclusion probability is batch/n.
        accountant = RDPAccountant(
            noise_multiplier=params['NOISE_MULTIPLIER'],
            sample_rate=params['BATCH_SIZE'] / z_norm.shape[0],
            delta=params['DP_DELTA'])
        train_model = wrapped
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['LR'],
                                     weight_decay=0)
        train_model = model

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=20)

    dataset = torch.utils.data.TensorDataset(z_norm)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=params['BATCH_SIZE'], shuffle=True,
        num_workers=0, drop_last=False)

    model.train()
    for epoch in range(params['EPOCHS']):
        epoch_loss, n_seen = 0.0, 0
        for (batch,) in loader:
            inputs = batch.to(device)
            loss = train_model(inputs).mean()

            optimizer.zero_grad()
            loss.backward()
            if params.get('USE_DP'):
                if optimizer.pre_step():
                    optimizer.original_optimizer.step()
                accountant.step()
            else:
                optimizer.step()

            epoch_loss += loss.item() * len(inputs)
            n_seen += len(inputs)

        scheduler.step(epoch_loss / max(1, n_seen))
        if epoch % 100 == 0 or epoch == params['EPOCHS'] - 1:
            msg = (f'  diffusion epoch {epoch:5d}  '
                   f'loss={epoch_loss / max(1, n_seen):.6f}')
            if accountant is not None and accountant.steps > 0:
                eps_d, _ = accountant.get_privacy_spent()
                msg += f'  eps_diffusion={eps_d:.2f}'
            print(msg)

    return model, mean, accountant


@torch.no_grad()
def sample_from_diffusion(diffusion_model, mean, n_gen, latent_shape,
                          params, device):
    """Sample new latents from the trained diffusion prior and undo the
    (x - mean)/2 normalization."""
    in_dim = int(np.prod(latent_shape))
    x = edm_sample(diffusion_model.denoise_fn_D, n_gen, in_dim,
                   num_steps=params['SAMPLE_STEPS'], device=device)
    x = x * 2 + mean.to(device)
    return x.reshape(n_gen, *latent_shape)


@torch.no_grad()
def decode_latents(vae_model, latents, device, X_num=None, params=None):
    """Decode sampled latents back to data space through the frozen VAE
    decoder + Reconstructor, with a DP-released range-clip backstop.

    PRIVACY: the original version clipped synthetic numerics to the
    real data's per-column min/max - a fresh non-private computation on
    raw records (min/max are especially sensitive: one extreme record
    moves them arbitrarily). Order statistics cannot be released with
    bounded sensitivity, so the clip bounds are instead a DP range
    proxy: mean +/- k*std released under the Gaussian mechanism
    (dp_release_mean_std), spent once into the privacy budget.
    """
    vae_model.eval()
    h = vae_model.VAE.decoder(latents[:, 1:])
    recon_num, recon_cat = vae_model.Reconstructor(h)

    syn_num = recon_num.cpu().numpy()
    syn_cat = np.stack([c.argmax(dim=-1).cpu().numpy() for c in recon_cat],
                       axis=1)

    if X_num is not None and params is not None:
        Xn = X_num.to(device)
        r_mean, r_std, stats_rdp = dp_release_mean_std(
            Xn, params['STATS_NOISE_MULTIPLIER'],
            params['STATS_MAX_NORM'], Xn.shape[0])
        k = params['STATS_RANGE_K']
        lo = (r_mean - k * r_std).cpu().numpy()
        hi = (r_mean + k * r_std).cpu().numpy()
        syn_num = np.clip(syn_num, lo[None, :], hi[None, :])
    # Domain-knowledge clamp: all numeric columns in this dataset (loan
    # amount, income, LTV, property value, tract population) are
    # non-negative by definition. Public knowledge, no privacy cost.
    syn_num = np.clip(syn_num, 0.0, None)
    if X_num is not None and params is not None:
        return syn_num, syn_cat, stats_rdp
    return syn_num, syn_cat, []


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Full FLIP pipeline (VAE + latent diffusion) on HMDA data')
    parser.add_argument('--all', action='store_true',
                        help='Train on the ENTIRE filtered dataset (all rows '
                             'of the three usable race groups, ~24.7k rows) '
                             'instead of the SAMPLE_SIZE subsample. Outputs '
                             'go to *_all.csv / *_all.html.')
    args = parser.parse_args()

    params = copy.deepcopy(TRADEOFF_PARAMS)
    dparams = copy.deepcopy(DIFFUSION_PARAMS)
    if args.all:
        # Use every row of the filtered groups
        params['SAMPLE_SIZE'] = None
        syn_path = SYNTHETIC_DATA_PATH_ALL
        report_path = REPORT_PATH_ALL
    else:
        syn_path = SYNTHETIC_DATA_PATH
        report_path = REPORT_PATH

    device = torch.device(params['DEVICE'])
    if device.type == 'cuda':
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('Using CPU (GPU not available)')
    print('=' * 70)
    print('FULL FLIP pipeline (VAE + latent diffusion) on bmo_nationwide.csv'
          + (' [ENTIRE DATASET]' if args.all else ''))
    print('=' * 70)
    print('\nStage-1 (VAE) tradeoff parameters:')
    for k, v in params.items():
        print(f'  {k:<24} = {v}')
    print('\nStage-2 (diffusion) parameters:')
    for k, v in dparams.items():
        print(f'  {k:<24} = {v}')

    # ---- Data -----------------------------------------------------------
    X_num, X_cat, categories, s_idx, t_idx, encoders, df, num_means, num_stds \
        = load_sample(params)
    print(f'\nSample: {len(X_num)} rows, {X_num.shape[1]} numeric, '
          f'{X_cat.shape[1]} categorical (categories={categories})')
    print(f'Protected attribute: {SENSITIVE_COL} '
          f'{dict(zip(range(len(encoders[SENSITIVE_COL])), encoders[SENSITIVE_COL]))}')

    # ---- Stage 1: FLIP VAE ----------------------------------------------
    print('\n[Stage 1] Training FLIP VAE...')
    vae_model, accountant, group_mech = train_flip(X_num, X_cat, categories,
                                                   s_idx, params, device)

    # ---- Stage 2: latent diffusion --------------------------------------
    print('\n[Stage 2] Encoding training data and training latent diffusion...')
    vae_model.eval()
    with torch.no_grad():
        enc = vae_model.forward_with_stages(X_num.to(device),
                                            X_cat.to(device))
    mu_z = enc['mu_z']                       # (n, tokens, d)
    latent_shape = mu_z.shape[1:]
    latents = mu_z.reshape(mu_z.shape[0], -1).cpu().numpy()  # (n, tokens*d)
    print(f'  latents: {latents.shape}')

    diffusion_model, z_mean, diff_accountant = train_diffusion(
        latents, dparams, device)

    # ---- Sample + decode -------------------------------------------------
    n_gen = dparams['N_GEN'] or len(X_num)
    print(f'\nSampling {n_gen} latents from the diffusion prior '
          f'({dparams["SAMPLE_STEPS"]} NFE)...')
    sampled = sample_from_diffusion(diffusion_model, z_mean, n_gen,
                                    latent_shape, dparams, device)
    syn_num, syn_cat, stats_rdp = decode_latents(vae_model, sampled, device,
                                                  X_num=X_num, params=params)

    # ---- Export ----------------------------------------------------------
    synthetic_df = decode_synthetic_data(syn_num, syn_cat, num_means,
                                         num_stds, encoders)
    os.makedirs(os.path.dirname(syn_path), exist_ok=True)
    synthetic_df.to_csv(syn_path, index=False)
    print(f'Synthetic data ({len(synthetic_df)} rows) saved to {syn_path}')

    # ---- Evaluate ---------------------------------------------------------
    fid = fidelity_metrics(X_num.numpy(), syn_num, X_cat.numpy(), syn_cat,
                           encoders)
    priv = privacy_metrics(X_num.numpy(), syn_num, params)
    fair = fairness_metrics(X_cat.numpy(), syn_cat, t_idx, s_idx, encoders,
                           real_num=X_num.numpy(), syn_num=syn_num)

    eps = None
    eps_group = None
    eps_diffusion = None
    eps_total = None
    if accountant is not None:
        eps, alpha = accountant.get_privacy_spent()
    if group_mech is not None and group_mech.steps > 0:
        eps_group, _ = group_mech.get_privacy_spent(params['DP_DELTA'])
    if diff_accountant is not None and diff_accountant.steps > 0:
        eps_diffusion, _ = diff_accountant.get_privacy_spent()
    # Compose ALL budgets into the true end-to-end guarantee: VAE
    # DP-SGD + fairness-gradient mechanism + statistics releases +
    # diffusion DP-SGD. Valid RDP composition sums the divergence at
    # each alpha and converts once - adding per-mechanism epsilons
    # (the old 'eps + eps_diffusion' line) is NOT a valid rule.
    if accountant is not None:
        budgets = [accountant.get_rdp_curve()]
        if group_mech is not None and group_mech.steps > 0:
            budgets.append(group_mech.get_rdp_curve())
        if diff_accountant is not None and diff_accountant.steps > 0:
            budgets.append(diff_accountant.get_rdp_curve())
        if stats_rdp:
            budgets.append(stats_rdp)
        eps_total, _ = compose_rdp_budgets(budgets, params['DP_DELTA'])

    print('\n' + '=' * 70)
    print('RESULTS (full pipeline: FLIP VAE + latent diffusion)')
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
        if eps_diffusion is not None:
            print(f'  DP guarantee (Stage 2 latent diffusion):        '
                  f'(epsilon={eps_diffusion:.2f}, '
                  f'delta={dparams["DP_DELTA"]})')
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

    # ---- HTML report ------------------------------------------------------
    # Merge both parameter dicts so the report documents the full run.
    report_params = copy.deepcopy(params)
    report_params.update({f'diffusion.{k}': v for k, v in dparams.items()})
    generate_html_report(report_params, fid, priv, fair, eps, encoders,
                         synthetic_df, report_path=report_path,
                         eps_group=eps_group)
    print(f'\nHTML report saved to {report_path}')


if __name__ == '__main__':
    main()
