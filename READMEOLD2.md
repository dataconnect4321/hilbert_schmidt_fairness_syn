# FLIP: Fair + Private TabSyn VAE

> **[2026-09-26 update]** This file describes the original FLIP
> integration. The design has since been substantially corrected —
> split record-level DP for the fairness losses, proper RDP budget
> composition, DP-released generation statistics, DP-SGD for the
> diffusion stage, and a fairness-metric encoding fix. **See
> `FINALREADME.md` (especially Part 9, "Limitations of the Guarantees")
> for the current, authoritative documentation.** The summary below
> is kept for historical context.

This document covers the changes made to adapt TabSyn's VAE training to the
method in *"Achieving Hilbert-Schmidt Independence Under Rényi Differential
Privacy for Fair and Private Data Generation"* (arXiv:2508.21815, the
"hilbert_schmidt" paper), which proposes **FLIP** (Fair Latent Intervention
under Privacy guarantees).

## Why

TabSyn's VAE learns a latent representation purely to reconstruct the data
well. If the training data is biased (e.g. loan origination rates differ by
race), the synthetic data generated from that VAE reproduces the same bias.
FLIP adds two things TabSyn doesn't have:

1. **Fairness** — an explicit training objective that makes the latent
   representation (and the reconstructed features) statistically
   independent of a chosen protected attribute, using the Hilbert-Schmidt
   Independence Criterion (HSIC), so synthetic data generated from it is
   less biased than the source data.
2. **Privacy** — a formal (ε, δ) differential-privacy guarantee via
   DP-SGD, so the trained model provably can't be used to reconstruct
   individual training records.

Both are optional and controlled by CLI flags / config knobs — the
unmodified TabSyn training path is unaffected if you don't opt in.

## What changed

### `tabsyn/vae/flip_fairness.py` (new file)

The math from the paper, implemented as standalone, testable functions:

| Function | Paper ref | What it does |
|---|---|---|
| `linear_cka_t` | Eq. 5-7 | Transposed linear CKA — the normalized HSIC used to compare feature-covariance patterns between two protected groups' representations. With centered inputs and a linear kernel this reduces to `‖AᵀB‖²_F / (‖AᵀA‖_F ‖BᵀB‖_F)`. |
| `disentanglement_loss` | Eq. 10 (D') | Negative mean pairwise CKAᵀ across all protected groups in a batch. Minimizing it pulls the groups' representations toward statistical independence from the protected attribute. |
| `multi_stage_disentanglement_loss` | Sec. 4.3.2 | Applies `disentanglement_loss` at three points — the latent space, the detokenizer output (feature-wise, protected column excluded), and the decoder output — then averages them, matching the paper's three-stage intervention. |
| `sliced_wasserstein_distance` | Eq. 10 (D) | The divergence penalty that keeps the Phase-2 encoder close to the frozen Phase-1 encoder, approximated via random 1-D projections (cheap, differentiable substitute for full Wasserstein distance). |
| `uniform_attribute_loss` | Eq. 3 (L_S) | ℓ2 distance between the mean softmax of the reconstructed protected-attribute logits and the uniform distribution — pushes the model to predict each protected group with equal probability regardless of its frequency in the training data. |
| `BalancedGroupSampler` | Eq. 11, Sec. 4.2.1 | A `torch.utils.data.Sampler` that builds each mini-batch with an equal number of samples per protected group, so gradient signal isn't dominated by the majority group. |
| `register_tabsyn_grad_samplers` | — (DP-SGD plumbing) | Registers Opacus per-sample-gradient functions for TabSyn's custom `Tokenizer` and `Reconstructor` modules. Without this, Opacus falls back to a `functorch`/vmap re-forward that crashes on `nn.Embedding` — see "Known issues" below. |
| `get_dp_trainable_parameters` | — (DP-SGD plumbing) | Filters out `Transformer.head` / `Transformer.last_normalization`, two submodules TabSyn defines but never calls in `forward()`. DP-SGD requires every optimized parameter to receive a per-sample gradient, and dead parameters never do. |
| `RDPAccountant` | Sec. 3.3/4.4 | Tracks Rényi-DP privacy loss across training steps and converts it to an (ε, δ) guarantee, so you can report how much privacy budget a training run spent. |

### `tabsyn/vae/model.py`

- Added `Model_VAE.forward_with_stages()`, which returns a dict with the
  latent (`mu_z`, `std_z`), decoder output (`h`), and reconstruction
  (`recon_x_num`, `recon_x_cat`) in one call — the disentanglement loss
  needs all three simultaneously, and the original `forward()` only
  returned the final reconstruction.
- `Tokenizer`'s `category_offsets` is now a plain Python list materialized
  into a tensor at forward time, instead of a registered buffer. Opacus's
  `GradSampleModule` rejects modules with buffers by default, and the
  buffer here is a constant that per-sample gradients don't need to see.

### `tabsyn/vae/main.py`

- `compute_loss()` gained an `s_idx` argument: when set, the protected
  attribute's column is **excluded from the reconstruction cross-entropy**
  (the paper's Sec. 4.2 — the protected attribute's value should be
  randomizable, not something the model tries hard to reconstruct) and its
  logits are instead scored with `uniform_attribute_loss`.
- The training loop is now **two-phase**:
  - **Phase 1 (quality)** — unchanged β-VAE loss (MSE + CE + β·KLD) plus
    `L_S`, for `--phase1_epochs` epochs.
  - **Phase 2 (disentanglement)** — freezes a copy of the Phase-1 encoder,
    then adds `SWD(frozen_latents, current_latents) + λ · disentanglement_loss`
    for the remaining epochs, where λ is `--lambda_fair`.
- Batching uses `BalancedGroupSampler` whenever `--sensitive_idx` is set.
- DP-SGD is wired in behind `--dp`: the model is wrapped in Opacus's
  `GradSampleModule`, the optimizer becomes a `DPOptimizer` (per-sample
  gradient clipping + Gaussian noise), and an `RDPAccountant` reports the
  spent (ε, δ) every epoch. **[Updated]** The loss is now split into two
  passes: the per-sample-decomposable terms go through Opacus DP-SGD,
  while the batch-coupled fairness gradient (SWD + CKAᵀ) goes through a
  separate Gaussian mechanism (`GroupLevelDPMechanism`) — whole-vector
  clipping, substitution sensitivity 2C — so the fairness terms carry a
  record-level guarantee that composes with the DP-SGD budget. The
  accountant uses the minority group's sampling rate (γ_max), not
  batch/N. See `FINALREADME.md` Part 9.
- New CLI flags: `--sensitive_idx`, `--phase1_epochs`, `--lambda_fair`,
  `--dp`, `--noise_multiplier`, `--max_grad_norm`, `--dp_delta`,
  `--group_noise_multiplier`, `--max_group_grad_norm`.

### `tests/test_flip_fairness.py` (new file)

61 unit tests covering every function above in isolation (CKAᵀ identity/
symmetry/boundedness and its covariance-cosine form, disentanglement sign
and gradient flow, SWD identity/shift sensitivity, `L_S` at uniform/skewed
logits plus its per-sample variant, sampler group balance and epoch
length, RDP accountant monotonicity and plausible magnitudes, the
multi-stage forward pass, the group-level mechanism's 2C sensitivity,
RDP budget composition, DP statistics releases, and end-to-end smoke
tests — plain and under DP-SGD). Run with:

```powershell
c:/Users/db234/OneDrive/Documents/Vector/.venv/Scripts/python.exe tests/test_flip_fairness.py
```

### `run_flip_demo.py` (new file, workspace root)

A self-contained script that trains FLIP on a sample of
`Data/bmo_nationwide_preprocessed.csv` and scores the result on all three
axes: fidelity (Wasserstein-1 / total-variation distance / correlation
preservation vs. the real data), privacy (distance-to-closest-record ratio
+ the DP accountant's ε), and fairness (demographic parity difference/ratio
of the synthetic data's loan-origination rate across protected groups, vs.
the real data's). All the fidelity/privacy/fairness trade-off knobs live in
one `TRADEOFF_PARAMS` dict at the top of the file, each documented with
which direction it pushes the three metrics.

## How to run

### Unit tests

```powershell
c:/Users/db234/OneDrive/Documents/Vector/.venv/Scripts/python.exe tests/test_flip_fairness.py
```

### The demo (fidelity/privacy/fairness report on real HMDA data)

```powershell
c:/Users/db234/OneDrive/Documents/Vector/.venv/Scripts/python.exe run_flip_demo.py
```

Runs on GPU automatically if `torch.cuda.is_available()` (edit
`TRADEOFF_PARAMS['DEVICE']` to force CPU). Key knobs to try:

- `SENSITIVE_COL` / `RACE_GROUPS` — which protected attribute and which of
  its values to include (defaults to `derived_race`, the most biased
  attribute in this dataset).
- `LAMBDA_FAIR` — raise for more fairness, at a fidelity cost.
- `NOISE_MULTIPLIER` / `MAX_GRAD_NORM` — raise `NOISE_MULTIPLIER` /
  lower `MAX_GRAD_NORM` for a tighter privacy guarantee, at a fidelity
  (and, per the paper, fairness-effectiveness) cost.
- `USE_DP = False` — isolate the fairness intervention's effect without
  DP noise.
- `MAX_DIV_PENALTY` / `PHASE2_LR_SCALE` — stability knobs; see below.

### The full TabSyn VAE training entry point

```powershell
cd tabsyn-main/tabsyn-main
python -m tabsyn.vae.main --dataname <your_dataset> --sensitive_idx <col> `
    --phase1_epochs 2000 --lambda_fair 1.0 `
    --dp --noise_multiplier 1.0 --max_grad_norm 1.0 --dp_delta 1e-5
```

Omit `--dp` to train fairness-only (no privacy noise). Omit
`--sensitive_idx` (or leave it at -1) to fall back to unmodified TabSyn
training.

## Known issues / things to watch

- **Loss spikes under DP-SGD at large batch sizes.** The reconstruction
  loss can spike by orders of magnitude for a few epochs before recovering
  — this is DP noise perturbing the β-VAE loss, not a bug in the fairness
  code. `MAX_DIV_PENALTY` (clamps the Phase-2 SWD term) and
  `PHASE2_LR_SCALE` (drops the learning rate when Phase 2 starts) reduce
  but don't fully eliminate this; lowering `LR` and/or `BATCH_SIZE`,
  or raising `NOISE_MULTIPLIER`'s companion `MAX_GRAD_NORM` clipping
  budget, helps further.
- **Opacus + `nn.Embedding` under vmap is broken** on the torch/Opacus
  versions in this environment (`functorch` re-forward raises
  `TypeError: only integer tensors of a single element can be converted
  to an index`). This is why `register_tabsyn_grad_samplers()` exists —
  it bypasses functorch entirely for the two custom modules that contain
  embeddings/elementwise ops Opacus doesn't have built-in support for.
  If you upgrade Opacus/torch and this starts working natively, the
  registration call becomes a no-op safety net, not a requirement.
- **Opacus 1.6 wants torch ≥ 2.6**; this environment runs torch
  2.5.1+cu121 (the newest CUDA 12.1 build available) to get GPU support
  with the installed driver. `pip` will warn about this dependency
  mismatch; in testing it did not cause functional problems, but keep it
  in mind if something looks wrong after a future `pip install --upgrade`.
- **The demo's generation step is not TabSyn's real sampler.** TabSyn
  normally trains a diffusion model on the VAE's latents and samples from
  that. `run_flip_demo.py` skips this (to stay a single self-contained
  script) and instead samples latents from a plain Gaussian — a rough
  stand-in, not the paper's actual generation quality. For a real
  fidelity number, train `tabsyn/main.py`'s diffusion stage on the FLIP
  VAE's latents and sample from it instead.
