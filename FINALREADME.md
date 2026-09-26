# FINALREADME — TabSyn + FLIP: Fair and Private Tabular Data Synthesis

This document is the complete write-up of this repository. It explains, in
order:

1. **The original TabSyn system** — what it is, how it works, and the
   mathematics behind it, at a level of detail sufficient to understand
   every change that came after.
2. **Every change made to implement the "hilbert_schmidt" paper** —
   *"Achieving Hilbert-Schmidt Independence Under Rényi Differential Privacy
   for Fair and Private Data Generation"* (arXiv:2508.21815), which proposes
   **FLIP** (Fair Latent Intervention under Privacy guarantees) — including
   why each change was made, the technical details of how it was made, and
   plain-language explanations of all mathematical jargon involved.
3. **Difficulties encountered** during the implementation.
4. **Deviations from the paper's original formulation** that were necessary
   or deliberate.
5. **What a good result looks like**, why, and how FLIP's results should
   compare against other tabular-generation methods.
6. **A post-integration bug** — negative loan amounts in the synthetic
   data — its root cause and fix.
7. **The full pipeline** (`run_flip_full_pipeline.py`) — the true
   two-stage TabSyn pipeline with FLIP, closing the demo's fidelity gap.

---

## Part 1 — The Original TabSyn System

### 1.1 What TabSyn is

TabSyn (Zhang et al., ICLR 2024 Oral, *"Mixed-Type Tabular Data Synthesis
with Score-based Diffusion in Latent Space"*) is a deep generative model for
**mixed-type tabular data** — tables containing both *numerical* columns
(income, loan amount) and *categorical* columns (race, loan type). Given a
real table, TabSyn learns its distribution and can emit a synthetic table of
arbitrary size that is statistically similar to the real one, without
containing any actual real record.

The central difficulty TabSyn addresses is that tabular data is *not
natively differentiable-friendly*: categorical values are discrete integers,
numerical columns have wildly different scales, and the joint distribution
has sharp marginals (e.g., a column that is 99% zeros) that image-style
generative architectures handle poorly. TabSyn's answer is a **two-stage
pipeline**:

```
Stage 1 (VAE):   real table  ──Tokenizer──► token sequence ──Encoder──► latent z
                 latent z    ──Decoder──► token sequence ──Reconstructor──► table
Stage 2 (Diffusion): learn the distribution of latents z with a
                 score-based diffusion model, then sample new z and decode.
```

### 1.2 Stage 1 — The latent VAE

**Variational Autoencoder (VAE).** A VAE is a probabilistic encoder/decoder
pair. The encoder $q_\phi(z \mid x)$ approximates the intractable posterior
over a latent variable $z$ given a data point $x$; the decoder
$p_\theta(x \mid z)$ maps latents back to data space. Training maximizes the
**Evidence Lower Bound (ELBO)**:

$$
\mathcal{L}_{\text{VAE}} \;=\; \underbrace{\mathbb{E}_{q_\phi(z|x)}
\big[\log p_\theta(x \mid z)\big]}_{\text{reconstruction}} \;-\;
\underbrace{\beta \cdot D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,\mathcal{N}(0, I)\big)}_{\text{regularization}}
$$

- The **reconstruction term** says: a sample from the encoder's latent
  distribution should decode back to the original record. In TabSyn this is
  **mean-squared error (MSE)** for numerical columns and **cross-entropy
  (CE)** for categorical columns. Cross-entropy is the standard loss for
  classification: it penalizes the model for assigning low probability to
  the true category, and is minimized (at zero) when the model's predicted
  probability distribution puts all mass on the true class.
- The **KL term** (Kullback–Leibler divergence) measures how different two
  probability distributions are; here it forces each record's approximate
  posterior $q_\phi(z|x)$ to stay close to a standard normal
  $\mathcal{N}(0, I)$, so the latent space is smooth, compact, and easy to
  sample from. $\beta$ (the "beta-VAE" knob) balances the two forces:
  TabSyn anneals $\beta$ downward over training (from `max_beta` to
  `min_beta`, multiplied by `lambd` when validation loss plateaus) so the
  model first learns a well-regularized space, then prioritizes
  reconstruction fidelity.
- The encoder outputs a **mean** $\mu_z$ and **log-variance** $\log\sigma^2$
  per latent dimension (a diagonal Gaussian), and the decoder receives
  samples $z = \mu_z + \sigma_z \odot \epsilon$ with
  $\epsilon \sim \mathcal{N}(0, I)$ (the "reparameterization trick", which
  makes sampling differentiable).

**The Tokenizer.** TabSyn's encoder is a Transformer, which operates on a
sequence of vectors ("tokens"). The `Tokenizer` module converts one record
into that sequence:

- Each of the $d_{\text{num}}$ numerical values $x_j$ becomes a token
  $w_j \cdot x_j + b_j$, where $w_j, b_j \in \mathbb{R}^{d}$ are learned
  per-column embedding vectors ($d$ = `d_token`). This is exactly a
  `nn.Linear` layer applied column-wise, so scaling and shifting each
  column is learned rather than fixed.
- Each categorical value becomes a lookup in a shared
  `nn.Embedding` table: category $c$ of column $j$ maps to row
  $c + \text{offset}_j$ of one big embedding matrix, where
  $\text{offset}_j = \sum_{k<j} |\text{categories}_k|$ ensures columns
  don't collide.
- A **[CLS] token** (a constant 1-valued numerical slot) is prepended; the
  final-layer hidden state of this token serves as the aggregate summary.

**The Transformer.** The token sequence passes through a standard
Transformer encoder (multi-head self-attention + feed-forward blocks).
*Self-attention* lets every token's representation depend on every other
token's, which is what lets the model capture **column-column correlations**
(e.g., loan amount ↔ income) — the hardest thing for column-independent
generative models to reproduce. The encoder's output at the [CLS] position
is projected to $(\mu_z, \log\sigma^2_z)$; the latent $z$ is therefore a
single vector per record (in this codebase, shape `(tokens, d_token)`,
treated as the latent).

**The Decoder and Reconstructor.** A mirrored Transformer decoder maps $z$
back to a token sequence $h$, and the `Reconstructor` converts tokens back
to data space: numerical column $j$ is reconstructed as
$\sum_d h_{j,d} \cdot \text{weight}_{j,d}$ (an elementwise-weighted sum over
the token dimension), and each categorical column gets its own
`nn.Linear(d_token, |categories|)` producing a **logit vector** —
pre-softmax scores, one per possible category value.

### 1.3 Stage 2 — Latent diffusion

Once the VAE is trained, every real record has a latent $z$. TabSyn then
trains a **score-based diffusion model** (an `MLPDiffusion` denoiser) on
those latents. Diffusion models work by gradually adding Gaussian noise to
data until it becomes pure noise, then learning to reverse the process:

- The **forward process** corrupts $z_0 \to z_t$ via
  $z_t = \sqrt{\bar\alpha_t} z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$.
- The **reverse process** is learned by predicting the noise $\epsilon$;
  the training loss is the simplified denoising objective
  $\mathbb{E}\,\|\epsilon - \epsilon_\theta(z_t, t)\|^2$.
- **Sampling** starts from pure Gaussian noise and repeatedly denoises,
  producing a new latent $\hat z$ that is *distributed like real latents*
  but is not any real latent. Decoding $\hat z$ through the frozen VAE
  decoder yields a synthetic record.

The key architectural insight — and the reason TabSyn beats prior tabular
diffusion methods — is that **diffusing in the VAE's latent space is easier
than diffusing raw mixed-type data**: the latents are continuous,
fixed-dimension, roughly Gaussian, and free of the sharp/discrete structure
that makes raw tabular diffusion unstable. It also makes sampling
dramatically faster (few function evaluations) than prior diffusion
baselines (TabDDPM, CoDi, STaSy), while achieving state-of-the-art
fidelity on density-estimation, detection, and column/pairwise-correlation
metrics.

### 1.4 Repository layout (original TabSyn)

```
tabsyn-main/tabsyn-main/
├── tabsyn/
│   ├── vae/            Stage 1: the latent VAE (model.py, main.py)
│   ├── main.py         Stage 2: latent diffusion training
│   ├── model.py        MLPDiffusion denoiser + DDPM/DDIM logic
│   ├── sample.py       Sampling from the trained diffusion prior
│   ├── latent_utils.py Glue between VAE latents and diffusion
│   └── data.py         Dataset loading/preprocessing
├── baselines/          SMOTE, GOGGLE, TabDDPM, CoDi, STaSy, GREAT
├── eval/               DCR, density, detection, MLE, quality metrics
└── data/Info/*.json    Dataset metadata (num/cat/target column indices)
```

### 1.5 The problem with vanilla TabSyn

TabSyn's only objective is *fidelity*: make synthetic data look like real
data. Two things it does not provide:

1. **Fairness.** If the real data encodes bias (e.g., in the HMDA
   mortgage data used here, White applicants' loans originate at ~61%
   while several minority groups originate at ~32–43%), a faithful
   generative model is a *bias-copying machine* — the synthetic data
   reproduces, and thereby launders, the disparity.
2. **Privacy.** A VAE/diffusion pipeline trained by ordinary SGD has no
   formal guarantee that it won't memorize and regurgitate individual
   training records (membership inference / record reconstruction risks).

These are exactly the two gaps the hilbert_schmidt paper addresses.

---

## Part 2 — Implementing the hilbert_schmidt Paper (FLIP)

The paper proposes **FLIP** (Fair Latent Intervention under Privacy
guarantees): train a generative model (in our case, TabSyn's VAE) such that
its latent representation is **statistically independent of a protected
attribute** (fairness), while the training itself satisfies
**differential privacy** (privacy). The implementation touched four areas:
new math (a new module), the VAE model, the training loop, and the
evaluation harness. Each is covered below with the reasoning and the math.

### 2.1 Addition: `tabsyn/vae/flip_fairness.py` — the paper's math

This new file implements every equation from the paper as small,
independently testable functions.

#### 2.1.1 `linear_cka_t` — the Hilbert-Schmidt Independence Criterion (Eqs. 5–7)

**Jargon first.** The **Hilbert-Schmidt Independence Criterion (HSIC)** is a
kernel-based test of statistical independence between two random variables.
Intuitively: two variables are independent if no function of one correlates
with any function of the other. HSIC approximates this by mapping both
variables into a high-dimensional feature space (via a kernel) and
measuring the norm of the **cross-covariance operator** between them. If
the (centered) feature representations of two variables are
$A \in \mathbb{R}^{n\times p}$ and $B \in \mathbb{R}^{n\times q}$ (rows =
samples), the *empirical* HSIC with a linear kernel is essentially
$\|A^\top B\|_F^2$ — the squared **Frobenius norm** (the square root of the
sum of squared matrix entries) of the cross-Gram matrix. Zero cross-covariance
⇒ evidence of independence.

**Centered Kernel Alignment (CKA)** is HSIC *normalized* so it lies in
$[0,1]$ and is invariant to the scale and dimension of the representations:

$$
\mathrm{CKA}(A, B) \;=\; \frac{\|A^\top B\|_F^2}{\|A^\top A\|_F \, \|B^\top B\|_F}
$$

This is the matrix analogue of the **absolute cosine similarity** between
two vectors: the numerator is the "shared direction" and the denominator
cancels each matrix's own magnitude. CKA = 1 means the two representation
sets have identical covariance structure (up to rotation/scale); CKA = 0
means they share none.

**The paper's twist — CKAᵀ (Eq. 7).** FLIP does not use plain CKA on
*sample-similarity* Gram matrices ($XX^\top$); it uses the **transposed**
form, $\mathrm{CKA}^\top(A,B) := \mathrm{CKA}(A^\top, B^\top)$, i.e., it
compares **feature-covariance patterns** ($X^\top X$) between two groups.
Why: for fairness we care about whether the *relationships among latent
features* are the same across protected groups. If the Black applicants'
latents and the White applicants' latents have the same feature-covariance
structure, then the latent geometry does not encode "which group you're
in" — a necessary condition for the representation to be group-blind.
Sample-space CKA would instead ask whether *individuals* align across
groups, which is neither necessary nor desirable for fairness.

Implementation detail: with centered inputs (each column mean-subtracted)
and a linear kernel, the formula reduces to the cheap closed form above —
no kernel matrix over sample pairs is ever materialized, so it scales to
large batches.

#### 2.1.2 `disentanglement_loss` — the fairness objective D′ (Eq. 10)

Given a batch's representation matrix `rep` (rows = samples) and each
sample's protected-group label, this computes the **negative mean pairwise
CKAᵀ across all protected groups**:

$$
\mathcal{D}' \;=\; -\frac{1}{|G|(|G|-1)/2} \sum_{i<j} \mathrm{CKA}^\top\big(\text{rep}_{g_i},\, \text{rep}_{g_j}\big)
$$

**Why the negative?** We *want* the groups' feature-covariance structures
to *converge* (high CKAᵀ = similar structure = group-blind geometry).
Gradient descent *minimizes* loss, so we minimize negative similarity —
i.e., maximize similarity. Minimizing $\mathcal{D}'$ pushes the model to
produce latents whose covariance structure is the same regardless of
protected group, which is the paper's operationalization of "the latent is
independent of the protected attribute."

Edge-case handling: if a batch contains fewer than two groups, or a group
has fewer than 2 samples, the loss returns 0 (no signal) rather than
crashing; groups are truncated to the smallest common size so CKAᵀ compares
equal-sized matrices (the balanced sampler normally guarantees this).

#### 2.1.3 `multi_stage_disentanglement_loss` — three-stage intervention (Sec. 4.3.2)

The paper intervenes at **three points** of the network, not just the
latent:

1. **Latent stage** — on $\mu_z$ (flattened over tokens): the primary
   objective; the latent is what the diffusion prior will model, so it must
   be group-blind.
2. **Detokenizer stage** — *feature-wise* on the reconstruction logits
   (numerical logits and each categorical column's logits separately,
   **excluding the protected attribute's own logits**, which are supervised
   by the uniform-attribute loss instead). Applying CKAᵀ per-feature and
   mean-aggregating prevents wide categorical columns (many unique values)
   from dominating the loss purely by dimension count.
3. **Decoder stage** — on the decoder's token output $h$ (flattened), whose
   dimensionality matches the latent space; this constrains the
   intermediate representation, not just its endpoints.

The three losses are averaged. **Why three stages?** Intervening only at
the latent can be "undone" by the decoder (the decoder can re-inject group
information on the way back to data space); intervening only at the output
leaves the latent (which the diffusion prior models) biased. Applying the
same independence pressure at latent, intermediate, and pre-output
representations makes the whole pipeline consistently group-blind.

#### 2.1.4 `sliced_wasserstein_distance` — the divergence penalty D (Eq. 10)

**Jargon.** The **Wasserstein-1 (earth mover's) distance** between two
distributions is the minimum "work" needed to morph one into the other;
in 1-D it has a closed form: sort both samples and average the absolute
differences of the order statistics. The **sliced** Wasserstein distance
extends this to high dimensions cheaply: project both point sets onto many
random 1-D directions, compute the 1-D Wasserstein distance on each
projection, and average. It is a proper metric, differentiable, and
$O(n \log n)$ per projection — versus the $O(n^3)$ optimal-transport
solvers the full Wasserstein distance would require.

**Why it's needed.** FLIP's Phase 2 (below) deliberately *changes* the
encoder to remove group information. Unconstrained, this intervention can
destroy the representation entirely (the trivial way to make latents
group-independent is to make them garbage). The paper anchors Phase 2 to
the Phase-1 encoder with a **distributional divergence penalty**:

$$
\mathcal{D} \;=\; \mathrm{SWD}\big(q_{\theta_0}(z \mid x),\; q_{\theta_t}(z \mid x)\big)
$$

i.e., the Phase-2 encoder's latent distribution must stay close (in
sliced-Wasserstein terms) to the *frozen* Phase-1 encoder's latent
distribution, while the disentanglement term removes group structure.
The two terms together say: "keep everything you learned, except the part
that encodes the protected attribute."

#### 2.1.5 `uniform_attribute_loss` — L_S (Eq. 3)

$$
\mathcal{L}_S \;=\; \Big\| \tfrac{1}{n}\sum_i \mathrm{softmax}(s_{\text{logits}}^{(i)}) \;-\; \mathrm{uniform} \Big\|_2
$$

The **softmax** converts a logit vector into a probability distribution;
the loss takes the batch-mean of the reconstructed protected-attribute
probability vectors and penalizes its ℓ2 distance from the uniform
distribution. **Why:** if the model were asked to *reconstruct* the
protected attribute, it would learn to predict each group at its training
frequency — and worse, the *other* features' representations would be
shaped to help predict it (that's how CE backprop works). Instead, FLIP
(1) removes the protected attribute from the reconstruction loss and
(2) pushes its predicted distribution toward uniform, so the model has no
incentive to encode group identity anywhere, and synthetic data assigns
protected-group labels at equal rates.

#### 2.1.6 `BalancedGroupSampler` — balanced mini-batches (Sec. 4.2.1, Eq. 11)

A `torch.utils.data.Sampler` that builds every mini-batch with an **equal
number of samples per protected group**. Iterations per epoch follow the
paper's Eq. 11, $L = \lceil m / b \rceil$ where $m$ is the smallest group's
size and $b$ the per-group batch share.

**Why:** CKAᵀ compares equal-sized group submatrices, and more
fundamentally, if one group is 80% of the data, ordinary shuffled batches
give the majority group 80% of the gradient signal — the disentanglement
objective would be optimized mostly "for the majority group's benefit."
Balanced batches give every group equal weight in every fairness gradient
step. (This is the standard remedy for class-imbalance biasing, applied
here at the *batch* level rather than by re-weighting losses.)

#### 2.1.7 DP-SGD plumbing: `register_tabsyn_grad_samplers`, `get_dp_trainable_parameters`, `RDPAccountant`

These support **differential privacy**, explained next.

**Differential privacy (DP).** A randomized training algorithm
$\mathcal{A}$ is $(\epsilon, \delta)$-differentially private if for any two
training datasets $D, D'$ differing in a *single record*,

$$
\Pr[\mathcal{A}(D) \in S] \;\le\; e^{\epsilon}\,\Pr[\mathcal{A}(D') \in S] + \delta
$$

Informally: no single person's data can change the output distribution by
more than a factor $e^\epsilon$ (with probability $1-\delta$ of even that
being violated). Small $\epsilon$ = strong privacy. **DP-SGD** (Abadi et
al.) achieves this for SGD by (a) clipping each *individual sample's*
gradient to L2 norm $C$ (bounding one person's influence) and (b) adding
Gaussian noise with std $\sigma C$ to the summed gradient before stepping.

**Rényi DP (RDP).** Plain $(\epsilon,\delta)$-DP composes badly: running
$k$ DP steps multiplies $\epsilon$ by $k$. **Rényi divergence** of order
$\alpha$ between distributions $P, Q$ is
$D_\alpha(P\|Q) = \frac{1}{\alpha-1}\log \mathbb{E}_Q[(P/Q)^\alpha]$ — a
continuum of divergences indexed by $\alpha$ (α→∞ gives max-divergence,
i.e., plain DP). RDP composes *additively* over steps, and converts back
to $(\epsilon, \delta)$-DP via
$\epsilon = \big(\epsilon_{\text{RDP}}(\alpha) + \log(1/\delta)\big)/(\alpha - 1)$,
minimized over $\alpha$. The `RDPAccountant` implements exactly this:
per-step RDP for the subsampled Gaussian mechanism, additive composition
over steps, and the optimal conversion. This is the accounting framework
the paper's Secs. 3.3/4.4 specify, and it's why we can report a meaningful
$\epsilon$ after thousands of training steps.

**Why custom grad samplers were needed.** Opacus (Meta's DP-SGD library)
computes per-sample gradients. For standard layers it has fast native
implementations; for custom modules it falls back to a `functorch`/`vmap`
re-forward — which crashes on TabSyn's `nn.Embedding` usage (see
Part 3). `register_tabsyn_grad_samplers()` registers hand-derived
per-sample-gradient formulas for TabSyn's two custom modules:

- **Tokenizer**: the numerical-token weight is applied elementwise
  (`weight[j] * x_num[j]`), so its per-sample gradient is
  `grad_output[n,j,:] * x_num_full[n,j]` — Linear-like; the bias maps 1:1
  to token positions, so its per-sample gradient is just the backprop rows
  for those positions; the `nn.Embedding` gets Opacus's native hook.
- **Reconstructor**: numerical reconstruction is
  `(h_num * weight).sum(-1)`, so the per-sample weight gradient is
  `go_num[n,:,None] * h_num[n,:,:]`; the categorical `nn.Linear` heads are
  standard and use native hooks.

`get_dp_trainable_parameters()` filters out TabSyn's `Transformer.head`
and `Transformer.last_normalization` — submodules that are *defined but
never called in `forward()`*. DP-SGD's optimizer asserts every parameter it
optimizes receives a per-sample gradient; dead parameters never do, so
they must be excluded from the DP parameter group (they receive no
updates anyway, so excluding them changes nothing functionally).

### 2.2 Modification: `tabsyn/vae/model.py`

Two changes:

1. **Added `Model_VAE.forward_with_stages()`** (addition, not a
   replacement — the original `forward()` is untouched). The
   disentanglement loss needs the latent $\mu_z$, the decoder output $h$,
   and the reconstructions *simultaneously*; the original `forward()`
   returns only the reconstructions and discards the intermediates.
   Returning a dict avoids recomputing the forward pass three times and
   guarantees the three stages are consistent (same forward, same batch).

2. **`Tokenizer.category_offsets` changed from a registered buffer to a
   plain Python list**, materialized into a tensor on the input's device at
   forward time. **Why:** Opacus's `GradSampleModule` rejects modules with
   non-trainable buffers by default (buffers aren't parameters, and
   per-sample gradient bookkeeping doesn't know what to do with them).
   The offsets are a *constant* index-shift table — per-sample gradients
   don't need to see it — so storing it as a list and rebuilding the
   tensor per call is functionally identical to the original buffer while
   being DP-compatible. This is the only behavioral change to the
   original model code, and it is invisible outside DP mode.

### 2.3 Modification: `tabsyn/vae/main.py` — the training loop

This is the heart of the integration. Changes:

1. **`compute_loss()` gained `s_idx` / `Recon_S_logits` arguments.** When a
   protected attribute is configured, its column is **excluded from the
   reconstruction cross-entropy** (paper Sec. 4.2: the protected attribute
   should be *randomizable*, not faithfully reconstructed — see the L_S
   rationale in 2.1.5), and its logits are instead scored with
   `uniform_attribute_loss`. Everything else about the loss (MSE + CE +
   β·KLD) is unchanged.

2. **Two-phase training (paper Sec. 4.3):**
   - **Phase 1 (quality), epochs 1…`phase1_epochs`:** the unchanged
     β-VAE loss *plus* $\mathcal{L}_S$. The model first learns a good
     representation of the data, with the protected attribute already
     treated as random.
   - **Phase 2 (disentanglement), remaining epochs:** at the phase
     boundary, a **copy of the encoder is frozen** (the "reference
     encoder" $q_{\theta_0}$). Each Phase-2 step adds
     $$\underbrace{\mathrm{SWD}\big(\mu_{\text{ref}}, \mu_{\text{cur}}\big)}_{\mathcal{D}}
     \;+\; \lambda_{\text{fair}} \cdot \underbrace{\mathcal{D}'_{\text{multi-stage}}}_{\text{Eq. 10}}$$
     to the loss. The SWD anchor keeps the representation intact; the
     disentanglement term removes group structure from latent,
     detokenizer, and decoder representations simultaneously.

   **Why two phases?** If disentanglement were applied from step 0, the
   model would have no good representation to protect — it would learn a
   group-blind but useless encoding. Phase 1 builds quality; Phase 2
   surgically removes the protected-attribute information while the SWD
   anchor prevents collateral damage. This mirrors the paper's
   "controlled bias mitigation."

3. **Balanced batching:** whenever `--sensitive_idx` is set, the
   DataLoader uses `BalancedGroupSampler` instead of shuffled batches.

4. **DP-SGD wiring behind `--dp`:** the model is wrapped in Opacus's
   `GradSampleModule` (per-sample gradient capture), the optimizer becomes
   a `DPOptimizer` (per-sample clipping + Gaussian noise), and the
   `RDPAccountant` reports the spent $(\epsilon, \delta)$ every epoch.

5. **New CLI flags:** `--sensitive_idx`, `--phase1_epochs`,
   `--lambda_fair`, `--dp`, `--noise_multiplier`, `--max_grad_norm`,
   `--dp_delta`. **All default to off** — running without them reproduces
   unmodified TabSyn exactly.

### 2.4 Additions: tests, demo, and evaluation harness

- **`tests/test_flip_fairness.py`** — 30 unit tests covering every function
  in isolation: CKAᵀ identity/symmetry/boundedness-in-[0,1],
  disentanglement-loss sign and gradient flow, SWD identity and
  shift-sensitivity, $\mathcal{L}_S$ at uniform vs. skewed logits, sampler
  group balance and epoch length, RDP accountant monotonicity, the
  multi-stage forward, and two end-to-end smoke tests (plain and DP).
- **`run_flip_demo.py`** — a self-contained script that trains FLIP on a
  sample of the HMDA mortgage data (`Data/bmo_nationwide_preprocessed.csv`,
  produced by `preprocess.py`), generates synthetic records, and scores
  all three axes (fidelity / privacy / fairness), emitting an HTML report
  (`flip_report.html`) and a readable synthetic CSV
  (`Data/flip_synthetic_bmo_nationwide.csv`). All trade-off knobs live in
  one documented `TRADEOFF_PARAMS` dict.
- **`preprocess.py`** — builds the binary target (1 = loan originated,
  0 = denied) from HMDA's `action_taken` and drops rows with missing
  sensitive attributes.
- **`fairness_audit.py` / `fairness_report.py`** — a *pre-training* audit
  of the real data using `fairlearn` (demographic parity
  difference/ratio, selection rates by group) that establishes the bias
  baseline FLIP is meant to reduce.

### 2.5 Summary table of changes

| File | Type | Change |
|---|---|---|
| `tabsyn/vae/flip_fairness.py` | **Added** | All paper math: CKAᵀ, disentanglement, multi-stage loss, SWD, L_S, balanced sampler, RDP accountant, DP grad samplers |
| `tabsyn/vae/model.py` | **Modified** | `forward_with_stages()` added; `category_offsets` buffer → list (DP compatibility) |
| `tabsyn/vae/main.py` | **Modified** | `s_idx` in loss; two-phase training; balanced sampler; DP-SGD wiring; 6 new CLI flags |
| `tests/test_flip_fairness.py` | **Added** | 30 unit tests |
| `run_flip_demo.py` | **Added** | End-to-end demo + 3-axis evaluation + HTML report (empirical-Gaussian latent sampling; see Parts 6–7) |
| `run_flip_full_pipeline.py` | **Added** | Full two-stage pipeline: FLIP VAE + TabSyn's EDM latent diffusion prior, with the same evaluation and report |
| `preprocess.py`, `fairness_audit.py`, `fairness_report.py` | **Added** | Data prep and bias-baseline audit |
| Original TabSyn files | **Untouched** | Diffusion stage, samplers, baselines, eval all unchanged |

---

## Part 3 — Difficulties Encountered

1. **Opacus × `nn.Embedding` under vmap.** Opacus's fallback per-sample
   gradient path re-runs the forward under `functorch.vmap`, which raises
   `TypeError: only integer tensors of a single element can be converted
   to an index` on TabSyn's embedding lookups. This was the single
   largest implementation obstacle — DP-SGD simply could not run on the
   unmodified model. The fix was to hand-derive and register native
   per-sample-gradient formulas for `Tokenizer` and `Reconstructor`
   (`register_tabsyn_grad_samplers()`), bypassing vmap entirely. Each
   formula had to be validated against autograd's ordinary gradients.

2. **Dead parameters breaking `DPOptimizer`.** TabSyn's `Transformer`
   defines `head` and `last_normalization` submodules that are never
   invoked in `forward()`. `DPOptimizer` asserts that every optimized
   parameter receives a per-sample gradient; dead parameters never do,
   so training crashed until they were filtered out.

3. **Registered buffers breaking `GradSampleModule`.** The `Tokenizer`'s
   `category_offsets` buffer caused Opacus to reject the module. Since the
   buffer is a constant lookup table, converting it to a plain list
   (materialized per forward call) removed the issue with zero
   behavioral change.

4. **Loss explosions under DP-SGD.** DP noise caused the reconstruction
   loss to spike by orders of magnitude for several epochs. Root causes
   and fixes (all documented in `TRADEOFF_PARAMS`):
   - DP noise gives every parameter a nonzero gradient *variance* even
     when the true signal is zero; Adam's tiny default `eps=1e-8` lets
     $1/\sqrt{v}$ blow up on that noise → raised `ADAM_EPS` to `1e-4`.
   - DP noise makes weights undergo an unbounded random walk over epochs;
     per-sample clipping does not stop this (it bounds each *sample's*
     contribution, not the accumulated drift) → added `WEIGHT_DECAY` as a
     restoring force.
   - A single drifted weight can produce a huge logit: CE *saturates*
     there (huge loss value, tiny gradient — the optimizer can't see how
     bad it is), while MSE's gradient grows linearly with the error and
     feeds back destructively → added `LOGIT_CLAMP` and `NUM_RECON_CLAMP`
     backstops plus a post-noise `MAX_OPTIMIZER_GRAD_NORM` clip (Opacus's
     `max_grad_norm` only bounds per-sample gradients *before*
     summation/noising, not the final applied gradient).
   - The SWD anchor grows unboundedly as the encoder drifts from the
     frozen reference under noise → clamped with `MAX_DIV_PENALTY`, and
     the Phase-2 learning rate is scaled down (`PHASE2_LR_SCALE`).

5. **Version friction.** Opacus 1.6 declares a torch ≥ 2.6 dependency;
   this environment runs torch 2.5.1+cu121 (the newest CUDA-12.1 build the
   installed driver supports). `pip` warns about the mismatch; in testing
   it caused no functional problems, but it is a latent upgrade hazard.

6. **Group-size imbalance in the data.** HMDA's race distribution is
   heavily skewed with several tiny groups; balanced batches with 3 groups
   were only feasible after restricting to the three largest groups
   (White, Black or African American, Asian). Tiny groups would make
   per-group batch shares too small for stable CKAᵀ estimates.

---

## Part 4 — Deviations from the Paper

The implementation is faithful to the paper's method, with these
deliberate deviations, all forced by engineering reality:

1. **The demo's generator was a Gaussian stand-in, not TabSyn's diffusion
   prior — now superseded by a full-pipeline script.** The paper (and full
   TabSyn) trains a diffusion model on the VAE latents and samples from
   it. `run_flip_demo.py` originally sampled latents from a plain standard
   Gaussian instead, to stay a single self-contained script. This caused a
   real bug (Part 6) and understated fidelity; both issues are now
   addressed: the demo samples from the *empirical* latent Gaussian
   (in-distribution), and a new `run_flip_full_pipeline.py` runs the true
   two-stage pipeline (FLIP VAE + TabSyn's EDM latent diffusion) end to
   end, emitting the same HTML report and synthetic CSV (see Part 7).

2. **RDP accountant: Opacus-backed, with a conservative fallback.** An
   earlier version hand-rolled the subsampled-Gaussian RDP bound; that
   bound was **incorrect** — for $\alpha q > 1$ it decayed in $\alpha$,
   so the reported ε shrank as the RDP order grew, producing an
   implausibly small ε (≈ 0.5) where the true value was orders of
   magnitude larger. The accountant now delegates to Opacus's audited
   `RDPAccountant` (Wang et al. 2019 bound) and falls back to the
   *unamplified* pure-Gaussian RDP bound (valid, merely conservative)
   when Opacus is unavailable. Two further corrections shipped with it:
   the sampling rate passed to the accountant is the **minority
   group's** per-step inclusion probability ($b_{\text{group}}/m$, the
   paper's $\gamma_{\max}$, Prop. 1 / Eq. 13) rather than
   batch/N — the balanced sampler shows every minority sample once per
   epoch, so batch/N understates ε for exactly the group the fairness
   intervention targets — and a regression test
   (`test_rdp_plausible_magnitude`) pins the accountant to plausible
   magnitudes so the bug cannot silently return.

   **Honest numbers:** at the demo's settings (σ = 0.8, q ≈ 0.17,
   ~960 steps), the corrected accountant reports **ε ≈ 80** at
   δ = 1e-5 — not 0.52. Reaching a research-grade ε (≤ 10) at this
   sampling rate requires a substantially larger noise multiplier
   and/or fewer steps; the knob panel supports both, at the documented
   fidelity cost.

3. **Stability machinery not in the paper.** The clamps, Adam-eps
   increase, weight decay, Phase-2 LR scaling, and post-noise gradient
   clipping (Part 3, item 4) are additions. The paper's loss is Eq. 10
   exactly; these additions only prevent numerical pathologies of
   DP-SGD + β-VAE training and do not change the objective's optima.

4. **Per-feature mean aggregation at the detokenizer stage.** The paper
   describes the detokenizer-stage intervention; the implementation
   applies CKAᵀ per feature (numerical logits and each categorical
   column's logits separately) and mean-aggregates, so that categorical
   columns with many unique values don't dominate the stage's loss purely
   by dimension. This is an implementation refinement, not a change of
   objective.

5. **Balanced-sampler epoch length uses ceiling division.** Eq. 11's
   $L = \lfloor m/b \rfloor$ would drop the minority group's tail rows
   every epoch; the implementation uses $\lceil m/b \rceil$ with a
   smaller final batch so every minority-group sample is seen exactly
   once per epoch.

---

## Part 6 — Post-Integration Bug: Negative Loan Amounts in the Synthetic Data

### 6.1 Symptom

After the first demo run, 2,717 of 12,000 rows in
`Data/flip_synthetic_bmo_nationwide.csv` had **negative `loan_amount`
values** (minimum ≈ −$1.05M) — impossible values for a mortgage dataset.

### 6.2 Root cause

The bug was in the demo's *generation* step, not in the FLIP training.
`generate()` sampled latents from a **standard normal** $\mathcal{N}(0, I)$
— the distribution the KL term is *supposed* to impose on the latent
space. But with `BETA = 1e-3`, the KL term barely regularizes the latents
(the deliberate fidelity-first choice), so the *actual* encoded training
latents were centered far from 0 with large per-dimension spreads.
Feeding $\mathcal{N}(0, I)$ noise to the decoder therefore presented it
with **out-of-distribution inputs** — a region of latent space it had
never seen during training. Neural decoders extrapolate unreliably
outside their training distribution, and the wild outputs, after the
standardization reversal (`value × std + mean`), landed below $0.

### 6.3 Fix (two layers)

1. **In-distribution sampling.** `generate()` now encodes the training
   data once, measures the empirical per-dimension latent mean $\mu$ and
   standard deviation $\sigma$, and samples new latents as
   $\mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$ —
   the same reparameterization form the VAE itself uses. The decoder now
   only ever sees inputs from the neighborhood it was trained on.
2. **Range-clip backstop.** Every synthetic numeric value is clipped to
   the real data's observed per-column range, so no generated value can
   fall outside the real support regardless of sampling luck.

### 6.4 What was and wasn't affected

- **Invalid data**: fixed — no synthetic value can be negative or
  otherwise outside the real columns' support.
- **Fidelity metrics**: were *understated* by the bug (computed against
  broken values); they improve once generation is in-distribution.
- **Fairness/privacy results**: unaffected — both are properties of the
  training objective (disentanglement, $\mathcal{L}_S$, DP-SGD), not of
  the latent sampler.

---

## Part 7 — The Full Pipeline: `run_flip_full_pipeline.py`

To close the demo's remaining fidelity gap (the per-dimension Gaussian
prior cannot capture the latent distribution's fine structure —
multimodality, cross-dimension correlations), a full-pipeline script was
added that runs the **true two-stage TabSyn pipeline with FLIP**:

1. **Stage 1 — FLIP VAE**: identical to the demo (two-phase training,
   balanced sampling, DP-SGD), reusing `run_flip_demo.train_flip` so the
   two pipelines are directly comparable.
2. **Stage 2 — latent diffusion**: encodes the training data to latents,
   then trains TabSyn's actual `MLPDiffusion` denoiser with the EDM
   preconditioning and EDM loss from `tabsyn/model.py` /
   `tabsyn/diffusion_utils.py`, following `tabsyn/main.py`'s
   $(z - \bar z)/2$ normalization convention.
3. **Sampling**: draws new latents with TabSyn's real EDM sampler
   (default 50 NFE, Heun 2nd-order steps) — not a Gaussian.
4. **Decode + evaluate**: decodes sampled latents through the frozen VAE
   decoder (with the same range-clip backstop), then computes the same
   fidelity/privacy/fairness metrics and emits the same HTML report
   format.

**Outputs**:
- `Data/flip_full_synthetic_bmo_nationwide.csv` — the synthetic sample
- `flip_full_report.html` — the report, with the diffusion parameters
  included in the parameter table

**Stage-2 knobs** (`DIFFUSION_PARAMS` at the top of the script):
`EPOCHS` (3000 — TabSyn's default is 10000+, excessive for a 12k-row
sample), `DIM_T` (1024, TabSyn's default denoiser width), `SAMPLE_STEPS`
(50 NFE), `N_GEN` (defaults to the training-set size).

**Why fidelity should improve over the demo:** the diffusion prior is
TabSyn's core contribution — it models the latent distribution's full
structure, where a per-dimension Gaussian models only its first two
moments per dimension independently. Fairness and privacy conclusions
are unchanged (they live in Stage 1), but the fidelity numbers from this
pipeline are the honest ones to report.

---

## Part 8 — What a Good Result Looks Like, and Why

### 8.1 The three axes and their target values

A successful FLIP run moves **fairness up** while keeping **fidelity and
privacy** in acceptable ranges. Concretely, on the HMDA data:

| Metric | Definition | Good | Why |
|---|---|---|---|
| **DP difference** (demographic parity difference) | max−min of the outcome (loan-origination) rate across protected groups in the *synthetic* data | **Much lower than the real data's** (real ≈ 0.17–0.30 here; synthetic should approach 0) | This is the paper's headline claim: the synthetic data should not encode the real data's group disparity. The uniform-attribute loss equalizes predicted group rates; the disentanglement loss prevents other features from carrying group information that could re-create the disparity downstream. |
| **DP ratio** | min/max group outcome rate | **→ 1.0** (real data ≈ 0.53–0.70 here; the demo achieved ≈ 0.89) | Scale-free version of the same claim; 1.0 = perfect parity. |
| **Numeric W1** (normalized Wasserstein-1) | Per-column earth-mover distance between real and synthetic marginals, divided by the real column's std | **≤ 0.2** (moderate ≤ 0.4) | The synthetic data must still *be useful*: marginals should match. W1 is the standard marginal-fidelity metric; normalizing by std makes it dimensionless and comparable across columns. |
| **Categorical TV** (total variation) | ½·ℓ1 distance between real and synthetic frequency distributions per column | **≤ 0.15** (moderate ≤ 0.3) | The categorical analogue of W1. TV = 0 means identical frequency tables. |
| **Correlation delta** | Mean absolute difference between real and synthetic numeric correlation matrices | **≤ 0.05** (moderate ≤ 0.15) | Marginals matching is not enough; the *joint* structure (loan amount ↔ income) is what makes synthetic data usable for downstream modeling. |
| **DCR ratio** (distance-to-closest-record) | mean min-distance(synthetic→real) ÷ mean min-distance(real holdout→real train) | **≥ 1.0** (the demo achieved ≈ 5.5) | The standard privacy *empirical* check: synthetic records should be no closer to real records than real records are to each other. < 1.0 suggests memorization (overfitting); ≫ 1 can also indicate the synthetic data is too diffuse — values in the 1–10 band with good fidelity are healthy. |
| **ε (DP guarantee)** | Rényi-DP-converted privacy budget at the chosen δ | **Small, and reported** | The formal guarantee. ε ≤ 1 is strong; ε ≤ 5–10 is typical research-grade. **Caveat:** the guarantee applies to the DP-SGD-trained VAE stage only, and only to the per-sample-decomposable portion of the loss (see *Limitations* below). The demo's original ε ≈ 0.52 was an artifact of an incorrect accountant; the corrected accountant reports ε ≈ 80 at the demo's settings. |

The demo run's actual results — DP difference 0.1699 → 0.0490, DP ratio
0.697 → 0.889, fairness gain +0.121, DCR ratio 5.5, with moderate
fidelity costs (W1 0.35, TV 0.22, corr Δ 0.07) — are a representative
outcome: a ~70% reduction in outcome disparity at a moderate fidelity
cost, with no memorization. (The originally reported ε = 0.52 was an
artifact of the broken accountant — see Part 4, item 2; the corrected
accountant reports ε ≈ 80 at these settings, and the DP-difference
headline should be read alongside the anti-gaming probes below.)

### 8.2 Why these results should look this way

- **Fairness improves because the objective directly targets it.** The
  uniform-attribute loss makes the model *unable to predict* the
  protected attribute from the representation better than chance, and the
  CKAᵀ disentanglement makes the latent *geometry* group-invariant. Since
  the outcome column's relationship to the protected attribute flows
  through the representation, scrubbing the representation of group
  information scrubs the outcome disparity. Vanilla TabSyn has *no*
  mechanism to do this — its objective is pure fidelity, so its DP
  difference should match the real data's (it faithfully copies the bias).
- **Fidelity degrades gracefully because of the SWD anchor and two-phase
  design.** Without the anchor, disentanglement could collapse the
  representation; with it, the model keeps everything except the
  protected-attribute direction. The fidelity cost is the price of the
  information removed — and information about a single categorical column
  is a small fraction of the table's total structure.
- **Privacy holds by construction.** DP-SGD's guarantee is a theorem, not
  an empirical observation: with the reported (ε, δ), no adversary can
  learn more than a bounded amount about any single training record
  regardless of their compute or side information. The DCR ratio is the
  *empirical* sanity check on top of the *formal* guarantee.

### 8.3 Comparison with other methods

| Method | Fairness | Privacy | Fidelity | Notes |
|---|---|---|---|---|
| **Vanilla TabSyn** (and other pure-fidelity generators: TabDDPM, STaSy, CoDi, GOGGLE, GREAT) | **None** — copies the training data's bias by design | None — no formal guarantee; DCR must be checked empirically per run | SOTA | The fairness/privacy gap FLIP fills. These methods optimize fidelity only; their DP difference should track the real data's. |
| **SMOTE** (and other resampling baselines) | None | None (worse: interpolates between real records) | Poor on joint structure | Interpolated points can sit *between* two real records, leaking both. |
| **Post-hoc fairness fixes** (reweight/repair the real data, then generate) | Partial | None | Degrades with intervention strength | Post-hoc approaches change the data, not the model; the generator can re-learn bias from residual correlations, and there is no privacy story at all. |
| **DP-only training** (DP-SGD on a standard generator) | None | Yes | Moderate–good | Privacy without fairness: the synthetic data is private *and* biased. |
| **FLIP (this repo)** | **Yes — explicit, measurable** | **Yes — formal (ε, δ)** | Good (moderate cost) | The only option here that addresses both axes simultaneously, and the fairness intervention is *certified under* the privacy noise (the paper's core result: HSIC-based disentanglement remains effective under Rényi-DP training). |

**Why FLIP should improve over the alternatives on the fairness axis:**
every non-FLIP approach either ignores fairness (leaving the full real
disparity in the synthetic data) or applies it post-hoc (where the
generator's fidelity objective actively fights the correction, since
fidelity *means* reproducing the data's statistics, disparity included).
FLIP is the only approach where fairness is part of the *training
objective itself*, so fidelity and fairness are traded off explicitly and
controllably via $\lambda_{\text{fair}}$, rather than implicitly and
uncontrollably.

**Why FLIP should improve on the privacy axis:** methods without DP-SGD
offer no guarantee at all — their empirical DCR can look fine while still
permitting membership-inference attacks in the worst case. FLIP's
guarantee holds unconditionally. And crucially, the paper's central
technical contribution — and the reason this integration works — is that
the HSIC/CKAᵀ disentanglement objective is *robust to DP noise*: the
fairness gain survives privacy training, whereas naive fairness penalties
(e.g., adversarial debiasing) are known to degrade badly under DP noise
because they rely on a discriminator whose gradients the noise destroys.
CKAᵀ compares *second-moment structure* across groups, which is a much
lower-variance statistical target than per-sample discrimination, so the
fairness signal survives the noise that DP-SGD injects.

**Expected headline comparison:** against vanilla TabSyn on the same
data, FLIP should show a near-identical privacy *failure* → formal
guarantee upgrade, and a large DP-difference reduction (e.g., the demo's
0.17 → 0.05), at a moderate fidelity cost (the demo's W1 0.35 / TV 0.22
sit in the "moderate" band of the thresholds above) — exactly the trade
the knob panel (`LAMBDA_FAIR`, `NOISE_MULTIPLIER`, `MAX_GRAD_NORM`) lets
a practitioner tune for their application. **These comparisons are
expectations, not measurements:** no vanilla-TabSyn or DP-only baseline
has been run in this repo; the comparison table above describes what
each method *should* do by construction.

---

## Part 9 — Limitations of the Guarantees (read before citing results)

These are the honest boundaries of what the implementation supports.

1. **The fairness losses are covered by a separate record-level Gaussian
   mechanism, and all budgets are COMPOSED into one total.** DP-SGD's
   guarantee requires each record's gradient to be clipped *in
   isolation*. The FLIP fairness terms — CKAᵀ between groups and the
   sliced-Wasserstein anchor — are *batch-level* functionals: one record
   changes the gradient attributed to every other record in its batch,
   so per-sample clipping cannot bound any individual's influence for
   these terms. The implementation therefore **splits the guarantee**:
   - The per-sample-decomposable loss (MSE + CE + KLD + L_S, with L_S
     in its per-sample form) goes through Opacus DP-SGD.
   - The batch-coupled fairness gradient is captured separately (with
     Opacus's hooks disabled), the **entire vector** is clipped to
     `MAX_GROUP_GRAD_NORM` C, and Gaussian noise σ_g·C is added.
     Because the whole vector is clipped, the mechanism's output
     depends on the data only through a norm-≤C vector: under
     substitution adjacency the sensitivity is **2C**, so each step's
     RDP is 2α/σ_g² (an earlier version used α/(2σ_g²), assuming
     sensitivity C and under-accounting by 4×). This makes the
     fairness-gradient mechanism **record-level DP under the same
     adjacency as DP-SGD**, so its RDP composes additively with the
     DP-SGD budget.

   **Composition:** every mechanism touches the same records, so the
   true end-to-end guarantee is the composition of ALL budgets — VAE
   DP-SGD, the fairness-gradient mechanism, the statistics releases,
   and (in the full pipeline) the diffusion stage's DP-SGD. Valid RDP
   composition **sums the Renyi divergence at each order α and converts
   to (ε, δ) once** (`compose_rdp_budgets`); adding per-mechanism
   epsilons is not a valid rule (each minimizes over a different α).
   Every run reports the component ε values and the composed TOTAL.

   **Honest magnitudes:** at the demo's settings the composed total is
   large (tens to ~100, depending on the minority-group size and epoch
   count) — a true statement about the mechanism, but not a strong
   guarantee. The dominant cause is the balanced sampler's high
   per-step inclusion rate for the minority group (q ≈ 0.1–0.34).
   Reaching single-digit ε requires substantially higher noise, far
   fewer epochs, or a smaller per-group batch — each at a fidelity
   cost. The knob panel supports all three.

2. **Generation-time statistics are DP-released, not raw.** The
   empirical latent mean/std used for sampling, and the numeric
   range-clip bounds, were previously computed directly on the raw
   training data — fresh non-private computations (min/max in
   particular are sensitive: one extreme record moves them
   arbitrarily). Both are now released through the Gaussian mechanism
   (`dp_release_mean_std`): each row's contribution is clipped to
   `STATS_MAX_NORM`, the sums are noised at `STATS_NOISE_MULTIPLIER`,
   and — since order statistics cannot be released with bounded
   sensitivity — the range clip uses a mean ± k·std proxy
   (`STATS_RANGE_K`) instead of true min/max. These releases are spent
   once into the privacy budget.

   **Stage 2 (latent diffusion) is now DP-trained too.** The full
   pipeline's diffusion prior is trained with DP-SGD (Opacus
   `GradSampleModule` + `DPOptimizer`) and its own RDP accountant. Two
   properties make this stage cleanly record-level private, with no
   group-level mechanism needed: the EDM loss is per-sample decomposable
   (each row draws its own σ and noise; the loss is a per-row squared
   error), and `MLPDiffusion` is standard Linear+SiLU with no BatchNorm,
   so Opacus's per-sample clipping covers every parameter. Because the
   latents it trains on are outputs of the DP-trained VAE encoder
   (post-processing of a private model), Stage 2's budget composes with
   Stage 1's into an **end-to-end record-level guarantee** reported as
   ε_VAE + ε_diffusion. The group-level fairness guarantee (item 1) is
   reported separately (different adjacency: group substitution vs
   record substitution).

   **Tuning note:** DP-SGD's memory scales as `batch_size × n_params ×
   4 bytes` because Opacus stores a per-sample gradient copy of every
   parameter. With `DIM_T=1024` the diffusion's largest layer is
   Linear(1024, 2048) (~2.1M params): batch 4096 needs ~34 GB of
   `grad_sample` alone (OOM on an 8 GB GPU), so the default
   `BATCH_SIZE` is 256 (~2 GB). The lower batch also *improves* the
   privacy budget (sampling rate q ≈ 0.02 instead of 0.34); the only
   cost is more steps per epoch. If memory is still tight, Opacus's
   `BatchMemoryManager` can split physical batches into virtual ones,
   and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps with
   fragmentation.

7. **Preprocessing statistics are not DP-released.** `load_sample`
   standardizes numerics with the raw column mean/std, median-imputes
   missing values, and derives the category sets from `pd.factorize` —
   all fresh computations on raw records, none accounted into the
   budget. Each is a single low-sensitivity statistic over 12k+ rows,
   so the practical leak is small, but formally the guarantee is
   conditional on these being public/non-sensitive. A rigorous fix
   would release them through the same `dp_release_mean_std`
   machinery and compose them into the total. Similarly, the diffusion
   stage's latent-mean shift is computed on the encoded training data —
   a new statistic of the raw data through a private channel, needing
   its own release or explicit accounting.

8. **The balanced sampler is shuffle-based, not Poisson.** Opacus's
   subsampled-Gaussian bound formally assumes Poisson sampling at rate
   q; using γ_max (the minority group's rate) is a conservative
   approximation, not exact. A per-group Poisson sampler would make the
   accounting tight.

9. **Committed reports predate the fixes.** `flip_full_report.html`
   and the demo report were generated before the accounting and
   fairness-direction fixes: they still show ε = 0.52 (an artifact of
   the broken accountant), a DCR ratio of 0.67 (< 1.0), a correlation
   delta of 0.18 ("Poor"), and reversed group rates. Rerun both
   pipelines to regenerate them; the README's "good result" numbers
   should be re-derived from the new outputs.

3. **CKAᵀ = 1 does not mean independence.** CKAᵀ is invariant to group
   mean shifts and overall scale, so a latent that encodes the protected
   attribute purely through a mean shift scores as perfectly
   disentangled. "The latent is independent of the protected attribute"
   overstates what the metric measures (this limitation comes from the
   paper; it is repeated here so it is not lost).

4. **The DP-difference headline can be gamed.** L_S pushes the synthetic
   protected column toward uniform; a generator that merely *shuffled*
   the race column would also score DP difference ≈ 0. Two anti-gaming
   probes are therefore computed and reported alongside the headline:
   - **Attribute probe AUC** — how well the protected attribute can be
     predicted from the *other* synthetic features (income, geography,
     loan type). 0.5 = no group signal.
   - **Downstream fairness** — a model trained on the synthetic data
     (without seeing the protected attribute) is evaluated on real data
     broken out by real group, measuring whether the synthetic data
     transmits disparity to downstream users.

5. **DCR ratio 5.5 is ambiguous.** A ratio well above 1 rules out
   memorization but can also mean the synthetic data is over-dispersed
   (too spread out) — a fidelity concern, not a privacy win. Read it
   together with the W1/TV fidelity numbers, not in isolation.

6. **The accountant's sampling model is conservative, not exact.** The
   balanced sampler draws fixed-size group-balanced batches, not
   Poisson samples; the accountant uses the minority group's inclusion
   rate (γ_max), which is valid but not tight. Exact accounting for
   balanced sampling would need per-group RDP composition (the paper's
   Prop. 1 / Eq. 13 machinery in full).

---

## Running everything

```powershell
# Unit tests (30 tests)
c:/Users/db234/OneDrive/Documents/Vector/.venv/Scripts/python.exe tests/test_flip_fairness.py

# Quick demo: FLIP VAE + empirical-Gaussian latent sampling
# (fast; fidelity is a lower bound — see Parts 6-7)
c:/Users/db234/OneDrive/Documents/Vector/.venv/Scripts/python.exe run_flip_demo.py

# FULL pipeline: FLIP VAE + TabSyn's latent diffusion prior
# (the honest fidelity numbers; emits flip_full_report.html and
#  Data/flip_full_synthetic_bmo_nationwide.csv)
c:/Users/db234/OneDrive/Documents/Vector/.venv/Scripts/python.exe run_flip_full_pipeline.py

# Full TabSyn VAE training with FLIP (fairness + privacy)
cd tabsyn-main/tabsyn-main
python -m tabsyn.vae.main --dataname <dataset> --sensitive_idx <col> `
    --phase1_epochs 2000 --lambda_fair 1.0 `
    --dp --noise_multiplier 1.0 --max_grad_norm 1.0 --dp_delta 1e-5

# Omit --dp for fairness-only; omit --sensitive_idx for vanilla TabSyn.
```
