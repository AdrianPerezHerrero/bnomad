# Ordered transition topology

`HDPConfig.transition_topology` selects the assignment topology:

- **`ergodic`** (default): the existing unrestricted model and inference paths.
- **`left_to_right`**: start in stage 0, remain there or advance to stage 1,
  then remain or advance to stage 2, and so on. No backward transitions or skips.

Each inferred stage occupies one contiguous interval in the decoded path. An
appearance can recur, but it must belong to a new stage; anatomical majority
labels may therefore repeat across different stage IDs. Increasing kappa
encourages persistence. It does not guarantee a minimum duration; conditional
on a row's stay probability, durations remain geometric.

## Configuration

```python
from hdpgpc.config import HDPConfig, InferenceConfig

hdp = HDPConfig(
    hdp_hyp='more_extra_sticky',
    hmm_switch=True,
    transition_topology='left_to_right',
)
inference = InferenceConfig(
    schedule='offline',
    offline_strategy='split_merge',
    hard_assignments=False,
    it_limit=30,
)
```

Use these sections in the ordinary `HDPGPCConfig`. Old serialized configurations
without a topology field resolve to `ergodic`. Emission and LDS configuration
remain independent of the assignment topology: PCA, MFA, fixed affine decoders
and ordinary GP observations use their existing filtering and matrix updates.

For CAS-Colon:

```bash
python examples/cas_colon_pca_clustering.py --records 1 --transition-topology left_to_right --offline-strategy split_merge --hdp-preset more_extra_sticky
```

For MFA with independent latent factors but ordered assignments:

```bash
python examples/cas_colon_mfa_clustering.py --records 1 --hmm-switch --transition-topology left_to_right --offline-strategy split_merge
```

Add `--dynamic` to the MFA command to learn A/Gamma. Selecting an ordered
assignment prior does not itself require or change latent LDS evolution.

## Proper prior on the allowed transitions

The constrained model has a different row prior from the unrestricted HDP-HMM.
It does not fit an unrestricted Dirichlet and mask/renormalize its posterior.
Keep the global GEM stick prior:

\[
v_i\sim\mathrm{Beta}(1,\gamma),\qquad
\beta_i=v_i\prod_{l<i}(1-v_l).
\]

Normalize the global base measure over the two allowed destinations:

\[
r_i=\frac{\beta_i}{\beta_i+\beta_{i+1}}
    =\frac{v_i}{v_i+(1-v_i)v_{i+1}}.
\]

Then the actual row distribution is two-dimensional:

\[
(\pi_{ii},\pi_{i,i+1})\mid v
\sim\mathrm{Dirichlet}\bigl(\alpha r_i+\kappa,\alpha(1-r_i)\bigr),
\quad\alpha=\mathrm{transAlpha}.
\]

Every other transition is structurally zero. This explicitly defines an
ordered hierarchical variant with constant row concentration alpha+kappa;
it is not the unrestricted prior with its old normalizer. Presets retain their
gamma/alpha/kappa values, but the normalized two-destination base changes their
interpretation and effect relative to the unrestricted model.

The initial stage is deterministic. Consequently the old startAlpha variable
is inactive here: no initial Dirichlet KL or expected-log term is omitted.
Each separate record/segment starts a fresh stage numbering.

## Variational updates and ELBO

Use q(v_i)=Beta(a_i,b_i) and q(pi_i)=Dirichlet(theta_i). For expected transition
counts N_ii,N_i,i+1, the exact row coordinate update is

\[
\theta_i=\bigl(\alpha E[r_i]+\kappa+N_{ii},
                 \alpha(1-E[r_i])+N_{i,i+1}\bigr).
\]

Expected log probabilities use digamma(theta_ij)-digamma(sum(theta_i)).
Forbidden edges have log probability minus infinity. Matrix-shaped storage
contains zeros for forbidden entries, but those entries are not dimensions of
a Dirichlet distribution and are never passed to digamma/log-Gamma.

Let A_i(v)=(alpha*r_i+kappa,alpha*(1-r_i)). The assignment linear term is

\[
\begin{aligned}
L_{\rm linear}={}&-\sum_i KL(q(v_i)\Vert\mathrm{Beta}(1,\gamma))\\
&+\sum_i\left\{
E\left[\log\Gamma(\alpha+\kappa)-\sum_j\log\Gamma(A_{ij}(v))\right]
+\sum_j(E[A_{ij}]+N_{ij}-1)E[\log\pi_{ij}]
+H[q(\pi_i)]\right\}.
\end{aligned}
\]

The existing chain entropy is added once, separately. Emission, latent and
LDS parameter terms retain their current definitions, including the existing
optional LDS-dimension scaling for proposal scoring. Raw ELBO reporting retains
the unscaled LDS term.

In particular, E[log Gamma(A)] is **not** replaced by log Gamma(E[A]). The
normalized row base makes the global Beta factors nonconjugate. Their coordinate
step numerically maximizes this ordered model's own ELBO contribution using
L-BFGS-B, not the unrestricted HDP optimizer or its surrogate top-level term.

Expectations use deterministic Gauss-Jacobi quadrature, with a one-dimensional
ratio density, analytic treatment of log endpoint terms and adaptive integration
near logarithmic density endpoints. Proposed global updates must improve their
objective and pass a 128-versus-256-order integration audit; otherwise the
previous global parameters are retained. This is numerical evaluation of the
stated expectation, not an analytic closed-form global posterior. Optimizer
diagnostics report the integration discrepancy and whether the step passed.
It adds computational work compared with the unrestricted global update.

### Truncation and the end of the record

An active final stage retains a possible advance to the next unoccupied stage.
The next unrepresented stick keeps its prior variational factor. Its transition
count is zero. No artificial exit is counted at the end of the record.

Native transition storage includes an extra reserve row/column. The active last
row's stay/advance probabilities sum to one **including the reserve column**.
The reserve's absorbing bookkeeping row is not an assertion that the physical
process has a known final anatomical region. This permits future stage births
without replacing the prior on the last active stage by a deterministic stay.

## Inference and structural proposals

- Banded, scaled log-domain forward/backward computes consistent state and pair
  marginals on stay/advance edges.
- Hard assignments use Viterbi. Soft inference also reports a Viterbi-decoded
  joint path for final labels. Per-frame marginal argmax can violate ordering
  and is not used as the ordered hard path.
- Splits propose a temporal cut within a stage. The new stage is inserted
  immediately after its parent; model arrays, representers and responsibilities
  follow that ordering.
- Cuts are screened by between-segment appearance gain and boundaries near the
  worst latent-scoring member. Splits initialize valid hard paths even if
  refinement uses soft responsibilities.
- Merges combine adjacent stages only. An empty interior stage cannot be deleted
  by swapping it with the final slot.
- Candidate fitting, parameter updates, shared-parameter handling, full objective
  comparisons, tolerances, cycle prevention and stopping retain the native
  transactional split/merge machinery.

This is ordered segmentation of one sequence. It is not a reusable anatomical
stage model fitted jointly across multiple videos or a model with repeated states.

## Supported combinations and limits

The initial implementation supports **CPU, offline split/merge with HMM
assignments**, for both hard and soft responsibilities. Configurations with
legacy offline proposals, online/online_fast, independent assignments or CUDA
are rejected explicitly. Legacy classify/fit-label utilities and birth-warp
diagnostic capture also reject ordered mode. The unrestricted default paths
remain available with their previous behavior.

No hard minimum duration, arbitrary transition graph or skip-forward topology
is implemented. No inference or cluster-recovery guarantee follows from this
structural prior alone. A tiny real-data smoke example is not a validation of
anatomical segmentation performance.

Focused mathematics, integration, compatibility evidence and example outputs
are recorded under the ignored `agents/tests/transition_topology` folder.
