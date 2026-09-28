> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

# HDP variational component: preservation record

## Pre-edit inventory and boundary

This is a structural extraction of the implemented variational approximation.
The existing code, including its unusual normalization and mutation behavior,
is the reference. No new HDP algorithm, pruning rule or optimizer is introduced.

| Quantity | Existing code name and behavior |
|---|---|
| Global stick mean/concentration | `rho`, `omega`; Beta shapes `eta1=rho*omega`, `eta0=(1-rho)*omega` |
| Expected global weights | `rho2beta`: cumulative products, either K entries or K+1 with remainder; no additional normalization |
| Stick prior concentration | `gamma` |
| Transition-row concentration | `transAlpha`; rows of `transTheta` are variational Dirichlet parameters |
| Initial-state concentration | `startAlpha`; `startTheta` is its variational Dirichlet vector |
| Sticky self-transition mass | `kappa`; all three constructor profiles set zero, but positive-kappa formulas exist |
| Effective truncation | `self.M`, responsibility width and caller-supplied theta dimensions; these are not always equal to the number of sticks |
| Optimizer diagnostics | `OptimizerInfo`, including objective, initial values, restart count and SciPy status |

Constructor profiles `(gamma,transAlpha,startAlpha,kappa)` are `less=(.01,.01,.01,0)`,
`balanced=(1,1,.1,0)`, and `more=(10,10,1,0)`. An unknown profile is not validated
or assigned replacement values. These concentrations are fixed during ordinary
updates, rather than learned. The unrelated GP process covariance and emission
cross-output correlation must not be confused with HDP gamma/rho.

### Initialization, dimensions and growth

`create_initrho(M)` uses `remMass=min(.1,1/M**2)` and the existing descending
denominator construction. `init_global_params` uses omega=1+gamma, all-one state
and transition counts, and `(M+1)` theta dimensions. Its `d_dim` argument is unused.
`reinit_global_params` resets sticks, creates omega on the default device and
requests theta dimension M. Temporary reinitialization copies old rho/omega into
the prefix of newly initialized arrays and returns `(M+1)` theta values without
committing them. Shrinking that prefix is not silently repaired.

`_calcThetaFull` adds counts and kappa only to the leading `(M-1)` block, even
when M equals the number of sticks and no remainder entry is returned. Last-row,
last-column and last-start counts are ignored. `_calcThetaPost` adds .8 times
stored transition theta, .2 times supplied transition counts, and kappa on the
whole new diagonal; start theta adds stored values and full supplied counts.
These formulas are different and remain so.

Cluster lifecycle remains in `GPI_HDP`: online birth is chosen when the selected
index equals `self.M`, after max-model and forced-assignment handling. It then
increments M, constructs GP/warp objects, reorders and updates HDP parameters.
Standard online runs four theta/stick alternations when M>=2, resetting first
when M>2. Fast online has its separate proposal/commit schedule. Offline proposals
grow responsibility arrays, evaluate emission/HDP objectives, and may remove the
last group; accepted widths update M in the driver. Offline global passes perform
two theta/stick alternations after reinitialization. `max_models` is not a bound
inside the stick optimizer or a new automatic truncation/pruning operation.

`new_group` zero-pads responsibilities, pairs and emission score arrays, and sets
new SNR entries to `-2*abs(min(snr,dim=1))`. `remove_last_group` returns slices.
`refill_resp` uses exact `resp==1`, moves the last state into the first empty slot
and performs the existing clone/assignment sequence on pair responsibilities.
The decisions to call these and to create/delete/reorder GP and warp objects
stay in orchestration. No component method schedules an emission or LDS update.

### Expected logs and normalization

These paths must remain distinct:

- Online/transition-message expectation: `digamma(theta)-digamma(sum(theta))`,
  retaining the extra remainder in the denominator when present.
- Batch local updates: `digamma(theta)-log(sum(exp(digamma(theta)))+1e-5)`.
- Stored transition refresh: the same log/exp expression without jitter.
- Nonlearning classification: `digamma(theta)-digamma(sum(theta)+1e-5)`.
- `compute_Pi`: exp of row-normalized digamma values over the entire theta matrix.
- Stick optimization: NumPy log/exp normalizations, with `1e-5` inside the log
  for both transition and start terms; whole theta arrays enter this calculation.

No log-sum-exp library replacement or normalized probability substitution is
made. `compute_trans_A` retains its `-inf` padding branch for one extra state.
Generic HMM forward/backward/pair-message recurrences and hard responsibility
selection remain in the inference layer, including their asymmetric thresholds,
cached-message behavior and float32 pair-hardening output.

### Objective, constrained optimization and external attribution

`OptimizerRhoOmega.py` explicitly attributes adapted code to
`https://bnpy.readthedocs.io/`. Preserve that attribution and every original
comment in its relocated implementation. This task does not substitute newer
bnpy code. The original `GPI_HDP` header/author attribution remains present.

The optimizer converts rho with inverse sigmoid and omega with log, uses SciPy
`fmin_l_bfgs_b` with analytical gradients by default, and reverses the transforms
with sigmoid/exp. EPS is `1e-8`. Rho clamps to `[EPS,1-EPS]`; initial omega clamps
in place to EPS (optional maximum only when explicitly supplied). Final rho is
clamped again; no new final-omega clamp is added. Scale-vector handling and the
existing chain-rule expression are preserved. Objective/gradient branches for
kappa, active/remainder statistics, nDoc=0 and approximate gradients remain.

Restart factr values are `[1e5,1e7,1e9,1e10,1e11]`. Only FAILURE/overflow
ValueErrors trigger retries; after exhausting trials with an explicit initial
rho it recurses with both initial vectors unset and the default factr list.
SciPy warnflag>1 becomes an error; warnflag==1 is accepted. Other stopping limits
are the existing SciPy defaults or supplied kwargs. The model wrapper catches
ValueError and retains current vectors when their length equals self.M, otherwise
uses the existing prior fallback. AssertionError is not newly caught.

The model objective's sticky `L_top` formula is not identical to the optimizer's
formula (`+eta1` and constant `1.9` are retained). Linear ELBO calculation mutates
the supplied NumPy transition-count array when adding prior mass. Higher-level
callers clone before passing that array. Entropy helpers and their epsilon stay
unchanged. Cached descending vectors and diagonal/flat-triangle index arrays
remain shared optimizer dictionaries; no cache invalidation/optimization is added.

### Intended extraction (recorded before editing)

`HDPVariational` will be a narrow live view over existing HDP state and conversion
operations. It will provide initialization, stick transforms, theta construction,
distinct expected-log calculations, objectives, optimization and responsibility
refilling. The main model retains its serialized fields and old callable methods
as delegates. No duplicate HDP state or new GP/LDS/warp ownership is introduced.
The adapted optimizer moves intact into `hdp._optimizer` with a legacy import
shim. Focused pre-edit fixtures will cover calculations, optimization, growth
and pruning-array contracts before any production edits.

## Implemented API and dependency boundary

```python
from hdpgpc.hdp import HDPVariational

hdp = model.hdp                 # live view; does not copy or initialize anything
beta = hdp.rho2beta(hdp.rho)     # includes remainder by default
pi = hdp.compute_Pi()           # original full-theta normalization
log_a = hdp.expected_log_transitions(model.M, hdp.transTheta)
log_pi = hdp.expected_log_start(model.M, hdp.startTheta)
rho, omega = hdp.find_optimum_rhoOmega()
# Returning optimized sticks does not commit them; the driver controls assignment.
```

`HDPVariational(state)` is also available for an explicit state/context object.
The view forwards only the nine fields `rho`, `omega`, `transTheta`, `startTheta`,
`gamma`, `transAlpha`, `startAlpha`, `kappa`, `OptimizerInfo`; context access is
limited to `M`, `device`, `cond_cpu`, `cond_cuda`, `cond_to_numpy`, `cond_to_torch`.
It exposes no GP, LDS, alignment or lifecycle operation. Reading a missing field
still raises AttributeError. The original conversion methods retain detach/device
effects; the view adds no coercion. M is read-only through this boundary.

The main model retains its plain fields. Each `model.hdp` access creates a small
view, so there is no stored back-reference cycle, duplicated arrays or additional
pickle field. Retaining a view reflects subsequent replacements of model fields.
Changing concentrations through that view writes the corresponding existing field.
The returned view is an operational handle, not a new persistence format.

Twenty old methods now delegate to `hdp._variational.HDPVariational`: global
initialization/reinitialization and temporary growth, stick conversions, full/
posterior theta calculations, Pi and message-padding calculations, objective/
entropy calculations, stick optimization and responsibility refilling. Stateless
entropy/cumulant/start-padding legacy methods delegate without
requiring a model instance; the existing owner-free call contract is retained.
The constructor concentration block moved with its original comments. Thirteen
remaining expected-log blocks now call seven separate component methods:
`expected_log_transitions`, `expected_log_start`, `batch_log_transitions`,
`batch_log_start`, `stored_log_transitions`, `classification_log_transitions`,
`classification_log_start`. None is a renamed alias for a different normalization.

`hdp._optimizer` contains the entire original optimizer source text, including
bnpy attribution, comments, constants and shared cache dictionaries.
`OptimizerRhoOmega.py` re-exports all its previous functions/imported names/caches,
including private helper names, and preserves function pickle module names.
The HDP component imports the private implementation directly. Importing the
`hdpgpc.hdp` namespace is lazy; requesting the concrete component loads numerical
libraries but no high-level model, emission, dynamics or alignment module and
does not change Torch's default dtype.

Ordinary imports and old model methods remain supported. Rebinding a function
global on a compatibility module does not rebind globals in the extracted
implementation; tests that patch internals should target `hdp._variational` or
`hdp._optimizer`. Arbitrary subclass overrides of legacy numerical methods are
not a newly supported component-injection API. Existing experiment files are
unchanged.

## Preservation evidence and focused tests

Before editing, all 15 workspace characterization cases passed (29.959 seconds),
and the repository suite passed 86 tests with one CUDA-only skip. The HDP fixtures
were then captured against the original sources in
[`tests/fixtures/hdp_reference.json`](../tests/fixtures/hdp_reference.json), with
source hashes. No existing golden file was regenerated.

The engine/global-method extraction and the expected-log extraction were separate
steps; all 15 original characterization cases passed after each (29.983 and
29.874 seconds). `tests/test_hdp.py` has 31 tests:

| Coverage | What it protects |
|---|---|
| 19 frozen snapshots and inventory | M=1/3/5 initialization, tail mass, rho/beta conversions, full/truncated/posterior theta, temporary growth/reset, seven expected-log variants, sticky/nonsticky objectives, mutation of count arrays, state/pair entropy, refill/growth/pruning arrays, analytical objectives/gradients, actual optimizer results/diagnostics and nDoc=0 solution |
| Live-state/persistence test | Same tensor identities, visible field replacement, no new serialized model field, narrow context access and read-only lifecycle count |
| Stateless legacy calls | Cumulant/start-padding calls still accept no owner; existing calculation tests additionally cover owner-free entropy calls |
| Concentration profiles | Exact less/balanced/more constants and the existing unknown-profile no-op |
| Expected logs | Extracted methods match the pre-edit fixture; incompatible normalization variants stay distinct |
| Temporary growth | Existing prefixes remain exact, no commit/M change or model construction, unsupported shrinking still raises |
| Theta construction | Last-row/column/start counts remain ignored, inputs stay unmodified |
| Bounds/cache/import tests | Rho clipping, in-place omega clipping, transform round trip, invalid beta length, shared optimizer cache/function identities and pickle resolution |
| Retry/status/fallback tests | Exact factr sequence, recursive reset to unset initial vectors, unrelated errors propagate, warnflag=1 accepted, warnflag=2 rejected, old/current prior fallback and stale diagnostic retention |

Snapshots use exact shape/dtype/discrete comparisons. Floating arrays use
`rtol=1e-7`, `atol=1e-9`, allowing small library roundoff without weakening existing
component tolerances; scalar diagnostics remain exact. Independently, two fresh
HDP captures in forward/reverse order matched all nineteen snapshots **exactly**,
including the newly extracted expected-log methods. See the
[component determinism report](../benchmarks/hdp_component_determinism.json).

The first full-suite run exposed that routing owner-free entropy calls through a
model property broke their existing contract. Those delegates were corrected;
the related owner-free cumulant/start-padding calls were checked too. No expected
values or numerical implementations were changed. The final repository suite
passed **118 tests, with one CUDA-only skip** (119 total). Two final fresh full
characterization captures matched original snapshots **byte for byte**, including
reversed case order; see the
[final full-inference report](../benchmarks/hdp_characterization_determinism_final.json).
The earlier capture report remains available as a record of the prior step.

The [final source audit](../benchmarks/hdp_source_audit_final.json) confirms the entire adapted
optimizer text is unchanged (apart from filesystem newline representation), all
twenty relocated methods are AST-identical, and the complete original model class
is AST-identical after inlining the delegates and concentration/expected-log
blocks and removing the added view property. Thus birth/deletion/reordering gates, max-model handling, update counts,
call order, GP/LDS/alignment calls and message recurrences retain their code.
The final standalone characterization run also passed all fifteen tests (29.934
seconds). An offline wheel built from a temporary copy contained the new HDP
modules and passed public/legacy imports, cache identity and function-pickle
checks. No build artifacts or dependency installations were added to the checkout.

Run from the enclosing workspace:

```text
python -B -m unittest discover -s HDP-GPC/tests -p test_hdp.py -v
python -B -m unittest discover -s HDP-GPC/tests -p "test_*.py" -q
python -B -m unittest discover -s tests/characterization -t . -q
python -B -m tests.characterization.verify_determinism --runs 2 --report NEW_REPORT.json
```

## Performance investigation

The dedicated benchmark uses M=3, float64, one numerical thread, fixed seed 42,
real SciPy optimization, and no GP fitting. A warmup precedes each workload;
samples average 1,000 stick conversions, 300 theta/Pi calculations, 50 linear
ELBOs or ten optimizations. Eleven samples form each primary median. Timings ran
sequentially without concurrent numerical tests. Environment: Python 3.11,
NumPy 1.26.4, SciPy 1.14.1, Torch 2.9.0+cpu.

| Operation | Before microseconds | After microseconds | Primary change |
|---|---:|---:|---:|
| Stick conversion | 9.045 | 10.585 | +17.02% |
| Full theta | 46.894 | 53.865 | +14.87% |
| Transition Pi | 22.967 | 25.685 | +11.83% |
| Linear ELBO | 197.064 | 215.322 | +9.27% |
| Stick optimization | 1894.370 | 1942.670 | +2.55% |

Because helper changes exceeded 5%, these were investigated and reported before
final validation. Alternating original/extracted/extracted/original fresh-process
runs used the same benchmark with an isolated pre-edit source package and seven
samples per run. Pooled fourteen-sample medians were:

| Operation | Original microseconds | Extracted microseconds | Change |
|---|---:|---:|---:|
| Stick conversion | 9.003 | 10.582 | +17.54% |
| Full theta | 47.944 | 54.870 | +14.45% |
| Transition Pi | 23.006 | 25.442 | +10.59% |
| Linear ELBO | 196.995 | 224.921 | +14.18% |
| Stick optimization | 1887.930 | 1892.235 | +0.23% |

The small-helper slowdown is reproducible, not dismissed as noise. Dispatch
profiling of 1,000 theta calls shows 1,000 new view constructions and 9,000
forwarded field/context lookups, with the same 1,000 theta and stick calculations.
Together with source equality, this identifies Python component/view overhead as
the structural cost. Absolute increases are approximately 1.6-27.9 microseconds
for these helpers. No inference arithmetic, optimizer tolerance, iteration count,
cache or initialization was changed to offset that cost. This is a measured
limitation of the extraction; it is not a claim of unchanged microbenchmark speed
or an end-to-end throughput estimate.

Raw evidence: [before](../benchmarks/hdp_before.json),
[after](../benchmarks/hdp_after.json),
`benchmarks/hdp_investigation_{1_old,2_new,3_new,4_old}.json`, and
[dispatch call counts](../benchmarks/hdp_dispatch_profile.json).
`benchmark_hdp.py --compare` reports changes; `HDPGPC_HDP_REFERENCE_SOURCE` can
select an isolated reference source tree. Reports refuse existing output paths.

## Limits

The CPU fixtures cover ordinary online/offline growth through original inference
snapshots plus direct temporary growth, refilling and pruning-array contracts.
They do not exercise every data-dependent offline proposal/deletion decision;
those branches are protected additionally by the source audit, not claimed as
fully exercised. CUDA was unavailable. Existing unsupported/known-failure
inference branches remain unchanged. Neither a new statistical normalization nor
compatibility with arbitrary third-party numerical-method monkeypatches is
implied by this extraction.
