> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

# Existing inference schedules: pre-edit pseudocode

Written from the current source **before modifying inference or test code** for
this extraction. Source: `hdpgpc/hdpgpc/GPI_HDP.py`, following the emission,
dynamics, alignment and HDP extractions. These are executable call orders, not
an idealized common variational algorithm. Statement order below is intentional.
The exact source will also be preserved temporarily for body/AST comparison.

Original anchors: `include_sample:1657`, `include_sample_fast:2040`,
`include_batch:697`, `variational_local_terms_batch:924`,
`estimate_q_first:994`, `estimate_q_all:2416`, `cluster_new_batch:2547`,
`reload_model_from_labels:3185`. Line numbers become historical after extraction;
method names remain compatibility entry points.

Notation: `local(q)` calls the existing HMM/local helper; `score(...)` calls
`compute_q_elbo`, which combines existing emission scores, HDP linear terms and
LDS parameter/latent terms. Neither notation substitutes new formulas. All
deepcopies, resets, clones, slices, dtype choices and early returns remain at
their source locations. Loops are sequential in the order shown.

## 1. Standard online: one incoming observation, one commit

```text
include_sample(x, y, warp, force_model, minibatch, classify):
  t <- old T
  unless classify: increment T; replace snr_norm with ones(T, outputs)
  M <- current M; convert y, then x
  resolve minibatch from configured batch; reset it to zero if minibatch >= t
  for each existing cluster m:
    copy the selected tail of raw observation history into y_mod[m]
  unless classify: append y to raw history; append x to x_train
  alignment.prepare_online(..., fast=False) once
  allocate q_aux(T, M+1, outputs) = -inf; q_lat = 0
  if t>0: copy prior q into old rows (or clone stored q for classification)

  for output ld, then existing cluster m:
    recompute that emission's latent scores across ALL stored x_train
    compute last-observation emission score using y_mod[m][-1]; add warp score

  if t>0:
    local(q_aux); score OLD samples/existing clusters, one_sample=True
  if t>0:                       # this duplicated block is real
    local(q_aux); score OLD samples/existing clusters, one_sample=True

  if t>0 and not classify:
    order existing candidates by descending weighted last-row emission score
    choose the LAST (worst-ranked) candidate as birth template
    clone q and q_lat for the birth proposal
    for output ld:
      copy template emission; reset GP; reset LDS (both save_last=False)
      estimate_new using last candidate waveform; add last warp score
      include weighted sample h=1 into proposal
      append proposal temporarily to owner's gpmodels[ld]
      recompute its latent scores over all stored observations
    local(birth q, warp scores)
    score birth proposal with post=True, one_sample=True; subtract old score
    for output ld: pop temporary proposal; restore M

    if flattened argmax(birth last-row q) == M:
      clone existing q and q_lat ONCE before absorb-candidate loop
      for candidate m in descending score order:
        save real emissions for all outputs
        for output ld:
          copy emission m; estimate_new plus warp score
          include new sample in copy; temporarily install copy on owner
          pair smooth; Bayesian parameter update (this branch is unconditional)
          recompute entire latent-score column
        local(absorb q, warp scores)
        score without birth column, post=False, one_sample=True
        subtract old score
        if absorb total > birth total:
          run local(absorb q, warp scores) AGAIN
          select absorb q/q_lat; restore all real emissions; break
        else:
          restore all real emissions; select birth proposal and its local terms
    else:
      retain original q/q_lat; local(original q, warp scores)
  else:
    retain original q/q_lat; local(original q, warp scores)

  derive last responsibility/log-responsibility arrays
  if classify: return existing sliced q/responsibility/warp-score tuple NOW
  apply near-tie hardening; choose model; apply max_models and forced assignment

  if selected model index == current M:             # committed birth
    increment M; append raw y history to y_mod
    for output ld: append default GP; append default warper
    append basis; reorder responsibilities/pairs/scores and GP/warp lists
    construct counts from current assignments
    if M>2: reset HDP for M-1 sticks
    if M>=2: repeat FOUR times: theta update, then rho/omega optimization
    refresh stored trans_A using the HDP stored-transition formula
  else:
    reorder only if not forced
    construct counts restricted to existing M
    if M>2: reset HDP for M-1 sticks
    if M>=2: repeat FOUR times: theta update, then rho/omega optimization
    refresh stored trans_A

  derive responsibilities AGAIN; repeat tie/cap/force handling
  forced assignment also patches final q/q_lat and final resp/pair entries
  assign actual_state; convert zero minibatch to None
  for output ld, then cluster m:
    include_weighted_sample(t, ..., y_mod[reorder[m]][-1], responsibility[m])
    assign returned basis immediately
    if responsibility>0.9: append selected waveform to y_train
    Bayesian parameter update OR existing weighted ML update
  score committed assignment once
  append full assignment history; append q_chos
  for output ld: alignment.update_selected(...)
  implicit return None
```

This is a single incoming-observation commit, not an offline outer EM loop.
Nevertheless, the existing standard online algorithm revisits past latent scores,
re-runs HMM messages over accumulated rows and evaluates temporary proposals. None
of that work may be replaced by a strictly last-row-only algorithm. In particular,
the fast engine below is not a semantics-preserving replacement for this one.

`local(q)` preserves this order: clone q; resolve theta/warp defaults; calculate
online HDP transition logs and assign trans_A; calculate start logs; add warp
scores to the last q row; apply classification exclusion if requested; weight
and max-shift q; forward; backward; normalize log(alpha*beta); coupled pair
messages and normalization; harden state and pair scores separately; return.

## 2. Fast online: a separate cached approximation

```text
include_sample_fast(...):
  t <- old T; unless classify: increment T and replace output weights
  convert y then x; resolve minibatch
  if t>0: ensure q/responsibility/latent caches
  else: initialize cold responsibility caches
  build candidate waveform histories; append raw observation/input unless classify
  alignment.prepare_online(..., fast=True)
  allocate extended q/q_lat; copy cached history into old rows
  for output ld, then existing cluster m: compute last emission score
      (do NOT add alignment score here: source differs from standard online)

  if first observation:
    choose initial state (0 unless forced)
    allocate hard state and self-transition responsibilities, including birth slot
    store detached real-cluster cache slices
    append assignments and q history HERE
  else:
    score previous cached state; detach baseline total
    choose best existing cluster by emission score
    append a hard assignment/pair to cached responsibilities
    pad to include birth slot

  if t>0 and not classify and no force_model:
    sort candidates; copy/reset worst-ranked template for each output
    estimate birth emission score; include sample into proposal
    for dynamic default only: update latent entry t with h_ini=.5, then multiply by 5
    if weighted birth emission wins:
      append hard birth step; build temporary output/cluster lists
      score birth, post=True; subtract detached old baseline
      for existing candidate m in sorted order:
        clone q/q_lat anew for this candidate
        for output ld:
          copy emission; estimate_new; include sample; pair smooth
          Bayesian OR weighted ML parameter update
          for dynamic model only: patch latent t, then t-1 when nonnegative
          install copy in temporary list, not owner's gpmodels
        append hard absorb step; score existing clusters; subtract baseline
        if absorb total improves: select, pad responsibilities, break
      select best q/q_lat/resp/pairs

  choose model with original tie/cap/force handling
  if birth: increment M; append GP and warp per output, basis and y_mod history
  unless forced: reorder
  construct counts; if M_eff>2 reset HDP with M_eff-1
  if M_eff>=2: FOUR theta-then-stick updates
      (no standard-online trans_A refresh is inserted here)
  assign actual_state
  for output ld, then all committed clusters m:
    weighted inclusion; update basis; append y_train only for h==1
    Bayesian OR weighted ML parameter update
  append assignments and q history AGAIN (including the first observation)
  replace detached q/q_lat/resp/pair caches
  for output ld: alignment.update_selected
  implicit return None
```

Cache reconstruction order is q_last, responsibilities/pairs, then q_lat_last.
Only a missing latent cache triggers output/cluster full-history latent scoring.
The tail patch iterates the provided time-index list in order and retains its
existing broad exception catch. Fast classification does not gain a new early
return or corrected cache lifecycle. Existing forced/unsupported branches are
preserved rather than silently repaired.

## 3. Offline outer schedule: repeated local/global passes

```text
include_batch(xs, ys, it_limit, warp):
  select effective alignment flag; print concentrations
  optionally reduce outputs
  increment T by number of supplied samples
  compute initial SNR weights BEFORE converting/storing xs and ys
  store x_train; initialize responsibilities/pairs to state zero for ALL samples
  allocate q/q_lat zeros; clone raw waveforms as initial warped data
  optionally re-estimate initial GP/LDS defaults
  initialize counts=None and reallocate=False

  repeat:
    refill responsibilities; read owner M
    if responsibility width==1:
      compute counts; reset HDP
      repeat TWO times: theta update(M+1), then stick optimization
    if refill says end: break             # after the one-column HDP block

    variational_local_terms_batch(...)    # complete local schedule below
    if responsibility width > old M: increment owner M by one
    clone responsibilities/pairs
    construct counts from assignments, or all-one counts when HMM is disabled
    reset HDP for M
    repeat TWO times: theta update(M+1), then stick optimization
    refresh trans_A using stored-transition formula

    if T>1:
      compute state/pair entropy; print iteration diagnostic
      score current q/q_lat/HDP/LDS; add entropy; increment iteration
      if it_limit reached:
        append objective and where(resp==1) labels; break BEFORE cache refresh
      if M == max_models:
        append objective and where(resp==1) labels; break BEFORE cache refresh
      append objective, argmax labels and q history
      assign q_last, q_lat_last, snr_last
      assign start/transition-count caches, resp/pair caches, elbo_last
      if >1 exactly-empty group OR consecutive label arrays are identical: break
      else retain latest objective and continue
    else: break
  implicit return None
```

The `iteration==14` message prints STOP but does not terminate. The outer loop is
not an ELBO-isclose loop. Final cache contents differ with the exit branch.

## 4. Offline local schedule and proposal order

```text
variational_local_terms_batch(...):
  resolve theta; compute batch transition logs; assign trans_A; compute start logs
  i=0; reparam=True
  if one group OR penultimate group mass>=1 OR first emission is unfitted:
    estimate_q_first(...)
    score returned state (post flag determined by width>owner.M)
    increment i
  else: score current state
  clone baseline score components
  unless reallocate:
    repeat:
      M <- responsibility width
      estimate_q_all(..., fixed baseline components)
      install returned gpmodels
      score returned state, using width-dependent post flag
      if (isclose(total_old,total_new,rtol=1e-5) and i>0) OR i==10:
        if not reparam: set reparam=True
        else: break
      replace previous score; increment i
  return arrays, warped data and reallocate flag
```

`estimate_q_first` has three sequential phases; acceptance returns immediately:

```text
  A. Bootstrap and representative scoring
    get all-candidate cached alignments before scoring
    if mean(q_)==0:
      for output ld:
        create default GP, or share/copy previous output's first GP as configured
        full_pass_weighted for cluster 0; add warp scores; compute SNR; install GP
    for cluster m: read indexes or exact-hard responsibility indexes
    clone representatives
    for output ld, then cluster m:
      copy GP; if fitted reset LDS then GP
      if nonempty, include representative raw sample
      compute all emission scores on candidate warped data; add warp scores
      compute SNR

  B. Try reallocating existing groups (M>1)
    clone scores/SNR; suppress empty final group using original minimum formulas
    weight/max-shift -> forward -> backward -> state/pair logs -> hardening
    rank by occupancy; reorder state columns (do not add missing pair reorder)
    for output ld, then cluster m:
      if assignments changed: copy/reset; full_pass_weighted; add warp score; SNR
      else reuse GP and saved columns exactly as source
    score proposal, then baseline
    repeat estimate_q_all + proposal score until isclose and i__>0 OR i__==20
    score baseline again, then proposal again
    if every group mass>=1 and total improves and observation score differs:
      install GPs; assign/reorder warp arrays; select assigned waveforms
      choose unique representatives in cluster order, ranking assigned samples
      normalize/store SNR; return reallocate=True
    elif some group mass<1 and emergency comparison improves:
      trim last GP per output; install; normalize SNR; slice all state arrays
      reorder representatives; return reallocate=True
    else: mark empty_estimation when applicable; continue to birth search

  C. Try splitting/creating a group
    compute three existing normalized rankings from representative, emission,
      and latent scores; retain their unguarded zero-range divisions
    build candidate list in TWO passes: first roughly half from representative
      ranking, remaining from emission+latent ranking; retain dedup/group gates
    call new_group twice on separate cloned score sets; increment LOCAL M
    for candidate representative in that list, respecting n_explore_steps:
      determine containing cluster, old representative and dedup eligibility
      unless empty_estimation:
        set candidate representative; get cached candidate alignments
        clone proposal arrays; increment exploration counter
        for output ld: copy/reset chosen GP, include new representative,
          compute its all-sample new-column score and SNR
        weight/max-shift -> forward -> backward -> state/pair logs -> hardening
      else: retain the existing alternate score forcing and message path
      reorder state responsibilities by occupancy
      for output ld, then proposed cluster m:
        new cluster: choose shared copy or new GP; reset when fitted; full pass
        changed old cluster: copy/reset; full pass
        unchanged old cluster: reuse GP and original score/latent/SNR columns
      score proposed state, post=True
      if largest group is final column: continue to next representative
      if any group mass<1:
        score baseline; if improved, trim GP and warp lists, slice arrays,
          commit state/representatives/warped data/SNR; return reallocate=True
        else continue
      score original baseline
      repeat estimate_q_all + proposal score until isclose and i__>0 OR i__==10
      score original baseline again, then proposal again
      if all groups occupied, final is not largest, and total improves:
        commit GPs; append default warp per output
        reorder/store warps and selected waveforms
        choose unique representatives in cluster order; normalize/store SNR
        return (the existing reallocate flag)
    if none accepted: return original arrays/data with reallocate=True
```

```text
estimate_q_all(...):
  resolve GP list and representatives; allocate candidate q/q_lat and clone SNR
  weighted q -> max shift -> forward -> backward -> log state/pair -> hardening
  rank occupancies; reorder state columns; get cached candidate alignments
  for output ld, then cluster m:
    compute assigned sample indexes
    existing GP with changed indexes:
      copy; reset LDS(save_last depends on reparam), reset GP
      if previously unfitted: create_gp_default(i=...)
      full_pass_weighted; add warp scores; compute SNR
    existing GP with unchanged indexes: reuse GP and cached score columns
    missing GP: create_gp_default(i=...); full pass only if assigned indexes exist
    append to temporary GP list
  score OLD state first; score proposed state second
  if all groups occupied and total improves:
    update warp fields, warped-data order, representatives (shape-gated), SNR
    return proposal and temporary GPs (caller installs the list)
  elif all occupied: return old state/list
  elif proposal beats BOTH fixed reference and current old state:
    trim last GP per output; install owner GPs NOW; normalize SNR
    slice responsibilities/pairs/scores/SNR; reorder representatives; return
  else: return old arrays with the source's current warped-data return value
```

The engine calls the existing emission `full_pass_weighted`; it must not copy
that method's mathematics or replace its sample order. Its current internal call
order is: select `resp>.99` indexes in ascending order; for each active sample,
weighted inclusion then (dynamic only) pair smoothing and Bayesian parameter
update; after all active samples, dynamic full smoothing; finally all-sample
emission scores and all-sample latent scores. No active samples returns supplied
q/q_lat unchanged. Static skips those dynamic smoothing/update blocks.

## 5. Related offline entry points remain distinct

`cluster_new_batch(learning=False)` converts inputs, then loops output -> cluster
-> sample for raw-waveform emission scores, followed by per-cluster SNR. It obtains
classification-specific HDP logs, runs existing HMM messages/hardening and returns
labels. The effective warp argument does not add alignment to this branch.

The learning branch first scores new data in that same nested order, using batch
HDP logs; appends old/new data, responsibilities and SNR; reorders state columns;
then loops output -> cluster to copy/reset/full-pass fit on accumulated data. It
installs those GPs, repeats HMM messages, and enters its own refill/local/two-global-
updates/objective loop. Its exit/cache sequence differs from include_batch. The
warp-on-convergence path references uninitialized `warp_computed` and also has a
three-versus-four-result unpacking mismatch; this extraction does not repair it.

`reload_model_from_labels` resolves warp, asserts outputs and converts data; if M
differs, rebuilds GP lists and appends warpers in output -> cluster order; then
sets M/T/data/type state. It constructs hard responsibilities and its original
pair indexing, then fits output -> cluster full passes. If requested, assigned-
state alignment precedes another output -> cluster refit. It appends q, constructs
counts, refills, resets HDP and performs two theta/stick updates; stores transition
and history/cache fields, normalizes SNR, chooses representatives, then scores
entropy plus objective and assigns elbo_last. This remains a separate operation.

## Extraction constraints and divergence gate

1. Start with green characterization and repository tests; freeze detailed
   source-derived schedule traces before changing production methods.
2. Introduce a minimal owner-binding `InferenceEngine` base and separate
   `OnlineInference` entry points for standard/fast online. Copy schedule bodies
   without rewriting their expressions or routing through a common algorithm.
3. Run all characterization tests immediately, plus detailed trace comparisons.
   Any changed posterior value stops further engine extraction. Report the first
   event/path difference and inspect that step against the preserved source;
   never regenerate a reference to accept the change.
4. Introduce `OfflineInference` with its own batch/local/proposal/classification/
   label-reload schedules. Preserve every legacy method as a delegate so nested
   callbacks and existing instrumentation still reach the same owner.
5. Repeat the characterization/divergence gate after offline introduction.
6. Compare exact source bodies, repeat fresh-process captures, run focused API/
   schedule tests, and measure runtime. Component mathematics, lifecycle helpers,
   model storage and experiment scripts are outside this extraction.
