# Offline convergence and reallocation

Existing-cluster reallocations require a proposal-score gain greater than
`1e-6 + 1e-8 * max(abs(baseline), abs(proposal))`. This rejects numerical
roundoff without changing the final birth acceptance rule.

Unchanged hard labels alone do not stop fitting. Normal convergence requires:

1. Consecutive full responsibility matrices and pairwise responsibilities
   agree within an absolute tolerance of `1e-6`, with the same dimensions.
2. Consecutive raw iteration ELBOs are finite and differ by at most
   `1e-6 + 1e-5 * max(abs(previous), abs(current))`.
3. The current iteration accepted neither a reallocation nor a new cluster.

Reallocation returns before birth exploration. After accepting one, normal
inference continues in the next iteration: it may reallocate again or explore
births through the existing selection path. Birth proposals are not forced.
Even when responsibilities and ELBO are stable, an iteration that accepted
a reallocation cannot terminate through the convergence check. Accepted
existing-cluster refinement and emergency deletion also count as reallocations;
temporary refinements of rejected birth proposals do not.

Debug output includes `reallocation_accepted`, `birth_search_complete`, `converged`, and the ELBO
change. Configured iteration and cluster limits remain separate stopping
conditions. Existing refill-failure, multiple-empty-slot and one-observation
exits also remain. Removing an early stop can increase runtime and change the
resulting clusters. Online and online_fast schedules are unaffected.

The raw reported ELBO and a dimension-normalized LDS proposal score are
different objectives when the experimental normalization is enabled. Birth
acceptance therefore need not imply a monotonic raw ELBO. The convergence
test uses the absolute change, so a substantial decrease is not stability.
