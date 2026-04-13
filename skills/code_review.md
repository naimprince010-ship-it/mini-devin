# Code review playbook

When reviewing or finishing a change set:

1. **Scope** — Prefer small, reviewable diffs; call out unrelated changes.
2. **Correctness** — Match requirements and edge cases; note race conditions and error paths.
3. **Tests** — Suggest or add tests for new behavior; run existing tests when feasible.
4. **Style** — Follow project conventions (formatter/linter); avoid drive-by refactors.
5. **Security** — Flag injection, authz, secret logging, and unsafe defaults.
6. **Output** — Summarize findings as: blocking / should-fix / nit; link to files/lines when possible.
