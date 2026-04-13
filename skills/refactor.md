# Refactor playbook

When refactoring existing code:

1. **Intent** — Preserve behavior unless the task explicitly changes it; document behavior shifts.
2. **Steps** — Prefer incremental steps with green tests between steps.
3. **APIs** — Avoid breaking public APIs without migration notes or shims.
4. **Dead code** — Remove only what is clearly unused; grep/tests before deleting.
5. **Naming** — Align names with domain language; keep diffs readable.
6. **Stop** — Do not expand scope into unrelated modules or new features.
