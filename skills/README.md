# Playbooks (skills)

Markdown guidelines at the repository root. The orchestrator can inject selected files into worker prompts (see `mini_devin.skills.playbook`).

- **Naming:** `skills/<tag>.md` — tag is the stem (e.g. `code_review` → `code_review.md`).
- **Activation:** set env `PLODDER_PLAYBOOK_TAGS` to a comma-separated list (e.g. `code_review,refactor`) or pass `playbook_tags=` when constructing `PlodderOrchestrator`.

Do not put secrets in these files.
