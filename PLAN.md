# Execution plan

## Goal
Make Plodder behave more like a capable agent by wiring planning, memory, recovery, and learning more tightly into the main run loop.

## Steps
- [x] **STEP-1**: Seed working memory and richer prompt context at task start and every model turn.
- [x] **STEP-2**: Record failures, recoveries, and successful lessons into working and conversation memory.
- [x] **STEP-3**: Verify the focused agent/memory flow with targeted tests or a syntax/import check.
- [x] **STEP-4**: Add retrieval-driven code and memory context to the initial task prompt.
- [x] **STEP-5**: Auto-replan after repeated failures by marking the active plan step failed and handing the recovery context to the planner.
- [x] **STEP-6**: Rank retrieval context by task type so bug-fix, testing, refactor, and feature tasks surface more relevant code and memory first.

> Tool calls during this task should include `plan_step` when using terminal/editor actions.
