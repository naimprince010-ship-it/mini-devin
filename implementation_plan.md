# Implement Workspace Seeding & Fix Issue Automation

## Open Questions

1. When a user clicks 'New Session' and leaves the working directory empty, it currently initializes an empty git repo. Should we change this so that an empty working directory automatically copies the mini-devin (Plodder) source code into the new workspace?
2. There is a bug in the "Let Plodder fix this issue" endpoint (/api/repos/{repo_id}/issues/{issue_number}/run). If the repo doesn't have a local path and relies on a clone URL, it passes the URL directly as the working directory instead of cloning it into a workspace folder. I will fix this so it clones the repo properly. Do you also want it to pull the latest code if it's using a local path?

## Proposed Changes

### [MODIFY] e:\mini devin\mini-devin\mini_devin\api\app.py
- Update create_session: If equested_dir is empty, instead of just _init_git_workspace(), we will copy the contents of the mini-devin repository into the new workspace, excluding gent-workspace and .git folders.
- Update start_issue_automation: Add the git cloning logic. If the repo needs cloning, we will generate a workspace_id, clone the repository into the workspace root, and then pass workspace_id and the new local working_dir to session_manager.create_session.

### [MODIFY] LLM Rate Limit Resilience
- Where is the LLM retry logic located? I will check mini_devin/core/llm_client.py or similar to increase the retry delay or implement exponential backoff so that RateLimitErrors don't instantly crash the loop.

## Verification Plan
1. Create a new session with an empty working directory and verify that the agent-workspace is populated with the Plodder source code.
2. Trigger "Fix this issue" from the UI and verify that it correctly clones the repo into a workspace and starts the agent.
