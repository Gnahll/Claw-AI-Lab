# Git deployment on the existing Linux server

Repository: `https://github.com/Gnahll/Claw-AI-Lab.git`, deployment branch: `preview-v1.1.0`.

The production directory is `/home/Disk/WuZihang/apps/Claw-AI-Lab`. Keep its `.venv`, Node runtime, `frontend/node_modules`, logs, queues, research outputs and resource links in place. These are runtime data, not source code to commit. Never use `git clean -fdx` or a forced reset on the production directory.

Server settings are separated from the tracked example:

- `.env`: private credentials and environment, mode 600.
- `examples/config_template.local.yaml`: actual server endpoint/models and Python path, mode 600, ignored by Git.
- `examples/config_template.yaml`: public example with placeholders, tracked by Git.
- `CLAW_CONFIG_TEMPLATE`: optional explicit override; a missing explicit file fails rather than silently using another provider.

The bridge and installer prefer the local template. The startup script reads `.env`. Store the new provider key only as `RESEARCHCLAW_API_KEY` in `.env`, keep `llm.api_key` empty in the local template, and use `llm.api_key_env: RESEARCHCLAW_API_KEY`. Do not commit private config or generated project configs. Keep old credentials out of requests to a new provider.

## Routine backend update

Merge reviewed changes into the deployment branch first. Then connect to the server:

```bash
ssh Server
cd /home/Disk/WuZihang/apps/Claw-AI-Lab
git status --short
.venv/bin/python deploy/git_deploy.py --check
.venv/bin/python deploy/git_deploy.py
```

The script fetches `origin`, requires a clean tracked worktree and a fast-forward update, validates the fetched commit in an isolated directory, waits for no jobs (it stops with a message if any are active), saves a source backup, advances the checkout with `git merge --ff-only`, restarts only the bridge and checks services. Network, validation and dirty-worktree failures stop before installation. A post-installation failure attempts a guarded `git reset --keep` to the previous deployed commit and restores the bridge. It never force-pushes or deletes untracked data. Backups are stored under `/home/Disk/WuZihang/backups/claw-ai-lab/git-deploy-*` for this server layout.

The current test set covers deployment safeguards, resource intake, training infrastructure, config, CLI, bridge template handling and earlier regression tests. It is not the entire test suite, a real GPU training run or an LLM check. Training workers must be stopped/checkpointed before updating their implementation.

The routine script deliberately stops for frontend, dependency or startup-script changes: these need an explicitly reviewed dependency/build/restart procedure. It does not automatically install packages or restart every service. For such releases, prepare and test a separate Git worktree, preserve local config, arrange idle queues, build using the installed compatible Node runtime, back up the previous commit, then fast-forward and restart the affected services with health checks. Do not bypass this stop with a blind pull-and-restart command.

## Fill or replace the API key

```bash
cd /home/Disk/WuZihang/apps/Claw-AI-Lab
nano .env
```

Set `RESEARCHCLAW_API_KEY='your-new-key'` without posting the key to chat. Then reload the bridge while queues are idle:

```bash
.venv/bin/python deploy/git_deploy.py --restart-bridge
```

This reloads the private `.env` and keeps existing bridge arguments. Service health checks do not send a paid model request. Blank keys leave model calls unavailable until the owner fills them. Actual model authentication is a separate check after the key is supplied.

## Network failures

Keep `origin` pointed at GitHub. Retry `git fetch origin` after restoring server-to-GitHub connectivity. For a server without direct network access, an authenticated maintenance workstation may transfer a Git bundle created from the same reviewed repository; verify its advertised commit, fetch that bundle into `refs/remotes/origin/preview-v1.1.0`, and use the same review/testing/restart procedure. A Git bundle transfers Git objects and history; it is not a manually assembled copy of project source files. Never treat a copied working directory as a verified Git revision.
