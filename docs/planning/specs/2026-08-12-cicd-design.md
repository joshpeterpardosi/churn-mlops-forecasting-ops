# CI/CD — Design Spec

**Sub-project 6 of 6.** Prior sub-projects (data layer, churn model, forecast model, serving, monitoring) are complete and merged to `master`. This is the final piece: get the repo onto GitHub with a working CI pipeline, and give it a public-facing README.

## Goal

On every push to `master` and every pull request: lint (ruff), run the full test suite, and build the Docker image. All three run in parallel and must pass. Create the GitHub repo and push, with a README a visitor can use to understand and run the project in under a minute.

## Non-goals

- No Docker image push to a registry (build-only smoke test — no registry auth to manage).
- No deployment step of any kind (this is a portfolio project, not a production service).
- No multi-Python-version matrix (CI matches the one dev environment exactly, see Dependency Pinning below).
- Not fixing the pre-existing Docker Compose end-to-end verification gap (flagged in project memory from sub-project 4, Docker Desktop was unavailable that session) — out of scope here, Dockerfile changes in this sub-project only affect the CI `docker build` smoke test, not `docker-compose.yml`.

## Components

### 1. Dependency pinning (`requirements-lock.txt`)

The monitoring sub-project's final review found that `evidently`'s own transitive dependencies (numpy, scipy, scikit-learn, etc.) are only lower-bounded in `pyproject.toml`, so a fresh `pip install` on a new machine (i.e., a GitHub Actions runner) could resolve a different, possibly broken, dependency set than what's proven to work locally — the exact failure mode that already happened once with evidently 0.4.37+/numpy<2.1 on Windows/cp313.

Fix: freeze a dependency closure to `requirements-lock.txt` at the repo root. **Amendment (discovered during plan-writing):** this machine has no project-local virtualenv — all development so far ran against the global `C:\Python313` interpreter, which has 335 packages installed (Jupyter, Streamlit, dbt, google-cloud-*, and other unrelated tooling). Freezing that directly would bloat the lockfile with irrelevant packages. Instead: create an isolated `.venv` (gitignored), `pip install -e ".[dev]"` into it (nothing else), then freeze *that* clean environment. This produces the actual minimal dependency closure — the same packages, versions resolved fresh against the pinned/lower-bounded ranges in `pyproject.toml` (numpy landed at 2.5.2, evidently stayed pinned at exactly 0.4.30 — already proven by the 101 passing tests, since those tests run the same resolved versions regardless of which interpreter runs them) without the unrelated global noise. CI and Docker both install from this file — no fresh resolution, ever.

`pyproject.toml` remains the source of truth for direct dependency *ranges* (for a human reading/adjusting them); `requirements-lock.txt` is what actually gets installed in CI/Docker. `ruff` is added to `[project.optional-dependencies] dev` and included in the lockfile freeze.

Install sequence (CI and Docker both):
```
pip install -r requirements-lock.txt
pip install --no-deps -e .
```
The `--no-deps` on the second step is load-bearing — it registers the local package in editable mode without triggering a second dependency resolution that could override the lockfile's pins.

### 2. Ruff lint

Add `[tool.ruff]` to `pyproject.toml` with the default rule set (E/F/W + isort/`I`). Run `ruff check src tests` locally, fix any real violations it surfaces (this codebase has never been linted), commit clean. Only then does CI's lint job become a real gate — it fails the build on any violation, no `continue-on-error`.

### 3. GitHub Actions workflow (`.github/workflows/ci.yml`)

Trigger: `push` to `master` and `pull_request`.

Three independent jobs, `runs-on: ubuntu-latest`, `python-version: "3.13"` (matches the local dev machine that produced `requirements-lock.txt` — see Non-goals):

- **`lint`** — checkout, `actions/setup-python`, install from lockfile, `ruff check src tests`.
- **`test`** — checkout, `actions/setup-python`, install from lockfile + local package, `pytest -q`. No fixture in the suite touches real local artifacts (`data/`, `mlflow.db`, `mlruns/`) — confirmed during sub-project 5's review that every test uses `tmp_path` or mocks; the suite is fully self-contained and needs nothing beyond a clean checkout.
- **`docker-build`** — checkout, `docker build .` (uses the Dockerfile from Component 4). No push, no registry, no tag beyond the build succeeding.

No job depends on another. A failure in any one job fails the workflow; the other two still run to completion and report independently, giving full signal instead of stopping at the first failure.

### 4. Dockerfile update

Current `Dockerfile` pins `python:3.11-slim` (mismatched against the 3.13 decision above) and does a fresh `pip install -e .` with no lockfile (same unpinned-resolution risk CI would otherwise have). Update:
- Base image → `python:3.13-slim`.
- `COPY requirements-lock.txt ./` and install from it before `pip install --no-deps -e .`, mirroring the CI install sequence — one dependency source of truth across local dev, CI, and the container.

`docker-compose.yml` is untouched — this sub-project doesn't touch the compose-based multi-service setup, only the standalone image build that CI smoke-tests.

### 5. README.md

Repo currently has none. Contents:
- One-paragraph project summary (churn + MRR forecasting MLOps loop, from `IDEA.md`'s Goal section).
- Architecture diagram (reuse/adapt the ASCII diagram already in `IDEA.md`).
- Setup/run instructions: clone, `pip install -r requirements-lock.txt`, `dagster dev` (orchestration UI), `uvicorn churn_mlops.serving.app:app` (API), `pytest` (tests).
- CI badge (GitHub Actions workflow status), added once the workflow exists and has run at least once for the badge URL to resolve.
- Brief per-sub-project summary (the 6-stage roadmap, all done) linking to `docs/planning/specs/` for anyone who wants the full design history.

### 6. Repo creation and publish

`gh repo create churn-mlops-forecasting-ops --public --source=. --remote=origin`, then push `master`. Repo name matches `pyproject.toml`'s `name` field and the local folder. Sole-author commits, no AI co-author trailer (standing global rule, already followed on every commit in this project). This step requires explicit go-ahead at execution time — it's a repo-creation + push action, not something to run unprompted mid-plan.

Two markdown/plan files were already scrubbed of internal AI-tooling references (commit `56021ac`) in preparation for this — the `docs/planning/plans/*.md` files no longer name the specific skill-invocation tooling used to build this project.

## Data Flow

1. Generate `requirements-lock.txt` from the current working environment.
2. Add ruff config, run it, fix violations, commit clean.
3. Add `.github/workflows/ci.yml`.
4. Update `Dockerfile` to match the lockfile + Python 3.13 decision.
5. Write `README.md` (CI badge added/fixed after step 6 once the workflow has a real run to link).
6. `gh repo create` + push (explicit confirmation gate).
7. Verify the real Actions run went green (`gh run watch` / `gh run view`) — not just that the YAML is syntactically valid. This is the final-review verification step, consistent with every prior sub-project's practice of confirming behavior against a real run rather than trusting the diff.

## Error Handling & Testing

- CI jobs use GitHub Actions' default behavior: a failing step fails the job, other jobs still run.
- No new application code in this sub-project — nothing to unit-test beyond what sub-projects 1-5 already cover. Verification is: ruff clean locally, `pytest -q` green locally (already true, 101 passed), `docker build .` succeeds locally, and — the real proof — the actual GitHub Actions run on the pushed repo goes green across all 3 jobs.

## Open Items Resolved During Brainstorming

- Repo visibility: public.
- Docker in CI: build-only, no registry push.
- CI trigger: push to `master` + pull requests.
- Python version: 3.13 only, matching the lockfile's source environment.
- Dependency reproducibility: closed via `requirements-lock.txt`, not deferred further.
- README: in scope for this sub-project.
