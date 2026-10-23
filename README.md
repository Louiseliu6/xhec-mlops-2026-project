# Abalone ring-count prediction — Group 4

MLOps course project with Artefact: reproducible training and inference, MLflow
tracking, scheduled Prefect retraining and a Dockerized FastAPI prediction service.
The target is ring count, not age in years.

**Contributors:** Yunxian LIU ([@louiseliu6](https://github.com/louiseliu6)),
Felix Meyer ([@felixmeyer6](https://github.com/felixmeyer6)),
Lucien Descazeaud ([@luciendescazeaud](https://github.com/luciendescazeaud)),
Anatole Jacquin de Margerie ([@Anatole-JDM](https://github.com/Anatole-JDM)).

| Exercise branch | Status |
| --- | --- |
| `0/environment_setup` | Complete |
| `1/eda_and_modelling_notebooks` | Complete |
| `2/from_notebooks_to_modules` | Complete |
| `3/use_prefect` | Implemented |
| `4/deploy_model_api` | Implemented |

## Setup

Use Python 3.11 and uv 0.12.20. All commands below assume the repository root.

```bash
uv sync --locked --python 3.11
uv run --no-sync pre-commit install
```

Dependencies are defined in `pyproject.toml` and locked in `uv.lock`. Prefect 2.x
is used for compatibility with the existing Pydantic 1.x and MLflow stack.
The [Abalone dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)
is included at `data/abalone.csv`; no download is needed.

## Train and predict

```bash
uv run --no-sync python -m src.modelling.main data/abalone.csv
uv run --no-sync python -m src.modelling.predicting data/abalone.csv
```

Prediction accepts the eight feature columns, ignores `Rings` if present and prints
JSON ring-count predictions. `Sex` must be M, F or I; measurements must be non-negative.
Missing values and zero height are handled by preprocessing.

Plain and Prefect training use seed 42 and 60/20/20 train/validation/test splits.
Candidates are a training-mean baseline, linear regression and a fixed random forest.
Preprocessing is fitted inside each pipeline; selection uses validation MAE before
refitting on train + validation and evaluating once on the held-out test set.

Both training entry points save the **full fitted pipeline** (preprocessing and regressor)
to `src/web_service/local_objects/abalone_model.pkl`, replacing the previous model:

| Training entry point | Results | MLflow experiment |
| --- | --- | --- |
| `src.modelling.main` | `artifacts/step2_results.json` | `abalone_step2` |
| `src.modelling.flows` | `artifacts/step3_results.json` | `abalone_step3` |

## Prefect orchestration

### Server and one-off training

In **each terminal**, set the same local configuration from the repository root:

```bash
export PREFECT_HOME="$PWD/.prefect"
export PREFECT_API_URL="http://127.0.0.1:4200/api"
```

**Terminal 1:** start the server and leave it running.

```bash
uv run --no-sync prefect server start --host 127.0.0.1 --port 4200
```

**Terminal 2:** run training once.

```bash
uv run --no-sync python -m src.modelling.flows
```

Open http://127.0.0.1:4200 → **Flow Runs** → the latest `abalone-training` run
to inspect task states, the graph and logs. Prefect's local state is in `.prefect/`.
Relative paths passed to the flow or deployment are resolved against the project root.

### Scheduled retraining

Instead of one-off training, keep **terminal 2** serving a deployment:

```bash
uv run --no-sync python -m src.modelling.deploy
```

This registers `abalone-training/daily-retraining` for **02:00 UTC daily**
(`0 2 * * *`). View its schedule and history under **Deployments** in the UI.
No separate worker or work pool is needed.

**Terminal 3:** trigger a run immediately and wait for completion.

```bash
uv run --no-sync prefect deployment run 'abalone-training/daily-retraining' --watch
```

To change the schedule, stop the runner and restart with, for example,
`--cron '0 6 * * 1'` (Mondays at 06:00 UTC). Run either Python module with `--help`
for input and output options.

**Operational notes:**
- Keep both the server and runner running; this is a local deployment.
- The runner allows one active run. Do not start another runner or one-off training
  using the same output paths: the limit is not shared across processes.
- Stop the runner with **Ctrl+C before stopping the server**. This pauses its schedule;
  restarting the runner reactivates it. Check queued runs after sleep or interruption.

## Prediction API

The FastAPI service in `src/web_service/` loads the trained pipeline once at startup
and validates each request with Pydantic before predicting.

### Run with Docker

Start Docker Desktop (or another Docker engine), then build and run the image:

```bash
docker build -f Dockerfile.app -t abalone-api .
docker run --rm --name abalone-api -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api
```

The build installs the locked runtime dependencies and trains the model, so no local
`.pkl` is needed; rebuild after code or data changes. `bin/run_services.sh` starts the
API on container port 8001 and a Prefect server on 4201, published on the host as:

| Service | URL |
| --- | --- |
| API docs (try `/predict` in the browser) | http://localhost:8000/docs |
| Prefect UI | http://localhost:4200 |

Keep these host ports: the containerized Prefect UI calls its API at `localhost:4200`.
Stop any local Prefect server first, since it uses the same port. Stop the container
with **Ctrl+C**. To record a training run in the container's Prefect server:

```bash
docker exec abalone-api python -m src.modelling.flows
```

The running API keeps the model loaded at startup; rebuild the image to serve a new one.

### Run locally

After training (see [Train and predict](#train-and-predict)):

```bash
uv run --no-sync uvicorn src.web_service.main:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001/docs. Set `MODEL_PATH` to serve a different pipeline file.

### Requests

`GET /` is a health check. `POST /predict` takes one abalone with the CSV column names
and returns its predicted ring count (age in years is roughly rings + 1.5):

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"Sex": "M", "Length": 0.455, "Diameter": 0.365, "Height": 0.095,
       "Whole weight": 0.514, "Shucked weight": 0.2245, "Viscera weight": 0.101,
       "Shell weight": 0.15}'
# {"rings": 10.62...}  (HTTP 201)
```

All eight fields are required; `Sex` must be M, F or I and measurements non-negative.
Invalid requests return HTTP 422 naming the offending field. On Windows, run `curl`
from Git Bash or use the `/docs` page instead.

## Notebooks and MLflow

Explore `notebooks/eda.ipynb`, then `notebooks/modelling.ipynb`, using `.venv/bin/python`
as the kernel, or execute them from the command line:

```bash
uv run --no-sync jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
uv run --no-sync jupyter nbconvert --to notebook --execute --inplace notebooks/modelling.ipynb
```

The modelling notebook writes outputs to `artifacts/` and uses experiment `abalone_step1`.
All experiments are tracked locally in `mlruns/`, including metrics and provenance:

```bash
uv run --no-sync mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5003
```

Open http://127.0.0.1:5003 and select the experiment for your training entry point.
Use this project's environment and tracking store, not the course tutorial's server.

## Development checks

```bash
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync pre-commit run --all-files
```

Tests use synthetic data, temporary outputs and an isolated Prefect database; no
running server or Docker is needed. Pre-commit checks Ruff linting/formatting and configuration;
GitHub Actions runs the same checks and tests on pushes and pull requests.
To fix Python files, use `uv run --no-sync ruff check --fix .` and
`uv run --no-sync ruff format .`, then rerun the checks.

When dependencies change, use `uv add` or `uv lock` and commit `uv.lock`.
Generated artifacts, tracking data, `.prefect/` and `.venv/` are ignored; do not commit credentials.

## Repository history

Use [this official team fork](https://github.com/Louiseliu6/xhec-mlops-2026-project)
and target `main` for PRs. Earlier PR discussions and CI records remain in the
[previous repository](https://github.com/Anatole-JDM/MLOPS_ARTEFACT_project).

---

## Original course instructions

The original README follows unchanged. Where it differs, use the fork-specific
setup and commands above. Completed exercise instruction files (`PR_i.md`) are
removed after their checkpoints are met.

<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

Welcome to your MLOps project! In this hands-on project, you'll build a complete machine learning system to predict the age of abalone (a type of sea snail) using physical measurements instead of the traditional time-consuming method of counting shell rings under a microscope.

**Your Mission**: Transform a simple ML model into a production-ready system with automated training, deployment, and prediction capabilities.

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

**Your Goal**: Use easier-to-obtain physical measurements (shell weight, diameter, etc.) to predict the age automatically.

📥 **Download**: Get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)


## 🚀 Quick Start

### Prerequisites
- GitHub account
- [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (for dataset download)
- Python 3.11

### Setup Steps

1. **Fork this repository**
   - ⚠️ **Important**: Uncheck "Copy the `main` branch only" to get all project branches

2. **Add your team members** as admins to your forked repository

3. **Set up your development environment**:
   ```bash
   # Create and activate a virtual environment
   uv sync
   source .venv/bin/activate # on Windows: .venv\Scripts\activate

   # Install pre-commit hooks for code quality
   uv run pre-commit install
   ```

## 📋 What You'll Build

By the end of this project, you'll have created:

### 🤖 **Automated ML Pipeline**
- Training workflows using Prefect
- Automatic model retraining on schedule
- Reproducible model and data processing

### 🌐 **Prediction API**
- REST API for real-time predictions
- Input validation with Pydantic
- Docker containerization

### 📊 **Production-Ready Code**
- Clean, well-documented code
- Automated testing and formatting
- Proper error handling

## 📝 How to Work on This Project

### The Branch-by-Branch Approach

This project is organized into numbered branches, each representing a step in building your MLOps system. Think of it like a guided tutorial where each branch teaches you something new!

**Here's how it works**:

1. **Each branch = One pull request** with specific tasks
2. **Follow the numbers** (branch_0, branch_1, etc.) in order
3. **Read the PR instructions** (PR_0.md, PR_1.md, etc.) before starting
4. **Complete all TODOs** in that branch's code
5. **Create a pull request** when done
6. **Merge and move to the next branch**

### Step-by-Step Workflow

For each numbered branch:

```bash
# Switch to the branch
git checkout branch_number_i

# Get latest changes (except for branch_0)
git pull origin main
# Note: A VIM window might open - just type ":wq" to close it

# Push your branch
git push
```

Then:
1. 📖 Read the PR_i.md file carefully
2. 💻 Complete all the TODOs in the code
3. 🔧 Test your changes
4. 📤 Open **ONE** pull request to your main branch
5. ✅ Merge the pull request
6. 🔄 Move to the next branch

> **💡 Pro Tip**: Always integrate your previous work when starting a new branch (except branch_0)!

### 🔍 Understanding Pull Requests

Pull Requests (PRs) are how you propose and review changes before merging them into your main codebase. They're essential for team collaboration!

**Important**: When creating a PR, make sure you're merging into YOUR forked repository, not the original:

❌ **Wrong** (merging to original repo):
![PR Wrong](assets/PR_wrong.png)

✅ **Correct** (merging to your fork):
![PR Right](assets/PR_right.png)

## 💡 Development Tips

### Managing Dependencies

Use uv to manage dependencies. Install or update packages with:

```bash
uv add <package>==<version>
```

Then sync the environment and regenerate the dependency files:

```bash
uv sync
```

### Code Quality
- The pre-commit hooks will automatically format your code
- Remove all TODOs and unused code before final submission
- Use clear variable names and add docstrings

## 📊 Evaluation Criteria

Your project will be evaluated on:

### 🔍 **Code Quality**
- Clean, readable code structure
- Proper naming conventions
- Good use of docstrings and type hints

### 🎨 **Code Formatting**
- Consistent style (automated with pre-commit)
- Professional presentation

### ⚙️ **Functionality**
- Code runs without errors
- All requirements implemented correctly

### 📖 **Documentation & Reproducibility**
- Clear README with setup instructions
- Team member names and GitHub usernames
- Step-by-step instructions to run everything

### 🤝 **Collaboration**
- Effective use of Pull Requests
- Good teamwork and communication

---

## 🎯 Final Deliverables Checklist

When you're done, your repository should contain:

✅ **Automated Training Pipeline**
- [x] Prefect workflows for model training
- [x] Separate modules for training and inference
- [x] Reproducible model and encoder generation

✅ **Automated Deployment**
- [x] Prefect deployment for regular retraining

✅ **Production API**
- [x] Working REST API for predictions
- [x] Pydantic input validation
- [x] Docker containerization

✅ **Professional Documentation**
- [x] Updated README with team info
- [x] Clear setup and run instructions
- [x] All TODOs removed from code

---

**Ready to start? Head to branch_0 and read PR_0.md for your first task! 🚀**
