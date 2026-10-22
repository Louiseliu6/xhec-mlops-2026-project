# MLOPS_ARTEFACT_project

Project for the mlops course with Artefact

## Repositories

Current team repository (official course fork):
https://github.com/Louiseliu6/xhec-mlops-2026-project

Previous repository (earlier pull request discussions, reviews, and CI records):
https://github.com/Anatole-JDM/MLOPS_ARTEFACT_project

The code, exercise branches, and Git commit history were transferred to the
current repository. Use the current repository for assessment and all future
pushes and pull requests. Earlier GitHub PR discussions and CI run logs remain
in the previous repository.

## Step 1: Data exploration and modelling

The Abalone dataset is included at `data/abalone.csv`.
Source: https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset.
The expectedcolumns are Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight, and Rings. The target is ring count, not years.

```bash
uv sync --locked --python 3.11
```

In VS Code open `notebooks/eda.ipynb`, then `notebooks/modelling.ipynb`.
Choose `.venv/bin/python` as the notebook kernel and use Run All.
Alternatively, execute from the project root:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/modelling.ipynb
```

The fixed seed is 42, with approximately 60/20/20 train/validation/test splits.
Preprocessing is fitted inside each model pipeline. Choose by validation MAE,
then evaluate the selected model once on held-out test data. Candidates are a
training-mean baseline, linear regression and a fixed random forest.
The notebooks compare ring-count MAE, RMSE and R-squared.

Experiments are local in `mlruns`; the selected pipeline and metrics are in
`artifacts`. Both directories and the CSV are excluded from Git.
To see this project's experiments, run from the project root:

```bash
uv run mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5003
```

Open http://127.0.0.1:5003 and select `abalone_step1`. This project uses MLflow
2.7.1 in its own environment; do not reuse the course tutorial's database or server.

## Step 0: Development environment

Use Python 3.11 and uv 0.12.20. Run these commands from the repository root:

```bash
uv sync --locked --python 3.11
source .venv/bin/activate
uv run pre-commit install
uv run pre-commit run --all-files
```

`uv sync` installs application and development dependencies into `.venv`.
Pydantic is an application dependency; Ruff and pre-commit are development tools.
Pydantic 1.x is selected for compatibility with the course's older library stack.
Dependencies for modelling, orchestration and the API will be added in their steps.

Pre-commit runs before each commit. GitHub Actions runs the same checks on pushes
and pull requests, using the committed `uv.lock` for reproducibility.
Ruff checks Python errors and import order, and checks formatting. To fix Python
files when needed, run:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pre-commit run --all-files
```

Some hooks fix whitespace automatically. Review their edits and run the checks again.
Whenever dependencies change, run `uv lock` (or `uv add`) and commit `uv.lock`.
Do not commit `.venv`, credentials, or downloaded datasets.

### Contributors

- Yunxian LIU (GitHub: louiseliu6)
- Felix Meyer (GitHub:felixmeyer6)
- Lucien Descazeaud (GitHub:luciendescazeaud)
- Anatole Jacquin de Margerie (GitHub:Anatole-JDM)

### Actual exercise branches

1. `0/environment_setup`
2. `1/eda_and_modelling_notebooks`
3. `2/from_notebooks_to_modules`
4. `3/use_prefect`
5. `4/deploy_model_api`

Read the corresponding `PR_i.md`, complete and check that step, then open a PR
to this team's `main` branch. Merge the previous work into each subsequent branch
before continuing. This repository uses `main`, not `master`.
The official course instructions below describe the remaining project goals.

---

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
- [ ] Prefect workflows for model training
- [ ] Separate modules for training and inference
- [ ] Reproducible model and encoder generation

✅ **Automated Deployment**
- [ ] Prefect deployment for regular retraining

✅ **Production API**
- [ ] Working REST API for predictions
- [ ] Pydantic input validation
- [ ] Docker containerization

✅ **Professional Documentation**
- [ ] Updated README with team info
- [ ] Clear setup and run instructions
- [ ] All TODOs removed from code

---

**Ready to start? Head to branch_0 and read PR_0.md for your first task! 🚀**
