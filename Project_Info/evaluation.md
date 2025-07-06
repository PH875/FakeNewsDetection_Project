# Project Specification and Evaluation Criteria

This document outlines the expectations, requirements, and grading criteria for the final course projects. Projects may be either **dataset-focused** or **model-focused** (see below). Students will complete the project over a **two-week period**, working in groups of **two**.

## ✅ General Requirements

### 1. Code Quality
- Write clean, modular Python code.
- Structure your code to integrate with the course's `courselib` package.
- Include a `README.md` with:
  - A brief project overview
  - Instructions for running experiments
  - A list of dependencies (ideally via `requirements.txt`)

Use the [README template](https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md) from [Tips for Publishing Research Code](https://github.com/paperswithcode/releasing-research-code).

### 2. Report
- Submit a 1–3 page PDF report written in LaTeX.
- The report should clearly summarize:
  - **What** task was addressed
  - **How** it was approached
  - **Why** certain methods/experiments were chosen
  - Key results and interpretations
- Include relevant figures (e.g., plots, tables).

Use the LaTeX report template provided on Moodle.

### 3. Reproducibility
- All experiments must be reproducible with a single script or notebook.
- Provide seeds, configurations, and system/environment requirements.

See [Tips for Publishing Research Code](https://github.com/paperswithcode/releasing-research-code) for reproducibility guidelines.

## 📊 Grading Rubric

| **Criterion**                         | **Max. Points** | **Requirements** |
|--------------------------------------|-----------------|------------------|
| **Code correctness and structure**   | 20              | Clean, modular, well-organized code. Integrates with `courselib`, avoids hardcoded constants, uses appropriate functions/classes. Includes a clear README with usage instructions and dependencies. |
| **Experiment setup and execution**   | 20              | Runs end-to-end without manual steps. Produces results shown in the report. Reproducible and efficient. Penalized if excessively slow or impractical. |
| **Report clarity and depth**         | 20              | Clear and concise writing. Explains *what* was done, *how*, and *why*. Includes figures. Well-structured and logically organized. |
| **Insightful analysis and discussion** | 20            | Goes beyond raw results. Compares models, includes relevant plots, and provides meaningful interpretation. |
| **Ambition and originality**         | 20              | Extends beyond course material. Adds new models, explores new datasets, or investigates novel ideas. |

> **Total**: 100 points  
> **Minimum to pass**: 50 points

## 📁 Project Types

### Type 1: Dataset-Focused Projects
- You are provided with a public dataset.
- **Goal**: Solve a meaningful ML task (e.g., classification, regression, clustering, dimensionality reduction, decision making).

**Minimum Requirements:**
- Choose appropriate models based on the task and justify your choices. Try at least **two models** from the course (e.g., SVM vs. MLP). Discuss hyperparameter settings in the report.
- Select suitable evaluation metrics and visualization strategies. Visualize learning curves, decision boundaries, or misclassifications.
- Interpret results and compare models. Clearly discuss conclusions in the report.

**For a higher grade:**
- Implement a model not covered in the course that suits the task.
- Achieve competitive performance.
- Use theoretical concepts (possibly beyond the course) to justify model design and analysis.

### Type 2: Model-Focused Projects
- You are assigned a specific model (e.g., soft-margin SVM, Lasso).
- **Goal**: Analyze the model’s behavior across different datasets, tasks, and hyperparameter settings.

**Minimum Requirements:**
- Design experiments to explore strengths and weaknesses of the model (consider performance, robustness, optimization dynamics, etc.)
- Use synthetic (e.g., 2D toy) and/or real datasets to demonstrate model behavior.
- Vary key hyperparameters and analyze the impact.
- Visualize important behaviors (e.g., training curves, decision boundaries).
- Report and interpret all findings.

**For a higher grade:**
- Link empirical findings to theoretical ideas (e.g., model's distributional assumptions, theoretical peroformance guarantees, etc.).
- Perform deeper or more original analysis (e.g., failure modes).
- Compare the assigned model to others on tasks that clearly demonstrate the difference. 
- Go beyond the course by reading a related paper, implementing a model's variation, or testing under novel conditions (e.g., label noise, data corruption, adversarial examples).

## 📌 Deliverables Summary

- ✅ Code: clean, modular, runs end-to-end
- ✅ README: includes overview, usage, and dependencies
- ✅ Report: max 3-page PDF (LaTeX)
- ✅ Figures: learning curves, accuracy plots, decision boundaries, confusion matrices, etc.
