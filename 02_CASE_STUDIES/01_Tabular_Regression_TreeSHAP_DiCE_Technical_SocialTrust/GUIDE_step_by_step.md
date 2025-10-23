# Pilot A · Step-by-Step Runbook (TreeSHAP + DiCE)

> Objective: run the pilot end-to-end, verify reproducibility, and validate trust metrics — technical (fidelity, completeness, stability) and social (actionability, diversity, plausibility).

---

## 0) Prerequisites
- Python ≥ 3.10, Jupyter/Colab
- Repository cloned with all `configs/` files
- GPU **not** required

---

## 1) Installation
```python
# Cell 1 — dependencies
# We install the core libraries for the pilot:
# - dice-ml: for counterfactual explanations (DiCE)
# - shap: for feature attribution explanations
# - scikit-learn: for the base model (RandomForestRegressor)
# - pandas/numpy: for data handling
# The --quiet flag suppresses verbose logs.
!pip install dice-ml shap scikit-learn pandas numpy --quiet
```
### Step 1.2 — Verify library versions

```python
import numpy, pandas, sklearn, shap, dice_ml
from importlib.metadata import version, PackageNotFoundError

print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("SHAP:", shap.__version__)

# dice-ml does not expose __version__; we read it from package metadata
try:
    print("dice-ml:", version("dice-ml"))
except PackageNotFoundError:
    print("dice-ml: (not found in metadata)")
```
#### Explanation:
This step confirms the environment versions used in the pilot.
Version logging ensures that SHAP and DiCE results can be reproduced under the same dependency setup.
Because dice-ml does not include a built-in __version__ attribute, we retrieve it through Python’s package metadata system.

#### Expected output (example):

NumPy: 1.26.4
Pandas: 2.2.2
scikit-learn: 1.5.2
SHAP: 0.46.0
dice-ml: 0.10


#### Checkpoint:
Make sure versions match (or are close to) those listed above.
If a package mismatch causes unexpected metric drift, document it in results/metadata.json.

### Step 1.3 — Create and verify the project directory structure

```python
from pathlib import Path
import json

# If this notebook is inside notebooks/, move up to the project root
BASE = Path.cwd().resolve().parent
PROJECT = BASE / "01_Tabular_Regression_TreeSHAP_DiCE_Technical_SocialTrust"

# Create subfolders for data, results, configs, and notebooks
for p in [
    PROJECT / "data/raw",
    PROJECT / "data/processed",
    PROJECT / "results/visuals",
    PROJECT / "configs",
    PROJECT / "notebooks",
]:
    p.mkdir(parents=True, exist_ok=True)

print("Project structure created at:", PROJECT)
```

#### Explanation:
This step ensures that the directory structure matches the repository layout.
It uses pathlib to handle paths consistently across operating systems.
Each folder (data, results, configs, notebooks) is created if missing — avoiding errors during file export.

#### Expected output (example):

Project structure created at: /content/repo_clone/01_Tabular_Regression_TreeSHAP_DiCE_Technical_SocialTrust


#### Checkpoint:
Confirm that the following folders now exist:

data/raw/
data/processed/
results/visuals/
configs/
notebooks/


If any folder is missing, check the relative path of the notebook — it must be executed from inside /notebooks/.

### Step 2.1 — Load seeds and tolerance priors

```python
# Load seeds and define SEED_SK (safe fallback)
import json, yaml

with open(PROJECT / "configs" / "seeds.json") as f:
    SEEDS = json.load(f)

SEED_SK = SEEDS.get("sklearn", SEEDS.get("data_split_seed", 42))

# (optional if using these parameters later)
try:
    with open(PROJECT / "configs" / "priors_clinical.yaml") as f:
        PRIORS = yaml.safe_load(f)
    TAU = PRIORS["tolerances"]["completeness_abs_tau"]
    EPS = PRIORS["tolerances"]["stability_perturbation"]["epsilon"]
except FileNotFoundError:
    TAU, EPS = 1e-6, 0.01  # default values if YAML not yet present
```
#### Important!
Before loading configuration values, download the reference configuration files from the GitHub repository.
This guarantees that the notebook runs with the same parameters used in the public experiment and ensures full reproducibility.

Files downloaded:

seeds.json → defines random seeds for data and model reproducibility

priors_clinical.yaml → defines clinical priors, tolerances (τ), and perturbation noise (ε)

dice_constraints.yaml → defines immutable variables, feature bounds, and counterfactual constraints for DiCE

If the files already exist locally, this step overwrites them with the latest repository version.

#### Explanation:
This step loads the random seeds and tolerance parameters used across the pilot:

seeds.json ensures reproducibility in data splits and model training.

priors_clinical.yaml defines tolerance thresholds and perturbation noise used in completeness (τ) and stability (ε) metrics.

τ (tau) = maximum absolute deviation allowed between model prediction and SHAP reconstruction.

ε (epsilon) = magnitude of noise used to test explanation stability.

If the YAML file is missing (e.g., first run), the notebook sets safe defaults (τ=1e-6, ε=0.01) to allow smooth execution.

#### Expected output (example):

Loaded seeds: {'sklearn': 123, 'data_split_seed': 42}
TAU = 0.001
EPS = 0.02


#### Checkpoint:
Verify that both files exist under configs/ and contain the correct parameters:

configs/seeds.json
configs/priors_clinical.yaml

If priors_clinical.yaml is missing, defaults will be used but must later be replaced with validated values for reproducible reporting.

> [!NOTE]
>
>**Conceptual note** — τ (tau) and ε (epsilon) in the XAI-TrustFramework
>
>`priors_clinical.yaml` does not define *trust notions* directly.  
>Instead, it stores **empirical thresholds** used to validate the **method-guaranteed properties** of XAI methods in practice.
>
>| Hierarchy | Example (TreeSHAP) | Relation | Parameter |
>|------------|--------------------|-----------|------------|
>| **Method-Guaranteed Property** | Additivity | theoretical guarantee | — |
>| **Trust Notion** | Completeness | empirical interpretation of Additivity | — |
>| **Metric** | % of instances where |f(x) − Σφᵢ(x)| ≤ τ | operationalizes Completeness |
>| **Parameter** | τ (tau) | acceptable deviation for Additivity | stored in `priors_clinical.yaml` |
>
>| Hierarchy | Example (TreeSHAP) | Relation | Parameter |
>|------------|--------------------|-----------|------------|
>| **Method-Guaranteed Property** | Consistency | theoretical guarantee | — |
>| **Trust Notion** | Stability | empirical interpretation of Consistency | — |
>| **Metric** | Cosine/Spearman similarity under ε-noise | operationalizes Stability |
>| **Parameter** | ε (epsilon) | perturbation level for Consistency | stored in `priors_clinical.yaml` |
>
>In code, these parameters are loaded as:
>
>```python
>TAU = PRIORS["tolerances"]["additivity_abs_tau"]   # τ — threshold for Additivity/Completeness
>EPS = PRIORS["tolerances"]["consistency_perturbation"]["epsilon"]  # ε — perturbation for Consistency/Stability
>```
>
>Interpretation:
>τ and ε are not trust notions themselves.
>They are empirical thresholds enabling the quantitative verification of TreeSHAP’s theoretical guarantees (Additivity and Consistency).
>
>For the full theoretical mapping between method-guaranteed properties, trust notions, and metrics, see the XAI-TrustFramework Conceptual Model
>
### Step 3.1 — Load and save the clinical dataset

```python
from sklearn.datasets import load_diabetes
import pandas as pd

# Use the public 'diabetes' dataset from scikit-learn for reproducibility
data = load_diabetes(as_frame=True)
df = data.frame.copy()
df["target"] = data.target  # make target column explicit for clarity

# Save a raw copy for experiment traceability
raw_path = PROJECT / "data/raw/diabetes.csv"
df.to_csv(raw_path, index=False)

print("Dataset saved at:", raw_path)
df.head()
```
#### Explanation:
This step loads a publicly available regression dataset (sklearn.diabetes) used widely in clinical ML demonstrations.
It ensures that all experiments are fully reproducible without requiring private data.

#### Workflow:

Load the dataset as a pandas DataFrame (as_frame=True).

Explicitly add the target column (regression label).

Save a copy to /data/raw/diabetes.csv for traceability and reproducibility.

#### Expected output (example):

Dataset saved at: /content/.../data/raw/diabetes.csv


#### Checkpoint:
Verify that the dataset file exists at:

data/raw/diabetes.csv

and that the first rows display expected columns such as:

['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'target']

### Step 3.2 — Upload the raw dataset to GitHub

Once the `diabetes.csv` file has been generated under `/data/raw/`,  
you must **download it from Colab and upload it to the GitHub repository**.

**Purpose:**  
Versioning the raw dataset ensures complete traceability — anyone rerunning the pilot will use the *exact same data*.

**Instructions:**
1. In the left panel of Google Colab, open the folder `data/raw/`.
2. Right-click on `diabetes.csv` → **Download**.
3. Upload the file to the repository at: 01_Tabular_Regression_TreeSHAP_DiCE_Technical_SocialTrust/data/raw/diabetes.csv

#### Checkpoint:
Confirm that the file now appears in the GitHub repository under
data/raw/diabetes.csv and matches the one generated locally.

This ensures that all future runs of the pilot will reference the same raw dataset version.

### Step 3.3 — Split data into train/validation/test sets

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# Load seeds for consistency across notebooks
with open(PROJECT / "configs/seeds.json") as f:
    SEEDS = json.load(f)

# Load the raw dataset (instead of fetching again from sklearn) for FAIR traceability
df = pd.read_csv(PROJECT / "data/raw/diabetes.csv")

X = df.drop(columns=["target"])
y = df["target"]

# Split: 70% train, 15% validation, 15% test
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=SEED_SK
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=SEED_SK
)

proc = PROJECT / "data/processed"
X_train.to_csv(proc / "X_train.csv", index=False)
y_train.to_csv(proc / "y_train.csv", index=False)
X_val.to_csv(proc / "X_val.csv", index=False)
y_val.to_csv(proc / "y_val.csv", index=False)
X_test.to_csv(proc / "X_test.csv", index=False)
y_test.to_csv(proc / "y_test.csv", index=False)

print("Splits saved in:", proc)
print("Shapes ->",
      "train:", X_train.shape,
      "| val:", X_val.shape,
      "| test:", X_test.shape)
```
#### Explanation:
This step divides the dataset into three subsets — training, validation, and test — to prevent data leakage and enable unbiased evaluation.

Data split:

70% training → used to fit the model.

15% validation → used for tuning and intermediate checks.

15% test → used only for final evaluation.

The random seeds loaded from seeds.json ensure consistent splits across different notebooks and runs, enabling reproducibility.

All subsets are saved in /data/processed/ so that subsequent notebooks can load them directly without re-splitting the data.

> [!NOTE]
> **What is the validation (val) set for, and where is it used later?**
>
> The **validation (val)** set is a *calibration* set: it is **not** used to train the model (train) and **not** reserved for the final report (test). You use it to tune methodological choices and sanity-check explainability before touching the test set.
>
> **Roles of each split**
> - **Train** → fit the model.
> - **Validation (val)** → calibrate thresholds and check explainability behavior.
> - **Test** → final, untouched evaluation for reporting.
>
> **How val is used in this pilot**
> 1) **TreeSHAP (technical metrics)**: try/adjust empirical thresholds from `priors_clinical.yaml`  
>    - τ (*additivity_abs_tau*) for Additivity → Completeness  
>    - ε (*consistency_perturbation.epsilon*) for Consistency → Stability  
>    Run on **val** first to ensure metrics behave reasonably before running on **test**.
>
> 2) **DiCE (counterfactuals)**: sandbox to validate constraints from `dice_constraints.yaml`  
>    - immutables, feature bounds, and relative costs  
>    Ensure counterfactuals are feasible/plausible on **val** before using **test**.
>
> 3) **Intermediate vs final results**  
>    - Save calibration metrics on **val** (e.g., `results/metrics_summary_val.csv`).  
>    - Only after calibration, compute final metrics on **test** (e.g., `results/metrics_summary_test.csv`).
>
> This separation prevents **data leakage** and keeps **test** as an honest, reproducible estimate of performance and trust properties.



#### Expected output (example):

Splits saved in: /content/.../data/processed
Shapes -> train: (309, 10) | val: (66, 10) | test: (67, 10)


#### Checkpoint:
Confirm that the following files exist under data/processed/:

X_train.csv
y_train.csv
X_val.csv
y_val.csv
X_test.csv
y_test.csv


>[!NOTE]
>Conceptual Note — FAIR reproducibility
>
>Loading and splitting data from the stored /data/raw/diabetes.csv (instead of fetching it again)
>enforces FAIR principles — Findable, Accessible, Interoperable, Reproducible —
>ensuring that the exact same dataset and partitions can be reused and audited later.
>
