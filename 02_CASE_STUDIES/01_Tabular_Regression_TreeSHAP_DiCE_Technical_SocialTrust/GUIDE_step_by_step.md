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

## 2) Configuration
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
    TAU = PRIORS["tolerances"]["additivity_abs_tau"]
    EPS = PRIORS["tolerances"]["consistency_perturbation"]["epsilon"]
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
## 3) Dataset
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

### Step 3.4 — Upload processed data to GitHub

After generating the processed splits, download them from Colab and upload them to the repository to ensure full reproducibility.

#### Folder to upload
01_Tabular_Regression_TreeSHAP_DiCE_Technical_SocialTrust/data/processed/

#### Files to include
X_train.csv
y_train.csv
X_val.csv
y_val.csv
X_test.csv
y_test.csv

#### Steps
1. In Colab, open the left sidebar → **data/processed/**.  
2. Right-click each `.csv` file and select **Download**.  
3. In your local GitHub repository, move the files to:
01_Tabular_Regression_TreeSHAP_DiCE_Technical_SocialTrust/data/processed/

Once uploaded, the processed splits will be available for all notebooks in the pilot (TreeSHAP + DiCE) to ensure consistent, reproducible results.

## 4) Train model
### Step 4.1 — Train the baseline model (RandomForestRegressor)

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import json

# Load processed splits
proc = PROJECT / "data/processed"
X_train = pd.read_csv(proc / "X_train.csv")
y_train = pd.read_csv(proc / "y_train.csv").squeeze("columns")
X_val   = pd.read_csv(proc / "X_val.csv")
y_val   = pd.read_csv(proc / "y_val.csv").squeeze("columns")

# Load seeds
with open(PROJECT / "configs/seeds.json") as f:
    SEEDS = json.load(f)

rf = RandomForestRegressor(
    n_estimators=300,      # enough trees for stability
    random_state=SEED_SK,
    n_jobs=-1
)
rf.fit(X_train, y_train)

pred_val = rf.predict(X_val)
mae_val = mean_absolute_error(y_val, pred_val)
print(f"MAE (validation): {mae_val:.3f}")
```
#### Explanation:
We train a simple and robust baseline using RandomForestRegressor.
This step provides a reference model before applying explainability techniques (TreeSHAP and DiCE).
The goal is not to optimize hyperparameters but to ensure the model behaves predictably and its explanations are interpretable.

#### Key points:

Fixed random seeds ensure reproducibility.

n_estimators=300 provides stability without overfitting.

The validation MAE serves as a baseline performance reference before any XAI analysis.

#### Expected output (example):

MAE (validation): 43.217

> [!NOTE]
> **Conceptual Note — Understanding MAE in this context**
>
> The **Mean Absolute Error (MAE)** is used here as a *neutral reference metric* for model performance before any explainability analysis.
>
> - **What MAE measures:**  
>   The average absolute difference between the model’s predicted values `ŷ` and the true targets `y`.  
>   $$\[
>   MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|
>   \]$$
>
> - **Why MAE (not R² or RMSE):**  
>   - MAE is **scale-consistent** and directly interpretable in the same units as the target (e.g., clinical progression score).  
>   - It is **less sensitive to outliers** than RMSE, avoiding distortions when the data are standardized (z-score).  
>   - It provides a **transparent baseline** for comparing XAI-derived metrics (like fidelity) later on.
>
> - **How it connects to the XAI-TrustFramework:**  
>   The MAE on the *validation set* establishes the **technical reliability baseline** of the predictive model.  
>   Only after confirming the model performs consistently can we interpret **fidelity**, **additivity**, or **stability** metrics as reflections of *explanation quality*, rather than of model error.
>
> *Reference:*  
>   In the framework’s hierarchy, MAE is part of the **pre-XAI reliability layer** — it assesses *model adequacy* before trust metrics are applied.


>[!NOTE]
>Why use a baseline first?
>Establishing a stable baseline avoids confusing the performance of the model itself
>with the quality of the explanations.
>Once fidelity, additivity, consistency, and other trust metrics are applied,
>we can attribute differences in behavior to the explainability methods — not to model instability.
>

## 5) SHAP
### Step 5.1 — Compute TreeSHAP attributions (interventional)

```python
import shap
import numpy as np

shap_explainer = shap.TreeExplainer(
    rf,
    data=X_train,                       # background dataset for expectations
    feature_perturbation="interventional"  # mitigates correlation artefacts
)

# SHAP on the validation split
shap_vals_val = shap_explainer.shap_values(X_val)
phi0 = shap_explainer.expected_value

# Ensure 2D ndarray (n_instances x n_features)
shap_matrix_val = np.asarray(shap_vals_val)
print("SHAP matrix shape:", shap_matrix_val.shape)
print("Expected value (phi0):", float(phi0))
```
> [!NOTE]
> ## Theoretical Foundation — TreeSHAP as Controlled System Decomposition
> 
> This analysis does **not** attempt to “open” the model or inspect its internals.  
> Instead, it provides a **functional characterization** of the model’s behavior through *input–output decomposition*.
> 
> TreeSHAP originates from **cooperative game theory**, formalized by Shapley’s *efficiency theorem* (1953).  
> Each feature acts as an **input channel** whose marginal influence on the output can be quantified under a controlled, fairness-preserving framework of *coalitional perturbations*.
> 
> In engineering terms, SHAP becomes a **system decomposition operator**:  
> it measures how the model distributes responsibility among inputs — providing **behavioral evidence** of structural reliability without introspection.
> 
> ---
> 
> ### Shapley Axioms and Empirical Trust Validation
> 
> | **Theoretical Property** | **Engineering Interpretation** | **Empirical Trust Notion** | **Metric / Evidence** | **Parameter / File** | **Operational Status** |
> |---------------------------|--------------------------------|-----------------------------|------------------------|----------------------|-------------------------|
> | **Additivity / Efficiency** | The total output equals the sum of individual feature contributions. | **Completeness** | % of instances where |f(x) − [φ₀ + Σφᵢ(x)]| ≤ τ | τ (`tolerances.additivity_abs_tau`) | ✅ Operationalized *(Additivity → Completeness, τ)* |
> | **Local Accuracy** | Each instance’s prediction equals its additive reconstruction. | **Fidelity** | MAE or R² between model output and [φ₀ + Σφᵢ(x)] | `metrics_summary_val.csv` | ✅ Operationalized |
> | **Consistency / Symmetry** | Inputs with equivalent impact receive equal attributions. | **Stability** | Cosine/Spearman similarity of SHAP vectors under ε-perturbations | ε (`tolerances.consistency_perturbation.epsilon`) / `stability_analysis.csv` | ✅ Operationalized *(Consistency → Stability, ε)* |
> | **Dummy / Missingness** | Inputs that have no influence receive zero attribution. | — *(not quantified)* | — | — | ⚪ Not included in this pilot |
> 
> 
> ---
> 
> ### Conceptual Summary — Controlled Behavioral Characterization
> 
> Within the *XAI-TrustFramework*, SHAP is **not** treated as a human-interpretability tool  
> but as a **non-intrusive diagnostic instrument** that validates the model’s internal coherence.
> 
> - **Additivity → Completeness:** verifies that outputs can be reconstructed from feature contributions.  
> - **Local Accuracy → Fidelity:** measures how faithfully those reconstructions reproduce predictions.  
> - **Consistency → Stability:** tests robustness of explanations under small perturbations.  
> 
> Together, these properties operationalize the **Technical Dimension of Trust**:  
> they ensure that the model behaves as a **stable, decomposable, and reproducible system**,  
> providing quantitative evidence of reliability — *without opening any black box*.



> [!NOTE]
> **Conceptual Note — Why TreeSHAP here? (Data → Task → Method → Trust Link)**
>
> **Data Type**  
> The dataset used (`sklearn.diabetes`) is *tabular*, composed of continuous clinical and biochemical variables  
> (e.g., BMI, blood pressure, serum indicators) with moderate inter-feature correlations.  
> Such data demand **local additive explanations** that can assign precise numerical contributions  
> to each input variable without assuming independence among them.
>
> **Task Type**  
> The task is **regression** — predicting a continuous disease progression score.  
> For regression, explanations must be:
> - **Additive**, so that the total prediction equals the sum of contributions plus a baseline.  
> - **Signed**, showing whether each feature increases or decreases the predicted outcome.  
> - **Quantitatively consistent**, producing comparable attributions across similar patients.
>
> **XAI Method — TreeSHAP**  
> TreeSHAP provides an *exact, model-consistent solution* for tree-based regressors (such as RandomForest).  
> It satisfies three theoretical guarantees — **Additivity**, **Consistency**, and **Missingness** —  
> making it particularly suited for quantifying *how much* each input feature contributes to a numeric output.  
> Unlike perturbation-based methods (e.g., LIME), TreeSHAP leverages the internal tree structure to compute  
> deterministic attributions without random sampling noise.
>
> **Trust Link — Technical Dimension**  
> Within the *XAI-TrustFramework*, TreeSHAP operationalizes the **Technical Dimension of Trust**:  
> it validates whether the model behaves as a *reliable, decomposable system* by empirically testing:
> - **Additivity → Completeness** — predictions can be reconstructed additively (τ threshold).  
> - **Local Accuracy → Fidelity** — reconstructed predictions align with the model’s outputs (MAE/R²).  
> - **Consistency → Stability** — explanations remain stable under ε-perturbations.  
>
> This data–task–method alignment ensures that TreeSHAP is not used for “interpretation,”  
> but as a **functional verification mechanism**, establishing *quantitative trust evidence*  
> grounded in the model’s observable, reproducible behavior.
>
> **Why Technical Dimension only?**  
> TreeSHAP explains *how the model behaves*, not *how a person can act upon it*.  
> Its goal is to ensure **internal reliability and mathematical coherence** of the model’s reasoning —  
> key aspects of the **technical dimension** of trust.  
> Human-centered notions like *actionability* or *plausibility* are addressed later through **DiCE**.
>
> *Summary:*  
> ```
> Data → Tabular  
> Task → Regression  
> Method → TreeSHAP  
> Theoretical Properties → Additivity, Consistency, Missingness  
> Trust Notions (operationalized) → Fidelity, Completeness, Stability  
> Dimension → Technical
> ```

> [!NOTE]
> **How SHAP’s axioms map to empirical trust validation in the XAI-TrustFramework**
>
> The table below shows how the formal axioms of the Shapley value —originating in cooperative game theory—  
> are **operationalized as empirical trust notions** within the *XAI-TrustFramework*.
>
> Each theoretical property defines a *guaranteed behavior* of the explanation model,  
> while its corresponding trust notion expresses that property as a **measurable hypothesis**  
> about the system’s observable outputs.
>
> - **Additivity (Efficiency)** → becomes **Completeness**, verifying whether each prediction can be reconstructed as  
>   the sum of individual feature contributions within a defined tolerance (τ).  
> - **Local Accuracy** → becomes **Fidelity**, assessing how closely the additive reconstruction matches the model’s actual prediction.  
> - **Consistency (Symmetry)** → becomes **Stability**, quantifying the robustness of explanations under ε-perturbations.  
> - **Dummy (Missingness)** → remains theoretically present, ensuring that non-influential features receive φ = 0,  
>   though it is not empirically tested in this pilot.
>
> This alignment converts **normative axioms into testable criteria**.  
> What was once a theoretical guarantee of fairness and coherence becomes a **set of trust metrics**  
> that can be validated empirically.  
>
> The resulting structure allows each trust dimension —technical, social, organizational, and institutional—  
> to be grounded in reproducible evidence rather than abstract claims.  
> In other words, the *XAI-TrustFramework* extends Shapley’s fairness principles  
> into a **methodology for assessing trustworthy behavior in AI systems**.
> 

#### Outputs captured:

expected_value (φ₀) — baseline model output with no feature contributions.

shap_matrix_val (φ) — attribution matrix with shape (n_instances × n_features) on the validation split.

#### Checks:

For regression, shap_values returns a 2D matrix; expected_value is a scalar.

Ensure the background data (data=X_train) matches your training distribution to keep expectations meaningful.

> [!NOTE]
> **Conceptual link — Additivity (→ Completeness)**
>
> TreeSHAP satisfies **Additivity** by design.  
> Empirically, we test its observable notion — **Completeness** — by verifying that:
>
> $$
> f(x) \approx \phi_0 + \sum_i \phi_i(x)
> $$
>
> within a tolerance threshold **τ**, defined in  
> `priors_clinical.yaml` → `tolerances.additivity_abs_tau`.  
> This validation is first performed on the **validation** split and later replicated on the **test** split.


> [!NOTE]
> **Conceptual link — Local Accuracy (Additivity → Fidelity)**
>
> TreeSHAP satisfies **Local Accuracy** (a consequence of Additivity/Efficiency) by design.  
> Empirically, we test its observable notion — **Fidelity** — by checking how well the additive reconstruction matches the model:
>
> $$
> f(x) \approx \phi_0 + \sum_i \phi_i(x)
> $$
>
> We quantify Fidelity on the **validation** split (and later on **test**) using error/agreement metrics such as **MAE** and **\(R^2\)** between \( f(x) \) and \( \phi_0 + \sum_i \phi_i(x) \).  
> Lower MAE and higher \(R^2\) indicate that the explanations **faithfully reproduce** the model’s behavior.
>

> [!NOTE]
> **Conceptual link — Consistency (→ Stability)**
>
> Shapley’s **Consistency/Symmetry** implies that explanations should vary **smoothly and coherently** under small input changes.  
> Empirically, we test its observable notion — **Stability** — by perturbing inputs with a small **ε** and comparing original vs. perturbed SHAP vectors:
>
> $$
> \text{Stability}(x;\,\varepsilon) \;=\; \cos\big(\,\boldsymbol{\phi}(x),\, \boldsymbol{\phi}(x+\delta)\,\big), 
> \quad \delta \sim \mathcal{N}(0,\varepsilon^2 I)
> $$
>
> We report the **mean (↑)** and **std (↓)** of cosine (or Spearman) similarity across instances.  
> The perturbation magnitude ε is configured in  
> `priors_clinical.yaml` → `tolerances.consistency_perturbation.epsilon`.  
> Higher similarity indicates **robust, consistent** explanations under realistic measurement noise.



### Step 5.2 — Save SHAP results for traceability

After computing the SHAP attributions, store both the baseline and the attribution matrix under `/results/shap_values/` for reproducibility.

```python
import pandas as pd
import json

results_dir = PROJECT / "results" / "shap_values"
results_dir.mkdir(parents=True, exist_ok=True)

# Save SHAP matrix (n_instances × n_features)
shap_df = pd.DataFrame(shap_matrix_val, columns=X_val.columns)
shap_df.to_csv(results_dir / "shap_matrix_val.csv", index=False)

# Save baseline expected value
with open(results_dir / "phi0.json", "w") as f:
    json.dump({"phi0": float(phi0)}, f, indent=2)

print("✅ SHAP values saved to:", results_dir)
```
Files generated:

results/shap_values/shap_matrix_val.csv
results/shap_values/phi0.json


> [!NOTE]
> Why save these files?
> Persisting SHAP outputs allows you to:
>
> Recompute trust metrics (Completeness, Fidelity) without re-running SHAP.
>
> Reuse the same attributions in stability or fairness analyses.
>
> Ensure the experiment is fully FAIR — Findable, Accessible, Interoperable, Reproducible.
>

### Step 5.3 — Upload SHAP results to GitHub

After running the SHAP explainer, two files are generated locally under:
01_Tabular_Regression_TreeSHAP_DiCE_Technical_SocialTrust/results/shap_values/

#### Files to include
shap_matrix_val.csv
phi0.json

#### Steps
1. In Colab, open the left sidebar → **results/shap_values/**.  
2. Download both files to your computer.  
3. In your local GitHub repository, move them to:
01_Tabular_Regression_TreeSHAP_DiCE_Technical_SocialTrust/results/shap_values/

Once uploaded, these SHAP outputs can be reused by later notebooks to compute trust metrics
(Completeness, Fidelity, and Stability) without recomputing explanations.

### Step 5.4 — Save and upload SHAP explanation sample

After combining the validation data with SHAP contributions,  
save a representative sample under `/results/` for traceability and future interpretability audits.
Files generated:

results/shap_explanations_sample.csv


This file shows, for each validation instance,
how every feature (φᵢ) contributes to the model’s final prediction relative to the baseline (φ₀ ≈ 153.02).

How to download from Colab

In Colab’s left sidebar, open the results/ folder.

Right-click the file shap_explanations_sample.csv → Download.

Upload to GitHub

Once downloaded, move the file into your local repository:

01_Tabular_Regression_TreeSHAP_DiCE_Technical_SocialTrust/results/shap_explanations_sample.csv

> [!TIP]
> **Result Analysis — SHAP Decomposition and Empirical Validation**
>
> **Core idea:**  
> SHAP values \( \phi_i(x) \) are **empirical observables** of the model’s response surface.  
> The decomposition
> $$
> f(x) = \phi_0 + \sum_i \phi_i(x)
> $$
> is not a visualization exercise but a **validation statement**:  
> it confirms whether the model behaves as a **stable, decomposable system** rather than an opaque artifact.
>
> **Computational meaning (TreeSHAP implementation)**  
> - \( \phi_0 \): model’s *expected prediction baseline* — the outcome when no specific feature shifts the result.  
>   In this pilot, **φ₀ ≈ 153.02**, aligned with the mean predicted progression score.  
> - \( \phi_i(x) \): *marginal contribution* of feature *i* across all possible feature coalitions.  
>   Positive values increase the prediction above φ₀; negative values decrease it.
>
> The resulting **SHAP matrix (66 × 10)** captures local attributions across all validation instances,  
> stored in `results/shap_explanations_sample.csv`.  
> Each φᵢ(x) quantifies the directional and quantitative influence of its feature.
>
> **Empirical observations (sample summary)**  
>
> | Variable | Mean φᵢ(x) | Interpretation |
> |-----------|-------------|----------------|
> | **s5** | +21.6 | Increases predicted progression. |
> | **bmi** | +13.5 | Strong positive contributor (higher BMI → higher risk). |
> | **bp** | +4.8 | Moderate positive contribution. |
> | **s3** | -8.5 | Reduces predicted progression. |
> | **age** | -2.9 | Slight negative influence. |
>
> These patterns align with clinical logic: higher **BMI** and **serum triglycerides (s5)** elevate risk,  
> while **age** and **s3** act as protective or inverse correlates.  
> This supports the model’s internal coherence and **plausibility of its reasoning**.
>
> **Integration with the XAI-TrustFramework**
>
> | Theoretical Property (Guarantee) | Empirical Trust Notion | Metric / Evidence | Meaning in this model |
> |----------------------------------|------------------------|------------------|-----------------------|
> | Additivity (Efficiency) | Completeness | % of instances where |f(x) − [φ₀ + Σφᵢ(x)]| ≤ τ | Validates additive reconstruction (τ = tolerance). |
> | Local Accuracy | Fidelity | MAE or R² between model output and reconstruction | Quantifies per-instance coherence. |
> | Consistency (Symmetry) | Stability | Cosine or Spearman similarity under ε-perturbations | Tests robustness of feature attributions. |

>
> **Relevance to the thesis methodology**  
> - Demonstrates how *normative fairness principles* (from Shapley’s theory) become **empirical trust evidence**.  
> - Establishes the **technical trust layer** of the framework: measurable coherence, stability, and completeness.  
> - Shows that *trustworthiness* is not abstract — it can be **quantified** through reproducible system behavior.


## 6) SHAP Metrics
### Step 6.1 — Compute Completeness (Additivity within tolerance)

Completeness measures the share of validation instances  
where the additive reconstruction \( \phi_0 + \sum_i \phi_i(x) \)  
matches the model’s prediction \( f(x) \) **within a tolerance threshold (τ)**.

The tolerance τ is defined in the configuration file:
configs/priors_clinical.yaml → tolerances.additivity_abs_tau

```python
import yaml

with open(PROJECT / "configs/priors_clinical.yaml") as f:
    PRIORS = yaml.safe_load(f)

tol = PRIORS["tolerances"]["additivity_abs_tau"]
ok = np.isclose(recon_val, pred_val, atol=tol)
completeness_ratio = float(ok.mean())

print(f"Completeness ratio (↑ better): {completeness_ratio:.3f} with tol={tol}")
```
Output example:

Completeness ratio (↑ better): 0.955 with tol=5.0

> [!NOTE]
> **Interpretation — Completeness as Additivity Validation**
>
> Completeness quantifies how often the SHAP additive identity  
> $$\\( f(x) \approx \phi_0 + \sum_i \phi_i(x) \\)$$  
> holds within an acceptable tolerance (τ).  
>
> A **high ratio** indicates that SHAP consistently preserves the additive reconstruction.  
> It operationalizes the **Additivity** property as measurable evidence of internal coherence.  
> τ acts as a **trust boundary**, defining the acceptable deviation between the explanation and the model output.  
>
> In the context of the **XAI-TrustFramework**, Completeness confirms that the technical decomposition  
> behaves predictably and remains stable within predefined tolerance ranges —  
> turning an abstract axiom into an **empirically testable trust guarantee**.

> [!TIP]
> **Result Analysis — Completeness (Additivity within τ = 5.0)**
>
> **Observed values:**
> | Metric | Value | Tolerance (τ) | Interpretation |
> |---------|--------|----------------|----------------|
> | **Completeness Ratio** | 1.000 | 5.0 | 100% of instances satisfy the additive reconstruction within tolerance. |
>
> **Interpretation:**
> - Every validation instance fulfills the identity  
>   \\( f(x) \approx \phi_0 + \sum_i \phi_i(x) \\)  
>   within the clinical tolerance defined in `priors_clinical.yaml`.  
> - No deviations exceed τ = 5.0, confirming that SHAP’s additive explanation  
>   is perfectly consistent with the model’s numerical outputs.  
> - This means that SHAP preserves the **Additivity property** not just theoretically,  
>   but *empirically* — the model’s logic and its explanation layer are fully aligned.
>
> **Methodological link:**
> - Demonstrates the **Additivity → Completeness** chain of validation  
>   within the *Technical Trust Dimension* of the XAI-TrustFramework.  
> - Confirms that the decomposition behaves predictably within predefined trust boundaries.  
> - Serves as a reproducible indicator that technical trust can be quantified  
>   through verifiable mathematical relations, not subjective interpretation.
>
> **Conclusion:**  
> Completeness = 1.0 (τ = 5.0) provides *strong empirical evidence*  
> that SHAP’s additive structure maintains total internal coherence.  
> In practice, this means the system’s explanations are **functionally faithful,  
> structurally stable, and technically trustworthy**.


### Step 6.2 — Compute Fidelity (Local Accuracy)

> [!NOTE]
> **Fidelity (Local Accuracy) as Empirical Trust Evidence**
>
> The analysis of φᵢ(x) contributions demonstrates **Local Accuracy**, one of the core properties of explainable models.  
> Local Accuracy ensures that each prediction \( f(x) \) can be reconstructed as the sum of all feature contributions plus the baseline:
>
> $$
> f(x) = \phi_0 + \sum_i \phi_i(x)
> $$
>
> When this relation holds across validation instances, the model’s explanations are not heuristic but **empirically verifiable**.  
> Fidelity therefore expresses *trust* as **quantitative alignment between model output and its additive reconstruction**.
>
> Within the *Technical Trust Dimension* of the XAI-TrustFramework:
> - **φ₀ (expected baseline)** anchors the additive space of predictions.  
> - **φᵢ(x)** represents measurable evidence of contribution for each feature.  
> - The **MAE or R²** between \( f(x) \) and \( \phi_0 + \sum_i \phi_i(x) \) quantifies how faithfully the decomposition holds.
>
> In this sense, Fidelity is not an abstract notion of interpretability,  
> but a **validation metric** confirming that the model’s internal logic behaves as a *stable, decomposable system* —  
> a prerequisite for technical trustworthiness.
>

The **Fidelity metric** quantifies how well the additive reconstruction  
\( \phi_0 + \sum_i \phi_i(x) \) matches the model’s actual predictions \( f(x) \).

A low MAE (Mean Absolute Error) and a high correlation indicate that  
the SHAP explanations are **faithful to the model’s behavior**.

```python
recon_val = shap_matrix_val.sum(axis=1) + phi0
pred_val  = rf.predict(X_val)

fidelity_mae = float(np.mean(np.abs(recon_val - pred_val)))
corr = float(np.corrcoef(recon_val, pred_val)[0, 1])

print(f"Fidelity MAE (↓ better): {fidelity_mae:.6f}")
print(f"Fidelity Corr (↑ better): {corr:.4f}")
```
#### Output example:

Fidelity MAE (↓ better): 0.006124
Fidelity Corr (↑ better): 0.9994


A high correlation indicates global alignment between the reconstructed and actual predictions.
Together, these metrics empirically verify the Local Accuracy property —
showing that SHAP explanations behave as a faithful, quantitative mirror of the model’s internal logic.

---

**Qué deja claro este bloque**
- Fidelidad ≠ interpretación; es **verificación cuantitativa** del principio de *Additivity/Local Accuracy*.  
- Cuanto menor sea el error (MAE) y mayor la correlación, más confiable es el modelo en su comportamiento explicativo.  
- Este paso convierte la teoría (axioma de eficiencia) en **evidencia empírica reproducible** de confianza técnica.

> [!TIP]
> **Result Analysis — Fidelity (Local Accuracy)**
>
> **Observed values:**
> | Metric | Value | Interpretation |
> |---------|--------|----------------|
> | **Fidelity MAE** | 0.2049 | Very low reconstruction error — the additive SHAP model reproduces predictions with high precision. |
> | **Fidelity Corr** | 0.99998 | Nearly perfect correlation between reconstructed and actual predictions. |
>
> **Interpretation:**
> - The additive identity \( f(x) \approx \phi_0 + \sum_i \phi_i(x) \) holds across validation instances.  
> - The SHAP decomposition behaves as a *stable and consistent mapping* of the model’s functional behavior.  
> - There is no evidence of drift or explanation noise between model output and its reconstructed form.
>
> **Methodological link:**
> - Confirms the **Local Accuracy** property — one of SHAP’s theoretical guarantees.  
> - Serves as **empirical evidence of Fidelity**, operationalizing *Technical Trust* through measurable alignment.  
> - Demonstrates that the explainability layer does not alter or distort model behavior,  
>   but rather mirrors it with quantitative consistency.
>
> **Conclusion:**  
> The model’s predictions and its additive explanations are numerically coherent.  
> SHAP can therefore be regarded as a *trust-preserving mechanism* within the Technical Trust Dimension  
> of the **XAI-TrustFramework**.

### Step 6.3 — Compute Stability (Consistency under ε-perturbation)

Stability measures how robust the SHAP explanations are  
when small perturbations (ε) are applied to the input features.  

A stable model should produce **similar explanation vectors**  
for original and slightly perturbed instances.

```python
with open(PROJECT / "configs/priors_clinical.yaml") as f:
    PRIORS = yaml.safe_load(f)
with open(PROJECT / "configs/seeds.json") as f:
    SEEDS = json.load(f)

SEED_NOISE = SEEDS.get("stability_noise_seed", SEEDS.get("numpy", SEEDS.get("global_seed", 42)))
rng = np.random.default_rng(SEED_NOISE)
eps = PRIORS["tolerances"]["consistency_perturbation"]["epsilon"]

X_val_pert = X_val.copy()
X_val_pert = X_val_pert + eps * rng.normal(size=X_val_pert.shape)

shap_vals_pert = shap_explainer.shap_values(X_val_pert)
shap_mat_pert = np.asarray(shap_vals_pert)

def cosine_sim(a, b):
    num = (a * b).sum(axis=1)
    den = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-12
    return num / den

cos_sims = cosine_sim(shap_matrix_val, shap_mat_pert)
stab_cosine_mean = float(np.mean(cos_sims))
stab_cosine_std  = float(np.std(cos_sims))

pd.DataFrame({"cosine_similarity": cos_sims}).to_csv("results/stability_analysis.csv", index=False)

print(f"Stability (cosine) mean±std: {stab_cosine_mean:.4f} ± {stab_cosine_std:.4f}")
```
Output example:

Stability (cosine) mean±std: 0.9812 ± 0.0074

> [!NOTE]
> **Interpretation — Stability as Consistency Validation**
>
> Stability quantifies how resistant the SHAP explanations are  
> to small random perturbations in the input data.  
>
> A **high mean cosine similarity (≈ 1.0)** means the explanations remain consistent.  
> A **low standard deviation (σ)** indicates uniform robustness across instances.  
> ε defines the perturbation magnitude — the smaller ε is, the more sensitive this test becomes.  
>
> This metric operationalizes the **Consistency → Stability** relation  
> under the *Technical Trust Dimension* of the **XAI-TrustFramework**,  
> ensuring that model explanations behave as *stable functions* of the input  
> rather than as *volatile or noise-sensitive artifacts*.

> [!TIP]
> **Result Analysis — Stability (Consistency under ε = configured value)**
>
> **Observed distribution:** cosine similarities between SHAP vectors  
> before and after applying ε-perturbations to input features.
>
> **Summary statistics (approx.):**
> - Mean cosine similarity ≈ **0.90–0.93**  
> - Standard deviation ≈ **0.10–0.12**  
> - Outliers: a few low (≤ 0.2) and one negative value (≈ –0.78)
>
> **Interpretation:**
> - Most instances show **high stability** (cos ≈ 0.95–0.99), confirming that explanations remain consistent.  
> - A small subset exhibits **moderate sensitivity**, indicating that minor input noise slightly alters local attributions.  
> - Negative or very low values likely correspond to instances near decision boundaries or numerical outliers.
>
> **Methodological link:**
> - Operationalizes the **Consistency → Stability** relation under the *Technical Trust Dimension*.  
> - Empirically tests whether SHAP explanations behave as *stable mappings* of model inputs.  
> - Confirms that trust cannot be assumed from theory but must be validated through perturbation-based testing.
>
> **Conclusion:**  
> The model demonstrates **overall robust and consistent explanatory behavior**,  
> with local sensitivity limited to a few edge cases.  
> This supports Stability as a measurable and reproducible trust property —  
> essential for verifying that the explainability layer behaves predictably  
> under realistic data variation (ε).

## 7) DICE
> [!NOTE]
> **Theoretical Foundations — Counterfactual Reasoning and Model Characterization**
>
> Counterfactual explanations do **not** attempt to “open” the model or inspect its internals.  
> Instead, they **characterize the model’s functional behavior** by observing how its outputs  
> respond to controlled and minimal input variations.
>
> In formal terms, a counterfactual instance \( x' \) is a nearby point in the input space such that  
> \( f(x') \) produces a desired outcome different from \( f(x) \).  
> By analyzing how the prediction changes between \( x \) and \( x' \),  
> we can describe *what transformations of the input space the model recognizes as meaningful*.
>
> From an engineering perspective, DiCE provides a **systematic perturbation framework** that:
> - Operates over the trained model as a *function* \( f: \mathbb{R}^n \to \mathbb{R} \),  
>   not over its architecture or parameters.  
> - Evaluates **sensitivity, controllability, and feasibility** of the model’s decision surface.  
> - Enables verification of *actionability* and *plausibility* under domain constraints,  
>   rather than introspection of internal weights.
>
> This aligns with the principle of **observable characterization**:  
> rather than “opening” the black box, we study its **external response to structured inputs**.  
> Counterfactual reasoning therefore becomes a *functional audit* —  
> mapping how real-world variables interact within the model’s learned representation.
>
> Within the *XAI-TrustFramework*, this technique supports the **Social Trust Dimension**,  
> translating technical model behavior into actionable, human-interpretable changes  
> that remain bounded by feasibility and domain knowledge.

> [!NOTE]
> **Conceptual Foundations — Counterfactual Constraints and Feasibility Logic**
>
> The `dice_constraints.yaml` file defines how counterfactual (CF) examples  
> are generated, validated, and bounded.  
> These constraints do not open the model but **govern the search space**  
> in which functional perturbations are tested — turning counterfactual reasoning  
> into a *controlled experiment* on the model’s behavior.
>
> ### Key sections:
>
> **1. Outcome definition**
> - Specifies the *target variable* and the *desired outcome condition*.  
>   In this pilot, `type: range` means CFs are generated so that  
>   \\( f(x') \\) falls within a tolerance band around the median of the validation target (±10 units).  
>   This approach treats CF generation as *bounded optimization*, not prediction reversal.
>
> **2. Search strategy**
> - `total_cfs_per_instance`: number of counterfactuals per instance (default = 3).  
> - `diversity_weight`: trade-off between **proximity** (minimal input change)  
>   and **diversity** (alternative plausible paths).  
> - `proximity_metric`: distance metric (L1 or L2) used to evaluate similarity between instances.  
> - `stop_when_k_found`: stops search when sufficient diverse CFs are found,  
>   optimizing computational efficiency.
>
> **3. Feasibility layer**
> - Defines which features are **immutable** (e.g., `sex`, `age`)  
>   and which are **actionable** (modifiable in realistic scenarios, e.g., `bmi`, `bp`).  
> - Bounds are derived from `priors_clinical.yaml` percentile ranges,  
>   ensuring that generated CFs remain within *data-informed limits*.  
> - Optional per-feature bounds or costs can override priors if domain knowledge requires.
>
> **4. Plausibility checks**
> - Post-generation filters enforce that CFs are *clinically and logically consistent*:  
>   - `enforce_monotonic_target_improvement`: CF must move prediction toward desired range.  
>   - `reject_if_outside_bounds`: CFs outside priors are invalidated.  
>   - `min_diversity_distance`: ensures counterfactuals represent distinct solution paths.
>
> **5. User profiles**
> - Defines constraints according to user context:  
>   - *Clinician profile*: can alter up to 5 features, maximizing interpretive power.  
>   - *Patient profile*: limited to 3 actionable features and smaller relative steps  
>     (≤ 0.5× the standard deviation) for realism and social feasibility.
>
> ---
>
> **Engineering interpretation**
>
> - The constraints transform DiCE from a heuristic generator into a **functional test bench**:  
>   each CF represents a controlled perturbation vector \\( \Delta x \\)  
>   satisfying feasibility and plausibility criteria.  
> - This approach characterizes how the model reacts to valid, bounded interventions,  
>   treating counterfactuals as **behavioral probes** — not as introspective tools.
>
> Within the *XAI-TrustFramework*, these rules operationalize the **Social Trust Dimension**,  
> ensuring that the model’s suggested changes remain **actionable, feasible, and explainable**  
> under realistic domain constraints.

> [!NOTE]  
> **Conceptual Note — Local Analysis as Functional Trust Probe**  
>
> The *XAI-TrustFramework* distinguishes between **global structural validation** and **local functional characterization**.  
> While global analyses (e.g., *TreeSHAP*) assess whether the model behaves as a *stable, decomposable system*,  
> local analyses (e.g., *DiCE*) observe how the system responds to controlled perturbations around a specific point in the input space.  
>
> In this pilot, selecting a **single patient (or instance)** does **not** aim at personal interpretation or clinical inference.  
> Instead, it provides a **functional probe** — a fixed, observable context from which to examine whether the model:  
>
> - Reacts smoothly to small, feasible input variations (*robustness*),  
> - Offers multiple realistic paths toward desirable outcomes (*diversity*), and  
> - Respects domain or ethical constraints during those transitions (*plausibility*).  
>
> This local behavior represents the **Social Dimension of Trust** within the framework —  
> translating abstract model sensitivity into measurable notions of *actionability*, *diversity*, and *plausibility*.  
> Through this lens, the individual instance serves as a **microcosm of model behavior**,  
> allowing trust to be empirically validated in a controlled, reproducible setting.

