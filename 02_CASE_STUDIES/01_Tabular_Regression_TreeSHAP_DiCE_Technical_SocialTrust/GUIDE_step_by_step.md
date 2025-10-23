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
> **Conceptual Note — Why TreeSHAP here? (Data → Task → Method → Trust Link)**
>
> **Data Type**  
> The dataset used (`sklearn.diabetes`) is *tabular*, with continuous numeric variables that are moderately correlated (e.g., BMI, blood pressure, glucose indicators).  
> Tabular data require **local additive explanations** that can attribute prediction outcomes to individual input features in a consistent, quantitative way.
>
> **Task Type**  
> The task is **regression** — predicting a continuous clinical progression score.  
> For regression models, we need explanations that are:
> - **additive** (sum of contributions equals prediction),
> - **signed** (each feature increases or decreases the prediction),
> - **quantitatively consistent** across similar inputs.
>
> **XAI Method: TreeSHAP**  
> TreeSHAP (a variant of SHAP specialized for tree-based models) provides these guarantees **by design**.  
> It decomposes each prediction \( f(x) \) into:
>
> $$
> f(x) = \phi_0 + \sum_i \phi_i(x)
> $$
>
> where:
> - \( \phi_0 \) is the model’s expected output (baseline prediction),
> - \( \phi_i(x) \) is the contribution of each feature \( i \).
>
> **Method-Guaranteed Properties (operationalized in this pilot)**  
> TreeSHAP satisfies three theoretical guarantees — *Additivity*, *Consistency*, and *Missingness* —  
> but in this pilot, only the first two are **empirically validated** through Technical Trust metrics:
>
> | **Method-Guaranteed Property** | **Trust Notion (empirical)** | **Trust Metric** | **Parameter** |
>|--------------------------------|-------------------------------|------------------|----------------|
>| **Additivity** | **Completeness** | % of instances where `| f(x) − (φ₀ + Σφᵢ(x)) | ≤ τ` | τ (`additivity_abs_tau`) |
>| **Local Accuracy (Additivity)** | **Fidelity** | `R²` or `MAE` between `f(x)` and reconstruction | — |
>| **Consistency** | **Stability** | Similarity of SHAP vectors under ε-perturbations | ε (`consistency_perturbation.epsilon`) |
>
> **Missingness** (features with no influence have φ=0) is acknowledged as a theoretical property of TreeSHAP  
> but is **not empirically tested** in this pilot — it remains a qualitative check for feature sparsity.
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

#### Why interventional?
It approximates do()-style interventions on individual features, which helps reduce artefacts from feature correlation compared to purely independent perturbations—particularly important with tabular data.

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

> [!NOTE]
> **Conceptual Note — SHAP as Controlled Behavioral Characterization**
>
> In this framework, SHAP (SHapley Additive Explanations) is not used as an interpretability tool but as a **controlled behavioral characterization method**.  
> Rather than “opening” the model or inspecting internal parameters, it **models the system’s response surface** under a set of fairness-preserving constraints.
>
> Each feature is treated as an **input variable in a cooperative system**, whose marginal influence on the output can be measured empirically.  
> Through this formalism, SHAP provides a way to **observe and quantify the functional structure** of the model without altering its internal mechanisms.
>
> The method is grounded in *Shapley’s efficiency theorem* (1953), which ensures that the decomposition of the prediction into additive components is **unique, stable, and invariant** under symmetrical transformations.  
> This transforms explainability into a form of **system-level accountability** — a measurable mapping between inputs and outputs that can be validated empirically.
>
> Within the *XAI-TrustFramework*, SHAP therefore acts as a **non-intrusive diagnostic instrument**, verifying whether the model’s predictions are:
> - **Additive** — decomposable into consistent feature contributions (→ Completeness);  
> - **Accurate locally** — reconstructable per instance (→ Fidelity);  
> - **Consistent** — stable under perturbations (→ Stability).
>
> In this sense, SHAP does not *explain* the model; it **tests its structural reliability**.  
> The focus shifts from interpretation to **empirical validation of trust properties** — aligning with the thesis premise that trust in AI systems emerges from verifiable behavioral evidence rather than introspection.
>

> [!TIP]
> **Result Analysis — SHAP baseline validation (φ₀, φᵢ(x), and their epistemic meaning)**
>
> **Theoretical foundation — from cooperative system decomposition to explainability**
>
> This analysis does **not** attempt to “open” or introspect the internal mechanics of the model.  
> Instead, it provides a **functional characterization** of the model’s observable behavior through *input–output contribution analysis*.
>
> SHAP originates from **cooperative game theory** but can be interpreted here as a **structured system-decomposition method**.  
> Each feature acts as an *input channel* whose marginal influence on the output can be quantified under a controlled framework of *coalitional perturbations*.
>
> The mathematical core is **Shapley’s efficiency theorem** (1953), which guarantees a *unique solution* for distributing a system’s output among its inputs, subject to three axioms:
>
>| **Shapley Axiom / Property** | **Engineering Interpretation** | **Trust Notion (empirical)** | **Metric / Formula** | **What it means in your model** | **File** | **Status in this pilot** |
>|-------------------------------|--------------------------------|------------------------------|----------------------|----------------------------------|-----------|---------------------------|
>| **Efficiency / Additivity** | The total output equals the sum of individual input contributions. | **Completeness** | % of instances where&nbsp;\\( \| f(x) - [\phi_0 + \sum_i \phi_i(x)] \| \leq \tau \\) >| Verifies that predictions can be **reconstructed** as the sum of all SHAP contributions plus the baseline. | — | ✅ Operationalized *(Additivity → Completeness, τ)* |
>| **Local Accuracy** | Each instance’s prediction equals the sum of its feature contributions. | **Fidelity** | \\( R^2 \\)&nbsp;or&nbsp;MAE&nbsp;between&nbsp;\\( f(x) \\)&nbsp;and&nbsp;\\( \phi_0 + \sum_i \phi_i(x) \\) | Evaluates how faithfully the additive reconstruction matches the model’s actual output. Ensures per-instance coherence. | `metrics_summary_val.csv` | ✅ Operationalized |
>| **Symmetry / Consistency** | Inputs with equivalent impact receive equal attributions. | **Stability** | Similarity of SHAP vectors under&nbsp;ε-perturbations&nbsp;(cosine,&nbsp;Spearman) | Tests whether small input perturbations preserve the relative ranking and direction of feature contributions. | `stability_analysis.csv` | ✅ Operationalized *(Consistency → Stability, ε)* |
>| **Dummy / Missingness** | Inputs that have no influence receive zero attribution. | — *(not quantified)* | — | Ensures that irrelevant or constant features have&nbsp;\\( \phi = 0 \\). This can be checked analytically but not empirically in this pilot. | — | ⚪ Not included in this pilot |
>
>
> These axioms make SHAP a **model-agnostic decomposition operator**:  
> it defines how to *represent and measure* contribution dynamics without accessing internal parameters.  
> What results is not a “peek inside the box,” but a **measurable mapping between features and outputs**, preserving the system’s functional integrity.
>
>> [!NOTE]
>> **How SHAP’s axioms map to empirical trust validation in the XAI-TrustFramework**
>>
>> The table above shows how the formal axioms of the Shapley value —originating in cooperative game theory—  
>> are **operationalized as empirical trust notions** within the *XAI-TrustFramework*.
>>
>> Each theoretical property defines a *guaranteed behavior* of the explanation model, while its corresponding trust notion  
>> expresses that property as a **measurable hypothesis** about the system’s observable outputs.
>>
>> - **Additivity (Efficiency)** → becomes **Completeness**, verifying whether each prediction can be reconstructed as  
> >  the sum of individual feature contributions within a defined tolerance (τ).  
>> - **Local Accuracy** → becomes **Fidelity**, assessing how closely the additive reconstruction matches the model’s actual prediction.  
>> - **Consistency (Symmetry)** → becomes **Stability**, quantifying the robustness of explanations under ε-perturbations.  
>> - **Dummy (Missingness)** → remains theoretically present, ensuring that non-influential features receive φ = 0,  
>>   though it is not empirically tested in this pilot.
>>
>
> This alignment converts **normative axioms into testable criteria**.  
> What was once a theoretical guarantee of fairness and coherence becomes a **set of trust metrics** that can be validated empirically.  
> The resulting structure allows each trust dimension —technical, social, organizational, and institutional—  
> to be grounded in reproducible evidence rather than abstract claims.
>
> In other words, the *XAI-TrustFramework* extends Shapley’s fairness principles from cooperative game theory  
> into a **methodology for assessing trustworthy behavior in AI systems**.
>
> **Key point:**  
> - SHAP values \( \phi_i(x) \) are *empirical observables* of the model’s response surface.  
> - The decomposition \( f(x) = \phi_0 + Σφ_i(x) \) is **a validation statement**, not a visual explanation.  
> - Additivity and consistency allow us to verify that the model behaves as a **stable, decomposable system**, rather than as an opaque artifact.
>
> In this sense, SHAP operationalizes the **principle of structural accountability**:  
> it converts the black-box system into a *traceable function space*, where every prediction can be validated by reconstruction.
>
>
> **Computational formalization — TreeSHAP and its expected value (φ₀)**  
> TreeSHAP implements this logic efficiently for tree-based models like `RandomForestRegressor`.  
> It computes, for each instance:
> $$
> f(x) = \phi_0 + \sum_i \phi_i(x)
> $$
> where:
> - \( \phi_0 \) is the **expected prediction** (baseline when no feature contributes).  
> - \( \phi_i(x) \) is the **marginal contribution** of feature *i* across all possible coalitions.
>
> The resulting **SHAP matrix (66 × 10)** captures these local attributions across all validation samples.  
> The computed baseline **φ₀ ≈ 153.02** represents the model’s expected output, aligning with the average clinical progression score predicted by the model over the training distribution.
>
> **Methodological integration — empirical trust in the technical dimension**  
> In the *XAI-TrustFramework*, this step operationalizes the **Technical Dimension of Trust**, linking formal guarantees with measurable evidence:
>
>| **Method-Guaranteed Property** (theoretical) | **Trust Notion** (empirical) | **Metric / Formula** | **What it means in your model** | **File** |
>|----------------------------------------------|-------------------------------|----------------------|----------------------------------|----------|
>| **Additivity (Efficiency)** | **Completeness** | % of instances where&nbsp;\\( \| f(x) - [\phi_0 + \sum_i \phi_i(x)] \| \leq \tau \\) | Verifies that each prediction can be **reconstructed** as the sum of all SHAP contributions plus the baseline within a tolerance&nbsp;τ. | — |
>| **Local Accuracy** | **Fidelity** | \\( R^2 \\)&nbsp;or&nbsp;MAE&nbsp;between&nbsp;\\( f(x) \\)&nbsp;and&nbsp;\\( \phi_0 + \sum_i \phi_i(x) \\) | Tests how **faithfully the additive reconstruction matches** the model’s actual prediction. Measures per-instance empirical coherence. | `metrics_summary_val.csv` |
>| **Consistency (Symmetry)** | **Stability** | Similarity of SHAP vectors under&nbsp;ε-perturbations&nbsp;(cosine,&nbsp;Spearman) | Checks whether **small input perturbations** produce **proportionally stable explanations**, ensuring invariance and reliability of attributions. | `stability_analysis.csv` |
>
>
>
> By generating SHAP values, we empirically access the *internal logic* of the model, allowing the theoretical properties of fairness and consistency to be **quantified** as reproducible data.  
> This makes SHAP a **bridge between cooperative fairness theory and technical trust measurement**.
>
> **Interpretation of results (φ₀ ≈ 153.02)**  
> - φ₀ is the model’s **expected prediction baseline** — the outcome when no specific feature shifts the result.  
> - Each φᵢ(x) describes **how much feature *i* pushes this instance’s prediction** above or below φ₀.  
> - In clinical terms, positive contributions (e.g., BMI, s5, bp) indicate higher predicted progression; negative ones (e.g., s3, age) lower progression.
>
> **Relevance to the thesis methodology**  
> - This analysis demonstrates how a **normative fairness principle** (from cooperative game theory) can be **empirically verified** in applied AI.  
> - It shows that *trustworthiness* is not abstract — it is **operationalized** by testing whether the model obeys fairness axioms in practice.  
> - It establishes the **first empirical layer** of the thesis methodology: from *theoretical guarantees* → *operational notions* → *measurable metrics*.

### Step 6.2 — Save and upload SHAP explanation sample

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



> **Next step:**  
> We will now compute the **Completeness (τ)** metric to quantify Additivity empirically, verifying how closely  
> \( f(x) \) matches \( φ₀ + Σφᵢ(x) \) within the tolerance set in `priors_clinical.yaml`.  
> This is the first validation of a *method-guaranteed property* as a **trust metric**.

