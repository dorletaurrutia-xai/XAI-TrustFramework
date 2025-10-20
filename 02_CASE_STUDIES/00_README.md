
# CASE_STUDIES — Empirical Validation of the XAI-TrustFramework

This directory contains three simulated case studies designed to **validate the methodological feasibility** of the *XAI-TrustFramework* through open, reproducible datasets.  
Each case represents a **distinct paradigm of explainability** and maps to **different dimensions of trust** — *technical*, *cognitive*, and *social*.

Together, they demonstrate that **trust in XAI systems can be operationalized and empirically measured** across data types, model architectures, and user perspectives.

---

## Purpose

Before applying the framework to real cooperative or institutional settings, this simulation stage aims to verify that the entire **experimental pipeline** — from model training to trust metric computation — is **technically viable** within an open, transparent, and replicable environment.

These pilots therefore act as **proof-of-concept experiments** showing that:

- The *XAI-TrustFramework* logic  
  `→ [Data Type] → [Task Type] → [Base Model] → [XAI Technique] → [Trust Notions / Metrics] → [Implementation]`  
  can be implemented end-to-end using standard Python tools.  
- Trust notions such as *fidelity*, *completeness*, *stability*, and *actionability* can be **measured quantitatively**.  
- The same methodology scales from **tabular regression** to **image classification** tasks.

---

## Structure

CASE_STUDIES/
├── 01_Tabular_Regression_TreeSHAP_TechnicalTrust/
│ └── Pilot A – Clinical regression using TreeSHAP
│ (fidelity, completeness, stability)
│
├── 02_Tabular_Classification_Anchors_CognitiveTrust/
│ └── Pilot B – Clinical classification using Anchors
│ (precision, coverage, local fidelity)
│
└── 03_Image_Classification_GradCAM_Technical_SocialTrust/
└── Pilot C – Biomedical imaging using Grad-CAM
(spatial fidelity, saliency consistency)


Each pilot includes:
- **`README.md`** explaining scope and rationale  
- **`data/`** folder with open datasets or download links  
- **`notebooks/`** with reproducible Colab notebooks  
- **`results/`** for metrics and visualization outputs  
- **`configs/`** for seeds, constraints, and priors  

---

## Experimental Overview

| Pilot | Domain | Task Type | Base Model | XAI Technique | Trust Dimensions | Core Metrics | Open Dataset |
|--------|----------|------------|--------------|----------------|------------------|---------------|----------------|
| **A** | Clinical (tabular) | Regression | RandomForestRegressor | **TreeSHAP** | Technical | Fidelity, Completeness, Stability | `sklearn.diabetes` |
| **B** | Clinical (tabular) | Classification | XGBoostClassifier | **Anchors** | Cognitive | Precision, Coverage, Local Fidelity | UCI Heart Disease |
| **C** | Biomedical imaging | Classification | ResNet50 | **Grad-CAM** | Technical + Social | IoU, SSIM, Saliency Correlation | COVID-19 Radiography (Kaggle) |

---

## Conceptual Alignment

Each case operationalizes one or more **trust dimensions** of the *XAI-TrustFramework*:

| Dimension | Description | Example in Pilot |
|------------|--------------|------------------|
| **Technical** | Internal reliability and stability of explanations. | Fidelity and completeness of TreeSHAP (Pilot A) |
| **Cognitive** | Human-centered interpretability and local consistency. | Rule precision and coverage in Anchors (Pilot B) |
| **Social** | Human empowerment, fairness, and actionability. | Grad-CAM saliency alignment with human attention (Pilot C) |

---

## Methodological Goals

1. **Reproducibility:**  
   Use open datasets and fixed seeds to ensure transparency.  
2. **Comparability:**  
   Apply a shared logic of evaluation and reporting across all pilots.  
3. **Scalability:**  
   Demonstrate that the same trust metrics can be extended across data modalities.  
4. **Traceability:**  
   Generate structured outputs (CSV + visuals) to feed future governance and audit studies.  

---

## Expected Outputs

- Standardized CSVs with computed trust metrics.  
- Visual summaries (e.g., SHAP beeswarm, Anchors rule coverage, Grad-CAM heatmaps).  
- Documented notebooks that trace each step of the methodology.  
- A meta-summary comparing the empirical behavior of each XAI method across trust dimensions.

---

## Next Step

Once the simulations confirm methodological viability, the framework will be extended to **real contexts**, linking empirical trust metrics to governance, transparency, and accountability requirements under EU AI regulation.

