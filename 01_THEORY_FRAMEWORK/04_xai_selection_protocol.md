# XAI Technique Selection Protocol

This document defines the **protocol for selecting and validating explainable AI (XAI) techniques** within the *XAI-TrustFramework*.  
The objective is not to compare models in terms of accuracy, but to evaluate **the trustworthiness of explanations** produced by different XAI methods across diverse data and task types.

The framework assumes that trust can be **measured empirically** when the relationship between a model’s predictions and its explanations is operationalized through measurable notions and metrics.  
Accordingly, the selection of techniques is not arbitrary or convenience-based: it follows a **methodological logic** aligned with the project’s conceptual model.

> **[Data Type] → [Task Type] → [Base Model] → [XAI Technique] → [Trust Notions / Metrics] → [Formulas / Implementation]**

Within this structure, each technique is chosen according to:
- Its **mathematical properties and theoretical guarantees**;  
- Its **compatibility with the data and model**; and  
- Its **ability to operationalize specific trust dimensions** (technical, cognitive, institutional, and social).

---

## Rationale for Technique Selection

The selection of XAI techniques in this framework is guided by methodological—not empirical—criteria.  
The goal is to ensure that the *XAI-TrustFramework* includes representative methods that:

1. **Possess formal theoretical guarantees** — properties such as *local accuracy*, *consistency*, *additivity*, *proximity*, and *feasibility* that can be empirically tested as measurable trust notions.  
2. **Cover complementary explanation paradigms** — ensuring representation of both *attributive* (why something happened) and *counterfactual* (what could be changed) reasoning, as well as *symbolic* and *visual* interpretability.  
3. **Map to different trust dimensions** — so that each method contributes to one or more of the framework’s dimensions: technical, cognitive, institutional, and social.  
4. **Are reproducible, interpretable, and widely adopted** — techniques with stable, open-source implementations enabling transparent replication and comparison.

This rationale reflects the logic of alignment:

> **[Data Type] → [Task Type] → [Base Model] → [XAI Technique] → [Trust Notions / Metrics] → [Implementation]**

Following this logic, the selected set of techniques satisfies four strategic goals:

- **Theoretical coverage** — include methods with formally guaranteed properties (e.g., SHAP, DiCE).  
- **Methodological diversity** — represent distinct types of explanations (additive, counterfactual, rule-based, visual).  
- **Dimensional balance** — operationalize at least one trust dimension per technique.  
- **Cross-domain scalability** — ensure generalization across regression, classification, and image-based domains.

Together, these techniques provide a **complementary and traceable foundation** for testing the multidimensional model of trust in XAI systems.

---

## Overview of Selected Techniques

| Technique | Type | Model dependency | Scope | Trust dimension addressed |
|------------|------|------------------|--------|----------------------------|
| **SHAP (SHapley Additive exPlanations)** | Post-hoc, model-specific | Requires access to internal model structure (TreeSHAP) | Local and global | Technical (fidelity, completeness, stability) |
| **DiCE (Diverse Counterfactual Explanations)** | Post-hoc, model-agnostic | Uses only input/output interface | Local | Social (actionability, feasibility, diversity) |
| **Anchors (Ribeiro et al., 2018)** | Post-hoc, model-agnostic | Rule-based local approximation | Local | Cognitive (local fidelity, precision, coverage) |
| **Grad-CAM (Selvaraju et al., 2017)** | Post-hoc, model-specific | Requires gradient access to CNN layers | Local | Technical + Social (spatial fidelity, saliency consistency) |

These techniques were chosen because they allow for a **layered validation** of trust:
- **SHAP** and **DiCE** constitute the *core experimental validation* of technical and social trust.  
- **Anchors** and **Grad-CAM** demonstrate the *scalability and cross-domain applicability* of the framework across classification and visual domains.

---

## Justification for SHAP

### Theoretical grounding
SHAP (Lundberg & Lee, 2017) is grounded in **cooperative game theory** and provides three **formally guaranteed properties**:
1. **Local accuracy (additivity)** — the sum of feature contributions equals the model’s prediction.  
2. **Consistency** — if a feature’s contribution increases in a new model, its SHAP value cannot decrease.  
3. **Missingness** — features not used by the model receive zero contribution.

These guarantees make SHAP a **mathematically rigorous baseline** for evaluating trust-related properties such as fidelity and completeness.

### Methodological alignment
| Criterion | Justification |
|-----------|----------------|
| **Data compatibility** | Designed for structured tabular data; TreeSHAP optimized for ensembles (RandomForest, XGBoost). |
| **Task type** | Regression tasks with continuous outcomes. |
| **Explainability scope** | Provides both local (instance-level) and global (dataset-level) explanations. |
| **Metrics supported** | Fidelity, Completeness, Stability, Monotonicity. |
| **Trust dimension** | Anchors the *technical dimension* of trust. |
| **Empirical precedent** | Extensively validated in biomedical and financial prediction contexts. |

### Expected contribution
SHAP enables the **empirical validation of its theoretical guarantees**, linking *local accuracy* and *consistency* to measurable trust notions such as *fidelity* and *stability*.  
It thus acts as a **reference technique** for the *technical trust dimension* and provides a benchmark for subsequent evaluations.

---

## Justification for DiCE

### Conceptual complementarity
While SHAP explains *why* a model predicted a certain value, DiCE (Mothilal et al., 2020) explains *what would need to change* to obtain a different outcome.  
This introduces **actionability** and **contestability**—two notions central to human-centred and responsible AI.

### Theoretical grounding
DiCE provides three **formally guaranteed properties**:
1. **Proximity** — counterfactuals are generated close to the original instance.  
2. **Feasibility** — counterfactuals respect domain constraints and realistic feature bounds.  
3. **Diversity** — multiple non-redundant counterfactuals can be generated.

### Methodological alignment
| Criterion | Justification |
|-----------|----------------|
| **Model dependency** | Model-agnostic; requires only input/output access. |
| **Complementary perspective** | Captures “what-if” scenarios instead of additive decomposition. |
| **Trust dimension** | Operationalizes the *social dimension* (actionability, feasibility, diversity). |
| **Metrics supported** | Actionability, Plausibility, Diversity. |
| **Compatibility** | Can be applied to the same models as SHAP, enabling cross-validation. |

### Expected contribution
DiCE expands the framework from *interpretation* to *intervention*, allowing evaluation of how explanations support **human decision-making**.  
It validates whether system outputs are **meaningful, feasible, and fair** to act upon, anchoring the *social trust dimension*.

---

## Justification for Anchors

### Theoretical grounding
Anchors (Ribeiro et al., 2018) is a **rule-based explanation method** that approximates local decision boundaries with high-precision if–then rules.  
Each “anchor” defines a minimal set of feature conditions that, if satisfied, guarantee the same model prediction with high probability.

Its key property is **precision under locality**, ensuring that within the anchor’s region, predictions remain consistent.

### Methodological alignment
| Criterion | Justification |
|-----------|----------------|
| **Data compatibility** | Tabular data with categorical or structured variables. |
| **Task type** | Classification. |
| **Explainability scope** | Local and discrete; produces human-readable rules. |
| **Metrics supported** | Local Fidelity, Precision, Coverage. |
| **Trust dimension** | Cognitive — measures understandability and rule consistency. |
| **Empirical precedent** | Commonly used in fairness auditing and human evaluation studies. |

### Expected contribution
Anchors introduces the **cognitive dimension of trust** into the framework, providing interpretable, rule-based reasoning that can be audited or contested by human users.  
It complements SHAP and DiCE by adding a **symbolic interpretability** layer.

---

## Justification for Grad-CAM

### Theoretical grounding
Grad-CAM (Selvaraju et al., 2017) produces **class-specific saliency maps** by combining gradients and feature activations in convolutional neural networks.  
Although it lacks formal axioms like SHAP, it provides **spatial faithfulness guarantees** through gradient-based consistency.

### Methodological alignment
| Criterion | Justification |
|-----------|----------------|
| **Data compatibility** | Visual (image) data; compatible with CNN architectures such as ResNet. |
| **Task type** | Image classification. |
| **Explainability scope** | Local; produces heatmaps showing pixel importance. |
| **Metrics supported** | Spatial Fidelity, Saliency Consistency, Robustness. |
| **Trust dimension** | Technical + Social (alignment with human visual attention). |
| **Empirical precedent** | Extensively validated in medical imaging and industrial inspection. |

### Expected contribution
Grad-CAM extends the framework to **perceptual trust**, testing whether visual explanations correspond to meaningful regions for humans.  
It bridges *technical reliability* and *social relevance*, validating trust in visual domains.

---

## Methodological Integration Across Pilots

| Pilot | Domain | Model | XAI Technique | Key Notions | Main Metrics | Trust Dimension |
|--------|--------|--------|----------------|--------------|---------------|----------------|
| **A** | Tabular – Regression | RandomForestRegressor | TreeSHAP | Fidelity, Completeness, Stability | Fidelity score, Additive residual, Correlation ρ | Technical |
| **B** | Tabular – Classification | XGBoostClassifier | Anchors | Local Fidelity, Precision, Coverage | Precision, Coverage, Local fidelity error | Cognitive |
| **C** | Image – Classification | ResNet50 | Grad-CAM | Spatial Fidelity, Saliency Consistency | IoU, SSIM, Saliency correlation | Technical + Social |
| **(Supplementary)** | Tabular – Counterfactual | Any classifier/regressor | DiCE | Actionability, Diversity, Plausibility | Distance(x,x′), % feasible CFs, Diversity index | Social |

All pilots follow a unified workflow:  
> **Model training → XAI explanation → Trust metric computation → Validation across roles and dimensions.**

This ensures methodological traceability across domains and scalability of the trust framework.

---

## Why Other Techniques Are Not Prioritized (Yet)

Although the framework is extensible, other techniques were **not prioritized** in the initial validation phase due to methodological misalignment or limited guarantees.

| Technique | Reason for exclusion (current phase) |
|------------|--------------------------------------|
| **LIME** | Lacks mathematical consistency; explanations depend on random local sampling. |
| **RuleFit** | Produces discrete rules less suitable for continuous regression tasks. |
| **Integrated Gradients / DeepLIFT** | Require differentiable models; incompatible with RandomForest in this phase. |
| **CEM / ProtoDash / ConceptSHAP** | Tailored for concept-based or multimodal reasoning, outside the scope of the baseline pilots. |

These methods remain **potential candidates for future phases**, particularly for extending the framework to **textual or multimodal** data.

---

## Methodological Implications

The integrated selection of SHAP, DiCE, Anchors, and Grad-CAM enables the framework to capture **multiple paradigms of explainability** and map them systematically to the four dimensions of trust:

| Paradigm | Representative Technique | Primary Dimension | Secondary Dimension | Type of Reasoning |
|-----------|--------------------------|-------------------|--------------------|------------------|
| **Attributive** | SHAP | Technical | Institutional | Explains *why* a prediction was made (feature contribution). |
| **Counterfactual** | DiCE | Social | Technical | Explains *what could be changed* to alter the outcome. |
| **Rule-based / Symbolic** | Anchors | Cognitive | Institutional | Explains *under what conditions* predictions hold true. |
| **Visual / Perceptual** | Grad-CAM | Social | Technical | Explains *where* the model focuses attention in the input space. |

Together, these techniques allow the *XAI-TrustFramework* to:

- Cover **four complementary paradigms** of explanation (attributive, counterfactual, symbolic, and visual).  
- Operationalize **all major trust dimensions** (technical, cognitive, institutional, and social).  
- Provide **cross-domain validation** across tabular, structured, and image-based data.  
- Offer **controlled comparisons** between model-specific and model-agnostic approaches.  
- Support **reproducible, auditable trust metrics** across roles and scenarios.

This multi-technique configuration ensures that the framework measures both **internal consistency** (technical reliability) and **external usability** (human interpretability, actionability, and contestability).  
It also demonstrates the **scalability and modularity** of the trust framework, showing how new techniques can be integrated following the same alignment logic.

---

## Future Extensions

The selection protocol remains **modular and extensible**.  
Future iterations may incorporate:
- **LIME or RuleFit** for intuitive but low-fidelity benchmarks.  
- **Integrated Gradients** for deep-learning explainability in text and vision.  
- **Concept-based methods (TCAV, CEM, ConceptSHAP)** for evaluating semantic or conceptual trust.  
- **TS4NLE or similar systems** for human-centred evaluation of comprehensibility and narrative quality.

Each new technique will be assessed using the same logic of alignment:  
> *Does it enable measurement of trust notions within one or more framework dimensions under realistic conditions?*

---

## Summary Table

| Technique | Type | Level | Dimension | Key Notions | Guaranteed Properties | Metrics |
|------------|------|--------|------------|--------------|------------------------|----------|
| **SHAP (TreeSHAP)** | Post-hoc, model-specific | Local + Global | Technical | Fidelity, Completeness, Stability | Local accuracy, Consistency, Missingness | Additive residual, Correlation (ρ), Fidelity score |
| **DiCE** | Post-hoc, model-agnostic | Local | Social | Actionability, Diversity, Feasibility | Proximity, Feasibility, Diversity | Distance(x,x′), % plausible CFs, Diversity index |
| **Anchors** | Post-hoc, model-agnostic | Local | Cognitive | Local Fidelity, Precision, Coverage | High-precision anchors | Precision, Coverage, Local fidelity error |
| **Grad-CAM** | Post-hoc, model-specific | Local | Technical + Social | Spatial Fidelity, Saliency Consistency | Gradient–activation consistency | IoU, SSIM, Saliency correlation |

---

## Expected Contributions

- A **methodologically grounded selection** of XAI techniques based on their formal guarantees and trust relevance.  
- Demonstration of **dual and cross-domain validation**: additive (SHAP), counterfactual (DiCE), rule-based (Anchors), and visual (Grad-CAM).  
- Empirical evidence on **how theoretical properties hold under real-world conditions**.  
- A validated set of **metrics for operationalizing trust** across data modalities.

---

*Last updated: October 2025*
