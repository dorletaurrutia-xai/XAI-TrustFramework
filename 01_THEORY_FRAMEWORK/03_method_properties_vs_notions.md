## Differentiation between Guaranteed Properties and Trust Notions

In this framework, XAI techniques are not validated by their ability to describe a predictive model correctly, but by the **trust** their explanations generate for different user profiles.
To evaluate this trust, it is necessary to distinguish between:

1. **Guaranteed theoretical properties** — derived from the mathematical formulation of each XAI technique.  
2. **Empirical trust notions** — defined in our multidimensional trust framework and verified through measurable metrics.

This distinction separates what the technique **assures by construction** from what must be **experimentally verified** under real conditions.

---

### Guaranteed Theoretical Properties  

These are formal conditions that an XAI technique satisfies **by definition**. They come from the original mathematical formulation and do not require empirical validation, although they can be checked numerically in practice.

#### Example 1: SHAP (Lundberg & Lee, 2017)
- **Local accuracy (additivity):**  
  The sum of feature contributions φᵢ exactly reproduces the prediction.  
  $$[
  \\hat{y}(x) = \\phi_0 + \\sum_i \\phi_i(x)
  \\]$$
- **Consistency:**  
  If the marginal effect of a variable increases, its contribution cannot decrease.  
- **Missingness:**  
  Features not used in the model receive zero contribution.

#### Example 2: DiCE (Mothilal et al., 2020)
- **Proximity:** Generates counterfactuals close to the original case.  
- **Feasibility:** Respects domain constraints (e.g., age cannot be negative).  
- **Diversity:** Produces multiple non-redundant counterfactuals.

These properties are **guaranteed by design**, but their empirical preservation depends on the data and implementation.

---

### Trust Notions (Empirically Evaluated)

**Trust notions** are observable attributes that verify to what extent explanations maintain their theoretical properties or how they behave in situations where theory alone is insufficient (e.g., correlation, noise, multi-user settings).

Each notion is linked to one or more **dimensions of trust** (technical, cognitive, institutional, social).

| Notion | What it Measures | Dimension | Technique | Related Property |
|---------|------------------|------------|------------|------------------|
| Fidelity | How well the explanation reconstructs the model's output | Technical | SHAP | Local accuracy |
| Completeness (additive residual) | Difference between prediction and sum of contributions | Technical | SHAP | Local accuracy |
| Stability / Robustness | Sensitivity of the explanation to small input perturbations | Technical / Institutional | SHAP | — |
| Faithfulness | Correlation between assigned importance and real effect on output | Technical | SHAP, LIME | Consistency |
| Monotonicity | Increase in predicted value with positive evidence | Technical | CEM, SHAP | Consistency |
| Actionability / Feasibility | Whether suggested variables are realistically modifiable | Social | DiCE | Feasibility |
| Comprehensibility / Readability | How understandable the generated explanation is | Cognitive | TS4NLE | — |

These notions are **operationalized through quantitative metrics**, allowing us to test whether the theoretical guarantees hold or degrade under realistic conditions.

---

### Metrics and Implementation

| Notion | Metric | Implementation |
|--------|---------|----------------|
| Fidelity | 1 − |ŷ − (b + Σφᵢ)| / range(ŷ) | Compare predicted vs reconstructed output |
| Completeness | mean(|ŷ − (b + Σφᵢ)|) | Mean additive residual |
| Stability | ρ(φ(x), φ(x+ε)) | Correlation between perturbed explanations |
| Monotonicity | Δŷ / Δφ⁺ > 0 | Sequential increase with positive variables |
| Faithfulness | Corr(importance, Δŷ) | Pearson correlation between relevance and output drop |
| Actionability | % of plausible counterfactuals | Ratio of CFs meeting domain constraints |
| Comprehensibility | #tokens / #variables / Flesch index | TS4NLE readability metrics |

---

### Integration into the Overall Framework

In the structure **[Data Type] → [Task Type] → [Base Model] → [XAI Techniques] → [Notions / Metrics] → [Formulas / Implementation]**, the distinction fits as follows:

| Block | Contents | Role in Validation |
|--------|-----------|-------------------|
| XAI Techniques | Theoretical guaranteed properties | Justify the technique selection and expected behavior |
| Notions / Metrics | Empirical trust properties | Evaluate if guarantees hold and how explanations behave in practice |
| Formulas / Implementation | Quantitative calculations | Reproduce and quantify the trust level achieved |

---

### Conclusion

> In summary, this framework distinguishes between **what an XAI technique guarantees mathematically** and **what it demonstrates empirically** under real conditions.  
> **Guaranteed properties** justify the technique choice, while **trust notions** evaluate its actual performance across user roles and trust dimensions.  
> This distinction ensures methodological traceability and consistency with European frameworks for *Trustworthy AI* (HLEG, 2019) and *Responsible AI Systems* (Díaz-Rodríguez et al., 2023).
