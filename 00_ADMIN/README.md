# 00_ADMIN — Project Administration and Overview

This repository supports the research project *“XAI-TrustFramework”*, part of Dorleta Urrutia’s doctoral work on **trust evaluation in explainable AI systems**.  
It aims to organize, document, and reproduce all conceptual, methodological, and experimental materials used throughout the thesis and related publications.

The project seeks to:
- Operationalize **trust dimensions** (technical, cognitive, institutional, social) into **quantifiable metrics**.
- Bridge **theoretical guarantees** of XAI techniques (e.g., SHAP, DiCE) with their **empirical validation** under real-world conditions.
- Provide a **reproducible framework** and reference repository for evaluating the reliability, transparency, and actionability of AI explanations.

---

## Structure of the Repository

| Folder | Purpose |
|--------|----------|
| `00_ADMIN/` | Administrative files, project overview, publication plan, and bibliography. |
| `01_THEORY_FRAMEWORK/` | Conceptual base of the multidimensional trust framework, roles, dimensions, and definitions. |
| `02_CASE_DIABETES/` | Experimental notebooks and results for the diabetes regression case study. |
| `03_ROLES_AND_DIMENSIONS/` | Mapping between user roles and trust dimensions. |
| `04_PUBLICATIONS/` | Outlines and abstracts for the three main papers (methodological, experimental, applied). |
| `05_FIGURES/` | Visual materials (taxonomies, diagrams, graphical abstracts). |
| `06_APPENDICES/` | Supporting documents (AI Act alignment, references, checklists). |

---

## Research Question
> To what extent do explainable AI techniques preserve their theoretically guaranteed properties of trust (fidelity, stability, completeness) when applied to real predictive models, and how does their validity and usefulness vary across user roles and trust dimensions?

---

## Research Axes

| Axis | Focus | Outcome |
|------|--------|----------|
| **Methodological** | Develop the multidimensional trust framework and its evaluation pipeline. | Formal methodological paper. |
| **Experimental** | Validate the framework with the diabetes case study (SHAP & DiCE). | Experimental paper with quantitative results. |
| **Applied / Governance** | Extend the framework to regulatory and ethical contexts. | Governance-oriented paper aligning with EU AI Act. |

---

## Bibliography Management
All references are stored in:
- `references_master.bib` (root reference file for all documents and notebooks).   
- Zotero: *XAI-TrustFramework (Dorleta Urrutia)*.

---

## Version Control and Workflow
- **Branches:**
  - `main` → stable, published materials  
  - `paper-methodological`, `paper-experimental`, `paper-applied` → per publication stream
- **Naming conventions:**  
  - Use semantic numbering (`01_`, `02_`, …) for logical order.  
  - Use lowercase, underscores `_` for filenames.  

---

## Publication Plan (summary)
| Paper | Focus | Status |
|--------|--------|---------|
| Methodological | Operationalizing trust framework | Drafting |
| Experimental | Diabetes case study | Setup |
| Applied | Governance and auditability | Concept phase |

Full version: [`04_PUBLICATIONS/publication_plan.md`](../04_PUBLICATIONS/publication_plan.md)

