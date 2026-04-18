# AI in Banking Risk Model

## 📌 Overview

This repository contains the **capstone project** for the **HKU SPACE** certificate course *"Artificial Intelligence and Machine Learning with Business & Financial Applications"* (awarded with **Distinction**).

**🎯 Project Focus:** A comparative analysis of **traditional statistical models** (Logistic Regression, Decision Tree) versus **advanced machine learning algorithms** (Random Forest, XGBoost) for **Probability of Default (PD) prediction** in retail credit risk.

The project demonstrates a complete **end-to-end machine learning pipeline** developed in **Python**, with methodology contextualized within the **HKMA Basel IRB framework** and a strong emphasis on **model explainability (SHAP)** .

## 📂 Repository Contents

| File | Description |
| :--- | :--- |
| `credit_risk_modelling.ipynb` | Jupyter Notebook with full Python implementation (EDA, preprocessing, modelling, SHAP). |
| `AI_Basel_HKMA_Presentation.pdf` | Main project deck covering Gen AI & HKMA compliance research. |
| `Appendix.pdf` | Technical supplement with detailed EDA visualizations and preprocessing steps. |
| `README.md` | This file. |

## 🛠️ Technologies & Libraries

| Area | Tools / Libraries |
| :--- | :--- |
| **Language** | Python 3 |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Traditional Models** | Logistic Regression, Decision Tree (pruned) |
| **Advanced Models** | Random Forest, **XGBoost** (with `GridSearchCV`) |
| **Model Explainability** | SHAP |
| **Statistical Analysis** | Statsmodels (VIF) |
| **Environment** | Google Colab |

## 📊 Project Workflow Summary

### 1. Exploratory Data Analysis (EDA)
- **Dataset:** Public Kaggle Credit Risk Dataset (~32k records).
- **Key Observations (see `Appendix.pdf` for visuals):**
  - Majority of applicants are young adults (median age ~26).
  - Income distribution is right-skewed with notable outliers.
  - ~17% of applicants have a prior default on file.
  - Renters constitute the largest home ownership category.

### 2. Data Cleansing & Preprocessing
- **Duplicate Removal:** 165 duplicate records excluded.
- **Missing Value Treatment:** `person_emp_length` and `loan_int_rate` missing values imputed with **mean** to preserve data volume.
- **Outlier Handling:** Values outside **mean ± 3σ** were removed (captures 99.7% of data, avoiding excessive data loss compared to IQR method).
- **Feature Engineering:** Created `Credit_Age_Ratio` and `Employment_Stability` to capture borrower behavior intuitively.

### 3. Feature Transformation
- **One-Hot Encoding:** Applied to `person_home_ownership`, `loan_intent`, `cb_person_default_on_file`.
- **Standardization:** Z-score normalization applied to all numeric features to ensure equal contribution during model training.

### 4. Model Development & Comparative Evaluation
| Model Category | Model | Key Preprocessing / Tuning |
| :--- | :--- | :--- |
| **Traditional** | Logistic Regression | Correlation screening + VIF ≤ 5 to address multicollinearity. |
| **Traditional** | Decision Tree | Cost-complexity pruning + max depth & leaf size constraints. |
| **Advanced** | Random Forest | Iterative feature elimination based on importance scores. |
| **Advanced** | **XGBoost** | `GridSearchCV` (5-fold) tuning `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`. |

### 5. Model Testing
- **Metric:** ROC-AUC on a held-out 30% test set.
- **Results:**

| Model | ROC-AUC (Test) |
| :--- | :--- |
| Logistic Regression | ~0.84 |
| Decision Tree (Pruned) | ~0.84 |
| Random Forest | ~0.90 |
| **XGBoost (Tuned)** | **~0.93** |

> **Key Insight:** XGBoost with highest AUC score due to its boosted ensemble approach, offers superior performance, demonstrating the tangible value of machine learning in credit risk assessment.

### 6. Model Explainability (SHAP)
- The tuned **XGBoost model** was selected for SHAP analysis due to its top-tier AUC.
- One-hot encoded SHAP values were **aggregated** to provide business-intuitive feature importance.
- **Top predictive features:** `loan_int_rate`, `loan_percent_income`, `person_income`, `person_home_ownership`, `loan_intent`.
- Detailed SHAP interpretations are available in the notebook and `Appendix.pdf`.

## 📄 Supplementary Research (Main Presentation Only)

The `AI in Banking and Credit Risk Modelling.pdf` file also includes strategic overviews of:
- **Federated Learning** for privacy-preserving cross-institutional modelling.
- **Generative AI governance** and HKMA's 2024 Ethical AI Framework.

*Note: These sections represent industry research and are not part of the hands-on Python implementation.*

## ⚠️ Important Notes

- **Data Source:** Publicly available, **synthetic** Kaggle dataset. **No real customer or proprietary bank data** is contained herein.
- **Academic Context:** This project demonstrates **technical proficiency and analytical thinking**; it is not intended as a production-ready credit decisioning system.

## 📎 References

- [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
