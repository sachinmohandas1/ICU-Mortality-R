# ICU Mortality Prediction — R Implementation

### A Feature Ensemble Approach to Patient Mortality Classification in the ICU

---

## Overview

This project investigates whether ICU patient mortality can be predicted using a subset of the MIMIC-III clinical database. Implemented entirely in **R**, the goal is to identify small ensembles of clinical features that rival or outperform a full-feature logistic regression classifier in accuracy.

The project was developed for MET CS555 – Foundations of Machine Learning at Boston University. A separate implementation was conducted in **Python** to compare language toolchains and modeling performance.

---

## Dataset

- **Source:** MIMIC-III ICU dataset (via Kaggle)  
- **Observations:** 1,177 patients entries  
- **Features:** 49 columns, including demographic, biochemical, and clinical measures  
- **Target Variable:** `outcome` → `1` = deceased, `0` = survived  
- **Preprocessing:**  
  - NAs imputed using column means  
  - ID and grouping columns removed  
  - Focus on 37 non-categorical variables for feature ensemble analysis  

---

## Methods

All modeling was done using **logistic regression** with a 50/50 train-test split. The workflow is twofold:

1. **Base Model Evaluation**
   - Logistic regression trained on all available predictors  
   - Achieved **85.7% accuracy**  

2. **Feature Ensemble Search**
   - Analyzed all 7,770 unique 3-feature combinations
   - Measured predictive accuracy for each mini-model  
   - Identified top-performing combinations and conducted statistical comparisons

---

## Results

| Model | Accuracy |
|-------|----------|
| Full Feature Logistic Regression | 85.7% |
| Top 3-Feature Ensemble | 87.8% |

**Best 3-feature ensemble:**  
`Respiratory rate`, `Bicarbonate`, `Lactic acid` → **87.8% accuracy**

**Additional insights:**
- 7,049 of the 7,770 ensembles (over 90%) outperformed the full model  
- Features most frequently appearing in top ensembles:
  - `Lactic acid`
  - `Anion gap`
  - `Respiratory rate`
  - `Blood potassium` (not as frequent, but high coefficient magnitudes: 0.58–1.2)

---

## Exploratory Data Analysis

- **Pairplots** were generated for the top 50 ensembles to visualize class separation.
- **Collinearity analysis** showed:
  - Mean collinearity for all ensembles: 0.0052
  - Mean for top 50 ensembles: 0.0312
  - t-test p-value: 0.455 → no statistically significant difference
- While collinearity wasn’t a driving factor, certain features showed strong influence on outcome prediction.

---

## Interpretation

- Smaller feature ensembles can outperform the full model without sacrificing much accuracy.  
- This suggests robust predictors exist among a limited subset of features - valuable when only partial patient data is available.  
- However, clinical interpretability and real-world uncertainty limit deployment without further refinement.

---

## Limitations

- Accuracy of ~88% may still be insufficient for ICU triage decisions.
- Clinical application requires:
  - Interpretability
  - External validation
  - Robustness across patient demographics
- Exploratory findings were not conclusive and require further medical contextualization.

---

## Technologies Used

- **Language:** R  
- **Libraries:** `caret`, `pROC`, `ROCR`, `GGally`, `itertools`, `ggplot2`  
- **EDA:** Custom collinearity calculations, pairplots  
- **Modeling:** Logistic regression using `caret` package

---

## Author

**Sachin Mohandas**  
Boston University – MET CS555: Foundations of Machine Learning  
Term Project (May 2024)

---

## Related Work

- **Python Implementation:** [ICU-Mortality-Python](https://github.com/sachinmohandas1/ICU-Mortality-Python)
