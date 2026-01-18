# 💰 Loan Default Prediction

**Advanced Machine Learning Classification with Contextual Loss Evaluation**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Overview

A production-grade binary classification system for predicting loan defaults using ensemble machine learning techniques and business-driven contextual loss functions. This project balances **predictive accuracy with real-world financial implications**, optimizing for both model performance and cost-sensitive decision making.

**Key Achievement:** Implemented custom contextual loss function (10×FN + 1×FP) to minimize financial risk while maintaining high classification accuracy across multiple ML models.

---

## 🎯 Business Problem

Financial institutions face the challenge of:
- **Approving risky loans** → Direct financial losses from defaults
- **Rejecting good loans** → Lost interest revenue and opportunity costs

This system provides data-driven loan approval decisions by:
✅ Identifying high-risk applicants with **high sensitivity**  
✅ Minimizing false rejections through **threshold optimization**  
✅ Providing **interpretable predictions** using SHAP analysis  

---

## 📊 Project Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Best Model** | Gradient Boosting (Log Loss) |
| **Test AUC** | 0.9574 |
| **Optimal Threshold** | 0.247 |
| **Sensitivity (Recall)** | 0.8543 |
| **Specificity** | 0.9012 |
| **F1-Score** | 0.7234 |

### Contextual Loss Optimization

- **Loss Function:** 10×FN + 1×FP
- **Financial Rationale:** 
  - FN (False Negative): Approved loan that defaults → **10× cost** (direct loss)
  - FP (False Positive): Rejected loan that would repay → **1× cost** (lost revenue)
- **Optimal Loss Achieved:** Minimized through threshold optimization on validation set

---

## 🔬 Technical Architecture

### Dataset Characteristics
- **Samples:** 255,347 loan applications
- **Features:** 18 (9 numerical, 9 categorical)
- **Target:** Binary classification (Default / No Default)
- **Class Imbalance:** Handled through stratified sampling and contextual loss
- **Data Split:** 60% Train / 20% Validation / 20% Test

### Models Implemented

#### Glass-box Models (Interpretable)
1. **Logistic Regression (L2 Loss)** - Ridge regularization approach
2. **Logistic Regression (Binomial Deviance)** - Standard maximum likelihood

#### Black-box Models (High Performance)
3. **Gradient Boosting (L2 Loss)** - Least squares objective
4. **Gradient Boosting (Log Loss)** - Binomial deviance ✅ **Best Model**
5. **Random Forest Classifier** - Ensemble bagging approach

### Model Selection Criteria
1. **Primary:** Validation AUC (discriminatory power)
2. **Secondary:** Contextual loss minimization
3. **Tertiary:** Test set generalization

---

## 🔍 Model Interpretability

### SHAP (SHapley Additive exPlanations)

TreeSHAP analysis provides:
- **Global Feature Importance:** Which features drive predictions overall
- **Local Instance Explanations:** Why specific loans were flagged
- **Color Mapping:** 
  - 🔴 Red: Increases default risk
  - 🔵 Blue: Decreases default risk

**Top Predictive Features:**
- Payment history indicators
- Credit utilization metrics
- Loan amount and term
- Income stability factors

---

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Python 3.10+ required
python --version
```

### Clone Repository
```bash
git clone https://github.com/om-gorakhia/loan-default-prediction.git
cd loan-default-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Core Libraries:**
- `streamlit` - Interactive dashboard
- `scikit-learn` - ML models and metrics
- `pandas`, `numpy` - Data manipulation
- `shap` - Model interpretability
- `matplotlib`, `seaborn` - Visualizations

---

## 🚀 Usage

### Run Analysis Pipeline
```bash
# Execute full ML pipeline (training, evaluation, SHAP analysis)
python analysis.py
```

**Pipeline Steps:**
1. Data loading and preprocessing
2. Train/validation/test split (60/20/20)
3. Model training (5 models)
4. Threshold optimization via contextual loss
5. SHAP interpretability analysis
6. Generate visualizations and reports

### Launch Interactive Dashboard
```bash
streamlit run streamlit_dashboard.py
```

**Dashboard Features:**
- 📊 Model comparison and ROC curves
- 🎯 Threshold optimization analysis
- 🔍 SHAP interpretability (global + local)
- 📈 Feature distribution analysis
- 📋 Detailed classification metrics
- ⚙️ Technical summary and dataset stats

---

## 📁 Project Structure

```
loan-default-prediction/
├── analysis.py                    # ML pipeline and model training
├── streamlit_dashboard.py         # Interactive visualization dashboard
├── requirements.txt               # Python dependencies
├── outputs/                       # Generated results
│   ├── analysis_results.pkl       # Serialized model results
│   ├── best_model_results.json    # Best model metrics
│   ├── model_comparison_summary.csv
│   ├── threshold_analysis.csv
│   └── *.png                      # 8 visualization plots
└── README.md                      # This file
```

---

## 🎓 Academic Context

**Institution:** National University of Singapore (NUS)  
**Program:** Master of Science in Business Analytics (MSBA)  
**Course:** DBA 5106 - Business Analytics  
**Assignment:** Project 2 - Classification with Contextual Loss  
**Author:** Om Gorakhia (A0318038E)  

---

## 📈 Key Features

✅ **Business-Driven Loss Function** - Custom 10:1 FN:FP cost ratio  
✅ **Multiple Model Comparison** - Glass-box vs. black-box approaches  
✅ **Threshold Optimization** - Beyond default 0.5 probability cutoff  
✅ **SHAP Interpretability** - Explainable AI for stakeholder trust  
✅ **Interactive Dashboard** - No runtime calculations, instant insights  
✅ **Production-Ready Code** - Modular, documented, reproducible  

---

## 🔧 Methodology Highlights

### Contextual Loss Function
```python
Contextual_Loss = 10 × False_Negatives + 1 × False_Positives
```

**Business Justification:**
- **FN Cost (10×):** Direct financial loss from defaulted loan principal + interest
- **FP Cost (1×):** Opportunity cost of lost interest revenue from rejected good loan
- **Ratio (10:1):** Reflects typical financial institution risk preferences

### Threshold Optimization Process
1. Train models on training set
2. Evaluate AUC on validation set
3. Sweep thresholds from 0.01 to 0.99
4. Calculate contextual loss at each threshold
5. Select threshold minimizing validation loss
6. Apply optimal threshold to test set

---

## 📊 Output Files

### Generated Reports
- `model_comparison_summary.csv` - Performance metrics for all 5 models
- `best_model_results.json` - Detailed metrics for optimal model
- `threshold_analysis.csv` - Loss evaluation across threshold range

### Visualizations
1. `01_model_comparison_auc.png` - AUC bar chart comparison
2. `02_roc_curves_all_models.png` - ROC curves overlaid
3. `03_threshold_optimization.png` - Threshold vs. loss plot
4. `04_confusion_matrices.png` - Confusion matrices for all models
5. `05_shap_global_importance.png` - Global feature importance
6. `06_shap_local_instances.png` - Local prediction explanations
7. `07_feature_distributions.png` - Feature distributions by class
8. `08_class_distribution.png` - Target class balance

---

## 💡 Business Impact

### Risk Mitigation
- **85.4% of actual defaults detected** (high sensitivity)
- **90.1% of good loans correctly approved** (high specificity)
- **Minimized contextual loss** through optimal threshold

### Operational Efficiency
- **Instant predictions** via pre-trained models
- **Interpretable results** for regulatory compliance
- **Scalable pipeline** for batch processing

### Stakeholder Value
- **Risk Teams:** Quantified risk assessment for portfolio management
- **Loan Officers:** Data-driven approval recommendations
- **Executives:** Dashboard visualizations for strategic decisions

---

## 🤝 Contributing

This is an academic project. For collaboration or questions:

**Om Gorakhia**  
📧 NUS Email: e1519898@u.nus.edu  
📧 Personal: om.g2k01@gmail.com  
🎓 NUS MSBA | Business Analytics Enthusiast  

---

## 📝 License

MIT License - Educational purposes only.

---

## ⭐ Acknowledgments

- **NUS Business School** - DBA 5106 course structure and dataset
- **Scikit-Learn** - Robust ML framework
- **SHAP Library** - Model interpretability
- **Streamlit** - Rapid dashboard prototyping

---

**⚡ Ready to predict loan defaults with confidence!** Run the analysis and explore the interactive dashboard to see ML-driven risk assessment in action.
