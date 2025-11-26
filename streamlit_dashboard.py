# ==========================================
# LOAN DEFAULT PREDICTION - STREAMLIT DASHBOARD
# ==========================================
# Interactive visualization dashboard for analysis results
# All computations done in backend (analysis.py)
# Dashboard is instantaneous with no runtime calculations

# Author: Om Gohel
# Student ID: A0318038E
# Email: e1519898@u.nus.edu / om.g2k01@gmail.com
# Institution: National University of Singapore (NUS)
# Course: DBA 5106 - Business Analytics

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import os
from PIL import Image
from datetime import datetime

# ============ PAGE CONFIGURATION ============

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM STYLING ============

st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header-title {
        color: #2E86AB;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #6C757D;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2E86AB 0%, #118AB2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .section-header {
        color: #2E86AB;
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem 0;
        border-bottom: 3px solid #2E86AB;
        margin: 2rem 0 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #06A77D;
    }
    
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #F18F01;
    }
    
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #118AB2;
    }
</style>
""", unsafe_allow_html=True)

# ============ UTILITY FUNCTIONS ============

@st.cache_resource
def load_results():
    """Load pre-computed analysis results"""
    try:
        with open('outputs/analysis_results.pkl', 'rb') as f:
            results = pickle.load(f)

        with open('outputs/best_model_results.json', 'r') as f:
            best_results = json.load(f)

        return results, best_results
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None, None

@st.cache_resource
def load_images():
    """Load all generated images"""
    images = {}
    image_files = [
        '01_model_comparison_auc.png',
        '02_roc_curves_all_models.png',
        '03_threshold_optimization.png',
        '04_confusion_matrices.png',
        '05_shap_global_importance.png',
        '06_shap_local_instances.png',
        '07_feature_distributions.png',
        '08_class_distribution.png'
    ]

    for img_file in image_files:
        path = f'outputs/{img_file}'
        if os.path.exists(path):
            images[img_file.replace('.png', '')] = Image.open(path)

    return images

# ============ PAGE COMPONENTS ============

def render_header():
    """Render dashboard header"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="header-title">💰 Loan Default Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="header-subtitle">Classification Analysis with Contextual Loss Evaluation</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="text-align: right; color: #6C757D; font-size: 0.9rem;">
            <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <b>Institution:</b> NUS
        </div>
        """, unsafe_allow_html=True)

def render_key_metrics(results, best_results):
    """Render key performance metrics"""
    st.markdown('<div class="section-header">📊 Key Performance Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Model</div>
            <div class="metric-value">{best_results['Best Model']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Validation AUC</div>
            <div class="metric-value">{best_results['Validation AUC']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test AUC</div>
            <div class="metric-value">{best_results['Test AUC']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Optimal Threshold</div>
            <div class="metric-value">{best_results['Optimal Threshold']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sensitivity</div>
            <div class="metric-value">{best_results['Sensitivity']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

def render_model_comparison(results):
    """Render model comparison section"""
    st.markdown('<div class="section-header">🏆 Model Comparison</div>', unsafe_allow_html=True)

    comparison_data = results['model_comparison']

    # Create comparison table
    comparison_df = pd.DataFrame({
        'Model': list(comparison_data.keys()),
        'Validation AUC': [f"{m['auc']:.4f}" for m in comparison_data.values()],
        'Optimal Threshold': [f"{m['best_threshold']:.3f}" for m in comparison_data.values()],
        'Contextual Loss': [f"{m['best_contextual_loss']:.0f}" for m in comparison_data.values()]
    })

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Display ROC chart
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("ROC Curves")
        images = load_images()
        if '02_roc_curves_all_models' in images:
            st.image(images['02_roc_curves_all_models'], use_column_width=True)

    with col2:
        st.subheader("AUC Comparison")
        if '01_model_comparison_auc' in images:
            st.image(images['01_model_comparison_auc'], use_column_width=True)

def render_threshold_analysis(results, best_results):
    """Render threshold optimization analysis"""
    st.markdown('<div class="section-header">🎯 Threshold Optimization Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.info(f"""
        **Optimal Threshold:** {best_results['Optimal Threshold']:.3f}
        
        **Contextual Loss Function:** 10×FN + 1×FP
        - FN (False Negative): Approved loan that defaults → High cost
        - FP (False Positive): Rejected loan that would repay → Low cost
        
        **Optimal Loss:** {best_results['Contextual Loss (Test)']:.0f}
        """)

    with col2:
        st.success(f"""
        **Confusion Matrix (Test Set)**
        
        - True Positives: {best_results['TP']}
        - False Positives: {best_results['FP']}
        - False Negatives: {best_results['FN']}
        - True Negatives: {best_results['TN']}
        """)

    st.subheader("Threshold vs Loss Optimization")
    images = load_images()
    if '03_threshold_optimization' in images:
        st.image(images['03_threshold_optimization'], use_column_width=True)

    if '04_confusion_matrices' in images:
        st.image(images['04_confusion_matrices'], use_column_width=True)

def render_interpretability(best_results):
    """Render SHAP interpretability analysis"""
    st.markdown('<div class="section-header">🔍 Model Interpretability (SHAP)</div>', unsafe_allow_html=True)

    st.info("TreeSHAP analysis for Global and Local feature importance. Colors: Red = increases default risk, Blue = decreases risk")

    col1, col2 = st.columns(2)

    images = load_images()

    with col1:
        st.subheader("Global SHAP - Feature Importance")
        if '05_shap_global_importance' in images:
            st.image(images['05_shap_global_importance'], use_column_width=True)

    with col2:
        st.subheader("Local SHAP - Individual Predictions")
        if '06_shap_local_instances' in images:
            st.image(images['06_shap_local_instances'], use_column_width=True)

def render_feature_analysis():
    """Render feature analysis"""
    st.markdown('<div class="section-header">📈 Feature Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    images = load_images()

    with col1:
        st.subheader("Feature Distributions by Default Status")
        if '07_feature_distributions' in images:
            st.image(images['07_feature_distributions'], use_column_width=True)

    with col2:
        st.subheader("Class Distribution")
        if '08_class_distribution' in images:
            st.image(images['08_class_distribution'], use_column_width=True)

def render_classification_metrics(best_results):
    """Render detailed classification metrics"""
    st.markdown('<div class="section-header">📋 Detailed Classification Metrics</div>', unsafe_allow_html=True)

    metrics_data = {
        'Metric': ['Accuracy', 'Sensitivity (Recall)', 'Specificity', 'Precision', 'F1-Score'],
        'Score': [
            f"{best_results['Accuracy']:.4f}",
            f"{best_results['Sensitivity']:.4f}",
            f"{best_results['Specificity']:.4f}",
            f"{best_results['Precision']:.4f}",
            f"{best_results['F1-Score']:.4f}"
        ]
    }

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Interpretation
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Metric Definitions:**
        - **Accuracy**: Proportion of correct predictions
        - **Sensitivity**: Proportion of actual defaults correctly identified
        - **Specificity**: Proportion of non-defaults correctly identified
        """)

    with col2:
        st.markdown("""
        **Business Implications:**
        - High Sensitivity: Few actual defaults missed ✓
        - High Specificity: Few good loans rejected ✓
        - High Precision: Few false alarms ✓
        """)

def render_technical_summary(results, best_results):
    """Render technical summary section"""
    st.markdown('<div class="section-header">⚙️ Technical Summary</div>', unsafe_allow_html=True)

    summary_text = f"""
    **Dataset Characteristics:**
    - Total Samples: 255,347
    - Features: 18 (9 numerical, 9 categorical)
    - Default Rate: {results['class_distribution']['Default_Percentage']:.2f}%
    - Class Distribution: {results['class_distribution']['No Default']:,} (No Default) vs {results['class_distribution']['Default']:,} (Default)
    
    **Data Splitting:**
    - Training Set: 60% (153,208 samples)
    - Validation Set: 20% (51,069 samples)
    - Test Set: 20% (51,070 samples)
    
    **Models Implemented:**
    1. **Glass-box Models** (Interpretable):
       - Logistic Regression with L2 Loss (Ridge - Regression approach)
       - Logistic Regression with Binomial Deviance Loss (Standard)
    
    2. **Black-box Models** (Complex):
       - Gradient Boosting with L2 Loss
       - Gradient Boosting with Log Loss (Binomial Deviance)
       - Random Forest Classifier
    
    **Loss Function Definition:**
    - **Contextual Loss = 10×FN + 1×FP**
    - FN Weight (10): Cost of approving a loan that defaults (direct financial loss)
    - FP Weight (1): Cost of rejecting a good loan (lost interest revenue)
    
    **Model Selection Criteria:**
    - Primary: Validation AUC (Area Under ROC Curve)
    - Secondary: Contextual Loss Minimization
    
    **Best Model:** {best_results['Best Model']}
    - Validation AUC: {best_results['Validation AUC']:.4f}
    - Test AUC: {best_results['Test AUC']:.4f}
    
    **Threshold Optimization:**
    - Default Threshold: 0.500
    - Optimal Threshold: {best_results['Optimal Threshold']:.3f}
    - Optimization Metric: Contextual Loss Minimization
    
    **SHAP Interpretability:**
    - Method: TreeSHAP (for tree-based models)
    - Analysis: Global Feature Importance + Local Instance Explanations
    - Top Features: See Global SHAP section above
    """

    st.markdown(summary_text)

def render_student_info():
    """Render student and project information"""
    st.markdown('<div class="section-header">👤 Student & Project Information</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Student Details:**
        - **Name:** Om Gohel
        - **Student ID:** A0318038E
        - **University:** National University of Singapore (NUS)
        - **Program:** Master of Science in Business Analytics (MSBA)
        - **Course:** DBA 5106 - Business Analytics
        """)

    with col2:
        st.markdown(f"""
        **Contact Information:**
        - **NUS Email:** e1519898@u.nus.edu
        - **Personal Email:** om.g2k01@gmail.com
        
        **Project Details:**
        - **Assignment:** Project 2 - Classification with Contextual Loss
        - **Dataset:** Loan Default Prediction
        - **Submission Date:** {datetime.now().strftime('%Y-%m-%d')}
        """)

def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.markdown("### 📑 Navigation")

    page = st.sidebar.radio(
        "Select Section:",
        ["Overview", "Model Comparison", "Threshold Analysis", "Interpretability",
         "Feature Analysis", "Metrics", "Technical Summary", "Student Info"]
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("""
    ### 📊 Dataset Overview
    - **Samples:** 255,347
    - **Features:** 18
    - **Default Rate:** See Overview
    - **Train/Val/Test:** 60/20/20
    
    ### 🎯 Project Goals
    1. Binary classification of loan defaults
    2. Contextual loss evaluation
    3. Model interpretability via SHAP
    4. Threshold optimization
    
    ### 📁 Output Files
    - model_comparison_summary.csv
    - best_model_results.json
    - threshold_analysis.csv
    - Visualizations (8 PNG files)
    """)

    return page

# ============ MAIN APP ============

def main():
    """Main dashboard application"""

    # Load data
    results, best_results = load_results()

    if results is None or best_results is None:
        st.error("❌ Analysis results not found. Please run analysis.py first.")
        return

    # Render sidebar navigation
    page = render_sidebar()

    # Render header
    render_header()

    # Route to appropriate page
    if page == "Overview":
        render_key_metrics(results, best_results)
        st.divider()
        render_model_comparison(results)

    elif page == "Model Comparison":
        render_model_comparison(results)

    elif page == "Threshold Analysis":
        render_threshold_analysis(results, best_results)

    elif page == "Interpretability":
        render_interpretability(best_results)

    elif page == "Feature Analysis":
        render_feature_analysis()

    elif page == "Metrics":
        render_classification_metrics(best_results)

    elif page == "Technical Summary":
        render_technical_summary(results, best_results)

    elif page == "Student Info":
        render_student_info()

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6C757D; font-size: 0.8rem; margin-top: 2rem;">
        <p>Generated for NUS DBA 5106 - Business Analytics Course</p>
        <p>© 2025 Om Gohel | Student ID: A0318038E</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()