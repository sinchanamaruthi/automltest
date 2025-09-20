import streamlit as st
import requests
import json
import pandas as pd
import zipfile
import io
import os
import numpy as np
from urllib.parse import quote
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.metrics import roc_curve, auc, confusion_matrix, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="AI ML Agent", layout="wide")

# Initialize session state for conversation
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = {}

# AI Agent Functions
def get_ai_response(user_message, context=""):
    """Enhanced AI agent responses with comprehensive understanding"""
    
    user_lower = user_message.lower()
    
    # Comprehensive response system
    if any(word in user_lower for word in ["hello", "hi", "hey", "start", "begin"]):
        return "👋 Hello! I'm your AI ML Assistant. I can help you with:\n\n" \
               "• **Dataset Selection** - Find the perfect dataset for your project\n" \
               "• **ML Pipeline Guidance** - Choose what analysis you need\n" \
               "• **Output Explanations** - Understand every chart and result\n" \
               "• **Step-by-step Guidance** - Walk through the entire ML process\n" \
               "• **Answer Any Questions** - Ask me anything about ML, data science, or your results\n\n" \
               "What would you like to work on today?"
    
    elif any(word in user_lower for word in ["correlation", "correlation matrix", "heatmap"]):
        return explain_correlation_matrix_detailed()
    
    elif any(word in user_lower for word in ["accuracy", "precision", "recall", "f1", "classification"]):
        return explain_classification_metrics()
    
    elif any(word in user_lower for word in ["mse", "rmse", "r2", "regression", "prediction"]):
        return explain_regression_metrics()
    
    elif any(word in user_lower for word in ["histogram", "distribution", "plot", "chart"]):
        return explain_visualizations()
    
    elif any(word in user_lower for word in ["missing", "null", "na", "empty"]):
        return explain_missing_data()
    
    elif any(word in user_lower for word in ["outlier", "anomaly", "extreme"]):
        return explain_outliers()
    
    elif any(word in user_lower for word in ["feature", "variable", "column"]):
        return explain_features()
    
    elif any(word in user_lower for word in ["model", "algorithm", "training"]):
        return explain_ml_models()
    
    elif any(word in user_lower for word in ["overfitting", "underfitting", "bias", "variance"]):
        return explain_model_performance()
    
    elif any(word in user_lower for word in ["cross validation", "validation", "test"]):
        return explain_validation()
    
    elif any(word in user_lower for word in ["preprocessing", "scaling", "encoding"]):
        return explain_preprocessing()
    
    elif any(word in user_lower for word in ["dataset", "data", "csv", "file"]):
        return "📊 **Dataset Selection Help:**\n\n" \
               "Tell me about your project:\n" \
               "• What type of data are you interested in? (e.g., sales, customer, medical, financial)\n" \
               "• What's your goal? (prediction, classification, analysis)\n" \
               "• Any specific domain? (business, healthcare, sports, etc.)\n\n" \
               "I'll recommend the best datasets for you!"
    
    elif any(word in user_lower for word in ["pipeline", "analysis", "workflow"]):
        return "🔧 **ML Pipeline Options:**\n\n" \
               "Choose what you need:\n" \
               "• **EDA Only** - Data exploration and visualization\n" \
               "• **Preprocessing** - Data cleaning and preparation\n" \
               "• **Model Training** - Build and train ML models\n" \
               "• **Full Pipeline** - Complete end-to-end analysis\n" \
               "• **Custom** - Tell me exactly what you want\n\n" \
               "What level of analysis do you need?"
    
    elif any(word in user_lower for word in ["explain", "what", "how", "why", "meaning", "understand"]):
        return "📚 **I'll Explain Everything:**\n\n" \
               "For each output, I'll provide:\n" \
               "• **What it shows** - Clear description\n" \
               "• **Why it matters** - Business/technical importance\n" \
               "• **How to interpret** - Reading the results\n" \
               "• **Next steps** - What to do with the insights\n\n" \
               "Just ask me about any chart or result you see!"
    
    elif any(word in user_lower for word in ["help", "stuck", "confused", "don't understand"]):
        return "🆘 **I'm here to help!**\n\n" \
               "You can ask me about:\n" \
               "• Any chart, graph, or visualization you see\n" \
               "• Model results and metrics\n" \
               "• Data preprocessing steps\n" \
               "• Next steps in your analysis\n" \
               "• General ML concepts\n\n" \
               "What specific part would you like me to explain?"
    
    else:
        # Intelligent response based on context
        return f"🤖 I understand you're asking about: '{user_message}'\n\n" \
               f"Let me help you with that! I can explain:\n" \
               f"• **Data Analysis Results** - Any charts, graphs, or statistics\n" \
               f"• **ML Model Outputs** - Accuracy, predictions, performance metrics\n" \
               f"• **Data Preprocessing** - Cleaning, scaling, encoding steps\n" \
               f"• **Next Steps** - What to do with your results\n\n" \
               f"Could you be more specific about what you'd like me to explain?"

# Comprehensive Explanation Functions
def explain_correlation_matrix_detailed():
    """Detailed explanation of correlation matrix"""
    return "📊 **Correlation Matrix - Complete Guide:**\n\n" \
           "**What it shows:**\n" \
           "• A heatmap showing relationships between all numeric variables\n" \
           "• Each cell shows correlation coefficient (-1 to +1)\n" \
           "• Colors: Red = positive, Blue = negative, White = no correlation\n\n" \
           "**How to read it:**\n" \
           "• **1.0**: Perfect positive correlation (variables move together)\n" \
           "• **0.7-0.9**: Strong positive correlation\n" \
           "• **0.3-0.7**: Moderate positive correlation\n" \
           "• **0.0-0.3**: Weak positive correlation\n" \
           "• **0.0**: No linear relationship\n" \
           "• **-0.3 to 0.0**: Weak negative correlation\n" \
           "• **-0.7 to -0.3**: Moderate negative correlation\n" \
           "• **-1.0**: Perfect negative correlation (variables move opposite)\n\n" \
           "**Why it matters:**\n" \
           "• **Feature Selection**: Remove highly correlated features to avoid redundancy\n" \
           "• **Multicollinearity**: High correlations can cause model instability\n" \
           "• **Business Insights**: Understand which factors influence each other\n" \
           "• **Model Building**: Choose features that are predictive but not redundant\n\n" \
           "**Red flags to watch:**\n" \
           "• Correlations > 0.8: Consider removing one variable\n" \
           "• Perfect correlations (1.0 or -1.0): Variables are identical\n" \
           "• Many high correlations: Data might be over-engineered"

def explain_correlation_matrix(df=None):
    """Explain correlation matrix results"""
    if df is not None:
        explanation = "📊 **Correlation Matrix Explanation:**\n\n"
        # Calculate correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Strong correlation threshold
                        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            explanation += "**What it shows:**\n"
            explanation += "• Colors represent correlation strength (-1 to +1)\n"
            explanation += "• Red = Strong positive correlation\n"
            explanation += "• Blue = Strong negative correlation\n"
            explanation += "• White = No correlation\n\n"
            
            if corr_pairs:
                explanation += "**Strong Correlations Found:**\n"
                for col1, col2, corr in corr_pairs[:3]:  # Show top 3
                    explanation += f"• {col1} ↔ {col2}: {corr:.2f}\n"
                explanation += "\n"
            
            explanation += "**Why it matters:**\n"
            explanation += "• Helps identify relationships between variables\n"
            explanation += "• Useful for feature selection\n"
            explanation += "• Reveals potential multicollinearity\n"
            explanation += "• Guides model building decisions\n\n"
            
            explanation += "**How to interpret:**\n"
            explanation += "• Values close to 1: Strong positive relationship\n"
            explanation += "• Values close to -1: Strong negative relationship\n"
            explanation += "• Values close to 0: No linear relationship\n"
        else:
            explanation += "No numeric columns found for correlation analysis."
        
        return explanation
    else:
        return explain_correlation_matrix_detailed()

def explain_classification_metrics():
    """Explain classification model metrics"""
    return "🎯 **Classification Metrics - Complete Guide:**\n\n" \
           "**Accuracy:**\n" \
           "• **What**: Percentage of correct predictions\n" \
           "• **Formula**: (True Positives + True Negatives) / Total Predictions\n" \
           "• **Good**: > 80% for most problems\n" \
           "• **Warning**: Can be misleading with imbalanced data\n\n" \
           "**Precision:**\n" \
           "• **What**: Of all positive predictions, how many were actually positive?\n" \
           "• **Formula**: True Positives / (True Positives + False Positives)\n" \
           "• **Use**: When false positives are costly (e.g., spam detection)\n" \
           "• **Good**: > 0.8 for most applications\n\n" \
           "**Recall (Sensitivity):**\n" \
           "• **What**: Of all actual positives, how many did we catch?\n" \
           "• **Formula**: True Positives / (True Positives + False Negatives)\n" \
           "• **Use**: When false negatives are costly (e.g., disease detection)\n" \
           "• **Good**: > 0.8 for most applications\n\n" \
           "**F1-Score:**\n" \
           "• **What**: Balance between precision and recall\n" \
           "• **Formula**: 2 × (Precision × Recall) / (Precision + Recall)\n" \
           "• **Use**: When you need balanced performance\n" \
           "• **Good**: > 0.7 for most applications\n\n" \
           "**Confusion Matrix:**\n" \
           "• **True Positives**: Correctly predicted positive cases\n" \
           "• **True Negatives**: Correctly predicted negative cases\n" \
           "• **False Positives**: Incorrectly predicted as positive\n" \
           "• **False Negatives**: Incorrectly predicted as negative"

def explain_regression_metrics():
    """Explain regression model metrics"""
    return "📈 **Regression Metrics - Complete Guide:**\n\n" \
           "**Mean Squared Error (MSE):**\n" \
           "• **What**: Average of squared differences between predicted and actual values\n" \
           "• **Formula**: Σ(Actual - Predicted)² / n\n" \
           "• **Units**: Same as target variable, but squared\n" \
           "• **Lower is better**: 0 = perfect predictions\n" \
           "• **Sensitive to outliers**: Large errors have big impact\n\n" \
           "**Root Mean Squared Error (RMSE):**\n" \
           "• **What**: Square root of MSE\n" \
           "• **Formula**: √MSE\n" \
           "• **Units**: Same as target variable\n" \
           "• **Lower is better**: 0 = perfect predictions\n" \
           "• **Interpretation**: Average prediction error in original units\n\n" \
           "**Mean Absolute Error (MAE):**\n" \
           "• **What**: Average absolute difference between predicted and actual values\n" \
           "• **Formula**: Σ|Actual - Predicted| / n\n" \
           "• **Units**: Same as target variable\n" \
           "• **Lower is better**: 0 = perfect predictions\n" \
           "• **Less sensitive to outliers**: More robust than MSE\n\n" \
           "**R² Score (Coefficient of Determination):**\n" \
           "• **What**: Proportion of variance in target variable explained by the model\n" \
           "• **Formula**: 1 - (SS_res / SS_tot)\n" \
           "• **Range**: 0 to 1 (can be negative for very bad models)\n" \
           "• **Interpretation**:\n" \
           "  - 1.0: Perfect predictions\n" \
           "  - 0.8-1.0: Excellent model\n" \
           "  - 0.6-0.8: Good model\n" \
           "  - 0.4-0.6: Moderate model\n" \
           "  - 0.0-0.4: Poor model\n" \
           "  - < 0.0: Worse than just using the mean"

def explain_visualizations():
    """Explain different types of visualizations"""
    return "📊 **Data Visualizations - Complete Guide:**\n\n" \
           "**Histograms:**\n" \
           "• **What**: Shows distribution of a single variable\n" \
           "• **Use**: Understand data shape, find patterns, detect outliers\n" \
           "• **Read**: Height = frequency, Width = value range\n" \
           "• **Patterns**: Normal (bell curve), Skewed, Bimodal, Uniform\n\n" \
           "**Scatter Plots:**\n" \
           "• **What**: Shows relationship between two variables\n" \
           "• **Use**: Find correlations, patterns, clusters\n" \
           "• **Read**: X-axis = variable 1, Y-axis = variable 2\n" \
           "• **Patterns**: Linear, Curved, Clustered, Random\n\n" \
           "**Box Plots:**\n" \
           "• **What**: Shows distribution summary (quartiles, outliers)\n" \
           "• **Use**: Compare groups, detect outliers, understand spread\n" \
           "• **Read**: Box = middle 50%, Line = median, Whiskers = range\n" \
           "• **Outliers**: Points beyond whiskers\n\n" \
           "**Bar Charts:**\n" \
           "• **What**: Compares categories or groups\n" \
           "• **Use**: Show counts, frequencies, comparisons\n" \
           "• **Read**: Height = value, Categories on X-axis\n" \
           "• **Types**: Vertical, Horizontal, Stacked, Grouped\n\n" \
           "**Line Charts:**\n" \
           "• **What**: Shows trends over time or sequence\n" \
           "• **Use**: Track changes, identify patterns, forecast\n" \
           "• **Read**: X-axis = time/sequence, Y-axis = values\n" \
           "• **Patterns**: Increasing, Decreasing, Seasonal, Cyclical"

def explain_missing_data():
    """Explain missing data analysis"""
    return "❓ **Missing Data - Complete Guide:**\n\n" \
           "**What is Missing Data?**\n" \
           "• Values that are not recorded or available\n" \
           "• Shown as NaN, None, empty cells, or special codes\n" \
           "• Can occur due to data collection issues, privacy, or errors\n\n" \
           "**Types of Missing Data:**\n" \
           "• **MCAR (Missing Completely at Random)**: No pattern, random\n" \
           "• **MAR (Missing at Random)**: Pattern exists but not in missing values\n" \
           "• **MNAR (Missing Not at Random)**: Pattern in missing values themselves\n\n" \
           "**Impact on Analysis:**\n" \
           "• **Reduces sample size**: Fewer observations for analysis\n" \
           "• **Biases results**: If missing data has patterns\n" \
           "• **Breaks algorithms**: Many ML models can't handle missing values\n" \
           "• **Reduces power**: Statistical tests become less reliable\n\n" \
           "**Handling Strategies:**\n" \
           "• **Deletion**: Remove rows/columns with missing data\n" \
           "• **Imputation**: Fill missing values with estimates\n" \
           "  - Mean/Median: For numeric data\n" \
           "  - Mode: For categorical data\n" \
           "  - Forward/Backward fill: For time series\n" \
           "  - Advanced: KNN, regression-based imputation\n" \
           "• **Flagging**: Create indicator variables for missing data\n\n" \
           "**When to worry:**\n" \
           "• > 5% missing: Investigate patterns\n" \
           "• > 20% missing: Consider if data is usable\n" \
           "• > 50% missing: May need to exclude variable"

def explain_outliers():
    """Explain outlier detection and handling"""
    return "🚨 **Outliers - Complete Guide:**\n\n" \
           "**What are Outliers?**\n" \
           "• Data points that are significantly different from others\n" \
           "• Values that fall outside the normal range\n" \
           "• Can be detected using statistical methods or visual inspection\n\n" \
           "**Types of Outliers:**\n" \
           "• **Point Outliers**: Individual data points\n" \
           "• **Contextual Outliers**: Unusual in specific context\n" \
           "• **Collective Outliers**: Group of points that are unusual together\n\n" \
           "**Detection Methods:**\n" \
           "• **IQR Method**: Q3 + 1.5×IQR or Q1 - 1.5×IQR\n" \
           "• **Z-Score**: Values with |z| > 3\n" \
           "• **Modified Z-Score**: Using median absolute deviation\n" \
           "• **Visual**: Box plots, scatter plots, histograms\n\n" \
           "**Causes of Outliers:**\n" \
           "• **Data Entry Errors**: Typos, wrong units\n" \
           "• **Measurement Errors**: Equipment malfunction\n" \
           "• **Natural Variation**: Extreme but valid values\n" \
           "• **Fraud/Anomalies**: Suspicious activities\n\n" \
           "**Impact on Analysis:**\n" \
           "• **Skew Statistics**: Mean, standard deviation\n" \
           "• **Affect Models**: Regression, clustering algorithms\n" \
           "• **Reduce Accuracy**: Can mislead model training\n" \
           "• **Mask Patterns**: Hide underlying relationships\n\n" \
           "**Handling Strategies:**\n" \
           "• **Investigate**: Understand why they exist\n" \
           "• **Remove**: If clearly errors\n" \
           "• **Transform**: Log, square root transformations\n" \
           "• **Cap/Winsorize**: Replace with threshold values\n" \
           "• **Keep**: If they represent important information"

def explain_features():
    """Explain feature engineering and selection"""
    return "🔧 **Features - Complete Guide:**\n\n" \
           "**What are Features?**\n" \
           "• Variables used to predict the target\n" \
           "• Input data for machine learning models\n" \
           "• Can be raw data or engineered/transformed variables\n\n" \
           "**Types of Features:**\n" \
           "• **Numeric**: Continuous values (age, price, temperature)\n" \
           "• **Categorical**: Discrete categories (color, brand, status)\n" \
           "• **Binary**: Two categories (yes/no, true/false)\n" \
           "• **Ordinal**: Ordered categories (rating, grade)\n" \
           "• **Text**: String data (reviews, descriptions)\n" \
           "• **Date/Time**: Temporal data (timestamps, dates)\n\n" \
           "**Feature Engineering:**\n" \
           "• **Creating New Features**:\n" \
           "  - Mathematical operations (sum, difference, ratio)\n" \
           "  - Binning continuous variables\n" \
           "  - Extracting from text (word count, sentiment)\n" \
           "  - Time-based features (day of week, season)\n" \
           "• **Transforming Features**:\n" \
           "  - Scaling (normalization, standardization)\n" \
           "  - Encoding (one-hot, label encoding)\n" \
           "  - Log transformation for skewed data\n\n" \
           "**Feature Selection:**\n" \
           "• **Filter Methods**: Statistical tests, correlation\n" \
           "• **Wrapper Methods**: Forward/backward selection\n" \
           "• **Embedded Methods**: Lasso, tree-based importance\n" \
           "• **Domain Knowledge**: Expert judgment\n\n" \
           "**Good Features:**\n" \
           "• **Relevant**: Related to the target variable\n" \
           "• **Non-redundant**: Not highly correlated with others\n" \
           "• **Complete**: Few missing values\n" \
           "• **Consistent**: Same format and meaning\n" \
           "• **Scalable**: Easy to obtain for new data"

def explain_ml_models():
    """Explain different ML algorithms"""
    return "🤖 **Machine Learning Models - Complete Guide:**\n\n" \
           "**Supervised Learning (with target variable):**\n" \
           "• **Classification**: Predict categories/classes\n" \
           "  - Logistic Regression: Linear decision boundary\n" \
           "  - Random Forest: Ensemble of decision trees\n" \
           "  - SVM: Finds optimal separating hyperplane\n" \
           "  - Naive Bayes: Probabilistic classifier\n" \
           "• **Regression**: Predict continuous values\n" \
           "  - Linear Regression: Linear relationship\n" \
           "  - Random Forest: Ensemble for regression\n" \
           "  - Polynomial Regression: Non-linear relationships\n\n" \
           "**Unsupervised Learning (no target variable):**\n" \
           "• **Clustering**: Group similar data points\n" \
           "  - K-Means: Spherical clusters\n" \
           "  - Hierarchical: Tree-like clustering\n" \
           "  - DBSCAN: Density-based clustering\n" \
           "• **Dimensionality Reduction**: Reduce feature count\n" \
           "  - PCA: Linear dimensionality reduction\n" \
           "  - t-SNE: Non-linear visualization\n\n" \
           "**Model Selection Criteria:**\n" \
           "• **Data Size**: Small data → simple models\n" \
           "• **Linearity**: Linear data → linear models\n" \
           "• **Interpretability**: Need explanations → linear/logistic\n" \
           "• **Performance**: Need accuracy → ensemble methods\n" \
           "• **Speed**: Need fast predictions → simple models\n\n" \
           "**Model Performance:**\n" \
           "• **Training**: How well model fits training data\n" \
           "• **Validation**: How well model generalizes\n" \
           "• **Overfitting**: Good on training, poor on new data\n" \
           "• **Underfitting**: Poor on both training and new data"

def explain_model_performance():
    """Explain model performance concepts"""
    return "📊 **Model Performance - Complete Guide:**\n\n" \
           "**Overfitting:**\n" \
           "• **What**: Model learns training data too well\n" \
           "• **Signs**: High training accuracy, low validation accuracy\n" \
           "• **Causes**: Too complex model, too little data, too many features\n" \
           "• **Solutions**:\n" \
           "  - Simplify model (reduce complexity)\n" \
           "  - More training data\n" \
           "  - Regularization (L1/L2)\n" \
           "  - Cross-validation\n" \
           "  - Early stopping\n\n" \
           "**Underfitting:**\n" \
           "• **What**: Model is too simple to capture patterns\n" \
           "• **Signs**: Low training and validation accuracy\n" \
           "• **Causes**: Too simple model, insufficient features\n" \
           "• **Solutions**:\n" \
           "  - Increase model complexity\n" \
           "  - Add more features\n" \
           "  - Feature engineering\n" \
           "  - Different algorithm\n\n" \
           "**Bias vs Variance:**\n" \
           "• **Bias**: Error from oversimplifying assumptions\n" \
           "  - High bias = underfitting\n" \
           "  - Low bias = model can capture complexity\n" \
           "• **Variance**: Error from sensitivity to training data\n" \
           "  - High variance = overfitting\n" \
           "  - Low variance = model is stable\n" \
           "• **Bias-Variance Tradeoff**: Balancing both errors\n\n" \
           "**Model Validation:**\n" \
           "• **Train/Test Split**: Separate data for training and testing\n" \
           "• **Cross-Validation**: Multiple train/test splits\n" \
           "• **Holdout Set**: Final test on unseen data\n" \
           "• **Time Series**: Use temporal splits for time data"

def explain_validation():
    """Explain validation techniques"""
    return "✅ **Model Validation - Complete Guide:**\n\n" \
           "**Why Validate?**\n" \
           "• **Prevent Overfitting**: Test on unseen data\n" \
           "• **Estimate Performance**: How well model will work in practice\n" \
           "• **Compare Models**: Choose best performing algorithm\n" \
           "• **Tune Parameters**: Find optimal hyperparameters\n\n" \
           "**Validation Methods:**\n" \
           "• **Holdout Validation**:\n" \
           "  - Split: 70% train, 15% validation, 15% test\n" \
           "  - Use: Quick validation, large datasets\n" \
           "  - Risk: Results depend on random split\n\n" \
           "• **K-Fold Cross-Validation**:\n" \
           "  - Split data into k folds (usually 5 or 10)\n" \
           "  - Train on k-1 folds, test on 1 fold\n" \
           "  - Repeat k times, average results\n" \
           "  - Use: More reliable estimates\n\n" \
           "• **Stratified K-Fold**:\n" \
           "  - Maintains class distribution in each fold\n" \
           "  - Use: Imbalanced datasets\n\n" \
           "• **Time Series Validation**:\n" \
           "  - Use past data to predict future\n" \
           "  - Expanding or rolling window\n" \
           "  - Use: Temporal data\n\n" \
           "**Validation Metrics:**\n" \
           "• **Classification**: Accuracy, Precision, Recall, F1\n" \
           "• **Regression**: MSE, RMSE, MAE, R²\n" \
           "• **Business Metrics**: Cost, profit, customer satisfaction\n\n" \
           "**Best Practices:**\n" \
           "• **Never use test set for model selection**\n" \
           "• **Use validation set for hyperparameter tuning**\n" \
           "• **Report final results on test set only**\n" \
           "• **Use multiple random seeds for stability**"

def explain_preprocessing():
    """Explain data preprocessing steps"""
    return "🔧 **Data Preprocessing - Complete Guide:**\n\n" \
           "**Why Preprocess?**\n" \
           "• **Clean Data**: Remove errors and inconsistencies\n" \
           "• **Format Data**: Make it suitable for algorithms\n" \
           "• **Improve Performance**: Better model accuracy\n" \
           "• **Handle Missing Values**: Deal with incomplete data\n\n" \
           "**Data Cleaning:**\n" \
           "• **Remove Duplicates**: Eliminate repeated records\n" \
           "• **Handle Outliers**: Detect and treat extreme values\n" \
           "• **Fix Inconsistencies**: Standardize formats\n" \
           "• **Validate Data**: Check for logical errors\n\n" \
           "**Handling Missing Data:**\n" \
           "• **Deletion**: Remove rows/columns with missing values\n" \
           "• **Imputation**: Fill missing values\n" \
           "  - Mean/Median: For numeric data\n" \
           "  - Mode: For categorical data\n" \
           "  - Forward/Backward fill: For time series\n" \
           "  - Advanced methods: KNN, regression imputation\n\n" \
           "**Feature Encoding:**\n" \
           "• **Label Encoding**: Convert categories to numbers\n" \
           "• **One-Hot Encoding**: Create binary columns for categories\n" \
           "• **Target Encoding**: Use target variable for encoding\n" \
           "• **Ordinal Encoding**: For ordered categories\n\n" \
           "**Feature Scaling:**\n" \
           "• **Standardization**: (x - mean) / std\n" \
           "  - Mean = 0, Std = 1\n" \
           "  - Use: Normal distribution, distance-based algorithms\n" \
           "• **Normalization**: (x - min) / (max - min)\n" \
           "  - Range: 0 to 1\n" \
           "  - Use: Neural networks, algorithms sensitive to scale\n" \
           "• **Robust Scaling**: (x - median) / IQR\n" \
           "  - Use: Data with outliers\n\n" \
           "**Feature Transformation:**\n" \
           "• **Log Transformation**: Reduce skewness\n" \
           "• **Square Root**: For count data\n" \
           "• **Box-Cox**: Optimal transformation\n" \
           "• **Polynomial**: Create interaction terms"

def explain_model_results(model_type, results):
    """Explain ML model results"""
    explanation = f"🤖 **{model_type} Model Results Explanation:**\n\n"
    
    if model_type == "Classification":
        if 'accuracy' in results:
            explanation += f"**Accuracy: {results['accuracy']:.2%}**\n"
            explanation += f"• This means the model correctly predicted {results['accuracy']:.1%} of all cases\n"
            explanation += f"• Higher is better (perfect = 100%)\n\n"
        
        if 'classification_report' in results:
            explanation += "**Classification Report:**\n"
            explanation += "• **Precision**: How many predicted positives were actually positive\n"
            explanation += "• **Recall**: How many actual positives were correctly identified\n"
            explanation += "• **F1-Score**: Balance between precision and recall\n\n"
    
    elif model_type == "Regression":
        if 'mse' in results:
            explanation += f"**Mean Squared Error: {results['mse']:.2f}**\n"
            explanation += f"• Average squared difference between predicted and actual values\n"
            explanation += f"• Lower is better (perfect = 0)\n\n"
        
        if 'r2' in results:
            explanation += f"**R² Score: {results['r2']:.2f}**\n"
            explanation += f"• Proportion of variance explained by the model\n"
            explanation += f"• Higher is better (perfect = 1.0)\n\n"
    
    explanation += "**What this means:**\n"
    explanation += "• The model has been trained and tested\n"
    explanation += "• These metrics show how well it performs\n"
    explanation += "• Use these to compare different models\n\n"
    
    explanation += "**Next steps:**\n"
    explanation += "• Try different algorithms to improve performance\n"
    explanation += "• Feature engineering might help\n"
    explanation += "• Consider ensemble methods for better results"
    
    return explanation

def recommend_datasets(user_preferences):
    """Recommend datasets based on user preferences"""
    recommendations = []
    
    # Popular datasets with descriptions
    all_datasets = {
        "House Prices": {
            "description": "Predict house prices using features like size, location, age",
            "type": "Regression",
            "domain": "Real Estate",
            "difficulty": "Intermediate",
            "tags": ["real estate", "regression", "prediction", "housing"]
        },
        "Titanic": {
            "description": "Predict passenger survival on the Titanic",
            "type": "Classification", 
            "domain": "Historical",
            "difficulty": "Beginner",
            "tags": ["classification", "survival", "beginner", "binary"]
        },
        "Wine Quality": {
            "description": "Predict wine quality based on chemical properties",
            "type": "Classification",
            "domain": "Food & Beverage", 
            "difficulty": "Intermediate",
            "tags": ["classification", "wine", "quality", "chemistry"]
        },
        "Credit Card Fraud": {
            "description": "Detect fraudulent credit card transactions",
            "type": "Classification",
            "domain": "Finance",
            "difficulty": "Advanced", 
            "tags": ["fraud", "detection", "finance", "imbalanced"]
        },
        "Customer Segmentation": {
            "description": "Segment customers based on purchasing behavior",
            "type": "Clustering",
            "domain": "Marketing",
            "difficulty": "Intermediate",
            "tags": ["clustering", "customers", "marketing", "segmentation"]
        }
    }
    
    # Simple matching logic
    user_text = " ".join(user_preferences.values()).lower()
    
    for name, info in all_datasets.items():
        score = 0
        for tag in info["tags"]:
            if tag in user_text:
                score += 1
        
        if score > 0:
            recommendations.append((name, info, score))
    
    # Sort by relevance score
    recommendations.sort(key=lambda x: x[2], reverse=True)
    
    return recommendations[:3]  # Return top 3

# Main App Layout
st.title("🤖 AI ML Agent - Your Personal Data Science Assistant")

# Sidebar for conversation
with st.sidebar:
    st.header("💬 Chat with AI Agent")
    
    # Display conversation history
    for message in st.session_state.conversation:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**AI:** {message['content']}")
    
    # Chat input
    user_input = st.text_input("Ask me anything about your ML project:", key="chat_input")
    
    if st.button("Send") and user_input:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": user_input})
        
        # Get AI response
        ai_response = get_ai_response(user_input)
        
        # Add AI response to conversation
        st.session_state.conversation.append({"role": "ai", "content": ai_response})
        
        # Store user preferences
        if "dataset" in user_input.lower() or "data" in user_input.lower():
            st.session_state.user_preferences["dataset_interest"] = user_input
        
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Your ML Project")
    
    # Show current status
    if st.session_state.user_preferences:
        st.subheader("📋 Project Preferences")
        for key, value in st.session_state.user_preferences.items():
            st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
    
    # Dataset recommendations
    if st.session_state.user_preferences:
        st.subheader("📊 Recommended Datasets")
        recommendations = recommend_datasets(st.session_state.user_preferences)
        
        if recommendations:
            for name, info, score in recommendations:
                with st.expander(f"📈 {name} (Relevance: {score}/5)"):
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Type:** {info['type']} | **Domain:** {info['domain']} | **Difficulty:** {info['difficulty']}")
                    
                    if st.button(f"Use {name}", key=f"use_{name}"):
                        st.session_state.current_dataset = name
                        st.success(f"Selected {name} dataset!")
                        st.rerun()
        else:
            st.info("Tell me more about your project to get dataset recommendations!")
    
    # Current dataset info
    if st.session_state.current_dataset:
        st.subheader(f"📊 Current Dataset: {st.session_state.current_dataset}")
        st.info("Ready to start analysis! Use the sidebar to load your dataset and begin the ML pipeline.")

with col2:
    st.header("🛠️ Quick Actions")
    
    # Quick start buttons
    if st.button("🚀 Start New Project"):
        st.session_state.conversation = []
        st.session_state.user_preferences = {}
        st.session_state.current_dataset = None
        st.rerun()
    
    if st.button("📊 Browse Datasets"):
        st.session_state.conversation.append({
            "role": "user", 
            "content": "I want to browse available datasets"
        })
        st.rerun()
    
    if st.button("🔧 ML Pipeline Help"):
        st.session_state.conversation.append({
            "role": "user",
            "content": "Help me understand ML pipeline options"
        })
        st.rerun()
    
    if st.button("📚 Explain Results"):
        st.session_state.conversation.append({
            "role": "user", 
            "content": "I need help understanding ML results and outputs"
        })
        st.rerun()

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Ask me about any part of the ML process - from data selection to result interpretation!")

# Add some sample conversation starters
st.subheader("💬 Conversation Starters")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("I want to predict house prices"):
        st.session_state.conversation.append({
            "role": "user",
            "content": "I want to predict house prices"
        })
        st.rerun()

with col2:
    if st.button("Help me analyze customer data"):
        st.session_state.conversation.append({
            "role": "user", 
            "content": "Help me analyze customer data"
        })
        st.rerun()

with col3:
    if st.button("Explain correlation matrix"):
        st.session_state.conversation.append({
            "role": "user",
            "content": "Explain correlation matrix"
        })
        st.rerun()
