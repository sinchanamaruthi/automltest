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
import openai
from datetime import datetime

st.set_page_config(page_title="Integrated ML Pipeline", layout="wide")

# Sidebar
st.sidebar.title("🤖 AutoML Pipeline")
st.sidebar.markdown("---")

# Dataset source selection
dataset_source = st.sidebar.selectbox(
    "📊 Select Dataset Source:",
    ["Sample Datasets", "Kaggle", "Upload CSV"]
)

st.title("🚀 Integrated ML Pipeline with Kaggle Datasets")

def search_kaggle_datasets(query, page=1):
    """Search Kaggle datasets using their public API"""
    try:
        url = f"https://www.kaggle.com/api/v1/datasets/list"
        params = {
            'search': query,
            'page': page,
            'pageSize': 20
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed with status: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

def redirect_to_kaggle_search(query):
    """Redirect to Kaggle search page with the query"""
    # URL encode the query
    encoded_query = quote(query)
    # Use the correct Kaggle search URL
    kaggle_search_url = f"https://www.kaggle.com/search?q={encoded_query}&type=datasets"
    
    # Create HTML to redirect to Kaggle
    redirect_html = f"""
    <script>
        window.open('{kaggle_search_url}', '_blank');
    </script>
    <div style="text-align: center; padding: 20px;">
        <h3>🔗 Redirecting to Kaggle...</h3>
        <p>Opening Kaggle search for: <strong>{query}</strong></p>
        <p>If the page doesn't open automatically, <a href="{kaggle_search_url}" target="_blank">click here</a></p>
    </div>
    """
    
    return redirect_html

def redirect_to_kaggle_dataset(dataset_name):
    """Redirect to specific Kaggle dataset page"""
    # Use the dataset reference from popular datasets instead of formatting the name
    # This will be passed from the calling function
    kaggle_dataset_url = f"https://www.kaggle.com/datasets/{dataset_name}"
    
    # Create HTML to redirect to Kaggle dataset
    redirect_html = f"""
    <script>
        window.open('{kaggle_dataset_url}', '_blank');
    </script>
    <div style="text-align: center; padding: 20px;">
        <h3>🔗 Redirecting to Kaggle Dataset...</h3>
        <p>Opening dataset: <strong>{dataset_name}</strong></p>
        <p>If the page doesn't open automatically, <a href="{kaggle_dataset_url}" target="_blank">click here</a></p>
    </div>
    """
    
    return redirect_html

def redirect_to_kaggle_login():
    """Redirect to Kaggle login page"""
    kaggle_login_url = "https://www.kaggle.com/account/login"
    
    redirect_html = f"""
    <script>
        window.open('{kaggle_login_url}', '_blank');
    </script>
    <div style="text-align: center; padding: 20px;">
        <h3>🔗 Redirecting to Kaggle Login...</h3>
        <p>Opening Kaggle login page</p>
        <p>If the page doesn't open automatically, <a href="{kaggle_login_url}" target="_blank">click here</a></p>
    </div>
    """
    
    return redirect_html

def get_dataset_details(dataset_ref):
    """Get detailed information about a specific dataset"""
    try:
        url = f"https://www.kaggle.com/api/v1/datasets/view/{dataset_ref}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        return None

def download_dataset_direct(dataset_ref):
    """Download dataset directly without redirecting to Kaggle"""
    try:
        url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_ref}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Download failed with status: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return None

def extract_csv_from_zip(zip_content):
    """Extract CSV files from zip content"""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
            csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
            
            if csv_files:
                largest_csv = max(csv_files, key=lambda x: zip_file.getinfo(x).file_size)
                csv_content = zip_file.read(largest_csv)
                return csv_content, largest_csv
            else:
                return None, None
                
    except Exception as e:
        st.error(f"Extraction error: {str(e)}")
        return None, None

def load_sample_datasets():
    """Load some sample datasets that work well for ML"""
    sample_datasets = {
        "Iris Dataset": {
            "description": "Classic classification dataset with 150 samples of iris flowers",
            "url": "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv",
            "type": "Classification",
            "size": "4.4 KB",
            "samples": "150",
            "features": "4",
            "tags": "classification, flowers, beginner, classic"
        },
        "Boston Housing": {
            "description": "Housing prices in Boston - regression dataset with 506 samples",
            "url": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
            "type": "Regression",
            "size": "15 KB",
            "samples": "506",
            "features": "13",
            "tags": "regression, housing, real estate, beginner"
        },
        "Wine Quality": {
            "description": "Wine quality prediction dataset with 1,599 samples",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "type": "Classification",
            "size": "100 KB",
            "samples": "1,599",
            "features": "11",
            "tags": "classification, wine, quality, food"
        },
        "Car Evaluation": {
            "description": "Car evaluation dataset for classification with 1,728 samples",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
            "type": "Classification",
            "size": "28 KB",
            "samples": "1,728",
            "features": "6",
            "tags": "classification, cars, evaluation, automotive"
        }
    }
    return sample_datasets

def get_popular_kaggle_datasets():
    """Get popular Kaggle datasets with detailed information"""
    popular_datasets = {
        "House Prices": {
            "ref": "c/house-prices-advanced-regression-techniques",
            "description": "Predict sales prices and practice feature engineering, RFs, and gradient boosting",
            "size": "1.2 MB",
            "downloads": "1,200,000+",
            "votes": "4,500+",
            "tags": "regression, real estate, feature engineering, gradient boosting"
        },
        "Titanic": {
            "ref": "c/titanic",
            "description": "Predict survival on the Titanic and get familiar with ML basics",
            "size": "60 KB",
            "downloads": "2,000,000+",
            "votes": "8,000+",
            "tags": "classification, survival analysis, beginner, binary classification"
        },
        "Wine Quality": {
            "ref": "uciml/red-wine-quality-cortez-et-al-2009",
            "description": "Predict the quality of wine based on physicochemical tests",
            "size": "25 KB",
            "downloads": "800,000+",
            "votes": "2,500+",
            "tags": "classification, wine, quality prediction, physicochemical"
        },
        "Credit Card Fraud": {
            "ref": "mlg-ulb/creditcardfraud",
            "description": "Detect fraudulent credit card transactions using machine learning",
            "size": "150 MB",
            "downloads": "1,500,000+",
            "votes": "3,200+",
            "tags": "fraud detection, imbalanced data, anomaly detection, finance"
        },
        "Customer Segmentation": {
            "ref": "vjchoudhary7/customer-segmentation-tutorial-in-python",
            "description": "Segment customers based on their purchasing behavior and demographics",
            "size": "2.5 MB",
            "downloads": "400,000+",
            "votes": "1,800+",
            "tags": "clustering, customer analytics, segmentation, marketing"
        },
        "Employee Attrition": {
            "ref": "pavansubhasht/ibm-hr-analytics-attrition-dataset",
            "description": "Predict employee attrition and understand factors that lead to turnover",
            "size": "45 KB",
            "downloads": "600,000+",
            "votes": "2,200+",
            "tags": "hr analytics, attrition prediction, employee retention, classification"
        }
    }
    return popular_datasets

def preprocess_data(df, target_col):
    """Enhanced preprocessing with better error handling"""
    try:
        df_processed = df.copy()
        df_processed = df_processed.dropna(subset=[target_col])
        
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Handle missing values in features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
        
        # Encode categorical features
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target variable if it's categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Ensure all data is numeric
        X = X.astype(float)
        y = y.astype(float)
        
        # Remove any remaining NaN values
        X = X.fillna(X.median())
        
        if X.empty or len(X) == 0:
            raise ValueError("No valid data after preprocessing")
        
        if len(X) < 2:
            raise ValueError("Not enough data for train/test split")
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
    except Exception as e:
        st.warning(f"Advanced preprocessing failed, using fallback: {e}")
        
        df_processed = df.dropna()
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))
        
        X = X.astype(float)
        y = y.astype(float)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train and evaluate models"""
    results = {}
    trained_models = {}
    is_classification = len(set(y_train)) < 20

    if is_classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(random_state=42)
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            acc = accuracy_score(y_test, preds)
            
            results[name] = {
                "accuracy": acc,
                "predictions": preds,
                "probabilities": pred_proba,
                "model": model
            }
            trained_models[name] = model
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42)
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            results[name] = {
                "MSE": mse, 
                "RMSE": np.sqrt(mse),
                "MAE": mae,
                "R2": r2,
                "predictions": preds,
                "model": model
            }
            trained_models[name] = model
    
    return results, trained_models, is_classification

def plot_correlation_matrix(df):
    """Correlation Heatmap for numeric features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, 
                square=True, cbar_kws={"shrink": .8})
    plt.title("📊 Correlation Heatmap", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# Popular datasets for quick access with detailed information
popular_datasets = {
    "House Prices": {
        "ref": "c/house-prices-advanced-regression-techniques",
        "description": "Predict sales prices and practice feature engineering, RFs, and gradient boosting",
        "size": "1.2 MB",
        "downloads": "1,200,000+",
        "votes": "4,500+",
        "tags": "regression, real estate, feature engineering, gradient boosting"
    },
    "Titanic": {
        "ref": "c/titanic",
        "description": "Predict survival on the Titanic and get familiar with ML basics",
        "size": "60 KB",
        "downloads": "2,000,000+",
        "votes": "8,000+",
        "tags": "classification, survival analysis, beginner, binary classification"
    },
    "Wine Quality": {
        "ref": "uciml/red-wine-quality-cortez-et-al-2009",
        "description": "Predict the quality of wine based on physicochemical tests",
        "size": "25 KB",
        "downloads": "800,000+",
        "votes": "2,500+",
        "tags": "classification, wine, quality prediction, physicochemical"
    },
    "Credit Card Fraud": {
        "ref": "mlg-ulb/creditcardfraud",
        "description": "Detect fraudulent credit card transactions using machine learning",
        "size": "150 MB",
        "downloads": "1,500,000+",
        "votes": "3,200+",
        "tags": "fraud detection, imbalanced data, anomaly detection, finance"
    },
    "Customer Segmentation": {
        "ref": "vjchoudhary7/customer-segmentation-tutorial-in-python",
        "description": "Segment customers based on their purchasing behavior and demographics",
        "size": "2.5 MB",
        "downloads": "400,000+",
        "votes": "1,800+",
        "tags": "clustering, customer analytics, segmentation, marketing"
    },
    "Employee Attrition": {
        "ref": "pavansubhasht/ibm-hr-analytics-attrition-dataset",
        "description": "Predict employee attrition and understand factors that lead to turnover",
        "size": "45 KB",
        "downloads": "600,000+",
        "votes": "2,200+",
        "tags": "hr analytics, attrition prediction, employee retention, classification"
    }
}

# Main content area based on sidebar selection
if dataset_source == "Sample Datasets":
    st.header("🎯 Quick Start - Sample Datasets")
    
    # Load sample datasets
    sample_datasets = load_sample_datasets()
    
    st.write("**Try these sample datasets that work perfectly for ML:**")

    cols = st.columns(3)
    for i, (name, info) in enumerate(sample_datasets.items()):
        with cols[i % 3]:
            with st.container():
                st.markdown(f"**{name}** ({info['type']})")
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Size:** {info['size']}")
                st.write(f"**Samples:** {info['samples']}")
                st.write(f"**Features:** {info['features']}")
                st.write(f"**Tags:** {info['tags']}")
                
                if st.button(f"🚀 Load & Run ML", key=f"sample_{i}"):
                    try:
                        with st.spinner(f"Loading {name}..."):
                            df = pd.read_csv(info['url'])
                            st.session_state['loaded_dataset'] = df
                            st.session_state['dataset_name'] = name
                            st.success(f"✅ {name} loaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load {name}: {str(e)}")

elif dataset_source == "Kaggle":
    st.header("🏆 Kaggle Datasets Explorer")
    st.write("**Browse and download popular Kaggle datasets for your ML projects:**")
    
    # Kaggle dataset selection method
    kaggle_method = st.radio(
        "Choose how to explore datasets:",
        ["Popular Datasets", "Search on Kaggle", "Download & Upload"],
        horizontal=True
    )
    
    if kaggle_method == "Popular Datasets":
        st.subheader("📊 Popular Kaggle Datasets")
        st.info("💡 **Tip:** Click 'Go to Kaggle' to view and download datasets directly from Kaggle.")
        
        # Display popular datasets
        for name, info in popular_datasets.items():
            with st.expander(f"📊 {name}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Dataset:** {name}")
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Size:** {info['size']}")
                    st.write(f"**Downloads:** {info['downloads']}")
                    st.write(f"**Votes:** {info['votes']}")
                    st.write(f"**Tags:** {info['tags']}")
                    
                    # Add Kaggle link
                    kaggle_url = f"https://www.kaggle.com/datasets/{info['ref']}"
                    st.markdown(f"**🔗 [View on Kaggle]({kaggle_url})**")
                
                with col2:
                    # Only show the Kaggle redirect button
                    if st.button(f"🔗 Go to Kaggle", key=f"popular_redirect_{info['ref']}"):
                        st.markdown(redirect_to_kaggle_dataset(info['ref']), unsafe_allow_html=True)
    
    elif kaggle_method == "Search on Kaggle":
        st.subheader("🔍 Search on Kaggle")
        st.write("**Search for specific datasets directly on Kaggle:**")

        # Simple search input and redirect
        search_query = st.text_input("Enter search terms (e.g., 'machine learning', 'stock prices', 'customer data'):", 
                                    placeholder="Type your search here...")

        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔗 Go to Kaggle Search", type="primary", key="search_button"):
                if search_query:
                    st.markdown(redirect_to_kaggle_search(search_query), unsafe_allow_html=True)
                else:
                    st.warning("Please enter search terms first!")
        
        with col2:
            if st.button("🔑 Login to Kaggle", type="secondary"):
                st.markdown(redirect_to_kaggle_login(), unsafe_allow_html=True)
        
        # Show some popular search suggestions
        st.subheader("💡 Popular Search Suggestions:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏠 Real Estate", key="suggest_real_estate"):
                st.markdown(redirect_to_kaggle_search("real estate"), unsafe_allow_html=True)
            if st.button("📈 Stock Market", key="suggest_stocks"):
                st.markdown(redirect_to_kaggle_search("stock market"), unsafe_allow_html=True)
        
        with col2:
            if st.button("🛒 E-commerce", key="suggest_ecommerce"):
                st.markdown(redirect_to_kaggle_search("e-commerce"), unsafe_allow_html=True)
            if st.button("🏥 Healthcare", key="suggest_healthcare"):
                st.markdown(redirect_to_kaggle_search("healthcare"), unsafe_allow_html=True)
        
        with col3:
            if st.button("🎮 Gaming", key="suggest_gaming"):
                st.markdown(redirect_to_kaggle_search("gaming"), unsafe_allow_html=True)
            if st.button("🌍 Climate", key="suggest_climate"):
                st.markdown(redirect_to_kaggle_search("climate"), unsafe_allow_html=True)
    
    elif kaggle_method == "Download & Upload":
        st.subheader("📥 Download & Upload Method")
        st.info("""
        **💡 How to use Kaggle datasets:**
        
        **Simple 3-step process:**
        - 🔗 **Browse & Download** - Use the options above to find datasets on Kaggle
        - 📁 **Upload CSV** - Use the file uploader below to load your data
        - 🚀 **Run ML Pipeline** - Your data will be processed automatically!
        """)
        
        # Show popular datasets with download links
        st.subheader("📊 Popular Datasets - Click to Download")
        
        # Create a grid layout for better organization
        cols = st.columns(2)
        
        for i, (name, info) in enumerate(popular_datasets.items()):
            with cols[i % 2]:
                with st.container():
                    st.markdown(f"### 📊 {name}")
                    
                    # Dataset info
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Size:** {info['size']} | **Downloads:** {info['downloads']} | **Votes:** {info['votes']}")
                    st.write(f"**Tags:** {info['tags']}")
                    
                    # Create download link
                    kaggle_url = f"https://www.kaggle.com/datasets/{info['ref']}"
                    
                    # Better styled download button
                    download_html = f"""
                    <div style="text-align: center; margin: 10px 0;">
                        <a href="{kaggle_url}" target="_blank" 
                           style="background-color: #20beff; color: white; padding: 12px 24px; 
                                  text-decoration: none; border-radius: 8px; display: inline-block; 
                                  font-weight: bold; font-size: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                            🔗 Open in Kaggle & Download
                        </a>
                    </div>
                    """
                    st.markdown(download_html, unsafe_allow_html=True)
                    
                    # Instructions
                    with st.expander("📋 Download Instructions"):
                        st.write("**Step-by-step guide:**")
                        st.write("1. **Click the blue button above** - Opens Kaggle in new tab")
                        st.write("2. **Sign in to Kaggle** (if not already signed in)")
                        st.write("3. **Click the 'Download' button** on the dataset page")
                        st.write("4. **Save the zip file** to your computer")
                        st.write("5. **Extract the zip file** to get the CSV")
                        st.write("6. **Upload the CSV below** using the file uploader")
                        st.write("7. **Run your ML pipeline!** 🚀")
                    
                    st.markdown("---")
        
        # File uploader for downloaded CSV
        st.subheader("📤 Upload Your Downloaded CSV File")
        st.info("💡 **Tip:** After downloading from Kaggle, extract the zip file and upload the CSV file here.")
        
        uploaded_file = st.file_uploader(
            "Choose your downloaded CSV file", 
            type=["csv"], 
            key="kaggle_upload",
            help="Upload the CSV file you downloaded from Kaggle"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['loaded_dataset'] = df
                st.session_state['dataset_name'] = uploaded_file.name
                st.success(f"✅ Successfully loaded {uploaded_file.name}!")
                
                # Show dataset info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Columns", f"{df.shape[1]:,}")
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                
                # Show first few rows
                st.subheader("📊 Dataset Preview")
                st.dataframe(df.head(10))
                
                st.success("🎉 **Ready for ML Pipeline!** Your dataset is loaded and ready to run machine learning analysis.")
                
            except Exception as e:
                st.error(f"❌ Failed to load CSV: {str(e)}")
                st.info("💡 **Troubleshooting:** Make sure you uploaded a valid CSV file. If the file is corrupted, try downloading it again from Kaggle.")

elif dataset_source == "Upload CSV":
    st.header("📤 Upload Your Own Dataset")
    st.write("**Upload a CSV file to run ML analysis:**")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['loaded_dataset'] = df
            st.session_state['dataset_name'] = uploaded_file.name
            st.success(f"✅ Successfully loaded {uploaded_file.name}!")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Failed to load CSV: {str(e)}")

# ML Pipeline Section
if 'loaded_dataset' in st.session_state:
    st.header("🤖 Machine Learning Pipeline")
    
    df = st.session_state['loaded_dataset']
    dataset_name = st.session_state.get('dataset_name', 'Unknown Dataset')
    
    st.success(f"✅ **{dataset_name}** is ready for ML analysis!")
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Dataset preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10))
    
    # Target column selection
    target_col = st.selectbox("🎯 Select target column", df.columns)
    
    if target_col:
        # Determine problem type
        unique_targets = df[target_col].nunique()
        is_classification = unique_targets < 20
        
        st.info(f"**Problem Type:** {'Classification' if is_classification else 'Regression'} ({unique_targets} unique values)")
        
        # Run ML Pipeline
        if st.button("🚀 Run ML Pipeline", type="primary"):
            try:
                with st.spinner("🔄 Preprocessing data..."):
                    X_train, X_test, y_train, y_test = preprocess_data(df, target_col)
                
                st.success(f"✅ Data preprocessed: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
                
                with st.spinner("🤖 Training models..."):
                    results, trained_models, is_classification = train_and_evaluate(X_train, X_test, y_train, y_test)
                
                # Display results
                st.subheader("📊 Model Results")
                
                if is_classification:
                    # Classification results
                    metrics_data = []
                    for model_name, metrics in results.items():
                        preds = metrics['predictions']
                        acc = metrics['accuracy']
                        
                        metrics_data.append({
                            'Model': model_name,
                            'Accuracy': round(acc, 3)
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Find best model
                    best_model_name = max(metrics_data, key=lambda x: x['Accuracy'])['Model']
                    best_accuracy = max(metrics_data, key=lambda x: x['Accuracy'])['Accuracy']
                    st.success(f"🏆 **{best_model_name}** is the best model with accuracy of {best_accuracy:.3f}")
                    
                else:
                    # Regression results
                    metrics_data = []
                    for model_name, metrics in results.items():
                        metrics_data.append({
                            'Model': model_name,
                            'R²': round(metrics['R2'], 3),
                            'RMSE': round(metrics['RMSE'], 3),
                            'MAE': round(metrics['MAE'], 3)
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Find best model
                    best_model_name = min(metrics_data, key=lambda x: x['RMSE'])['Model']
                    best_rmse = min(metrics_data, key=lambda x: x['RMSE'])['RMSE']
                    st.success(f"🏆 **{best_model_name}** is the best model with RMSE of {best_rmse:.3f}")
                
                # EDA Section
                st.subheader("📊 Exploratory Data Analysis")
                
                # Correlation Matrix
                corr_plot = plot_correlation_matrix(df)
                if corr_plot:
                    st.pyplot(corr_plot)
                else:
                    st.warning("Not enough numeric features for correlation analysis")
                
                # Feature Distributions
                st.subheader("📈 Feature Distributions")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    n_cols = min(3, len(numeric_cols))
                    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                    if n_rows == 1:
                        axes = [axes] if n_cols == 1 else axes
                    else:
                        axes = axes.flatten()
                    
                    for i, col in enumerate(numeric_cols):
                        if i < len(axes):
                            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                            axes[i].set_title(f"📊 {col}", fontweight='bold')
                            axes[i].set_xlabel(col)
                            axes[i].set_ylabel("Frequency")
                    
                    # Hide unused subplots
                    for idx in range(len(numeric_cols), len(axes)):
                        axes[idx].set_visible(False)
                    
                    plt.suptitle("📈 Feature Distributions", fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.success("🎉 ML Pipeline completed successfully!")
                
            except Exception as e:
                st.error(f"❌ ML Pipeline failed: {str(e)}")
                st.info("💡 Try selecting a different target column or check your data for issues")
    
    # Clear dataset
    if st.button("🗑️ Clear Dataset"):
        del st.session_state['loaded_dataset']
        del st.session_state['dataset_name']
        st.rerun()

st.header("ℹ️ How to Use")

st.info("""
**🚀 Integrated ML Pipeline:**
1. **Load Dataset** - Click "Load & Run ML" on sample datasets or "Download & Run ML" on Kaggle datasets
2. **Select Target** - Choose the column you want to predict
3. **Run ML Pipeline** - Click "Run ML Pipeline" to train models and get results
4. **View Results** - See model performance, EDA, and visualizations

**🎯 Features:**
- **Direct ML Pipeline** - No need to download and re-upload
- **Automatic Preprocessing** - Handles missing values and encoding
- **Multiple Models** - Logistic Regression, Random Forest, Linear Regression
- **Comprehensive EDA** - Correlation matrix, feature distributions
- **Model Comparison** - See which model performs best

**💡 Pro Tips:**
- Start with sample datasets for quick testing
- Try popular Kaggle datasets for real-world data
- Use custom search for specific datasets
- Check the target column selection for best results
""")

st.header("🚀 Benefits")

st.success("""
**🎉 What Makes This Special:**
- **No Downloads Needed** - Datasets go directly to ML pipeline
- **One-Click ML** - From dataset to results in seconds
- **Real-World Data** - Access to thousands of Kaggle datasets
- **Complete Pipeline** - Preprocessing, training, evaluation, and visualization
- **User-Friendly** - Simple interface for complex ML tasks

**Your integrated ML pipeline is ready to work with any dataset!** 🚀
""")
