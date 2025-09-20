import streamlit as st
import os
import pandas as pd
import shutil

# Set Kaggle configuration BEFORE any imports
os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath('.kaggle')

st.set_page_config(page_title="Kaggle Test", layout="wide")
st.title("🏆 Kaggle API Test")

def test_kaggle_connection():
    """Test Kaggle API connection"""
    try:
        # Set up Kaggle API configuration
        local_kaggle_dir = ".kaggle"
        kaggle_file = os.path.join(local_kaggle_dir, "kaggle.json")
        
        if os.path.exists(kaggle_file):
            st.success("✅ Found kaggle.json in local .kaggle folder")
            os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath(local_kaggle_dir)
        else:
            st.error("❌ No kaggle.json found in .kaggle folder")
            return False
        
        # Test API connection
        with st.spinner("🔗 Testing API connection..."):
            import kaggle
            kaggle.api.authenticate()
        st.success("✅ Kaggle API authenticated successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return False

def load_kaggle_dataset(dataset_name, max_rows=1000):
    """Load dataset from Kaggle"""
    try:
        # Create temp directory
        temp_dir = "temp_kaggle_test"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        # Download dataset
        with st.spinner(f"📥 Downloading {dataset_name}..."):
            import kaggle
            kaggle.api.dataset_download_files(dataset_name, path=temp_dir, unzip=True)
        
        # Find CSV files
        csv_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            st.error("❌ No CSV files found in dataset")
            return None
        
        # Load the largest CSV file
        csv_file = max(csv_files, key=os.path.getsize)
        st.info(f"📊 Loading: {os.path.basename(csv_file)}")
        
        df = pd.read_csv(csv_file)
        
        # Limit rows for performance
        if len(df) > max_rows:
            df = df.head(max_rows)
            st.info(f"📈 Showing first {max_rows} rows (dataset has {len(df)} total rows)")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Download Error: {str(e)}")
        if os.path.exists("temp_kaggle_test"):
            shutil.rmtree("temp_kaggle_test")
        return None

# Main app
st.header("🔧 Kaggle API Test")

# Test connection
if st.button("🔗 Test Kaggle Connection", type="primary"):
    test_kaggle_connection()

st.header("📊 Load Kaggle Dataset")

# Popular datasets
popular_datasets = {
    "House Prices (Regression)": "c/house-prices-advanced-regression-techniques",
    "Titanic (Classification)": "c/titanic",
    "Wine Quality (Classification)": "uciml/red-wine-quality-cortez-et-al-2009",
    "Credit Card Fraud (Classification)": "mlg-ulb/creditcardfraud"
}

# Dataset selection
dataset_choice = st.selectbox("Choose a dataset:", list(popular_datasets.keys()))
dataset_name = popular_datasets[dataset_choice]

# Max rows
max_rows = st.slider("Max rows to load", 100, 5000, 1000)

# Load button
if st.button("📥 Load Dataset", type="primary"):
    df = load_kaggle_dataset(dataset_name, max_rows)
    
    if df is not None:
        st.success(f"✅ Successfully loaded {dataset_name}!")
        
        # Display dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Show dataset preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10))
        
        # Show column info
        st.subheader("📊 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null': df.count(),
            'Null': df.isnull().sum()
        })
        st.dataframe(col_info)
        
        # Show basic statistics
        st.subheader("📈 Basic Statistics")
        st.dataframe(df.describe())

st.header("ℹ️ Instructions")
st.info("""
1. **First**: Click "Test Kaggle Connection" to verify your API works
2. **Then**: Select a dataset and click "Load Dataset"
3. **Finally**: Explore the loaded data!

**Note**: Make sure you have a valid kaggle.json file in the .kaggle folder.
""")
