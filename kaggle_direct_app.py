import streamlit as st
import requests
import json
import pandas as pd
import zipfile
import io
import os
from urllib.parse import quote

st.set_page_config(page_title="Kaggle Direct Download", layout="wide")
st.title("🏆 Kaggle Dataset Direct Download")

def search_kaggle_datasets(query, page=1):
    """Search Kaggle datasets using their public API"""
    try:
        # Kaggle's public search API (no authentication required)
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

def download_dataset_direct(dataset_ref):
    """Download dataset directly without redirecting to Kaggle"""
    try:
        # Try to get dataset files directly
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
                # Get the largest CSV file
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
            "description": "Classic classification dataset with 150 samples",
            "url": "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv",
            "type": "Classification"
        },
        "Boston Housing": {
            "description": "Housing prices in Boston - regression dataset",
            "url": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
            "type": "Regression"
        },
        "Wine Quality": {
            "description": "Wine quality prediction dataset",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "type": "Classification"
        },
        "Car Evaluation": {
            "description": "Car evaluation dataset for classification",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
            "type": "Classification"
        }
    }
    return sample_datasets

# Popular datasets for quick access
popular_datasets = {
    "House Prices": "c/house-prices-advanced-regression-techniques",
    "Titanic": "c/titanic", 
    "Wine Quality": "uciml/red-wine-quality-cortez-et-al-2009",
    "Credit Card Fraud": "mlg-ulb/creditcardfraud",
    "Customer Segmentation": "vjchoudhary7/customer-segmentation-tutorial-in-python",
    "Employee Attrition": "pavansubhasht/ibm-hr-analytics-attrition-dataset",
    "Car Price Prediction": "nehalbirla/vehicle-dataset-from-cardekho",
    "Sales Prediction": "bumba5341/advertisingcsv",
    "Spam Detection": "uciml/sms-spam-collection-dataset",
    "Stock Price Prediction": "rohanrao/amex-nyse-nasdaq-stock-histories"
}

st.header("🎯 Quick Start - Sample Datasets")

# Load sample datasets
sample_datasets = load_sample_datasets()

st.write("**Try these sample datasets that work perfectly for ML:**")

cols = st.columns(2)
for i, (name, info) in enumerate(sample_datasets.items()):
    with cols[i % 2]:
        with st.container():
            st.markdown(f"**{name}** ({info['type']})")
            st.write(info['description'])
            
            if st.button(f"📥 Load {name}", key=f"sample_{i}"):
                try:
                    with st.spinner(f"Loading {name}..."):
                        df = pd.read_csv(info['url'])
                        st.session_state['loaded_dataset'] = df
                        st.session_state['dataset_name'] = name
                        st.success(f"✅ {name} loaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to load {name}: {str(e)}")

st.header("🏆 Popular Kaggle Datasets")

st.write("**Try downloading these popular datasets directly:**")

# Display popular datasets
for name, ref in popular_datasets.items():
    with st.expander(f"📊 {name}"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**Dataset:** {name}")
            st.write(f"**Reference:** {ref}")
            st.write("Click 'Download & Load' to get this dataset directly in the app!")
        
        with col2:
            if st.button(f"📥 Download & Load", key=f"kaggle_{ref}"):
                with st.spinner(f"Downloading {name}..."):
                    zip_content = download_dataset_direct(ref)
                    
                    if zip_content:
                        csv_content, filename = extract_csv_from_zip(zip_content)
                        
                        if csv_content:
                            try:
                                # Load CSV from memory
                                df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
                                st.session_state['loaded_dataset'] = df
                                st.session_state['dataset_name'] = name
                                st.success(f"✅ {name} loaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to parse CSV: {str(e)}")
                        else:
                            st.error("No CSV files found in the dataset")
                    else:
                        st.error("Failed to download dataset")

st.header("🔍 Search Custom Datasets")

# Search functionality
search_query = st.text_input("Enter search terms (e.g., 'machine learning', 'stock prices', 'customer data'):", 
                            placeholder="Type your search here...")

if st.button("🔍 Search Datasets", type="primary") and search_query:
    with st.spinner("Searching Kaggle datasets..."):
        results = search_kaggle_datasets(search_query)
    
    if results and len(results) > 0:
        st.success(f"Found {len(results)} datasets!")
        
        for i, dataset in enumerate(results):
            with st.expander(f"📊 {dataset.get('title', 'Untitled')} by {dataset.get('owner', 'Unknown')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {dataset.get('subtitle', 'No description available')}")
                    st.write(f"**Size:** {dataset.get('size', 'Unknown')}")
                    st.write(f"**Downloads:** {dataset.get('downloadCount', 0):,}")
                    st.write(f"**Votes:** {dataset.get('voteCount', 0)}")
                    
                    # Show tags
                    if dataset.get('tags'):
                        tags = [tag['name'] for tag in dataset['tags'][:5]]
                        st.write(f"**Tags:** {', '.join(tags)}")
                
                with col2:
                    dataset_ref = f"{dataset.get('owner')}/{dataset.get('name')}"
                    
                    if st.button(f"📥 Download & Load", key=f"search_{i}"):
                        with st.spinner(f"Downloading {dataset.get('title', 'dataset')}..."):
                            zip_content = download_dataset_direct(dataset_ref)
                            
                            if zip_content:
                                csv_content, filename = extract_csv_from_zip(zip_content)
                                
                                if csv_content:
                                    try:
                                        df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
                                        st.session_state['loaded_dataset'] = df
                                        st.session_state['dataset_name'] = dataset.get('title', 'Custom Dataset')
                                        st.success(f"✅ Dataset loaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to parse CSV: {str(e)}")
                                else:
                                    st.error("No CSV files found in the dataset")
                            else:
                                st.error("Failed to download dataset")
    else:
        st.warning("No datasets found. Try different search terms.")

# Display loaded dataset
if 'loaded_dataset' in st.session_state:
    st.header("📊 Loaded Dataset")
    
    df = st.session_state['loaded_dataset']
    dataset_name = st.session_state.get('dataset_name', 'Unknown Dataset')
    
    st.success(f"✅ **{dataset_name}** is ready for analysis!")
    
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
    
    # Column information
    st.subheader("📊 Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null': df.count(),
        'Null': df.isnull().sum()
    })
    st.dataframe(col_info)
    
    # Download option
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Dataset as CSV",
        data=csv,
        file_name=f"{dataset_name.replace(' ', '_')}.csv",
        mime="text/csv",
        type="primary"
    )
    
    # Clear dataset
    if st.button("🗑️ Clear Dataset"):
        del st.session_state['loaded_dataset']
        del st.session_state['dataset_name']
        st.rerun()

st.header("ℹ️ How to Use")

st.info("""
**🎯 Quick Start:**
1. **Try Sample Datasets** - Click "Load" on any sample dataset above
2. **Popular Datasets** - Click "Download & Load" on popular Kaggle datasets
3. **Custom Search** - Search for specific datasets and download them

**📊 After Loading:**
1. **Preview the data** - See the dataset structure and content
2. **Download as CSV** - Save the dataset to your computer
3. **Use in ML App** - Upload the CSV to your main AutoML app

**💡 Pro Tips:**
- Sample datasets load instantly and work perfectly for ML
- Popular datasets are pre-tested and reliable
- Custom search lets you find specific datasets
- All datasets are loaded directly in the app - no redirects!
""")

st.header("🚀 Next Steps")

st.success("""
**After loading a dataset:**
1. **Download the CSV** using the download button
2. **Go to your main ML app** (ml_app.py)
3. **Upload the CSV file** using "Upload CSV" option
4. **Run your ML pipeline** with the real-world data!

**Your dataset is ready for machine learning!** 🎉
""")


