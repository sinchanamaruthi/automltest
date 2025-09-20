import streamlit as st
import requests
import json
import pandas as pd
from urllib.parse import quote

st.set_page_config(page_title="Kaggle Dataset Search", layout="wide")
st.title("🔍 Kaggle Dataset Search & Download")

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

def get_dataset_info(dataset_ref):
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

def create_download_links(dataset_ref, dataset_title):
    """Create download links for the dataset"""
    base_url = "https://www.kaggle.com/datasets"
    
    return {
        'dataset_page': f"{base_url}/{dataset_ref}",
        'download_all': f"{base_url}/{dataset_ref}/download",
        'download_zip': f"{base_url}/{dataset_ref}/download?datasetVersionNumber=1"
    }

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

st.header("🏆 Popular Datasets")

# Display popular datasets in a grid
cols = st.columns(3)
for i, (name, ref) in enumerate(popular_datasets.items()):
    with cols[i % 3]:
        with st.container():
            st.markdown(f"**{name}**")
            links = create_download_links(ref, name)
            
            col1, col2 = st.columns(2)
            with col1:
                st.link_button("📊 View", links['dataset_page'])
            with col2:
                st.link_button("⬇️ Download", links['download_zip'])
            
            st.markdown("---")

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
                    links = create_download_links(dataset_ref, dataset.get('title', ''))
                    
                    st.link_button("📊 View Dataset", links['dataset_page'], use_container_width=True)
                    st.link_button("⬇️ Download ZIP", links['download_zip'], use_container_width=True)
                    
                    # Show dataset reference
                    st.code(dataset_ref)
    else:
        st.warning("No datasets found. Try different search terms.")

st.header("ℹ️ How to Use")

st.info("""
**🔍 Search for Datasets:**
1. Enter keywords in the search box (e.g., "machine learning", "stock prices")
2. Click "Search Datasets" to find relevant datasets
3. Click "View Dataset" to see details on Kaggle
4. Click "Download ZIP" to download the dataset

**🏆 Popular Datasets:**
- Click on any popular dataset above for quick access
- These are commonly used datasets for ML projects

**📥 Download Instructions:**
1. Click the download link
2. You'll be redirected to Kaggle
3. Sign in to Kaggle (free account)
4. Download the dataset
5. Upload the CSV file to this app for analysis

**💡 Pro Tips:**
- Use specific keywords for better results
- Check the dataset size before downloading
- Look at download counts and votes for quality indicators
""")

st.header("🚀 Next Steps")

st.success("""
**After downloading a dataset:**
1. **Upload the CSV file** using the "Upload CSV" option in the main app
2. **Run your ML pipeline** with real-world data
3. **Explore and analyze** the dataset with our AutoML features

**Your AutoML app is ready to work with any Kaggle dataset!** 🎉
""")


