import os
import kaggle
import pandas as pd
import shutil

def test_kaggle():
    print("🔍 Testing Kaggle API...")
    
    try:
        # Set up Kaggle API configuration
        local_kaggle_dir = ".kaggle"
        kaggle_file = os.path.join(local_kaggle_dir, "kaggle.json")
        
        if os.path.exists(kaggle_file):
            print("✅ Found kaggle.json in local .kaggle folder")
            os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath(local_kaggle_dir)
        else:
            print("❌ No kaggle.json found in .kaggle folder")
            return False
        
        # Test API connection
        print("🔗 Testing API connection...")
        kaggle.api.authenticate()
        print("✅ Kaggle API authenticated successfully!")
        
        # Test dataset download
        print("📥 Testing dataset download...")
        dataset_name = "c/house-prices-advanced-regression-techniques"
        
        # Create temp directory
        temp_dir = "temp_test"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        # Download dataset
        kaggle.api.dataset_download_files(dataset_name, path=temp_dir, unzip=True)
        print("✅ Dataset downloaded successfully!")
        
        # Find and load CSV
        csv_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if csv_files:
            csv_file = max(csv_files, key=os.path.getsize)  # Largest file
            print(f"📊 Loading dataset: {csv_file}")
            
            df = pd.read_csv(csv_file)
            print(f"✅ Dataset loaded successfully!")
            print(f"📈 Shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)[:5]}...")  # First 5 columns
            
            # Clean up
            shutil.rmtree(temp_dir)
            print("🧹 Cleaned up temporary files")
            
            return True
        else:
            print("❌ No CSV files found in dataset")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_kaggle()
    if success:
        print("\n🎉 Kaggle integration test PASSED!")
        print("✅ Your Kaggle API is working correctly!")
    else:
        print("\n💥 Kaggle integration test FAILED!")
        print("❌ Please check your kaggle.json file and API credentials")




