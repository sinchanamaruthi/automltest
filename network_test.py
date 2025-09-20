import requests
import socket

def test_network():
    print("🔍 Testing network connectivity...")
    
    # Test 1: DNS Resolution
    try:
        ip = socket.gethostbyname('www.kaggle.com')
        print(f"✅ DNS Resolution: www.kaggle.com → {ip}")
    except Exception as e:
        print(f"❌ DNS Resolution failed: {e}")
        return False
    
    # Test 2: HTTP Connection
    try:
        response = requests.get('https://www.kaggle.com', timeout=10)
        print(f"✅ HTTP Connection: Status {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ HTTP Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_network()
    if success:
        print("\n🎉 Network is working! Kaggle should be accessible.")
    else:
        print("\n💥 Network issue detected. Check your internet connection.")




