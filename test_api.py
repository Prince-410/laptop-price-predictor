import requests
import json

# Test the API
url = 'http://localhost:5000/predict'

# Sample laptop configuration
laptop_config = {
    'brand': 'ASUS',
    'processor_brand': 'Intel',
    'processor_name': 'Core i5',
    'ram_gb': 8,
    'ram_type': 'DDR4',
    'ssd': 512,
    'hdd': 0,
    'os': 'Windows',
    'graphic_card_gb': 2,
    'weight': 'Casual',
    'warranty': 1,
    'Touchscreen': 'No',
    'msoffice': 'Yes'
}

print("Testing Laptop Price Prediction API...")
print(f"\nInput Configuration:")
print(json.dumps(laptop_config, indent=2))

try:
    response = requests.post(url, json=laptop_config)
    result = response.json()
    
    print(f"\nAPI Response:")
    print(json.dumps(result, indent=2))
    
    if result.get('success'):
        print(f"\n✓ Predicted Price: ₹{result['predicted_price']:,.2f}")
    else:
        print(f"\n✗ Error: {result.get('error')}")
        
except Exception as e:
    print(f"\n✗ Connection Error: {e}")
    print("Make sure the Flask server is running (python app.py)")
