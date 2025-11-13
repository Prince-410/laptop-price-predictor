from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model pipeline
try:
    with open('model_pipeline.pkl', 'rb') as file:
        model = pickle.load(file)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("Run the training code first to generate model_pipeline.pkl")
    model = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction (JSON input)"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Train the model first.'}), 500
        
        # Get JSON data
        data = request.get_json()
        
        # Create DataFrame with exact column names from your notebook
        input_df = pd.DataFrame([{
            'brand': data.get('brand', 'ASUS'),
            'processor_brand': data.get('processor_brand', 'Intel'),
            'processor_name': data.get('processor_name', 'Core i5'),
            'ram_gb': int(data.get('ram_gb', 8)),
            'ram_type': data.get('ram_type', 'DDR4'),
            'ssd': int(data.get('ssd', 512)),
            'hdd': int(data.get('hdd', 0)),
            'os': data.get('os', 'Windows'),
            'graphic_card_gb': int(data.get('graphic_card_gb', 2)),
            'weight': data.get('weight', 'Casual'),
            'warranty': int(data.get('warranty', 1)),
            'Touchscreen': data.get('Touchscreen', 'No'),
            'msoffice': data.get('msoffice', 'No')
        }])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'predicted_price': round(prediction, 2),
            'currency': 'INR',
            'input_features': data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/predict_form', methods=['POST'])
def predict_form():
    """Endpoint for form-based prediction"""
    try:
        if model is None:
            return render_template('index.html', 
                                 prediction_text='Error: Model not loaded')
        
        # Create DataFrame from form data
        input_df = pd.DataFrame([{
            'brand': request.form.get('brand'),
            'processor_brand': request.form.get('processor_brand'),
            'processor_name': request.form.get('processor_name'),
            'ram_gb': int(request.form.get('ram_gb')),
            'ram_type': request.form.get('ram_type'),
            'ssd': int(request.form.get('ssd')),
            'hdd': int(request.form.get('hdd')),
            'os': request.form.get('os'),
            'graphic_card_gb': int(request.form.get('graphic_card_gb')),
            'weight': request.form.get('weight'),
            'warranty': int(request.form.get('warranty')),
            'Touchscreen': request.form.get('Touchscreen'),
            'msoffice': request.form.get('msoffice')
        }])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return render_template('index.html', 
                             prediction_text=f'Predicted Laptop Price: ₹{prediction:,.2f}')
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
