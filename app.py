import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Load model
try:
    with open('model_pipeline.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")

def predict_price(brand, processor_brand, processor_name, processor_gnrtn, ram_gb, ram_type, 
                  ssd, hdd, os, graphic_card_gb, weight, warranty, touchscreen, msoffice):
    
    # Mapping for processor generation to match training data
    map_gnrtn = {
        '10th': 10, '11th': 11, '12th': 12, '7th': 7, 
        '8th': 8, '9th': 9, '4th': 4, 'Not Available': 0
    }
    gnrtn_value = map_gnrtn.get(processor_gnrtn, 0)
    
    input_data = pd.DataFrame([{
        'brand': brand,
        'processor_brand': processor_brand,
        'processor_name': processor_name,
        'processor_gnrtn': gnrtn_value,
        'ram_gb': int(ram_gb),
        'ram_type': ram_type,
        'ssd': int(ssd),
        'hdd': int(hdd),
        'os': os,
        'graphic_card_gb': int(graphic_card_gb),
        'weight': weight,
        'warranty': int(warranty),
        'Touchscreen': touchscreen,
        'msoffice': msoffice
    }])
    
    try:
        prediction = model.predict(input_data)[0]
        # Ensure prediction is positive
        prediction = max(0, prediction)
        return f"Predicted Price: ₹{prediction:,.2f}"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Create interface
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(["ASUS", "Lenovo", "HP", "DELL", "acer", "MSI", "Avita", "APPLE"], label="Brand"),
        gr.Dropdown(["Intel", "AMD", "M1"], label="Processor Brand"),
        gr.Dropdown(["Core i5", "Core i3", "Core i7", "Core i9", "Ryzen 5", "Ryzen 7", "Ryzen 3", "M1"], label="Processor Name"),
        gr.Dropdown(["10th", "11th", "12th", "7th", "8th", "9th", "4th", "Not Available"], label="Processor Generation"),
        gr.Dropdown([4, 8, 16, 32, 64], label="RAM (GB)"),
        gr.Dropdown(["DDR4", "DDR5", "LPDDR3", "LPDDR4", "LPDDR4X"], label="RAM Type"),
        gr.Number(label="SSD (GB)", value=512),
        gr.Number(label="HDD (TB)", value=0),
        gr.Dropdown(["Windows", "Mac", "DOS"], label="Operating System"),
        gr.Dropdown([0, 2, 4, 6, 8], label="Graphics Card (GB)"),
        gr.Dropdown(["Casual", "ThinNlight", "Gaming"], label="Weight Category"),
        gr.Dropdown([0, 1, 2, 3], label="Warranty (Years)"),
        gr.Dropdown(["No", "Yes"], label="Touchscreen"),
        gr.Dropdown(["No", "Yes"], label="MS Office")
    ],
    outputs=gr.Textbox(label="Predicted Price"),
    title="🚀 Advanced Laptop Price Predictor",
    description="This version uses a Gradient Boosting Regressor with Log-Transformation for high accuracy predictions.",
    theme="glass" # Using a nice theme
)

if __name__ == "__main__":
    demo.launch()