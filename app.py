import gradio as gr
import pandas as pd
import pickle

# Load model
with open('model_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_price(brand, processor_brand, processor_name, ram_gb, ram_type, 
                  ssd, hdd, os, graphic_card_gb, weight, warranty, touchscreen, msoffice):
    
    input_data = pd.DataFrame([{
        'brand': brand,
        'processor_brand': processor_brand,
        'processor_name': processor_name,
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
    
    prediction = model.predict(input_data)[0]
    return f"Predicted Price: ₹{prediction:,.2f}"

# Create interface
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(["ASUS", "Lenovo", "HP", "DELL", "acer", "MSI", "Avita"], label="Brand"),
        gr.Dropdown(["Intel", "AMD", "M1"], label="Processor Brand"),
        gr.Dropdown(["Core i5", "Core i3", "Core i7", "Core i9", "Ryzen 5", "Ryzen 7", "Ryzen 3", "M1"], label="Processor Name"),
        gr.Dropdown([4, 8, 16, 32], label="RAM (GB)"),
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
    outputs=gr.Textbox(label="Predicted Laptop Price"),
    title="💻 Laptop Price Predictor",
    description="Get instant laptop price predictions based on specifications"
)

demo.launch()