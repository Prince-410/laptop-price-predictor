import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('model_pipeline.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# App title and description
st.title("💻 Laptop Price Predictor")
st.markdown("### Get instant price predictions based on laptop specifications")
st.markdown("---")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Specifications")
    
    brand = st.selectbox(
        "Brand",
        ["ASUS", "Lenovo", "HP", "DELL", "acer", "MSI", "Avita"]
    )
    
    processor_brand = st.selectbox(
        "Processor Brand",
        ["Intel", "AMD", "M1"]
    )
    
    processor_name = st.selectbox(
        "Processor Name",
        ["Core i5", "Core i3", "Core i7", "Core i9", "Ryzen 5", "Ryzen 7", 
         "Ryzen 3", "Ryzen 9", "M1", "Celeron Dual", "Pentium Quad"]
    )
    
    ram_gb = st.selectbox(
        "RAM (GB)",
        [4, 8, 16, 32]
    )
    
    ram_type = st.selectbox(
        "RAM Type",
        ["DDR4", "DDR5", "LPDDR3", "LPDDR4", "LPDDR4X"]
    )
    
    os = st.selectbox(
        "Operating System",
        ["Windows", "Mac", "DOS"]
    )

with col2:
    st.subheader("Storage & Graphics")
    
    ssd = st.number_input(
        "SSD (GB)",
        min_value=0,
        max_value=2048,
        value=512,
        step=128
    )
    
    hdd = st.number_input(
        "HDD (TB)",
        min_value=0,
        max_value=2,
        value=0,
        step=1
    )
    
    graphic_card_gb = st.selectbox(
        "Graphics Card (GB)",
        [0, 2, 4, 6, 8]
    )
    
    weight = st.selectbox(
        "Weight Category",
        ["Casual", "ThinNlight", "Gaming"]
    )
    
    warranty = st.selectbox(
        "Warranty (Years)",
        [0, 1, 2, 3]
    )
    
    touchscreen = st.selectbox(
        "Touchscreen",
        ["No", "Yes"]
    )
    
    msoffice = st.selectbox(
        "MS Office",
        ["No", "Yes"]
    )

st.markdown("---")

# Predict button
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    if model is None:
        st.error("Model not loaded. Please ensure model_pipeline.pkl exists.")
    else:
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'brand': brand,
            'processor_brand': processor_brand,
            'processor_name': processor_name,
            'ram_gb': ram_gb,
            'ram_type': ram_type,
            'ssd': ssd,
            'hdd': hdd,
            'os': os,
            'graphic_card_gb': graphic_card_gb,
            'weight': weight,
            'warranty': warranty,
            'Touchscreen': touchscreen,
            'msoffice': msoffice
        }])
        
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success("### Prediction Complete!")
            st.metric(
                label="Predicted Laptop Price",
                value=f"₹{prediction:,.2f}",
                delta=None
            )
            
            # Show input summary
            with st.expander("📊 View Input Summary"):
                st.json({
                    "Brand": brand,
                    "Processor": f"{processor_brand} {processor_name}",
                    "RAM": f"{ram_gb} GB {ram_type}",
                    "Storage": f"{ssd} GB SSD + {hdd} TB HDD",
                    "Graphics": f"{graphic_card_gb} GB",
                    "OS": os,
                    "Weight": weight,
                    "Warranty": f"{warranty} years",
                    "Touchscreen": touchscreen,
                    "MS Office": msoffice
                })
                
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Footer
st.markdown("---")
st.markdown("Built with ❤ using Streamlit | Machine Learning Model: RandomForest")