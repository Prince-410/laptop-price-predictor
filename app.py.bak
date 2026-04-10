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
        prediction = max(0, prediction)
        return f"₹{prediction:,.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================
#  Premium Dark Theme — works WITH Gradio, not against it
# ============================================================
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    # Global canvas
    body_background_fill="linear-gradient(160deg, #0a0a1a 0%, #0d1025 40%, #0f0a20 100%)",
    body_text_color="#cbd5e1",

    # Blocks / panels
    block_background_fill="rgba(15, 23, 42, 0.65)",
    block_border_color="rgba(99, 102, 241, 0.15)",
    block_border_width="1px",
    block_label_text_color="#a5b4fc",
    block_title_text_color="#e2e8f0",
    block_shadow="0 4px 24px rgba(0,0,0,0.4)",
    block_radius="16px",
    block_label_background_fill="transparent",

    # Inputs
    input_background_fill="rgba(15, 23, 42, 0.8)",
    input_border_color="rgba(99, 102, 241, 0.25)",
    input_border_width="1px",
    input_radius="10px",
    input_shadow="0 2px 8px rgba(0,0,0,0.25)",

    # Buttons
    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",
    button_primary_shadow="0 4px 20px rgba(139, 92, 246, 0.4)",

    button_secondary_background_fill="rgba(30, 41, 59, 0.6)",
    button_secondary_text_color="#c7d2fe",
    button_secondary_border_color="rgba(99, 102, 241, 0.3)",

    # Borders & shadows
    border_color_primary="rgba(99, 102, 241, 0.2)",
    shadow_spread="8px",
)

custom_css = """
/* ── Header ───────────────────────────────────────── */
.header-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5rem 0 0.25rem 0;
    letter-spacing: -0.5px;
    animation: titleIn 0.7s ease-out both;
}
.header-sub {
    text-align: center;
    color: #64748b;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 1.5rem;
    animation: titleIn 0.7s 0.15s ease-out both;
}
@keyframes titleIn {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Section Headers ──────────────────────────────── */
.section-label {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: #818cf8 !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.15) !important;
    padding-bottom: 8px !important;
    margin-bottom: 4px !important;
}

/* ── Predict Button ───────────────────────────────── */
.predict-btn {
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    padding: 14px 0 !important;
    border-radius: 12px !important;
    transition: all 0.25s ease !important;
}
.predict-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.5) !important;
}

/* ── Price Output ─────────────────────────────────── */
.price-output textarea {
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    text-align: center !important;
    color: #34d399 !important;
    background: transparent !important;
    border: none !important;
    text-shadow: 0 0 30px rgba(52, 211, 153, 0.3);
}

/* ── Info Card ────────────────────────────────────── */
.info-card {
    margin-top: 16px;
    padding: 14px 16px;
    border-left: 3px solid #6366f1;
    background: rgba(99, 102, 241, 0.06);
    border-radius: 0 10px 10px 0;
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.5;
}

/* ── Feature chips (for radio buttons) ────────────── */
.gradio-container .wrap .wrap-inner {
    gap: 8px !important;
}
"""

# ============================================================
#  App Layout
# ============================================================
with gr.Blocks(css=custom_css, theme=theme, title="Laptop Price Predictor") as demo:

    # ── Header
    gr.HTML("<div class='header-title'>💻 Laptop Price Predictor</div>")
    gr.HTML("<div class='header-sub'>AI-powered valuation using Gradient Boosting · Trained on real market data</div>")

    with gr.Row(equal_height=False):

        # ═══════════════ LEFT — Input Form ═══════════════
        with gr.Column(scale=3):

            # ── Core Specs
            gr.Markdown("🏷️ &nbsp; BRAND & PROCESSOR", elem_classes="section-label")
            with gr.Row():
                brand = gr.Dropdown(
                    ["ASUS", "Lenovo", "HP", "DELL", "acer", "MSI", "Avita", "APPLE"],
                    label="Brand", value="ASUS"
                )
                proc_brand = gr.Dropdown(
                    ["Intel", "AMD", "M1"],
                    label="Processor Brand", value="Intel"
                )
            with gr.Row():
                proc_name = gr.Dropdown(
                    ["Core i5", "Core i3", "Core i7", "Core i9", "Ryzen 5", "Ryzen 7", "Ryzen 3", "M1"],
                    label="Processor", value="Core i5"
                )
                proc_gnrtn = gr.Dropdown(
                    ["10th", "11th", "12th", "7th", "8th", "9th", "4th", "Not Available"],
                    label="Generation", value="11th"
                )

            # ── Memory & Storage
            gr.Markdown("🧠 &nbsp; MEMORY & STORAGE", elem_classes="section-label")
            with gr.Row():
                ram_gb = gr.Dropdown(["4", "8", "16", "32", "64"], label="RAM (GB)", value="8")
                ram_type = gr.Dropdown(["DDR4", "DDR5", "LPDDR3", "LPDDR4", "LPDDR4X"], label="RAM Type", value="DDR4")
            with gr.Row():
                ssd = gr.Dropdown(["0", "128", "256", "512", "1024", "2048"], label="SSD (GB)", value="512")
                hdd = gr.Dropdown(["0", "1", "2"], label="HDD (TB)", value="0")

            # ── Build & Extras
            gr.Markdown("⚙️ &nbsp; BUILD & EXTRAS", elem_classes="section-label")
            with gr.Row():
                os = gr.Dropdown(["Windows", "Mac", "DOS"], label="OS", value="Windows")
                gpu = gr.Dropdown(["0", "2", "4", "6", "8"], label="GPU VRAM (GB)", value="0")
            with gr.Row():
                weight = gr.Dropdown(["Casual", "ThinNlight", "Gaming"], label="Weight Category", value="Casual")
                warranty = gr.Dropdown(["0", "1", "2", "3"], label="Warranty (Years)", value="1")
            with gr.Row():
                touchscreen = gr.Radio(["No", "Yes"], label="Touchscreen", value="No")
                msoffice = gr.Radio(["No", "Yes"], label="MS Office Included", value="No")

        # ═══════════════ RIGHT — Output Panel ═══════════════
        with gr.Column(scale=2):

            gr.Markdown("💰 &nbsp; ESTIMATED VALUE", elem_classes="section-label")
            output_price = gr.Textbox(
                label="Market Price Estimate",
                elem_classes="price-output",
                interactive=False,
                lines=1,
                placeholder="Click predict..."
            )
            predict_btn = gr.Button(
                "🔮  Predict Price",
                variant="primary",
                elem_classes="predict-btn",
                size="lg"
            )

            gr.HTML("""
                <div class='info-card'>
                    <strong>ℹ️ How it works:</strong><br>
                    This model uses a <strong>Gradient Boosting Regressor</strong> trained on
                    real laptop market data. It analyses 14 hardware & software features to
                    estimate a fair market price. Predictions may vary by ±5-10% based on
                    regional pricing and availability.
                </div>
            """)

    # ── Event binding
    predict_btn.click(
        fn=predict_price,
        inputs=[brand, proc_brand, proc_name, proc_gnrtn, ram_gb, ram_type,
                ssd, hdd, os, gpu, weight, warranty, touchscreen, msoffice],
        outputs=output_price
    )

if __name__ == "__main__":
    demo.launch()