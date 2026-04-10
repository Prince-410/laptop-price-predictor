document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // UI Elements
    const btn = document.querySelector('.btn-predict');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.loading-spinner');
    const resultCard = document.getElementById('resultCard');
    const priceOutput = document.getElementById('priceOutput');

    // Show Loading
    btn.disabled = true;
    btnText.style.display = 'none';
    spinner.style.display = 'block';
    
    // Collect Data from all fields
    const data = {
        brand: document.getElementById('brand').value,
        processor_brand: document.getElementById('processor_brand').value,
        processor_name: document.getElementById('processor_name').value,
        processor_gnrtn: document.getElementById('processor_gnrtn').value,
        ram_gb: parseInt(document.getElementById('ram_gb').value),
        ram_type: document.getElementById('ram_type').value,
        ssd: parseInt(document.getElementById('ssd').value),
        hdd: parseInt(document.getElementById('hdd').value),
        os: document.getElementById('os').value,
        graphic_card_gb: parseInt(document.getElementById('graphic_card_gb').value),
        weight: document.getElementById('weight').value,
        warranty: parseInt(document.getElementById('warranty').value),
        touchscreen: document.querySelector('input[name="touchscreen"]:checked').value,
        msoffice: document.querySelector('input[name="msoffice"]:checked').value
    };

    console.log("Input data:", data);

    try {
        // Change this URL to your Backend API (Render/HuggingFace) once deployed
        const API_URL = 'https://YOUR_BACKEND_URL/predict'; 
        
        // This is where you connect to your Gradio/FastAPI backend
        /*
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        const formattedPrice = result.formatted_price;
        */
        
        // Simulation for visual testing until backend is connected
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        const mockPrice = Math.floor(Math.random() * (150000 - 30000) + 30000);
        const formattedPrice = new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            maximumFractionDigits: 0
        }).format(mockPrice);

        priceOutput.innerText = formattedPrice;
        resultCard.classList.add('active');
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'end' });

    } catch (error) {
        console.error('Error:', error);
        alert('Could not connect to the Prediction AI. Please ensure the backend is running.');
    } finally {
        // Hide Loading
        btn.disabled = false;
        btnText.style.display = 'block';
        spinner.style.display = 'none';
    }
});
