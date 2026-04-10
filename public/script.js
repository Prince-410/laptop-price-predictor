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
    
    // Collect Data
    const data = {
        brand: document.getElementById('brand').value,
        processor_brand: "Intel", // Default or extract from another field
        processor_name: document.getElementById('processor').value,
        processor_gnrtn: "11th",
        ram_gb: parseInt(document.getElementById('ram').value),
        ram_type: "DDR4",
        ssd: parseInt(document.getElementById('ssd').value),
        hdd: 0,
        os: document.getElementById('os').value,
        graphic_card_gb: parseInt(document.getElementById('gpu').value),
        weight: "Casual",
        warranty: 1,
        touchscreen: "No",
        msoffice: "No"
    };

    try {
        // Change this URL to your Backend API (Render/HuggingFace)
        // For demonstration, we'll simulate a response if the URL is not set
        const API_URL = '/api/predict'; 
        
        // This is where you connect to your Gradio/FastAPI backend
        // Since we are deploying a static site, we'd normally call the external host
        
        // Simulation for visual testing
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        const mockPrice = Math.floor(Math.random() * (120000 - 35000) + 35000);
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
