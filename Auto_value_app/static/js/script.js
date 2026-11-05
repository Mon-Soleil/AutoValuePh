document.addEventListener("DOMContentLoaded", async () => {
    const makeSelect = document.getElementById("make");
    const modelSelect = document.getElementById("model");
    const transmissionSelect = document.getElementById("transmission");
    const fuelTypeSelect = document.getElementById("fuel_type");
    
    // 1. Fetch dropdown options from Flask API on load
    try {
        const response = await fetch("/api/v1/options");
        const data = await response.json();
        
        if (data.success) {
            populateDropdown(makeSelect, data.options.makes, "Select Make...");
            populateDropdown(modelSelect, data.options.models, "Select Model...");
            populateDropdown(transmissionSelect, data.options.transmissions, "Select Transmission...");
            populateDropdown(fuelTypeSelect, data.options.fuel_types, "Select Fuel Type...");
        } else {
            console.error("Failed to load options:", data.error);
        }
    } catch (err) {
        console.error("Error connecting to options API:", err);
    }

    // Helper function to fill options into a <select> element
    function populateDropdown(selectElement, items, defaultText) {
        selectElement.innerHTML = `<option value="">${defaultText}</option>`;
        items.forEach(item => {
            const opt = document.createElement("option");
            opt.value = item;
            opt.textContent = item;
            selectElement.appendChild(opt);
        });
    }

    // 2. Handle form submission for valuation prediction (if you have a submit button/form wrapper)
    const form = document.querySelector("form");
    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            
            const payload = {
                make: makeSelect.value,
                model: modelSelect.value,
                year: parseInt(document.getElementById("year").value),
                mileage: parseFloat(document.getElementById("mileage").value),
                transmission: transmissionSelect.value,
                fuel_type: fuelTypeSelect.value
            };

            try {
                const res = await fetch("/api/v1/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const result = await res.json();

                if (result.success) {
                    alert(`Estimated Car Valuation: ₱${result.predicted_valuation.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`);
                } else {
                    alert("Prediction failed: " + result.error);
                }
            } catch (err) {
                console.error("Error running prediction:", err);
                alert("Something went wrong with the prediction request.");
            }
        });
    }
});