from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

# Load the model and scaler
with open('diabetes_model.pkl', 'rb') as model_file, open('scaler.pkl', 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# Initialize the Flask application
app = Flask(__name__)

# Define the home route with HTML template as a string
@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Diabetes Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(to right, #060053, #007E2E);
                color: #333;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                padding: 20px;
                box-sizing: border-box;
            }
            h1 {
                color: #fff;
                text-align: center;
            }
            form {
                background: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                width: 100%;
                max-width: 500px;
            }
            label {
                display: block;
                margin-top: 10px;
                font-weight: bold;
            }
            input[type="number"], input[type="submit"] {
                width: 100%;
                padding: 10px;
                margin-top: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
            }
            input[type="submit"] {
                background: #009A36;
                color: #fff;
                border: none;
                cursor: pointer;
                transition: background 0.3s ease;
            }
            input[type="submit"]:hover {
                background: #026800;
            }
            .modal {
                display: none;
                position: fixed;
                z-index: 1;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                overflow: auto;
                background-color: rgb(0,0,0);
                background-color: rgba(0,0,0,0.4);
                justify-content: center;
                align-items: center;
            }
            .modal-content {
                background-color: #00BBFF;
                margin: auto;
                padding: 20px;
                border: 1px solid #888;
                width: 80%;
                max-width: 500px;
                border-radius: 8px;
                text-align: center;
            }
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
            }
            .close:hover,
            .close:focus {
                color: black;
                text-decoration: none;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <div>
            <h1>Diabetes Prediction</h1>
            <form id="prediction-form">
                <label for="Pregnancies">Pregnancies:</label>
                <input type="number" id="Pregnancies" name="Pregnancies" required><br>
                
                <label for="Glucose">Glucose:</label>
                <input type="number" id="Glucose" name="Glucose" required><br>
                
                <label for="BloodPressure">Blood Pressure:</label>
                <input type="number" id="BloodPressure" name="BloodPressure" required><br>
                
                <label for="SkinThickness">Skin Thickness:</label>
                <input type="number" id="SkinThickness" name="SkinThickness" required><br>
                
                <label for="Insulin">Insulin:</label>
                <input type="number" id="Insulin" name="Insulin" required><br>
                
                <label for="BMI">BMI:</label>
                <input type="number" step="any" id="BMI" name="BMI" required><br>
                
                <label for="DiabetesPedigreeFunction">Diabetes Pedigree Function:</label>
                <input type="number" step="any" id="DiabetesPedigreeFunction" name="DiabetesPedigreeFunction" required><br>
                
                <label for="Age">Age:</label>
                <input type="number" id="Age" name="Age" required><br>
                
                <input type="submit" value="Predict">
            </form>
            <div id="result-modal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="document.getElementById('result-modal').style.display='none'">&times;</span>
                    <h2 id="result"></h2>
                </div>
            </div>
        </div>
        
        <script>
            document.getElementById('prediction-form').onsubmit = async function (e) {
                e.preventDefault();
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData.entries());
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                document.getElementById('result').innerText = result.prediction;
                document.getElementById('result-modal').style.display = 'flex';
            };
            
            window.onclick = function(event) {
                const modal = document.getElementById('result-modal');
                if (event.target == modal) {
                    modal.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    ''')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()
    
    # Convert the data to a list of values
    features = [float(data[field]) for field in data]
    
    # Convert the list to a numpy array and reshape for prediction
    features_array = np.array(features).reshape(1, -1)
    
    # Scale the features
    features_scaled = scaler.transform(features_array)
    
    # Make the prediction
    prediction = model.predict(features_scaled)
    
    # Return the prediction
    result = 'Patient has Diabetes' if prediction[0] == 1 else "Patient doesn't have Diabetes"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
