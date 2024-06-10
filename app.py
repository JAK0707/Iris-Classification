from flask import Flask, send_file, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the Iris classification model from the .sav file
model = joblib.load('saved_model.sav')

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()
    
    # Convert input data to a numpy array
    input_data = np.array(data['input']).reshape(1, -1)
    
    # Perform prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Return the predicted class as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
