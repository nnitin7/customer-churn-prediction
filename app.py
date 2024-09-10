from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model and scaler
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = [float(x) for x in request.form.values()]
    
    # Scale the input
    scaled_data = scaler.transform([data])
    
    # Make prediction
    prediction = model.predict(scaled_data)
    
    result = "Churn" if prediction[0] == 1 else "Not Churn"
    
    return render_template('index.html', prediction_text=f'Customer will {result}')

if __name__ == '__main__':
    app.run(debug=True)
