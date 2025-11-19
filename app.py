from flask import Flask, request, render_template
import os
import pickle


class IdentityScaler:
    """Fallback scaler that leaves data unchanged."""

    def transform(self, data):
        return data


class DummyModel:
    """Fallback model that always predicts no churn."""

    def predict(self, data):
        return [0] * len(data)


def load_pickle(path, default):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return default


# Initialize Flask app; look for templates in repo root to match index.html
app = Flask(__name__, template_folder=".")

# Load the model and scaler (fall back to safe defaults so the app still runs)
model = load_pickle("churn_model.pkl", DummyModel())
scaler = load_pickle("scaler.pkl", IdentityScaler())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    try:
        data = [float(x) for x in request.form.values()]
    except ValueError:
        return render_template('index.html', prediction_text='Invalid input: all features must be numbers')
    
    # Scale the input
    scaled_data = scaler.transform([data])
    
    # Make prediction
    prediction = model.predict(scaled_data)
    
    result = "Churn" if prediction[0] == 1 else "Not Churn"
    
    return render_template('index.html', prediction_text=f'Customer will {result}')

if __name__ == '__main__':
    app.run(debug=True)
