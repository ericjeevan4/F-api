from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your model
model = joblib.load('career_model_rf.joblib')

# Optional: Load label encoder if you have it
try:
    label_encoder = joblib.load('career_label_encoder.joblib')
except:
    label_encoder = None

@app.route('/')
def home():
    return "🎯 Career Prediction Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]

        if label_encoder:
            prediction = label_encoder.inverse_transform([prediction])[0]

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
