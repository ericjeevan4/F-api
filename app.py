from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load model and label encoder
model = joblib.load('career_model_rf.joblib')
label_encoder = joblib.load('career_label_encoder.joblib')

app = Flask(__name__)

@app.route('/')
def home():
    return '🎯 Career Prediction API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        return jsonify({'career_prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
