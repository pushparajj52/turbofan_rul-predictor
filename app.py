from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app      = Flask(__name__)
model    = joblib.load('model.pkl')
scaler   = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

@app.route('/')
def home():
    return render_template('index.html', features=list(features))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data received'}), 400

        missing = [f for f in features if f not in data]
        if missing:
            return jsonify({'error': f'Missing sensors: {missing}'}), 400

        values = pd.DataFrame([[float(data[f]) for f in features]], columns=features)
        scaled = scaler.transform(values)
        rul    = float(model.predict(scaled)[0])

        print("=== DEBUG ===")
        print("Received:", {f: data[f] for f in features})
        print("Scaled:", scaled)
        print("Predicted RUL:", rul)
        print("=============")

        if rul < 30:
            status = 'CRITICAL'
        elif rul < 60:
            status = 'WARNING'
        else:
            status = 'HEALTHY'

        return jsonify({'rul': round(rul, 1), 'status': status})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)