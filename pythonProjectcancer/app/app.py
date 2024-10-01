from flask import Flask, request, jsonify
import numpy as np
from models.combined_model import create_combined_model
from data_loader import load_images, preprocess_genomic_data, preprocess_ehr_data
from alert_system import send_alert


app = Flask(__name__)


model = create_combined_model()


RISK_THRESHOLD = 0.75
@app.route('/')
def index():
    return "Welcome to OncoAI Cancer Detection System API"


@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.json

        image_data = np.array(data['image']).reshape(1, 224, 224, 1)
        genomic_data = np.array(data['genomic']).reshape(1, 1000)

        ehr_data = np.array(data['ehr']).reshape(1, 200)
        prediction = model.predict([image_data, genomic_data, ehr_data])
        risk_score = float(prediction[0][0])

        if risk_score >= RISK_THRESHOLD:
            send_alert(patient_id=data['patient_id'], risk_score=risk_score, provider_email=data['provider_email'])

        return jsonify({
            'patient_id': data['patient_id'],
            'risk_score': risk_score,
            'alert_triggered': risk_score >= RISK_THRESHOLD
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Main block to run the app
if __name__ == '__main__':
    app.run(debug=True)
