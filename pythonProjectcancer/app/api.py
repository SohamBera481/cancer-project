from flask import Flask, request, jsonify
from models.combined_model import create_combined_model
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = create_combined_model()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Preprocess data here (images, genomics, ehr)
    # Assuming preprocessed input data
    image_data = np.array(data['image']).reshape(1, 224, 224, 1)
    genomic_data = np.array(data['genomic']).reshape(1, 1000)
    ehr_data = np.array(data['ehr']).reshape(1, 200)

    prediction = model.predict([image_data, genomic_data, ehr_data])
    return jsonify({'risk_score': float(prediction[0][0])})


if __name__ == '__main__':
    app.run(debug=True)
