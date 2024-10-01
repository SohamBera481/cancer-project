import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import cv2

# Load Medical Images
def load_images(image_folder):
    images = []
    labels = []
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        label = img_name.split('_')[0]  # Assuming the label is part of the image file name
        images.append(img)
        labels.append(int(label))
    return np.array(images), np.array(labels)

# Load Genomic Data
def load_genomic_data(genomic_file):
    return pd.read_csv(genomic_file)

# Load EHR Data
def load_ehr_data(ehr_file):
    return pd.read_csv(ehr_file)

# Standardize Genomic Data
def preprocess_genomic_data(genomic_data):
    scaler = StandardScaler()
    return scaler.fit_transform(genomic_data)

# Preprocess EHR Data
def preprocess_ehr_data(ehr_data):
    ehr_data.fillna(0, inplace=True)
    return pd.get_dummies(ehr_data)
