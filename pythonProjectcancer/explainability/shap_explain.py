import shap
import numpy as np
import tensorflow as tf
from models.combined_model import create_combined_model


# Initialize SHAP explainer
def explain_with_shap(model, X):
    explainer = shap.DeepExplainer(model, X)
    shap_values = explainer.shap_values(X)

    # Plot SHAP values
    shap.summary_plot(shap_values, X)


# Sample Usage
if __name__ == '__main__':
    # Load model
    model = create_combined_model()
    X_test = np.random.rand(10, 224, 224, 1)  # Dummy test data for images
    explain_with_shap(model, X_test)
