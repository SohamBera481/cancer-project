import lime
import lime.lime_tabular
import numpy as np
from models.ehr_model import create_ehr_model


# Lime explanation for EHR data
def explain_with_lime(model, X):
    explainer = lime.lime_tabular.LimeTabularExplainer(X, mode='classification')
    explanation = explainer.explain_instance(X[0], model.predict)

    explanation.show_in_notebook()


# Sample Usage
if __name__ == '__main__':
    model = create_ehr_model(200)
    X_test = np.random.rand(10, 200)  # Dummy test data for EHR
    explain_with_lime(model, X_test)
