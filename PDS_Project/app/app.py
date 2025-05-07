from flask import Flask, render_template, request
import joblib
import os
import sys
import numpy as np
import pandas as pd
import markdown
# from transformers import pipeline


# LLM integration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'llm')))
from llm_analysis import generate_llm_insight

app = Flask(__name__)

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
model = joblib.load(os.path.join(PARENT_DIR, 'models', 'churn_model.pkl'))
scaler = joblib.load(os.path.join(PARENT_DIR, 'models', 'scaler.pkl'))
encoders = joblib.load(os.path.join(PARENT_DIR, 'models', 'encoders.pkl'))
# Load once at the top
# sentiment_pipeline = pipeline("sentiment-analysis")

df = pd.read_csv(os.path.join(PARENT_DIR, 'data', 'Telco-Customer-Churn.csv'))
feature_names = df.drop(columns=['Churn','customerID']).columns.tolist()

# Dynamically build feature_options from encoders
feature_options = {}
for feature in feature_names:
    if feature in encoders:
        encoder = encoders[feature]
        feature_options[feature] = [(label, int(encoder.transform([label])[0])) for label in encoder.classes_]
    else:
        feature_options[feature] = []  # Numerical or unencoded binary fields

# Identify feature types
binary_features = [f for f in feature_names if df[f].nunique() == 2 and f not in encoders]
numerical_features = [f for f in feature_names if df[f].dtype in [np.float64, np.int64] and f not in encoders]
categorical_features = [f for f in feature_names if f not in binary_features + numerical_features]

@app.route('/')
def index():
    return render_template(
        'index.html',
        feature_names=feature_names,
        feature_options=feature_options
    )

@app.route('/predict', methods=['POST'])
def predict():
    form_data = [request.form.get(feature) for feature in feature_names]
    prompt_text = request.form.get('prompt')
    form_values = dict(zip(feature_names, form_data))
    
    # Clean input and cast appropriately
    data_processed = []
    for feature, val in zip(feature_names, form_data):
        if feature in encoders:
            data_processed.append(int(val))  # Already encoded
        elif feature in binary_features:
            data_processed.append(int(val))  # 0 or 1 from input
        elif feature in numerical_features:
            try:
                data_processed.append(float(val))
            except ValueError:
                return render_template(
                    'index.html',
                    prediction_text='Invalid numerical input.',
                    feature_names=feature_names,
                    feature_options=feature_options,
                    form_values=form_values
                )
        else:
            data_processed.append(0)

    # support_text = request.form.get("support_text", "")
    # sentiment_result = ""
    # if support_text:
    #     sentiment_result = sentiment_pipeline(support_text)[0]

    prediction = model.predict([data_processed])[0]
    print(prediction)
    if prediction == 0:
        prediction_text = "The customer will not churn"
    else:
        prediction_text = "The customer will churn"

    llm_output = generate_llm_insight(form_values, prompt_text, prediction)
    llm_output = markdown.markdown(llm_output)
    
    # print(form_data['tenure'])
    # customer_journey = generate_customer_journey(form_data)

    return render_template(
        'index.html',
        prediction_text=prediction_text,
        llm_output=llm_output,
        feature_names=feature_names,
        feature_options=feature_options,
        form_values=form_values,
        # sentiment_result=sentiment_result,
        # customer_journey=customer_journey
    )

# def generate_customer_journey(data):
#     journey = []
#     if int(data["tenure"]) < 6:
#         journey.append("Recently joined customer.")
#     else:
#         journey.append("Loyal customer with moderate tenure.")

#     if float(data["MonthlyCharges"]) > 70:
#         journey.append("High monthly spender.")
#     else:
#         journey.append("Moderate monthly charges.")

#     if data["Contract"] == "0":
#         journey.append("On a month-to-month contract, may leave anytime.")

#     return " ".join(journey)

if __name__ == "__main__":
    app.run(debug=True)
