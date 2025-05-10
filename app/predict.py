import joblib
import pandas as pd


bundle = joblib.load("E:\html-boilerplate\Fraud-detection-handling\model/fraud_rf_bundle.pkl")
model = bundle['model']
encoder = bundle['label_encoder']

def predict_fraud(input_dict):
    df = pd.DataFrame([input_dict])
    df['type'] = encoder.transform(df['type'])
    prediction = model.predict(df)
    return bool(prediction[0])
