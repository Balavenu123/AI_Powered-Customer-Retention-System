from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# =============================
# Load pickled artifacts
# =============================
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
freq_maps = pickle.load(open("freq_maps.pkl", "rb"))

# =============================
# Feature order (MUST match training)
# =============================
FEATURE_COLUMNS = [
    'tenure','MonthlyCharges','TotalCharges',
    'gender','Partner','Dependents','PhoneService','Contract','PaperlessBilling',
    'PaymentMethod','Sim','Region',
    'MultipleLines_No phone service','MultipleLines_Yes',
    'InternetService_Fiber optic','InternetService_No',
    'OnlineSecurity_No internet service','OnlineSecurity_Yes',
    'OnlineBackup_No internet service','OnlineBackup_Yes',
    'DeviceProtection_No internet service','DeviceProtection_Yes',
    'TechSupport_No internet service','TechSupport_Yes',
    'StreamingTV_No internet service','StreamingTV_Yes',
    'StreamingMovies_No internet service','StreamingMovies_Yes'
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    f = request.form
    data = dict.fromkeys(FEATURE_COLUMNS, 0)

    # -------- Numeric --------
    data['tenure'] = float(f['tenure'])
    data['MonthlyCharges'] = float(f['MonthlyCharges'])
    data['TotalCharges'] = float(f['TotalCharges'])

    # -------- Binary --------
    data['gender'] = 1 if f['gender'] == 'Male' else 0
    data['Partner'] = 1 if f['Partner'] == 'Yes' else 0
    data['Dependents'] = 1 if f['Dependents'] == 'Yes' else 0
    data['PhoneService'] = 1 if f['PhoneService'] == 'Yes' else 0
    data['PaperlessBilling'] = 1 if f['PaperlessBilling'] == 'Yes' else 0

    # -------- Ordinal --------
    contract_map = {
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    }
    data['Contract'] = contract_map[f['Contract']]

    # -------- Frequency Encoding --------
    data['PaymentMethod'] = freq_maps['PaymentMethod'].get(f['PaymentMethod'], 0)
    data['Sim'] = freq_maps['Sim'].get(f['Sim'], 0)
    data['Region'] = freq_maps['Region'].get(f['Region'], 0)

    # -------- Multiple Lines --------
    if f['MultipleLines'] == 'Yes':
        data['MultipleLines_Yes'] = 1
    elif f['MultipleLines'] == 'No phone service':
        data['MultipleLines_No phone service'] = 1

    # -------- Internet Service --------
    if f['InternetService'] == 'Fiber optic':
        data['InternetService_Fiber optic'] = 1
    elif f['InternetService'] == 'No':
        data['InternetService_No'] = 1

    # -------- Helper --------
    def yes_no_internet(prefix, value):
        if value == 'Yes':
            data[f'{prefix}_Yes'] = 1
        elif value == 'No internet service':
            data[f'{prefix}_No internet service'] = 1

    yes_no_internet('OnlineSecurity', f['OnlineSecurity'])
    yes_no_internet('OnlineBackup', f['OnlineBackup'])
    yes_no_internet('DeviceProtection', f['DeviceProtection'])
    yes_no_internet('TechSupport', f['TechSupport'])
    yes_no_internet('StreamingTV', f['StreamingTV'])
    yes_no_internet('StreamingMovies', f['StreamingMovies'])

    # -------- DataFrame + Scaling --------
    df = pd.DataFrame([data])
    df = df[FEATURE_COLUMNS]   # enforce column order
    df_scaled = scaler.transform(df)

    # -------- Prediction --------
    y_pred = model.predict(df_scaled)[0]
    y_prob = model.predict_proba(df_scaled)[0][1]

    prediction_label = "Churn" if y_pred == 1 else "No Churn"

    return render_template(
        "index.html",
        result={
            "prediction": prediction_label,
            "probability": round(y_prob * 100, 2)
        }
    )

if __name__ == "__main__":
    app.run(debug=True)
