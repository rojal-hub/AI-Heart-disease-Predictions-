from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ===============================
# LOAD TRAINED MODELS
# ===============================

cad_model = joblib.load("models/cad_model.pkl")
hf_model = joblib.load("models/heart_failure_model.pkl")
arr_model = joblib.load("models/arrhythmia_model.pkl")


# ===============================
# HOME PAGE
# ===============================

@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# PREDICTION ROUTE
# ===============================

@app.route("/predict", methods=["POST"])
def predict():

    # ===============================
    # CAD MODEL INPUT
    # ===============================

    cad_features = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["trestbps"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["restecg"]),
        float(request.form["thalach"]),
        float(request.form["exang"]),
        float(request.form["oldpeak"]),
        float(request.form["slope"]),
        float(request.form["ca"]),
        float(request.form["thal"])
    ]

    cad_array = np.array(cad_features).reshape(1, -1)

    cad_prediction = cad_model.predict(cad_array)
    cad_prob = cad_model.predict_proba(cad_array)[0][1] * 100

    if cad_prediction[0] == 1:
        cad_result = "⚠ Possible Coronary Artery Disease"
    else:
        cad_result = "✔ No Coronary Artery Disease Detected"


    # ===============================
    # HEART FAILURE MODEL
    # ===============================

    hf_features = [
        float(request.form["age"]),
        float(request.form["anaemia"]),
        float(request.form["creatinine_phosphokinase"]),
        float(request.form["diabetes"]),
        float(request.form["ejection_fraction"]),
        float(request.form["high_blood_pressure"]),
        float(request.form["platelets"]),
        float(request.form["serum_creatinine"]),
        float(request.form["serum_sodium"]),
        float(request.form["sex"]),
        float(request.form["smoking"]),
        float(request.form["time"])
    ]

    hf_array = np.array(hf_features).reshape(1, -1)

    hf_prediction = hf_model.predict(hf_array)
    hf_prob = hf_model.predict_proba(hf_array)[0][1] * 100

    if hf_prediction[0] == 1:
        hf_result = "⚠ High Heart Failure Risk"
    else:
        hf_result = "✔ Low Heart Failure Risk"


    # ===============================
    # ARRHYTHMIA DETECTION
    # ===============================

    heart_rate = float(request.form["thalach"])

    if heart_rate < 50 or heart_rate > 120:
        arr_prediction = 1
    else:
        arr_prediction = 0

    if arr_prediction == 1:
        arr_result = "⚠ Possible Arrhythmia Detected"
    else:
        arr_result = "✔ Normal Heart Rhythm"


    # ===============================
    # AI RECOMMENDATION ENGINE
    # ===============================

    if cad_prob > 60 or hf_prob > 60:
        recommendation = "High cardiovascular risk detected. Immediate cardiologist consultation recommended."

    elif arr_prediction == 1:
        recommendation = "Possible arrhythmia detected. ECG monitoring and cardiologist consultation advised."

    elif cad_prob > 30 or hf_prob > 30:
        recommendation = "Moderate cardiovascular risk detected. Lifestyle monitoring and regular health checkups recommended."

    else:
        recommendation = "Low cardiovascular risk. Maintain healthy lifestyle, balanced diet, and regular exercise."


    # ===============================
    # EXPLAINABLE AI FEATURES
    # ===============================

    feature_names = [
        "Age",
        "Sex",
        "Chest Pain",
        "Resting BP",
        "Cholesterol",
        "Fasting Sugar",
        "Rest ECG",
        "Max Heart Rate",
        "Exercise Angina",
        "Oldpeak",
        "Slope",
        "Major Vessels",
        "Thalassemia"
    ]

    feature_values = cad_features


    # ===============================
    # RETURN RESULT PAGE
    # ===============================

    return render_template(
        "result.html",
        cad_result=cad_result,
        cad_prob=round(cad_prob,2),
        hf_result=hf_result,
        hf_prob=round(hf_prob,2),
        arr_result=arr_result,
        recommendation=recommendation,
        feature_names=feature_names,
        feature_values=feature_values
    )


# ===============================
# RUN SERVER
# ===============================

if __name__ == "__main__":
    app.run(debug=True)