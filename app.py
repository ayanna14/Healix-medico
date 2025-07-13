from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("models.pkl", "rb") as file:
    model = pickle.load(file)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load list of all symptoms
with open("symptoms_list.pkl", "rb") as f:
    all_symptoms = pickle.load(f)

# Load CSV files
precautions_df = pd.read_csv("datasets/precautions_df.csv")
workout_df = pd.read_csv("datasets/workout_df.csv")
workout_df["Disease_clean"] = workout_df["Disease"].str.strip().str.lower()
description_df = pd.read_csv("datasets/disease_descriptions.csv")
medications_df = pd.read_csv("datasets/medication_cleaned.csv")

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}




@app.route("/")
def home():
    return render_template("home.html")

@app.route("/log")
def symptom_form():
    return render_template("symptom_form.html", symptoms=all_symptoms)

@app.route("/predict", methods=["POST"])
def predict():
    selected_symptoms = request.form.getlist("feature")

    # Convert selected symptoms to input vector
    input_vector = [0] * len(all_symptoms)
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            input_vector[index] = 1

    if sum(input_vector) == 0:
        return "Error: No symptoms selected."

    # Predict and decode
    predicted_index = model.predict([input_vector])[0]
    disease_name = diseases_list.get(predicted_index, "Unknown Disease")

    # Get description
    try:
        row = description_df[description_df["Disease"].str.strip().str.lower() == disease_name.strip().lower()]
        description = row["Description"].values[0] if not row.empty else "Description not available."
    except Exception as e:
        description = f"Error: {e}"

    return render_template(
        "predict.html",
        disease=disease_name,
        description=description,
    )


@app.route("/precautions/<disease>")
def make_precautions(disease):
    row = precautions_df[precautions_df["Disease"] == disease]
    tips = row.iloc[0, 1:].dropna().tolist() if not row.empty else ["No precautions found."]
    return render_template("precautions.html", disease=disease, tips=tips)

@app.route("/medications/<disease>")
def show_medications(disease):
    disease_clean = disease.strip().lower()
    medications_df["Disease_clean"] = medications_df["Disease"].astype(str).str.strip().str.lower()

    row = medications_df[medications_df["Disease_clean"] == disease_clean]
    medications = row.drop(columns=["Disease", "Disease_clean"]).dropna(axis=1).values.flatten().tolist() if not row.empty else []

    return render_template("medications.html", disease=disease, medications=medications)

@app.route("/workout/<disease>")
def show_workout(disease):
        disease_clean = disease.strip().lower()
        row = workout_df[workout_df["Disease_clean"] == disease_clean]

        # Exclude "Disease" and "Disease_clean" columns
        workouts = row.drop(columns=["Disease", "Disease_clean"], errors="ignore").values.flatten().tolist()
        # Filter out empty and repeated disease name entries
        workouts = [w for w in workouts if w.strip().lower() != disease_clean and w.strip() != '']

        return render_template("workout.html", disease=disease, workouts=workouts)


if __name__ == "__main__":
    app.run(debug=True)
