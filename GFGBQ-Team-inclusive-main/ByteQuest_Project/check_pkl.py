import joblib

# Load the files
model = joblib.load('medical_rf_model.pkl')
symptoms = joblib.load('symptom_features_list.pkl')

# Inspect the symptoms list
print(f"Total symptoms tracked: {len(symptoms)}")
print("First 10 symptoms:", symptoms[:10])

# Inspect the model
print("\n--- Model Information ---")
print(f"Model Type: {type(model)}")
print(f"Classes (Diseases) it can predict: {len(model.classes_)}")
print(f"Disease Names: {model.classes_[:5]}... (and more)")