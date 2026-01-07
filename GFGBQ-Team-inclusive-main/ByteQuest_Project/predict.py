import joblib
import numpy as np
import pandas as pd

# Load the brain and the vocabulary
model = joblib.load('medical_rf_model.pkl')
symptoms_list = joblib.load('symptom_features_list.pkl')

def get_detailed_diagnosis(user_symptoms, top_k=5):
    # 1. Prepare input vector (132 zeros)
    input_vector = np.zeros(len(symptoms_list))
    
    # 2. Process inputs
    found_any = False
    for s in user_symptoms:
        clean_s = s.strip().replace('_', ' ').lower()
        if clean_s in symptoms_list:
            idx = symptoms_list.index(clean_s)
            input_vector[idx] = 1
            found_any = True
        else:
            print(f"⚠️ Warning: Symptom '{s}' not recognized.")

    if not found_any:
        return "No recognized symptoms provided.", []

    # 3. Get Probabilities for ALL 41 diseases
    # predict_proba returns an array of probabilities
    probabilities = model.predict_proba(input_vector.reshape(1, -1))[0]
    
    # 4. Map probabilities to disease names
    disease_probs = []
    for i, prob in enumerate(probabilities):
        if prob > 0: # Only keep diseases with > 0% chance
            disease_probs.append({
                "disease": model.classes_[i],
                "probability": round(prob * 100, 2)
            })
    
    # 5. Sort by highest probability
    disease_probs = sorted(disease_probs, key=lambda x: x['probability'], reverse=True)
    
    return disease_probs[:top_k]

# --- Interactive Test Section ---
if __name__ == "__main__":
    print("\n--- Disease Predictor ---")
    print("Enter symptoms separated by commas (e.g., itching, skin rash, fatigue)")
    user_input = input("Enter symptoms: ").split(',')
    
    results = get_detailed_diagnosis(user_input)
    
    print("\n📊 DIAGNOSIS RESULTS:")
    print("-" * 30)
    for res in results:
        print(f"{res['probability']}% | {res['disease']}")
    print("-" * 30)