import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Setup Paths
# Pointing to the 'data_sets' subfolder
DATA_DIR = "data_sets"

print("Reading CSV files...")
try:
    # Adding the folder prefix to the filenames
    df = pd.read_csv(os.path.join(DATA_DIR, 'dataset.csv'))
    severity = pd.read_csv(os.path.join(DATA_DIR, 'Symptom-severity.csv'))
except FileNotFoundError as e:
    print(f"Error: Could not find the files. Check if you are running the script from the root folder. \n{e}")
    exit()

# 2. Clean text data
# We use .fillna('') to prevent errors if there are empty cells in your CSV
severity['Symptom'] = severity['Symptom'].str.replace('_', ' ').str.strip()

for col in df.columns[1:]:
    df[col] = df[col].fillna('').str.replace('_', ' ').str.strip()

# 3. Create the Binary Matrix
print("Processing symptoms into binary matrix...")
all_symptoms = severity['Symptom'].unique().tolist()
data_dict = {symptom: [] for symptom in all_symptoms}
data_dict['Disease'] = []

for _, row in df.iterrows():
    # Get all symptoms this patient has in one list, excluding empty strings
    patient_list = [s for s in row[1:].values.tolist() if s != '']
    
    for s in all_symptoms:
        data_dict[s].append(1 if s in patient_list else 0)
    data_dict['Disease'].append(row['Disease'].strip())

# Convert to DataFrame
train_df = pd.DataFrame(data_dict)

# 4. Train the Model
print("Training the Random Forest model...")
X = train_df.drop('Disease', axis=1)
y = train_df['Disease']

# Random Forest is excellent for this because it handles high-dimensional binary data well
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Export for Deployment
# We save the model and the specific order of symptoms (features)
joblib.dump(model, 'medical_rf_model.pkl')
joblib.dump(all_symptoms, 'symptom_features_list.pkl')

print("-" * 30)
print("SUCCESS!")
print(f"Model trained on {len(train_df)} rows and {len(all_symptoms)} symptoms.")
print("Files 'medical_rf_model.pkl' and 'symptom_features_list.pkl' are ready.")
print("-" * 30)