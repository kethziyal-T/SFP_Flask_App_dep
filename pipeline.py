import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# 1. Load data and drop unused features cleanly
df = pd.read_csv('saved_models/SensorFault_Detector.csv', nrows=20000)
drop_cols = ['Machine_ID', 'Laser_Intensity', 'Hydraulic_Pressure_bar', 'Coolant_Flow_L_min', 'Heat_Index']
df = df.drop(columns=drop_cols, errors='ignore')

# 2. Preprocess data types & save clean intermediate file
df = pd.get_dummies(df, drop_first=True, dtype=int)
df['AI_Supervision'] = df['AI_Supervision'].astype('int64')
df['Failure_Within_7_Days'] = df['Failure_Within_7_Days'].astype('int64')
df.to_csv('saved_models/sensor_failure_analysis.csv', index=False)

# 3. Isolate features and target (natively flat 1D)
features = ['Operational_Hours', 'Temperature_C', 'Vibration_mms', 'Oil_Level_pct', 'Remaining_Useful_Life_days']
X = df[features]
y = df['Failure_Within_7_Days']

# 4. Train-Test Split (25% validation fold)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 5. Apply SMOTE to handle training imbalance safely (guaranteed 1D output)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Balanced Class Counts:\n", y_train_res.value_counts())

# 6. Apply Feature Scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train_res)
X_test_scaled = sc.transform(X_test)

# 7. Fit the Decision Tree Model (using optimized hyper-parameters)
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X_train_scaled, y_train_res)

# 8. Save production assets out to 'saved_models/' folder
os.makedirs('saved_models', exist_ok=True)
with open('saved_models/scalerDT.pkl', 'wb') as f:
    pickle.dump(sc, f)
with open('saved_models/finalized_model_SFP_DecisionTree.sav', 'wb') as f:
    pickle.dump(model, f)
print("Scaler and Model assets successfully saved to 'saved_models/' folder.")

# ==========================================
# 9. DEPLOYMENT INFERENCE TESTING BLOCK 
# ==========================================
print("\n--- Running Live Prediction Test ---")

# Raw operational input data sample
raw_input_sample = [[94006, 49.63, 23.78, 42.96, 0]]

# CRITICAL FIX: Transform the raw data to the exact scale the model expects
scaled_input = sc.transform(raw_input_sample)

# Predict using the fitted model instance
prediction = model.predict(scaled_input)[0]  # Extract scalar prediction integer

if prediction == 0:
    print(f"Prediction: {prediction} -> Embedded Machine operates with Sensor Healthy")
else:
    print(f"Prediction: {prediction} -> Embedded Machine operates with Sensor Faulty")
