import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("../datasets/heart_failure/heart_failure_clinical_records_dataset.csv")

# Features and target
X = data.drop("DEATH_EVENT", axis=1)
y = data["DEATH_EVENT"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Heart Failure Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "../models/heart_failure_model.pkl")

print("Heart Failure model saved successfully")