import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Column names for Cleveland dataset
columns = [
"age","sex","cp","trestbps","chol","fbs",
"restecg","thalach","exang","oldpeak",
"slope","ca","thal","target"
]

# Load dataset
data = pd.read_csv("../datasets/cleveland/processed.cleveland.data", names=columns)

# Replace missing values
data = data.replace("?", np.nan)
data = data.dropna()

# Convert target to binary
data["target"] = data["target"].apply(lambda x: 1 if int(x) > 0 else 0)

# Split features and labels
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "../models/cad_model.pkl")

print("Model saved successfully")