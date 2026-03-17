import wfdb
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_path = r"C:\Users\rojal\OneDrive\Desktop\Donell project\datasets\mit-bih-arrhythmia-database-1.0.0"
signals = []
labels = []

records = ['100','101','102','103','104','105']

for record in records:
    
    record_data = wfdb.rdrecord(os.path.join(data_path, record))
    annotation = wfdb.rdann(os.path.join(data_path, record), 'atr')

    signal = record_data.p_signal[:,0]

    for i in range(len(annotation.sample)):

        index = annotation.sample[i]

        if index + 100 < len(signal):

            segment = signal[index:index+100]

            signals.append(segment)

            if annotation.symbol[i] == 'N':
                labels.append(0)  # Normal
            else:
                labels.append(1)  # Arrhythmia


X = np.array(signals)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Arrhythmia Model Accuracy:", acc)

joblib.dump(model, "../models/arrhythmia_model.pkl")

print("Arrhythmia model saved successfully")