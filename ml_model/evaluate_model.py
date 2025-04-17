import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load processed dataset
df = pd.read_csv("dataset/processed/processed_dataset.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Load saved scaler and model
scaler = joblib.load("ml_model/scaler.joblib")
model = joblib.load("ml_model/best_model.joblib")

# Scale features
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)

# Classification Report
print("📋 Classification Report:")
print(classification_report(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred, labels=sorted(y.unique()))
labels = sorted(y.unique())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Save plot
plt.savefig("results/confusion_matrix.png")
print("\n✅ Confusion matrix saved to results/confusion_matrix.png")
plt.show()
