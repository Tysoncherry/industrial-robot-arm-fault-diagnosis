import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load processed dataset
df = pd.read_csv("dataset/processed/processed_dataset.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Split into train + hold-out test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_trainval_scaled = scaler.fit_transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "ml_model/scaler.joblib")

# Define models to compare
models = {
    "SVM": SVC(probability=True, kernel="rbf", C=1),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_score = 0.0
best_model_name = ""

print("\n🔍 Cross-Validation Results:")

for name, model in models.items():
    scores = []
    for train_idx, val_idx in cv.split(X_trainval_scaled, y_trainval):
        X_train, X_val = X_trainval_scaled[train_idx], X_trainval_scaled[val_idx]
        y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)

    avg_score = np.mean(scores)
    print(f"{name:<15} Average CV Accuracy: {avg_score:.4f}")

    if avg_score > best_score:
        best_score = avg_score
        best_model = model
        best_model_name = name

# Save the best model
joblib.dump(best_model, "ml_model/best_model.joblib")

print(f"\n✅ Best Model Selected: {best_model_name} (CV Accuracy = {best_score:.4f})")

# Final evaluation on hold-out test set
print("\n🧪 Final Evaluation on Test Set:")
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
