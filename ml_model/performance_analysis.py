import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1) Load data
df = pd.read_csv("dataset/processed/processed_dataset.csv")
X = df.drop("label", axis=1)
y = df["label"]

# 2) Define classifiers
classifiers = {
    "SVM": SVC(probability=True, kernel="rbf", C=1, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
}

# 3) Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) Cross‑validation performance (5‑fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores
    print(f"{name}: CV accuracy mean = {scores.mean():.4f}, std = {scores.std():.4f}")

# 5) Plot CV accuracies
plt.figure()
names = list(cv_results.keys())
means = [cv_results[n].mean() for n in names]
stds  = [cv_results[n].std()  for n in names]
plt.bar(names, means, yerr=stds, capsize=5)
plt.ylabel("Accuracy")
plt.title("5‑Fold CV Accuracy Comparison")
plt.tight_layout()
plt.savefig("results/cv_accuracy_comparison.png")
print("✅ CV accuracy plot saved to results/cv_accuracy_comparison.png")
plt.show()

# 6) Train/test split for ROC
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, stratify=y, random_state=42
)

# 7) Load best model (as selected previously)
best_model = joblib.load("ml_model/best_model.joblib")

# 8) Binarize labels for ROC
classes = sorted(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

# 9) Fit and predict probabilities
best_model.fit(X_train, y_train)
y_score = best_model.predict_proba(X_test)

# 10) Compute ROC curve and AUC for each class
fprs, tprs, aucs = {}, {}, {}
for i, cls in enumerate(classes):
    fprs[cls], tprs[cls], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    aucs[cls] = auc(fprs[cls], tprs[cls])
    print(f"Class {cls}: AUC = {aucs[cls]:.4f}")

# 11) Plot all ROC curves
plt.figure()
for cls in classes:
    plt.plot(fprs[cls], tprs[cls], label=f"{cls} (AUC {aucs[cls]:.2f})")
plt.plot([0,1], [0,1], linestyle="--")  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/roc_curves.png")
print("✅ ROC curves plot saved to results/roc_curves.png")
plt.show()
