# Explainable-ML-Project
An Explainable Machine Learning project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report

from lime.lime_tabular import LimeTabularExplainer

# 1. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Explainable ML: Feature Importance Comparison")
print("Dataset:", X.shape)
print("Features:", len(data.feature_names))

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train ML model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

# 4. Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions,
                            target_names=data.target_names))

# 5. Built-in Feature Importance
built_in = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Built-in Features:")
print(built_in.head(10))

built_in.head(10).sort_values().plot(
    kind="barh", title="Built-in Feature Importance"
)
plt.tight_layout()
plt.savefig("built_in_importance.png")
plt.show()

# 6. Permutation Importance
perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=5, random_state=42,
    scoring="accuracy"
)

permutation = pd.Series(
    perm.importances_mean, index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Permutation Features:")
print(permutation.head(10))

permutation.head(10).sort_values().plot(
    kind="barh", title="Permutation Importance"
)
plt.tight_layout()
plt.savefig("permutation_importance.png")
plt.show()

# 7. SHAP Explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Handle SHAP output formats across versions
if isinstance(shap_values, list):
    values = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    values = shap_values[:, :, 1]
else:
    values = shap_values

if hasattr(values, "values"):
    values = values.values

if values.ndim == 3:
    values = values[:, :, 1]

shap.summary_plot(values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.show()

# 8. LIME Explanation for one prediction
lime_explainer = LimeTabularExplainer(
    X_train.to_numpy(),
    feature_names=list(X.columns),
    class_names=list(data.target_names),
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

sample = X_test.iloc[0].to_numpy()

lime_exp = lime_explainer.explain_instance(
    sample,
    model.predict_proba,
    num_features=10
)

print("\nLIME Explanation:")
for feature, contribution in lime_exp.as_list():
    print(feature, ":", round(contribution, 4))

lime_exp.save_to_file("lime_explanation.html")

# 9. Compare top features
comparison = pd.DataFrame({
    "Built-in Importance": built_in,
    "Permutation Importance": permutation,
    "SHAP Mean Absolute Value": np.abs(values).mean(axis=0)
})

comparison = comparison.fillna(0)
comparison = comparison.sort_values(
    "Permutation Importance", ascending=False
)

comparison.to_csv("feature_importance_comparison.csv")

print("\nComparison of Top 10 Features:")
print(comparison.head(10).round(4))

print("\nProject completed successfully!")
print("Check generated PNG, CSV and HTML files.")
