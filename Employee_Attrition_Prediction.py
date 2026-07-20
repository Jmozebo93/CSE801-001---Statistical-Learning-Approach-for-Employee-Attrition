#!/usr/bin/env python
# coding: utf-8

# ## IMPORT LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
import json
import joblib
import os
import subprocess
from datetime import datetime, timezone


# In[2]:


# Create a folder for figures
os.makedirs("results/figures", exist_ok=True)


# In[3]:


# Read HR Employee-Attrition Dataset
emp_dataset = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(emp_dataset.head())
print(emp_dataset.tail())
print(emp_dataset.info())
print(emp_dataset.describe())


# In[4]:


print(emp_dataset.isnull())


# In[5]:


emp_dataset["Attrition"].value_counts()


# In[6]:


emp_dataset.columns


# In[7]:


emp_dataset.nunique()


# In[8]:


# Dropping useless columns
emp_dataset.drop(columns=["EmployeeCount", "Over18", "StandardHours"], inplace=True)


# In[9]:


# Dropping employee Number column
emp_dataset.drop(columns=["EmployeeNumber"], inplace=True)


# In[10]:


emp_dataset.shape


# In[11]:


# Converting Attrition to binary 0 or 1
emp_dataset["Attrition"] = emp_dataset["Attrition"].map({"Yes":1, "No":0})


# In[12]:


# Verify that the conversion was successful
emp_dataset["Attrition"].value_counts()


# In[13]:


# Identify categorical columns
emp_dataset.select_dtypes(include=["object"]).columns


# In[14]:


# Apply one-hot encoding
emp_dataset = pd.get_dummies(emp_dataset, drop_first=True)


# In[15]:


# Verify that the encoding worked
emp_dataset.select_dtypes(include=["object"]).columns


# In[16]:


emp_dataset.shape


# In[17]:


# Check the Attrition
emp_dataset["Attrition"].value_counts()


# In[18]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
X = emp_dataset.drop("Attrition", axis=1)
y = emp_dataset["Attrition"]
feature_columns = list(X.columns)


# ## Train/Test Split

# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[20]:


# Verify the split
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


# ## Train Logistic Regression 

# In[21]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight="balanced", max_iter=5000)
model.fit(X_train, y_train)

os.makedirs("results/artifacts", exist_ok=True)
joblib.dump(model, "results/artifacts/model.joblib")
joblib.dump(scaler, "results/artifacts/scaler.joblib")
with open("results/artifacts/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    git_commit = "unknown"

model_metadata = {
    "model_name": "LogisticRegression",
    "model_version": f"logreg-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "feature_count": len(feature_columns),
    "source_dataset": "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    "git_commit": git_commit,
}
with open("results/artifacts/model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)


# In[22]:


# Make prediction
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# In[23]:


# Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))


# In[24]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")

plt.savefig("results/figures/logistic_confusion_matrix.png", bbox_inches="tight")
plt.show()


# In[25]:


from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.savefig("results/figures/logistic_roc_curve.png", bbox_inches="tight")
plt.show()


# # TRAINING RANDOM FOREST

# In[26]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 100, class_weight = "balanced", random_state = 42)
rf_model.fit(X_train, y_train)


# In[27]:


# Make prediction
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:,1]


# In[28]:


# Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))


# In[29]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.savefig("results/figures/RF_confusion_matrix.png", bbox_inches="tight")
plt.show()


# In[30]:


from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.savefig("results/figures/RF_roc_curve.png", bbox_inches="tight")
plt.show()


# # TRAINING XGBOOST

# In[31]:


from xgboost import XGBClassifier
xgb_model = XGBClassifier(scale_pos_weight = 1233/237, # handle imbalance 
                          n_estimators = 200,
                          learning_rate = 0.05,
                          max_depth = 5,
                          random_state = 42
)
xgb_model.fit(X_train, y_train)


# In[32]:


# Make prediction
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]


# In[33]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_xgb))


# In[34]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.savefig("results/figures/XGBoost_confusion_matrix.png", bbox_inches="tight")
plt.show()


# In[35]:


from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(y_test, y_proba_xgb)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba_xgb):.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.savefig("results/figures/XGBoost_roc_curve.png", bbox_inches="tight")
plt.show()


# # FEATURES IMPORTANCE USING XGBoost

# In[36]:


# Get feature importance 
importances = xgb_model.feature_importances_

# Feature names
feature_names = X.columns

# Create series
feat_imp = pd.Series(importances, index=feature_names)

# Sort and take top 10
feat_imp = feat_imp.sort_values(ascending=False).head(10)

# Plot
plt.figure()
feat_imp.plot(kind="barh")
plt.title("Top 10 Important Features - XGBoost")
plt.gca().invert_yaxis()
plt.savefig("results/figures/xgboost_feature_importance.png", bbox_inches="tight")
plt.show()


# # IMPLEMENTING SHAP FOR EXPLAINABILITY

# In[37]:


import shap
shap.initjs()

# Get the feature names
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# Initilize explainer
explainer = shap.TreeExplainer(xgb_model)

# Compute shap value
shap_values = explainer.shap_values(X_test)


# In[38]:


# Shap summary
shap.summary_plot(shap_values, X_test_df, show=False)
plt.savefig("results/figures/shap_summary_plot.png", bbox_inches="tight")
plt.show()


# In[39]:


# Bar plot
shap.summary_plot(shap_values, X_test_df, plot_type = "bar", show=False)
plt.savefig("results/figures/shap_bar_plot.png", bbox_inches="tight")
plt.show()


# In[40]:


# Individual explanation
shap.plots.waterfall(shap.Explanation(values = shap_values[0], base_values=explainer.expected_value, data=X_test_df.iloc[0], feature_names=X_test_df.columns),show=False)
plt.savefig("results/figures/shap_waterfall.png", bbox_inches=False)
plt.show()


# # MODEL STABILITY - CROSS-VALIDATION

# In[41]:


cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)


# In[42]:


# Cross-validation with logistic regression
log_model = LogisticRegression(max_iter=1000)

#Accuracy
log_acc = cross_val_score(log_model, X, y, cv=cv, scoring='accuracy')

#ROC-AUC
log_auc = cross_val_score(log_model, X, y, cv=cv, scoring='roc_auc')

print("Logistic Regression")
print("Accuracy: ", log_acc.mean(), "+/-", log_acc.std())
print("ROC-AUC: ", log_auc.mean(), "+/-", log_auc.std())


# In[43]:


xgb_model = XGBClassifier(scale_pos_weight = 1233/237, n_estimators = 200, learning_rate = 0.05, max_depth = 5, random_state = 42)

# Accuracy
xgb_acc = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy')

# ROC-AUC
xgb_auc = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc')

print("\nXGBoost")
print("Accuracy: ", xgb_acc.mean(), "+/-", xgb_acc.std())
print("ROC-AUC: ", xgb_auc.mean(), "+/-", xgb_auc.std())


# In[ ]:




