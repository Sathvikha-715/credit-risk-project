import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier

# ---------------- SETTINGS ----------------
USE_CLASS_WEIGHTS = False   # set True only if evaluate_baseline.py showed a clear PR-AUC gain
COST_FN = 5                 # assumed cost of approving a loan that defaults
COST_FP = 1                 # assumed cost of rejecting a loan that would be repaid
# ------------------------------------------

# ---- LOAD + CLEAN DATA ----
df = pd.read_csv("dataset/credit_risk_dataset.csv")
df = df[df["person_age"] <= 100]
df = df[df["person_emp_length"].isna() | (df["person_emp_length"] <= 60)]

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
num_cols = [c for c in X.columns if c not in cat_cols]

# ---- SPLITS: train / validation (picks threshold) / test (final report) ----
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)

spw = (y_train == 0).sum() / (y_train == 1).sum()

# ---- MODEL ----
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

rf = RandomForestClassifier(
    n_estimators=100, random_state=42,
    class_weight="balanced" if USE_CLASS_WEIGHTS else None,
)
xgb = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=4,
    eval_metric="logloss", random_state=42,
    scale_pos_weight=spw if USE_CLASS_WEIGHTS else 1,
)
ensemble = VotingClassifier([("rf", rf), ("xgb", xgb)], voting="soft")

model = Pipeline([("preprocessor", preprocessor), ("classifier", ensemble)])

print("Training...")
model.fit(X_train, y_train)

# ---- PICK THRESHOLD BY COST (on the validation set) ----
val_proba = model.predict_proba(X_val)[:, 1]
best_t, best_cost = 0.5, float("inf")
for t in np.linspace(0.05, 0.95, 91):
    tn, fp, fn, tp = confusion_matrix(y_val, (val_proba >= t).astype(int)).ravel()
    cost = COST_FN * fn + COST_FP * fp
    if cost < best_cost:
        best_t, best_cost = float(t), cost
print(f"\nBest threshold on validation set: {best_t:.2f}  (cost {COST_FN}:{COST_FP})")


# ---- FINAL REPORT ON THE UNTOUCHED TEST SET ----
def report(threshold):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 3),
        "precision": round(float(precision_score(y_test, pred)), 3),
        "recall": round(float(recall_score(y_test, pred)), 3),
        "f1": round(float(f1_score(y_test, pred)), 3),
    }


at_default = report(0.5)
at_tuned = report(best_t)

print("\n=== TEST SET RESULTS ===")
print("Default threshold 0.5 :", at_default)
print("Cost-tuned threshold  :", at_tuned)
print("Default rate in data  :", round(float(y.mean()), 3))

# ---- SAVE ----
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/credit_model.pkl", "wb"))
pickle.dump((X_test, y_test), open("models/test_data.pkl", "wb"))
json.dump(
    {
        "threshold": round(best_t, 3),
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "default_rate": round(float(y.mean()), 3),
        "metrics_at_0.5": at_default,
        "metrics_at_tuned": at_tuned,
    },
    open("models/config.json", "w"),
    indent=2,
)
print("\n✅ Saved models/credit_model.pkl and models/config.json")