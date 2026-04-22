import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# ---- LOAD DATA ----
df = pd.read_csv("credit_risk_dataset.csv")  # use your dataset

# ---- FEATURES ----
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# ---- COLUMN TYPES ----
cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
num_cols = [col for col in X.columns if col not in cat_cols]

# ---- PREPROCESSING ----
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# ---- MODELS ----
rf = RandomForestClassifier(n_estimators=100, random_state=42)

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    use_label_encoder=False,
    eval_metric='logloss'
)

# ---- ENSEMBLE ----
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft'
)

# ---- PIPELINE ----
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", ensemble)
])

# ---- TRAIN ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)

# ---- SAVE ----
pickle.dump(model, open("models/credit_model.pkl", "wb"))

print("✅ Ensemble model saved!")