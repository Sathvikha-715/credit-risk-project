import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix

model = pickle.load(open("models/credit_model.pkl", "rb"))
X_test, y_test = pickle.load(open("models/test_data.pkl", "rb"))
proba = model.predict_proba(X_test)[:, 1]

COST_FN, COST_FP = 5, 1
rows = []
for t in [0.5, 0.2]:
    tn, fp, fn, tp = confusion_matrix(y_test, (proba >= t).astype(int)).ravel()
    rows.append({"threshold": t, "missed defaulters (FN)": fn,
                 "false alarms (FP)": fp, "total cost": COST_FN * fn + COST_FP * fp})

df = pd.DataFrame(rows)
print(df.to_string(index=False))
drop = 1 - df.loc[1, "total cost"] / df.loc[0, "total cost"]
print(f"\nCost reduction at 0.2 vs 0.5: {drop:.1%}")