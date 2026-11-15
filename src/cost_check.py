import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix

model = pickle.load(open("models/credit_model.pkl", "rb"))
X_test, y_test = pickle.load(open("models/test_data.pkl", "rb"))
proba = model.predict_proba(X_test)[:, 1]

COST_FN, COST_FP = 5, 1
rows = []
costs = []
for t in [0.5, 0.2]:
    tn, fp, fn, tp = confusion_matrix(y_test, (proba >= t).astype(int)).ravel()
    cost = int(COST_FN * fn + COST_FP * fp)
    costs.append(cost)
    rows.append({
        "threshold": t,
        "missed defaulters (FN)": int(fn),
        "false alarms (FP)": int(fp),
        "total cost": cost,
    })

print(pd.DataFrame(rows).to_string(index=False))
drop = 1 - costs[1] / costs[0]
print(f"\nCost reduction at 0.2 vs 0.5: {drop:.1%}")