# train_auto_columns_smote_only.py
import os, json
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, average_precision_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import joblib


# ------------------ Paths & load ------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
DATA_PATH = "data/processed/training_data.csv"   # change if needed
df = pd.read_csv(DATA_PATH).copy()



# ------------------ Target ------------------
if "target" not in df.columns:
    raise ValueError("Expected a 'target' column with values {'healthy','diseased'}")
y = df["target"].map({"healthy": 0, "diseased": 1}).astype(int)

# ------------------ Column inference ------------------
# Categorical (object/str)
obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Known ordinals by name (if present)
known_ordinal_text = []
if "smoking_level" in df.columns and df["smoking_level"].dtype == "object":
    levels = ["Non-smoker", "Light", "Heavy"]
    if set(levels).issubset(set(df["smoking_level"].dropna().unique().tolist())):
        known_ordinal_text.append("smoking_level")

# Numeric columns
num_all = df.select_dtypes(include=["number"]).columns.tolist()
num_all = [c for c in num_all if c != "target"]

# Ordinal candidates among numeric: small integer scales (<=12 unique)
ordinal_numeric = []
for c in num_all:
    s = df[c].dropna()
    if len(s) == 0:
        continue
    if np.allclose(s, np.round(s)) and s.nunique() <= 12:
        ordinal_numeric.append(c)

# Ensure stress_level is ordinal if present
if "stress_level" in df.columns and "stress_level" not in ordinal_numeric:
    ordinal_numeric.append("stress_level")

# Final groups
cat_cols = [c for c in obj_cols if c not in known_ordinal_text and c != "target"]
ord_cols = known_ordinal_text + ordinal_numeric
num_cols = [c for c in num_all if c not in ordinal_numeric]

# ------------------ Minimal imputation ------------------
for c in cat_cols:
    df[c] = df[c].fillna("Unknown")
for c in ord_cols:
    if df[c].dtype == "object":
        df[c] = df[c].fillna("Unknown")
    else:
        df[c] = df[c].fillna(df[c].median())
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

# ------------------ Build ordinal category orders ------------------
ordinal_orders = []
for c in known_ordinal_text:
    if c == "smoking_level":
        ordinal_orders.append(["Non-smoker", "Light", "Heavy"])
    else:
        vals = list(pd.Categorical(df[c].dropna()).categories)
        ordinal_orders.append(vals)
for c in ordinal_numeric:
    vals = sorted(df[c].dropna().unique().tolist())
    ordinal_orders.append(vals)

# ------------------ Split ------------------
X_cat = df[cat_cols].copy()
X_ord = df[ord_cols].copy()
X_num = df[num_cols].copy()

X_train_cat, X_test_cat, \
X_train_ord, X_test_ord, \
X_train_num, X_test_num, \
y_train, y_test = train_test_split(
    X_cat, X_ord, X_num, y, test_size=0.30, random_state=42, stratify=y
)

# ------------------ Fit preprocessors on TRAIN only ------------------
onehot = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
onehot.fit(X_train_cat)
oh_names = onehot.get_feature_names_out(cat_cols).tolist()

ordinal = None
if len(ord_cols) > 0:
    ordinal = OrdinalEncoder(
        categories=ordinal_orders, handle_unknown='use_encoded_value', unknown_value=-1
    )
    ordinal.fit(X_train_ord)

scaler = StandardScaler()
scaler.fit(X_train_num)

def transform(cat, ord_, num):
    parts = []
    # numeric
    Xn = pd.DataFrame(scaler.transform(num), columns=num_cols, index=num.index)
    parts.append(Xn)
    # categorical
    if len(cat_cols) > 0:
        Xc = pd.DataFrame(onehot.transform(cat), columns=oh_names, index=cat.index)
        parts.append(Xc)
    # ordinal
    if ordinal is not None and len(ord_cols) > 0:
        Xo = pd.DataFrame(ordinal.transform(ord_), columns=ord_cols, index=ord_.index)
        parts.append(Xo)
    return pd.concat(parts, axis=1)

X_train = transform(X_train_cat, X_train_ord, X_train_num)
X_test  = transform(X_test_cat,  X_test_ord,  X_test_num)

# ------------------ SMOTE (ONLY imbalance strategy) ------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ------------------ Models (SMOTE-only; NO class weights) ------------------
candidates_smote = [
    ("LR_SMOTE",  LogisticRegression(max_iter=1000)),
    ("RF_SMOTE",  RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=42)),
    ("ET_SMOTE",  ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, random_state=42)),
    ("GB_SMOTE",  GradientBoostingClassifier(n_estimators=600, learning_rate=0.05, max_depth=2, random_state=42)),
    ("HGB_SMOTE", HistGradientBoostingClassifier(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=20, random_state=42)),
]

def evaluate(model, X_tr, y_tr, X_te, y_te, label):
    model.fit(X_tr, y_tr)
    y_score = (model.predict_proba(X_te)[:, 1]
               if hasattr(model, "predict_proba")
               else model.decision_function(X_te))
    y_pred = (y_score >= 0.5).astype(int)  # temp cutoff; tuned below
    return {
        "label": label,
        "Accuracy":  accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall":    recall_score(y_te, y_pred, zero_division=0),
        "F1":        f1_score(y_te, y_pred, zero_division=0),
        "ROC_AUC":   roc_auc_score(y_te, y_score),
        "PR_AUC":    average_precision_score(y_te, y_score),
        "model":     model,
        "y_score":   y_score
    }

# ------------------ Train & evaluate all models ------------------
results = []
for name, mdl in candidates_smote:
    results.append(evaluate(mdl, X_train_smote, y_train_smote, X_test, y_test, name))


metrics_table = pd.DataFrame([
    {
        "Model":     r["label"],
        "Accuracy":  r["Accuracy"],
        "Precision": r["Precision"],
        "Recall":    r["Recall"],
        "F1":        r["F1"],
        "ROC_AUC":   r["ROC_AUC"],
        "PR_AUC":    r["PR_AUC"],
    }
    for r in results
]).sort_values("ROC_AUC", ascending=False)

print("\n=== All models (sorted by ROC_AUC) ===")
print(metrics_table.to_string(index=False))

# ================== Per-model classification reports (tuned) ==================
print("\n=== Per-model classification reports (each with tuned threshold) ===")
tuned_summaries = []
for r in results:
   
    fpr_m, tpr_m, thr_m = roc_curve(y_test, r["y_score"])
    youden_m = tpr_m - fpr_m
    thr_best_m = thr_m[int(np.argmax(youden_m))]

  
    y_pred_m = (r["y_score"] >= thr_best_m).astype(int)

 
    print(f"\n--- {r['label']} (tuned threshold = {thr_best_m:.3f}) ---")
    print(classification_report(y_test, y_pred_m, digits=2))

    tuned_summaries.append({
        "Model":     r["label"],
        "Threshold": float(thr_best_m),
        "Accuracy":  accuracy_score(y_test, y_pred_m),
        "Precision": precision_score(y_test, y_pred_m, zero_division=0),
        "Recall":    recall_score(y_test, y_pred_m, zero_division=0),
        "F1":        f1_score(y_test, y_pred_m, zero_division=0),
        "ROC_AUC":   r["ROC_AUC"],   # threshold-free, same as earlier
        "PR_AUC":    r["PR_AUC"],
    })

tuned_table = pd.DataFrame(tuned_summaries).sort_values("ROC_AUC", ascending=False)
tuned_table.to_csv("reports/model_metrics_tuned.csv", index=False)
print("\nSaved tuned summary table: reports/model_metrics_tuned.csv")
# ============================================================================

# ------------------ Pick best by ROC-AUC, tune threshold ------------------
best = max(results, key=lambda r: r["ROC_AUC"])
best_model = best["model"]
best_scores = best["y_score"]

fpr, tpr, thr = roc_curve(y_test, best_scores)
youden = tpr - fpr
best_thr = thr[int(np.argmax(youden))]

y_pred_tuned = (best_scores >= best_thr).astype(int)
final_report = {
    "Chosen_model": best["label"],
    "Tuned_threshold": float(best_thr),
    "Accuracy":  accuracy_score(y_test, y_pred_tuned),
    "Precision": precision_score(y_test, y_pred_tuned, zero_division=0),
    "Recall":    recall_score(y_test, y_pred_tuned, zero_division=0),
    "F1":        f1_score(y_test, y_pred_tuned, zero_division=0),
    "ROC_AUC":   roc_auc_score(y_test, best_scores),
    "PR_AUC":    average_precision_score(y_test, best_scores)
}
print("\nFINAL (tuned) REPORT:", json.dumps(final_report, indent=2))


print("\n=== Classification report (tuned threshold, chosen model) ===")
print(classification_report(y_test, y_pred_tuned, digits=2))


cm = confusion_matrix(y_test, y_pred_tuned)
print("\nConfusion matrix [[TN FP]\n                 [FN TP]]:")
print(cm)


others_acc = metrics_table.loc[metrics_table["Model"] != final_report["Chosen_model"], ["Model", "Accuracy"]] \
                          .sort_values("Accuracy", ascending=False)
print("\nOther models' accuracies (descending):")
print(others_acc.to_string(index=False))

# Save reports
metrics_table.to_csv("reports/model_metrics.csv", index=False)
pd.DataFrame(classification_report(y_test, y_pred_tuned, output_dict=True, digits=2)).T \
  .to_csv("reports/classification_report_chosen.csv", index=True)

# ------------------ Save artifacts + threshold + column config ------------------
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump({
    "onehot": onehot,
    "ordinal": ordinal,  # may be None
    "scaler": scaler,
    "categorical_columns": cat_cols,
    "ordinal_columns": ord_cols,
    "numerical_columns": num_cols,
    "ordinal_orders": ordinal_orders
}, "models/preprocessing.joblib")

with open("models/threshold.json", "w") as f:
    json.dump({"threshold": float(best_thr)}, f)

with open("models/column_config.json", "w") as f:
    json.dump({
        "categorical_columns": cat_cols,
        "ordinal_columns": ord_cols,
        "numerical_columns": num_cols
    }, f, indent=2)


print("\nSaved: models/best_model.pkl, models/preprocessing.joblib, models/threshold.json, "
      "reports/model_metrics.csv, reports/model_metrics_tuned.csv, reports/classification_report_chosen.csv")
