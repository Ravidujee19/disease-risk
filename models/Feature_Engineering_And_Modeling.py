import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib

# paths
os.makedirs("models", exist_ok=True)
DATA_PATH = "data/processed/training_data.csv"

# load
data = pd.read_csv(DATA_PATH)

# columns
categorical_columns = ['gender', 'alcohol_consumption', 'family_history']
ordinal_columns = ['smoking_level', 'stress_level']

numerical_columns = [
    'age','waist_size','blood_pressure','cholesterol','glucose','insulin',
    'sleep_hours','physical_activity','calorie_intake','sugar_intake'
]

# clean / coerce
# make sure stress_level is numeric
if not np.issubdtype(data['stress_level'].dtype, np.number):
    data['stress_level'] = pd.to_numeric(data['stress_level'], errors='coerce')

# define category orders 
smoking_order = ['Non-smoker', 'Light', 'Heavy']
stress_order  = list(range(0, 11))

# encoders
onehot_encoder   = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
ordinal_encoder  = OrdinalEncoder(
    categories=[smoking_order, stress_order],
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

# fit encoders
onehot_encoder.fit(data[categorical_columns])
ordinal_encoder.fit(data[ordinal_columns])

# transform
onehot_data  = onehot_encoder.transform(data[categorical_columns])
ordinal_data = ordinal_encoder.transform(data[ordinal_columns])

# back to DataFrame
onehot_df  = pd.DataFrame(onehot_data,
                          columns=onehot_encoder.get_feature_names_out(categorical_columns),
                          index=data.index)
ordinal_df = pd.DataFrame(ordinal_data,
                          columns=ordinal_columns,
                          index=data.index)

# combine features
X = pd.concat([data[numerical_columns], onehot_df, ordinal_df], axis=1)

# target (0: healthy, 1: diseased)
y = data['target'].map({'healthy': 0, 'diseased': 1}).astype(int)

# sanity check
assert len(X) == len(y), "X and y must have same rows"

# split (stratified) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# scale numeric for LR stability (fit on train only) 
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_columns])
X_test_num  = scaler.transform(X_test[numerical_columns])

X_train_scaled = pd.concat([
    pd.DataFrame(X_train_num, columns=numerical_columns, index=X_train.index),
    X_train.drop(columns=numerical_columns)
], axis=1)

X_test_scaled = pd.concat([
    pd.DataFrame(X_test_num, columns=numerical_columns, index=X_test.index),
    X_test.drop(columns=numerical_columns)
], axis=1)

# imbalance handling 
# class weights from ORIGINAL y_train (not SMOTE)
classes = np.array([0, 1])
cw = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, cw))

# SMOTE on training set only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight=class_weight_dict, random_state=42)
}

def evaluate(model, X_te, y_te):
    y_pred = model.predict(X_te)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_te)
    else:
        y_score = y_pred
    return {
        'Accuracy':  accuracy_score(y_te, y_pred),
    }

# train & evaluate
results = {}
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    results[name] = evaluate(model, X_test_scaled, y_test)

results_df = pd.DataFrame(results).T.sort_values(by='Accuracy', ascending=False)
print(results_df)

# Save the best
best_model_name = results_df.index[0]
best_model = models[best_model_name]
print(best_model)
joblib.dump(best_model, "models/best_model.pkl")

joblib.dump({
    "onehot": onehot_encoder,
    "ordinal": ordinal_encoder,
    "scaler": scaler,
    "categorical_columns": categorical_columns,
    "ordinal_columns": ordinal_columns,
    "numerical_columns": numerical_columns
}, "models/preprocessing.joblib")
