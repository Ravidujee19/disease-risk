from __future__ import annotations
from pathlib import Path
import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Model path
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "best_model.pkl"
PREP_PATH = MODELS / "preprocessing.joblib"

st.set_page_config(page_title="Disease Risk Predictor", page_icon="🩺", layout="wide")
st.markdown("""
<style>
/* cards */
.card {background: #11131a; border: 1px solid #202434; border-radius: 16px; padding: 18px 18px; margin-bottom: 12px;}
.card h3 {margin: 0 0 8px 0}

/* stepper */
.stepper {display:flex; align-items:center; gap:8px; margin: 10px 0 16px;}
.step {width:26px; height:26px; border-radius:50%; display:flex; align-items:center; justify-content:center;
       background:#1d2233; border:1px solid #2a2f45; color:#aab2d5; font-size:12px;}
.step.active {background:#3b82f6; color:white; border-color:#3b82f6;}
.step-line {height:2px; width:40px; background:#2a2f45;}
.small {color:#9aa3c3; font-size:13px}
.kbd {background:#22263a; padding:2px 6px; border-radius:6px; border:1px solid #2f3550; font-size:12px;}
</style>
""", unsafe_allow_html=True)

missing = []
if not MODEL_PATH.exists(): missing.append(str(MODEL_PATH))
if not PREP_PATH.exists(): missing.append(str(PREP_PATH))
if missing:
    st.error("Required file(s) not found:\n- " + "\n- ".join(missing))
    st.stop()

# Load model and preprocessing 
model = joblib.load(MODEL_PATH)
prep = joblib.load(PREP_PATH)
onehot  = prep["onehot"]
ordinal = prep["ordinal"]
scaler  = prep["scaler"]
categorical_cols = prep["categorical_columns"]          
ordinal_cols     = prep["ordinal_columns"]              
numerical_cols   = prep["numerical_columns"]            

onehot_feature_names = list(onehot.get_feature_names_out(categorical_cols))
TRAIN_COL_ORDER = numerical_cols + onehot_feature_names + ordinal_cols
LABEL_MAP = {0: "healthy", 1: "diseased"}

# Session
if "wizard_step" not in st.session_state: st.session_state.wizard_step = 1
if "inputs" not in st.session_state: st.session_state.inputs = {}

def reset_all():
    st.session_state.wizard_step = 1
    st.session_state.inputs = {}

def build_features_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    req = set(numerical_cols) | set(categorical_cols) | set(ordinal_cols)
    miss = [c for c in req if c not in raw_df.columns]
    if miss: raise ValueError(f"Missing required input column(s): {miss}")

    df = raw_df.copy()
    if "stress_level" in df.columns and not np.issubdtype(df["stress_level"].dtype, np.number):
        df["stress_level"] = pd.to_numeric(df["stress_level"], errors="coerce")

    X_num = pd.DataFrame(scaler.transform(df[numerical_cols]), columns=numerical_cols, index=df.index)
    X_cat = pd.DataFrame(onehot.transform(df[categorical_cols]), columns=onehot_feature_names, index=df.index)
    X_ord = pd.DataFrame(ordinal.transform(df[ordinal_cols]), columns=ordinal_cols, index=df.index)
    X = pd.concat([X_num, X_cat, X_ord], axis=1)[TRAIN_COL_ORDER]
    return X

def predict_with_threshold(X: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= thr).astype(int)
        return pd.DataFrame({"pred": pred, "prob_1": prob}, index=X.index)
    pred = model.predict(X)
    return pd.DataFrame({"pred": pred}, index=X.index)

def csv_template() -> bytes:
    """Generate a blank CSV template with the raw input columns in the right order."""
    cols = numerical_cols + categorical_cols + ordinal_cols
    df = pd.DataFrame(columns=cols)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# side nav
st.sidebar.title("🩺 Disease Risk")
page = st.sidebar.radio("Navigation", ["Single (Wizard)", "Batch CSV"], index=0)
with st.sidebar.expander("Model & Schema", expanded=False):
    st.write("**Numerical (10):**", ", ".join(numerical_cols))
    st.write("**Categorical (3):**", ", ".join(categorical_cols))
    st.write("**Ordinal (2):**", ", ".join(ordinal_cols))
    st.caption("Model expects scaled numericals + one-hot(categorical) + encoded ordinals.")

st.sidebar.markdown("---")
if st.sidebar.button("Reset session"):
    reset_all()
    st.sidebar.success("Cleared 👌")

# Single prediction
if page.startswith("Single"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="stepper">'
                f'<div class="step {"active" if st.session_state.wizard_step>=1 else ""}">1</div>'
                '<div class="step-line"></div>'
                f'<div class="step {"active" if st.session_state.wizard_step>=2 else ""}">2</div>'
                '<div class="step-line"></div>'
                f'<div class="step {"active" if st.session_state.wizard_step>=3 else ""}">✓</div>'
                '</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Step 1
    if st.session_state.wizard_step == 1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Step 1 · Lifestyle & Family History")

        c1, c2, c3 = st.columns(3)
        gender_opts  = list(onehot.categories_[categorical_cols.index("gender")])
        alcohol_opts = list(onehot.categories_[categorical_cols.index("alcohol_consumption")])
        famhist_opts = list(onehot.categories_[categorical_cols.index("family_history")])

        gender = c1.selectbox("Gender", gender_opts, help="Biological sex as recorded.")
        alcohol_consumption = c2.selectbox("Alcohol consumption", alcohol_opts, help="Typical consumption level.")
        family_history = c3.selectbox("Family history of disease", famhist_opts, help="Immediate family history.")

        smoking_cats = list(ordinal.categories_[ordinal_cols.index("smoking_level")])
        stress_cats  = list(ordinal.categories_[ordinal_cols.index("stress_level")])
        c4, c5 = st.columns(2)
        smoking_level = c4.radio("Smoking level", smoking_cats, horizontal=True)
        stress_level  = c5.slider("Stress level", min_value=int(min(stress_cats)), max_value=int(max(stress_cats)),
                                  value=int(min(stress_cats)), step=1, help="0 = none, 10 = extreme")

        col_a, col_b = st.columns([1,1])
        if col_a.button("Use example"):
            gender, alcohol_consumption, family_history = gender_opts[0], alcohol_opts[0], famhist_opts[0]
            smoking_level, stress_level = smoking_cats[0], int(min(stress_cats))
            st.toast("Example filled")
        go = col_b.button("Next ➜", type="primary")

        if go:
            st.session_state.inputs.update({
                "gender": gender,
                "alcohol_consumption": alcohol_consumption,
                "family_history": family_history,
                "smoking_level": smoking_level,
                "stress_level": stress_level,
            })
            st.session_state.wizard_step = 2
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 2
    elif st.session_state.wizard_step == 2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Step 2 · Clinical & Activity Metrics")

        # 10 numericals in a neat grid
        grid = st.columns(5)
        values = {}
        helps = {
            "age": "Years",
            "waist_size": "Inches or cm (consistent with training)",
            "blood_pressure": "Systolic or combined score per your dataset",
            "cholesterol": "Standardized value",
            "glucose": "Fasting glucose level",
            "insulin": "Fasting insulin level",
            "sleep_hours": "Average hours per night",
            "physical_activity": "Minutes/day or your dataset’s unit",
            "calorie_intake": "Daily kcal",
            "sugar_intake": "Daily grams"
        }
        for i, col in enumerate(numerical_cols):
            values[col] = grid[i % 5].number_input(col, value=float(st.session_state.inputs.get(col, 0.0)),
                                                   step=1.0, format="%.4f", help=helps.get(col))

        thr = st.slider("Decision threshold (for 1 = diseased)", 0.0, 1.0, 0.5, 0.01)
        back, predict = st.columns([1,1])
        if back.button("◀ Back"):
            st.session_state.wizard_step = 1
            st.rerun()
        if predict.button("Predict ▶", type="primary"):
            st.session_state.inputs.update(values)
            st.session_state.inputs["__threshold__"] = thr
            st.session_state.wizard_step = 3
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 3
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Result")
        row = {k: st.session_state.inputs[k] for k in (set(numerical_cols) | set(categorical_cols) | set(ordinal_cols))}
        raw_df = pd.DataFrame([row])
        try:
            X = build_features_df(raw_df)
            pred_df = predict_with_threshold(X, st.session_state.inputs.get("__threshold__", 0.5))
            pred = int(pred_df.iloc[0]["pred"])
            st.metric("Prediction", f"{pred} · {LABEL_MAP.get(pred)}")
            if "prob_1" in pred_df.columns:
                p = float(pred_df.iloc[0]["prob_1"])
                st.progress(min(max(p, 0.0), 1.0), text=f"Probability of diseased: {p:.3f}")
            c1, c2 = st.columns([1,1])
            if c1.button("◀ Back to Step 2"): 
                st.session_state.wizard_step = 2; st.rerun()
            if c2.button("Start over"):
                reset_all(); st.rerun()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# batch prediction
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Batch Prediction")
    st.write("Upload a CSV containing **raw** feature columns (no target):")
    st.code(
        "numerical (10): " + ", ".join(numerical_cols) + "\n" +
        "categorical (3): " + ", ".join(categorical_cols) + "\n" +
        "ordinal (2): " + ", ".join(ordinal_cols),
        language="text",
    )

    ctmpl, cthr = st.columns([1,1])
    ctmpl.download_button("Download CSV template", data=csv_template(),
                          file_name="batch_template.csv", mime="text/csv")
    thr = cthr.slider("Decision threshold (for 1 = diseased)", 0.0, 1.0, 0.5, 0.01)

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            raw_df = pd.read_csv(up)
            X = build_features_df(raw_df)
            out = predict_with_threshold(X, thr)
            res = pd.concat([raw_df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
            st.success(f" Predicted {len(res)} rows.")
            st.download_button("Download predictions", data=res.to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
