from __future__ import annotations
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]  
MODEL_PATH = ROOT / "models" / "best_model.pkl"
PREP_PATH  = ROOT / "models" / "preprocessing.joblib"

st.set_page_config(page_title="Disease Risk – Predictor", page_icon="🩺", layout="centered")

# Style
st.markdown("""
<style>
:root{
  --bg:#0b0f16;
  --card:#0f1320cc;         /* glass */
  --border:#1b2236;
  --muted:#aab2d5;
  --text:#e6eaff;
  --accent:#60a5fa;         /* blue */
  --accent-2:#22d3ee;       /* cyan */
  --good:#34d399;           /* green */
  --bad:#f87171;            /* red */
  --shadow:0 10px 30px rgba(0,0,0,.35);
}

/* Background & container width */
#root, .main, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 800px at 15% -10%, #18213a40, transparent 60%),
    radial-gradient(1000px 700px at 110% 10%, #0e1a2e40, transparent 60%),
    var(--bg) !important;
  color: var(--text);
}
[data-testid="stAppViewBlockContainer"]{
  max-width: 1120px; padding: 12px 12px 40px 12px;
}

/* Title (prevent clipping / half-visible text) */
h1{
  margin-top: 8px !important;
  line-height: 1.1 !important;
  word-break: break-word !important;
  overflow: visible !important;
  text-wrap: balance;
  letter-spacing:.2px;
  background:linear-gradient(90deg,#eef2ff,#93c5fd);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}

/* Subtitle */
header + div p, .stMarkdown p { color: var(--muted) !important; }

/* Cards */
.card{
  background: var(--card);
  border:1px solid var(--border);
  border-radius: 20px;
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  transition: border .2s ease, transform .2s ease;
}
.card:hover{ border-color:#2a3553; transform: translateY(-1px); }
.card:empty{ display:none; }

/* Stepper */
.stepper{ display:flex; align-items:center; gap:12px; margin:8px 0 12px; }
.step{
  width:32px; height:32px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  color:#cbd5ff; font-size:12px; font-weight:700;
  background: linear-gradient(180deg,#151b2d,#0f1424);
  border:1px solid #253052;
  box-shadow: inset 0 1px 0 #263257, 0 4px 10px rgba(0,0,0,.25);
}
.step.active{
  color:#001021;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  border-color: transparent;
  box-shadow: 0 0 0 4px #60a5fa22, 0 8px 20px #22d3ee33;
}
.step-line{ height:2px; width:56px; border-radius:999px; background:#29324c; position:relative; overflow:hidden; }
.step-line::after{
  content:""; position:absolute; inset:0;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  transform: translateX(-100%); animation: flow 2.2s linear infinite;
}
@keyframes flow{ to{ transform: translateX(100%);}}

/* Subheaders */
h2, h3, .stSubheader, .stMarkdown h2{
  color:#fff !important; letter-spacing:.2px;
}

/* Labels */
label, .stMarkdown p { color: var(--muted) !important; }

/* Inputs */
input[type="number"], input[type="text"], textarea{
  background:#0c1222 !important; color:var(--text) !important;
  border:1px solid #24304e !important; border-radius:14px !important;
  padding:10px 12px !important; height:44px; box-shadow:none !important;
  transition:border .2s ease, box-shadow .2s ease;
}
input:focus, textarea:focus{
  border-color:#3b82f6 !important; box-shadow:0 0 0 4px #3b82f622 !important; outline:none !important;
}

/* Remove Streamlit's +/- buttons on number_input */
[data-testid="stNumberInput"] button{ display:none !important; }
[data-testid="stNumberInput"] div[role="spinbutton"]{ padding-right: 6px !important; }

/* Remove browser spin buttons */
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button{ -webkit-appearance:none; margin:0;}
input[type=number]{ -moz-appearance:textfield; }

/* Selects & Radios */
.stSelectbox, [data-baseweb="select"]>div{
  border-radius:14px !important; background:#0c1222 !important; border:1px solid #24304e !important;
}
.stRadio [role="radio"]{
  padding:8px 12px; border-radius:999px; border:1px solid #233050;
  background:#0f1628; color:#c7d2fe; margin-right:8px;
}
.stRadio [aria-checked="true"]{
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  color:#0b1020; border-color: transparent;
}

/* Slider – hide ticks / min-max labels */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
.stSlider [data-testid="stTickBar"] span{ display:none !important; }
.stSlider [data-baseweb="slider"]>div>div{
  background:linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"]{ box-shadow:0 0 0 4px #60a5fa33; }

/* Buttons */
.stButton>button{
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  color:#0a0f1c !important; font-weight:700; letter-spacing:.2px;
  border:none !important; border-radius:14px !important;
  padding:10px 16px !important; width:100%;
  box-shadow: 0 10px 20px #22d3ee2e, inset 0 0 0 1px #ffffff22;
  transition: transform .08s ease, filter .2s ease, box-shadow .2s ease;
}
.stButton>button:hover{ filter:brightness(1.05); }
.stButton>button:active{ transform: translateY(1px); }
.stButton:has(button):not(:has(button[type="primary"])) > button{
  background:#121a2c !important; color:#c7cff9 !important;
  border:1px solid #273356 !important; box-shadow:none !important;
}

/* Metric & Progress */
[data-testid="stMetric"]{
  background: var(--card); border:1px solid var(--border);
  border-radius: 18px; padding: 14px 16px; box-shadow: var(--shadow);
}
[data-testid="stMetricValue"]{ color:#ffffff !important; font-weight:800 !important; text-shadow:0 2px 20px #22d3ee33; }
[data-testid="stMetricLabel"]{ color:#9fb0e6 !important; }
.stProgress > div > div{
  background:#0f1628 !important; border:1px solid #24304e; border-radius:999px;
}
.stProgress > div > div > div{
  background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
}

/* Result badge */
.result-badge{
  display:inline-flex; align-items:center; gap:10px;
  padding:10px 14px; border-radius:14px; font-weight:800;
  letter-spacing:.2px; box-shadow: var(--shadow);
}
.result-ok   { background:#052e1e; color:#a7f3d0; border:1px solid #0b3a29; }
.result-bad  { background:#3a0b12; color:#fecaca; border:1px solid #4c0f17; }

/* Uniform vertical spacing for widgets */
[data-testid="stVerticalBlock"] > div:has(> label),
.stRadio, .stSelectbox, [data-testid="stNumberInput"], .stSlider { margin-bottom: 10px !important; }

/* Links & hr */
a{ color:#7dd3fc; } hr{ border:none; border-top:1px solid #1b2440;}
</style>
""", unsafe_allow_html=True)

# Load Artifacts
missing = []
if not MODEL_PATH.exists(): missing.append(str(MODEL_PATH))
if not PREP_PATH.exists():  missing.append(str(PREP_PATH))
if missing:
    st.error("Required file(s) not found:\n- " + "\n- ".join(missing))
    st.stop()

model = joblib.load(MODEL_PATH)
prep  = joblib.load(PREP_PATH)

onehot  = prep["onehot"]
ordinal = prep["ordinal"]
scaler  = prep["scaler"]

categorical_cols = prep["categorical_columns"]
ordinal_cols     = prep["ordinal_columns"]
numerical_cols   = prep["numerical_columns"]

onehot_names = list(onehot.get_feature_names_out(categorical_cols))
TRAIN_COL_ORDER = numerical_cols + onehot_names + ordinal_cols

LABEL_MAP = {0: "Healthy Person", 1: "Diseased Person"}
FIXED_THRESHOLD = 0.5  

# Session 
if "step" not in st.session_state: st.session_state.step = 1
if "inputs" not in st.session_state: st.session_state.inputs = {}

def reset_all():
    st.session_state.step = 1
    st.session_state.inputs = {}

def build_features_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    req = set(numerical_cols) | set(categorical_cols) | set(ordinal_cols)
    miss = [c for c in req if c not in raw_df.columns]
    if miss: raise ValueError(f"Missing required column(s): {miss}")

    df = raw_df.copy()
    if "stress_level" in df.columns and not np.issubdtype(df["stress_level"].dtype, np.number):
        df["stress_level"] = pd.to_numeric(df["stress_level"], errors="coerce")

    X_num = pd.DataFrame(scaler.transform(df[numerical_cols]), columns=numerical_cols, index=df.index)
    X_cat = pd.DataFrame(onehot.transform(df[categorical_cols]), columns=onehot_names, index=df.index)
    X_ord = pd.DataFrame(ordinal.transform(df[ordinal_cols]), columns=ordinal_cols, index=df.index)
    X = pd.concat([X_num, X_cat, X_ord], axis=1)[TRAIN_COL_ORDER]
    return X

def predict_threshold_0_5(X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(X)[:, 1][0])
        pred = int(p >= FIXED_THRESHOLD)
        return pred, p
    pred = int(model.predict(X)[0])
    return pred, None

# Header
st.title("🩺 Disease Risk – Predictor")
st.caption("Two steps → instant prediction")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    '<div class="stepper">'
    f'<div class="step {"active" if st.session_state.step>=1 else ""}">1</div>'
    '<div class="step-line"></div>'
    f'<div class="step {"active" if st.session_state.step>=2 else ""}">2</div>'
    '<div class="step-line"></div>'
    f'<div class="step {"active" if st.session_state.step>=3 else ""}">✓</div>'
    '</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Step 1
if st.session_state.step == 1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Step 1 · Lifestyle & Family History")

    with st.form("step1"):
        c1, c2, c3 = st.columns(3, gap="small")

        gender_opts  = list(onehot.categories_[categorical_cols.index("gender")])
        alcohol_opts = list(onehot.categories_[categorical_cols.index("alcohol_consumption")])
        famhist_opts = list(onehot.categories_[categorical_cols.index("family_history")])

        gender = c1.selectbox("Gender", gender_opts, key="gender")
        alcohol_consumption = c2.selectbox("Alcohol consumption", alcohol_opts, key="alcohol")
        family_history = c3.selectbox("Family history of disease", famhist_opts, key="famhist")

        smoking_cats = list(ordinal.categories_[ordinal_cols.index("smoking_level")])
        stress_cats  = list(ordinal.categories_[ordinal_cols.index("stress_level")])

        c4, c5 = st.columns(2, gap="small")
        smoking_level = c4.radio("Smoking level", smoking_cats, horizontal=True, key="smoking")
        stress_level  = c5.slider("Stress level",
                                  min_value=int(min(stress_cats)),
                                  max_value=int(max(stress_cats)),
                                  value=int(min(stress_cats)), step=1, key="stress")

        _, _, btn_col = st.columns([5, 1, 1])
        next_clicked = btn_col.form_submit_button("Next ➜", use_container_width=True)

        if next_clicked:
            st.session_state.inputs.update({
                "gender": gender,
                "alcohol_consumption": alcohol_consumption,
                "family_history": family_history,
                "smoking_level": smoking_level,
                "stress_level": stress_level,
            })
            st.session_state.step = 2
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Step 2
elif st.session_state.step == 2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Step 2 · Clinical & Activity Metrics")

    labels = {
        "age":"Age","waist_size":"Waist size","blood_pressure":"Blood pressure",
        "cholesterol":"Cholesterol","glucose":"Glucose","insulin":"Insulin",
        "sleep_hours":"Sleep hours","physical_activity":"Physical activity",
        "calorie_intake":"Calorie intake","sugar_intake":"Sugar intake"
    }

    with st.form("step2"):
        values = {}
        for i, col_name in enumerate(numerical_cols):
            if i % 2 == 0:
                col1, col2 = st.columns(2, gap="medium")

            target_col = col1 if i % 2 == 0 else col2
            values[col_name] = target_col.number_input(
                labels.get(col_name, col_name),
                value=float(st.session_state.inputs.get(col_name, 0.0)),
                step=1.0, format="%.0f", key=col_name
            )
        left, right = st.columns([1,1])
        back_clicked    = left.form_submit_button("◀ Back", use_container_width=True)
        predict_clicked = right.form_submit_button("Predict ▶", use_container_width=True)

        if back_clicked:
            st.session_state.step = 1
            st.rerun()

        if predict_clicked:
            st.session_state.inputs.update(values)
            st.session_state.step = 3
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# Step 3
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Result")

    row = {k: st.session_state.inputs[k] for k in (set(numerical_cols)|set(categorical_cols)|set(ordinal_cols))}
    raw_df = pd.DataFrame([row])

    try:
        X = build_features_df(raw_df)
        pred, prob = predict_threshold_0_5(X)

        label = LABEL_MAP.get(pred, "Unknown")
        badge_class = "result-ok" if pred == 0 else "result-bad"
        icon = "✅" if pred == 0 else "⚠️"

        st.markdown(f"""
            <div class="result-badge {badge_class}">{icon} {label}</div>
        """, unsafe_allow_html=True)

        if prob is not None:
            p = min(max(prob, 0.0), 1.0)
            st.progress(p, text=f"Estimated probability of disease: {p:.3f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

    c1, c2 = st.columns([1,1])
    if c1.button("◀ Back to Step 2"): st.session_state.step = 2; st.rerun()
    if c2.button("Start over"):        reset_all(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
