# Disease Risk Predictor 🩺

## Overview
This project is a **Machine Learning-based Disease Risk Prediction System** built using Python, Scikit-learn, and Streamlit.  
It enables end-users to input their daily lifestyle and health-related attributes to predict the likelihood of developing certain diseases.

The project is divided into multiple phases following a **data science pipeline**:
1. **Dataset Collection & Preparation** – Raw health and lifestyle dataset is stored in `data/raw/`.
2. **Preprocessing & Feature Engineering** – Cleaned and transformed data stored in `data/processed/`.
3. **Exploratory Data Analysis (EDA)** – Insights and visualizations created in Jupyter notebooks (`notebooks/eda_only.ipynb`).
4. **Modeling** – Multiple ML models (Random Forest, Logistic Regression) trained and evaluated.
5. **Streamlit Deployment** – Interactive web UI for predictions (`app/streamlit_app.py`).

---

## Project Structure
```
disease-risk-main/
│── app/
│   ├── streamlit_app.py       # Streamlit frontend app
│   └── .streamlit/config.toml # UI configuration
│
│── data/
│   ├── raw/                   # Original dataset
│   ├── processed/             # Preprocessed data
│   └── .gitkeep
│
│── models/
│   ├── preprocessing.joblib   # Preprocessing pipeline
│   └── best_model.pkl         # Saved best trained model
│
│── notebooks/
│   ├── Feature_Engineering_And_Modeling.py
│   ├── eda_only.ipynb         # Exploratory Data Analysis
│   └── preprocess.py          # Preprocessing scripts
│
│── reports/
│   ├── figures/               # Visualizations
│   ├── before_preprocessed/   # Pre-processed analysis
│   └── after_preprocessed/    # Post-processed analysis
│
│── requirements.txt           # Python dependencies
│── instruction.txt            # Instructions for running
│── .gitignore
```

---

## Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/Ravidujee19/disease-risk
   cd disease-risk-main
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # For Linux/Mac
   .venv\Scripts\activate    # For Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the App
Run the Streamlit app locally:
```bash
streamlit run app/streamlit_app.py
```

Then open your browser at `http://localhost:8501`.

---

## Features
- User-friendly **Streamlit UI** for single or batch predictions.
- Handles **categorical and numerical preprocessing** with saved pipeline (`preprocessing.joblib`).
- **Multiple ML models** trained, best one saved as `best_model.pkl`.
- Supports **Exploratory Data Analysis (EDA)** via Jupyter notebooks.
- Modular codebase following **MLOps best practices** (data → preprocess → model → deploy).

---

## Future Improvements
- Integration with cloud storage for model artifacts.
- Docker-based deployment for reproducibility.
- Enhanced model explainability with SHAP or LIME.
- Continuous retraining pipeline for new data.

---

## License
This project is licensed under the MIT License.  
Feel free to use, modify, and distribute with attribution.
