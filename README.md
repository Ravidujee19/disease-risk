# Disease Risk from Daily Habits

Mini Project – Fundamentals of Data Mining (IT3051)  
Team Predictora | SLIIT – 2025  

## 📌 Overview
This project predicts lifestyle-related **disease risk** (Low/Medium/High) from daily habits and biometrics using machine learning.

## 🚀 Setup

1. Clone repo:
   ```bash
   git clone https://github.com/Ravidujee19/disease-risk
   cd disease-risk
   ```

2. Create virtual environment (Windows PowerShell):
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Place Kaggle dataset into:
   ```
   data/raw/disease_risk.csv
   ```

4. Run notebooks or scripts:
   ```bash
   python -m src.data_prep
   python -m src.train
   python -m src.evaluate
   streamlit run app/streamlit_app.py
   ```

## 📂 Structure
- `data/` → datasets (raw, processed)
- `notebooks/` → Jupyter notebooks for EDA/modeling
- `src/` → scripts for preprocessing, training, evaluation
- `app/` → Streamlit app for predictions
- `reports/` → figures + report docs
- `models/` → trained models

---
