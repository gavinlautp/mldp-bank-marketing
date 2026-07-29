# TargetCall — Bank Marketing Term Deposit Predictor

Machine Learning for Developers (CAI2C08) — individual project.
A machine learning solution that predicts whether a bank customer will
subscribe to a **term deposit**, so a telemarketing team can prioritise
its call list and stop wasting calls on unlikely leads.

**Live app:** https://mldp-bank-marketing-ys2vyzpdvk6xezksydqysb.streamlit.app/

---

## Business problem

A Portuguese retail bank runs outbound telemarketing campaigns to sell term
deposits. Historically only about **12% of calls succeed** — nearly nine in ten
are wasted call-centre time. This project scores customers *before* the call so
the bank can focus effort on high-probability leads and lift campaign ROI
without extra headcount.

- **Task:** binary classification (`y` = will subscribe: yes / no)
- **Primary metric:** F1-score on the "yes" class (the target is imbalanced ~88% no / 12% yes, so plain accuracy is misleading)

## Dataset

- **Bank Marketing** — UCI Machine Learning Repository (ID 222), 45,211 records
- Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Citation: Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems, 62, 22–31.

## Approach (CRISP-DM)

1. **Data understanding** — EDA on distributions, outliers, correlation, and subscription rate by feature.
2. **Data preparation** — drop `duration` (data leakage); engineer `previously_contacted`, `previous_success`, and `age_band`; one-hot encode; stratified 70/30 train-test split.
3. **Modelling** — Dummy baseline vs Logistic Regression vs Random Forest (`class_weight="balanced"`), scikit-learn only.
4. **Evaluation & tuning** — `RandomizedSearchCV` on the Random Forest (`n_estimators`, `max_depth`), feature selection by cumulative importance, and a decision-threshold analysis framed in business terms.
5. **Deployment** — Streamlit web app for scoring a single customer.

**Final model:** tuned Random Forest, chosen on F1 on the "yes" class. The
notebook documents every version tried (v0–v4) with a fair comparison on the
same held-out test set.

## Repository structure

| File | Description |
|------|-------------|
| `MLDP_Program_Codes.ipynb` | Full analysis: EDA, preparation, modelling, tuning, evaluation |
| `streamlit_app.py` | Streamlit web app that loads the model and scores a customer |
| `bank_marketing_rf_model.pkl` | Trained Random Forest model (saved with joblib) |
| `model_features.csv` | Column order the model expects (for input alignment) |
| `bank-full.csv` | Dataset (UCI Bank Marketing) |
| `requirements.txt` | Python dependencies |

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

**Notebook** — open `MLDP_Program_Codes.ipynb` in Jupyter and run all cells.
This trains the models and saves `bank_marketing_rf_model.pkl`.

**Web app** — from the repo folder:

```bash
streamlit run streamlit_app.py
```

The app rebuilds the same engineered features from the user's inputs, aligns
them to the trained model, and returns a subscription-likelihood prediction
with a call/skip recommendation.

## Author

Lau Zheng Yuan Gavin — 2501290G

