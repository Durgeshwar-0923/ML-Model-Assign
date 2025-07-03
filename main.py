import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from collections import Counter

# ─── CONFIGURATION ──────────────────────────────────────────────
REQUIRED_COLUMNS = ['Income', 'Age', 'Experience', 'CCAvg', 'Mortgage']
RB_COLS = ['CCAvg', 'Mortgage']
STD_COLS = ['Income', 'Experience', 'Age']

st.set_page_config(
    page_title="🏦 Loan Predictor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── LOADING ARTIFACTS ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts(models_dir='artifacts'):
    pt = joblib.load(os.path.join(models_dir, 'pt.pkl'))
    rs = joblib.load(os.path.join(models_dir, 'rs.pkl'))
    ss = joblib.load(os.path.join(models_dir, 'ss.pkl'))
    selector = joblib.load(os.path.join(models_dir, 'selector.pkl'))

    models = {}
    for fn in os.listdir(models_dir):
        if fn.endswith('_model.pkl'):
            name = fn.replace('_model.pkl', '')
            models[name] = joblib.load(os.path.join(models_dir, fn))
    return pt, rs, ss, selector, models

# ─── PREPROCESSING ─────────────────────────────────────────────
def preprocess(df, pt, rs, ss, selector):
    df = df.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1, errors='ignore')
    df[RB_COLS] = pt.transform(df[RB_COLS])
    df[RB_COLS] = rs.transform(df[RB_COLS])
    df[STD_COLS] = ss.transform(df[STD_COLS])
    selected = selector.transform(df)
    return pd.DataFrame(selected, columns=[f'F{i}' for i in range(selected.shape[1])])

# ─── MAJORITY VOTING ───────────────────────────────────────────
def compute_majority_vote(df, model_names):
    preds = df[[f'Prediction_{m}' for m in model_names]]
    df['Final_Prediction'] = preds.mode(axis=1)[0]
    return df

# ─── MAIN ──────────────────────────────────────────────────────
def main():
    st.title("🏦 Loan Predictor Dashboard")
    st.markdown("""
    Upload a **CSV** file with required fields and get predictions from multiple models, along with majority vote results.
    """)

    pt, rs, ss, selector, models = load_artifacts()

    with st.sidebar:
        st.header("🔧 Instructions")
        st.markdown("""
        - Required Columns:
          - `Income`, `Age`, `Experience`, `CCAvg`, `Mortgage`
        - Exclude: `Personal Loan`
        - Output includes:
          - Model-wise Predictions
          - Confidence Scores
          - Final Majority Decision
        """)

    uploaded = st.file_uploader("📤 Upload your CSV file", type=['csv'])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")
            return

        # Validate
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
            return

        st.success("✅ File uploaded successfully!")
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        with st.spinner("🔄 Processing and predicting..."):
            X = preprocess(df.copy(), pt, rs, ss, selector)

            for name, model in models.items():
                pred = model.predict(X)
                df[f'Prediction_{name}'] = pred

                if hasattr(model, "predict_proba"):
                    conf = model.predict_proba(X)[:, 1]
                    df[f'Confidence_{name}'] = (conf * 100).round(2)

            df = compute_majority_vote(df, list(models.keys()))

        st.success("🎯 Prediction complete!")

        # Summary stats
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Final Prediction Summary")
            counts = df['Final_Prediction'].value_counts().sort_index()
            labels = ['Not Approved', 'Approved']
            fig = px.pie(
                names=[labels[int(i)] for i in counts.index],
                values=counts.values,
                title="Loan Approval Distribution",
                color_discrete_sequence=['#ff6b6b', '#1dd1a1']
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🧠 Model Performance (Sample-wise)")
            vote_df = df[[col for col in df.columns if col.startswith('Prediction_')]]
            st.dataframe(vote_df.head(), use_container_width=True)

        st.subheader("📑 Full Results")
        st.dataframe(df, use_container_width=True)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Results CSV",
            data=csv,
            file_name="loan_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("📁 Please upload a CSV file to begin.")


if __name__ == "__main__":
    main()
