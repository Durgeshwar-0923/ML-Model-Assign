# app.py
import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ─── CONFIGURATION ────────────────────────────────────────────────────────
REQUIRED_COLUMNS = ['Income', 'Age', 'Experience', 'CCAvg', 'Mortgage']
RB_COLS = ['CCAvg', 'Mortgage']
STD_COLS = ['Income', 'Experience', 'Age']

st.set_page_config(
    page_title="🏦 Bank Loan Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── LOADING ARTIFACTS ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts(models_dir='saved_models'):
    try:
        pt       = joblib.load(os.path.join(models_dir, 'pt.pkl'))
        rs       = joblib.load(os.path.join(models_dir, 'rs.pkl'))
        ss       = joblib.load(os.path.join(models_dir, 'ss.pkl'))
        selector = joblib.load(os.path.join(models_dir, 'selector.pkl'))

        models = {}
        for fn in os.listdir(models_dir):
            if fn.endswith('_model.pkl'):
                model_name = fn.replace('_model.pkl', '')
                models[model_name] = joblib.load(os.path.join(models_dir, fn))
        return pt, rs, ss, selector, models
    except Exception as e:
        st.error(f"❌ Failed to load model artifacts: {e}")
        st.stop()

# ─── PREPROCESSING ────────────────────────────────────────────────────────
def preprocess(df, pt, rs, ss, selector):
    df = df.copy()

    # Drop non-feature columns
    df.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1, inplace=True, errors='ignore')

    # Transform specific columns
    df[RB_COLS] = pt.transform(df[RB_COLS])
    df[RB_COLS] = rs.transform(df[RB_COLS])
    df[STD_COLS] = ss.transform(df[STD_COLS])

    # Feature selection
    selected = selector.transform(df)
    return pd.DataFrame(selected, columns=[f"F{i}" for i in range(selected.shape[1])])

# ─── MAIN APP ─────────────────────────────────────────────────────────────
def main():
    st.title("🏦 Bank Loan Predictor")
    st.markdown("""
    Upload a **CSV** file with these columns:  
    `Income`, `Age`, `Experience`, `CCAvg`, `Mortgage`  
    and get predictions from multiple machine learning models.
    """)

    pt, rs, ss, selector, models = load_artifacts()

    uploaded = st.file_uploader(
        "📁 Upload your CSV file",
        type=['csv'],
        help="Do not include the target column 'Personal Loan'"
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ Failed to read the CSV: {e}")
            return

        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
            return

        st.subheader("📊 Preview of Uploaded Data")
        st.dataframe(df.head(), use_container_width=True)

        with st.spinner("🔄 Running preprocessing and prediction..."):
            X_transformed = preprocess(df, pt, rs, ss, selector)

            prediction_results = []
            for name, model in models.items():
                try:
                    preds = model.predict(X_transformed)
                    if hasattr(model, 'predict_proba'):
                        confs = model.predict_proba(X_transformed)[:, 1]
                        df[f'Confidence_{name}'] = np.round(confs * 100, 2)
                    df[f'Prediction_{name}'] = preds
                except Exception as e:
                    st.warning(f"⚠️ Model {name} failed to predict: {e}")

        st.success("✅ Predictions completed!")

        st.subheader("📈 Prediction Results")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv,
            file_name="loan_predictions.csv",
            mime="text/csv"
        )

    else:
        st.info("🔽 Upload a CSV file to begin.")


if __name__ == "__main__":
    main()
