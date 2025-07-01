# MLFLOW Project

This project contains two web applications:

- **Flask App** (`app.py`)
- **Streamlit App** (`main.py`)

## Setup

1. **Clone the repository**
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Flask App

1. Navigate to the project directory.
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open your browser and go to [http://localhost:5000](http://localhost:5000)

---

## Running the Streamlit App

1. Navigate to the project directory.
2. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
3. Streamlit will provide a local URL (e.g., [http://localhost:8501](http://localhost:8501)).

---

## Running MLflow Tracking Server

1. Start the MLflow tracking server:
   ```bash
   mlflow ui
   ```
2. Open your browser and go to [http://localhost:5000](http://localhost:5000) (or the port shown in your terminal).

---

## Production Deployment

### Serve the Model

After saving/training your model with MLflow, you can serve it using:

```bash
python serve_model.py
```

This will start a server for model inference.

### Test the Deployed Model

To test the deployed model, run:

```bash
python test.py
```

This script will send test requests to your served model and display the results.

---
