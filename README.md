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

## Execute the DefaulterNotebook.ipynb file

### Evidently reports
evidently reports are saved in the folder DataDrift_reports 
## Running MLflow Tracking Server

1. Start the MLflow tracking server:
   ```bash
   mlflow ui
   ```
2. Open your browser and go to [http://localhost:5000](http://localhost:5000) (or the port shown in your terminal).


## Running the Flask App

1. Navigate to the project directory.
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open your browser and go to [http://localhost:8080](http://localhost:8080)

---

## Running the Streamlit App

1. Navigate to the project directory.
2. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
3. Streamlit will provide a local URL (e.g., [http://localhost:8501](http://localhost:8501)).

---

---

## Flask Deployment
### 1. Find the Best Model

Run the following to select the best model from your MLflow experiments:

```bash
python find_best_model.py
```

### 2. Promote the Best Model

Promote the selected model to production using:

```bash
python promote.py
```

### 3. Serve the Model

After promoting, serve the model for inference:

```bash
python serve_model.py
2. Open your browser and go to [http://localhost:8000](http://localhost:8000) (or the port shown in your terminal).
```

### 4. Test the Deployed Model

Test the served model with:

```bash
python test.py

```

## ouputs are present in ouputs_screenshots folder
