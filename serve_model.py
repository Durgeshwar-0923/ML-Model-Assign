from flask import Flask, request, jsonify
import mlflow

# Set the tracking URI to the running MLflow server (on port 8080)
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Load the model using alias
model_uri = "models:/BankLoanBestModel@champion"
model = mlflow.pyfunc.load_model(model_uri)

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Model is ready at /predict"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        import pandas as pd
        df = pd.DataFrame([data])  # expects a dict of features
        prediction = model.predict(df)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
