import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, flash, url_for, make_response
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'replace-with-a-secure-random-key'

# ─── Configurations ───────────────────────────
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Load Transformers & Models ───────────────
try:
    pt = joblib.load('artifacts/pt.pkl')
    rs = joblib.load('artifacts/rs.pkl')
    ss = joblib.load('artifacts/ss.pkl')
    selector = joblib.load('artifacts/selector.pkl')
except Exception as e:
    raise RuntimeError(f"❌ Failed to load preprocessing artifacts: {e}")

models = {}
for fn in os.listdir('saved_models'):
    if fn.endswith('_model.pkl'):
        name = fn.replace('_model.pkl', '')
        try:
            models[name] = joblib.load(os.path.join('saved_models', fn))
        except Exception as e:
            print(f"⚠️ Warning: Could not load model '{name}': {e}")

# ─── Helpers ──────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_and_select(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['ID', 'ZIP Code', 'Personal Loan'], errors='ignore')

    rb_cols = ['CCAvg', 'Mortgage']
    std_cols = ['Income', 'Experience', 'Age']

    for col in rb_cols + std_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Keep DataFrame format to retain column names
    df[rb_cols] = pt.transform(df[rb_cols])
    df[rb_cols] = rs.transform(df[rb_cols])
    df[std_cols] = ss.transform(df[std_cols])

    selected = selector.transform(df)
    return pd.DataFrame(selected, columns=[f'F{i}' for i in range(selected.shape[1])])

# ─── Routes ───────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('⚠️ No file part in the request.')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('⚠️ No file selected.')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                df = pd.read_csv(filepath)

                # Preprocess
                X = preprocess_and_select(df.copy())

                # Predict using all models
                for name, model in models.items():
                    df[f'Pred_{name}'] = model.predict(X)

                    if hasattr(model, 'predict_proba'):
                        try:
                            proba = model.predict_proba(X)[:, 1]
                            df[f'Conf_{name}'] = (np.array(proba) * 100).round(2)
                        except Exception as e:
                            print(f"⚠️ Confidence score failed for {name}: {e}")

                # Majority voting
                pred_cols = [col for col in df.columns if col.startswith('Pred_')]
                df['Final_Prediction'] = df[pred_cols].mode(axis=1)[0]

                # Save results for download
                csv_path = os.path.join(UPLOAD_FOLDER, 'latest_results.csv')
                df.to_csv(csv_path, index=False)

                return render_template(
                    'results.html',
                    records=df.to_dict(orient='records'),
                    models=list(models.keys()),
                    total=len(df),
                    timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                )

            except Exception as e:
                flash(f'❌ Error during processing: {str(e)}')
                return redirect(request.url)
        else:
            flash('❌ Invalid file format. Please upload a .csv file.')
            return redirect(request.url)

    return render_template('index.html')

@app.route('/download', methods=['GET'])
def download():
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'latest_results.csv')
    if not os.path.exists(csv_path):
        flash("⚠️ No file available for download.")
        return redirect(url_for('index'))

    with open(csv_path, 'r', encoding='utf-8') as f:
        content = f.read()

    response = make_response(content)
    response.headers["Content-Disposition"] = "attachment; filename=loan_predictions.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

if __name__ == '__main__':
    app.run(debug=True, port=8080)
