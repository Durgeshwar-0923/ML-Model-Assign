import os
import io
import joblib
import pandas as pd
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
pt       = joblib.load('saved_models/pt.pkl')
rs       = joblib.load('saved_models/rs.pkl')
ss       = joblib.load('saved_models/ss.pkl')
selector = joblib.load('saved_models/selector.pkl')

models = {}
for fn in os.listdir('saved_models'):
    if fn.endswith('_model.pkl'):
        name = fn.replace('_model.pkl', '')
        models[name] = joblib.load(os.path.join('saved_models', fn))

# ─── Helpers ──────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_and_select(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1, errors='ignore')

    rb_cols  = ['CCAvg', 'Mortgage']
    std_cols = ['Income', 'Experience', 'Age']

    df[rb_cols]  = pt.transform(df[rb_cols])
    df[rb_cols]  = rs.transform(df[rb_cols])
    df[std_cols] = ss.transform(df[std_cols])

    X_sel = selector.transform(df)
    return pd.DataFrame(X_sel, columns=[f'F{i}' for i in range(X_sel.shape[1])])

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
                X = preprocess_and_select(df.copy())

                for name, model in models.items():
                    df[f'Pred_{name}'] = model.predict(X)

                # Save processed results for download
                csv_path = os.path.join(UPLOAD_FOLDER, 'latest_results.csv')
                df.to_csv(csv_path, index=False)

                return render_template(
                    'results.html',
                    table=df.to_html(classes='table table-striped table-bordered', index=False, border=0),
                    models=list(models.keys()),
                    timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            except Exception as e:
                flash(f'❌ Error processing file: {e}')
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
        csv_content = f.read()

    response = make_response(csv_content)
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

# ─── Run Server ───────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
