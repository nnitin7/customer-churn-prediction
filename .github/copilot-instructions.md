<!-- .github/copilot-instructions.md - guidance for AI coding agents -->
# Customer Churn Prediction — Copilot Instructions

This file contains focused, actionable information to help AI coding agents be immediately productive in this repository.

- **Big picture:** `model.py` trains a Logistic Regression from `Telco-Customer-Churn.csv` and writes `churn_model.pkl` and `scaler.pkl`. `app.py` is a Flask web app that loads those two pickled objects, expects a numeric feature vector, scales it with `scaler.transform(...)`, and calls `model.predict(...)` to return a result rendered by `index.html`.

- **Key files:** `model.py` (training pipeline), `app.py` (Flask app), `index.html` (form UI), `Telco-Customer-Churn.csv` (training data), `churn_model.pkl`, `scaler.pkl` (artifacts created by training).

- **Run / debug commands (Windows PowerShell):**
  - Create a venv and install dependencies:
    - ``python -m venv .venv``
    - ``.venv\Scripts\Activate.ps1``
    - ``pip install flask pandas scikit-learn numpy``
  - Train model (produces `churn_model.pkl` and `scaler.pkl`):
    - ``python model.py``
  - Run the app (from project root):
    - ``python app.py`` and open `http://127.0.0.1:5000/`

- **Important implementation details & gotchas (discoverable):**
  - `model.py` uses `pd.get_dummies(..., drop_first=True)`. The training pipeline therefore creates a set of encoded feature columns. The Flask app currently expects a raw numeric vector (list of floats) provided in the same order and length as the scaled training features. There is no saved list of feature column names — this can cause runtime mismatches if the input order or count differs from training.
  - `app.py` calls `render_template('index.html')`. Flask expects templates to live in a `templates/` folder. In this repo `index.html` currently sits in the repository root; either move it to `templates/index.html` or set `Flask(__name__, template_folder=...)` accordingly to avoid `TemplateNotFound` errors.
  - The app loads `churn_model.pkl` and `scaler.pkl` from the working directory. Always run the Flask app from the repo root (or ensure paths are updated) so these files can be found.

- **Quick verification snippets:**
  - Check scaler and model input size (PowerShell one-liner):
    - ``python -c "import pickle; s=pickle.load(open('scaler.pkl','rb')); m=pickle.load(open('churn_model.pkl','rb')); print('scaler.n_features_in_=', getattr(s,'n_features_in_',None)); print('model coef shape=', getattr(m,'coef_',None).shape)"``
  - If you get a mismatch between `scaler.n_features_in_` and the input length the app sends, update the UI or pipeline to match training features.

- **Recommended low-effort fixes an AI may propose (and examples to edit):**
  - Persist the training feature columns: after creating `X` in `model.py`, save `X.columns.to_list()` to `feature_order.json` and load it in `app.py` to validate and order inputs before scaling.
  - Move `index.html` into `templates/` (create `templates/index.html`) to match Flask conventions.
  - Replace the simple form in `index.html` with named inputs that map to the `feature_order.json` names; build the vector in the same order the scaler expects.

- **Conventions observed in this repo:**
  - Artifacts are simple pickles: `churn_model.pkl` and `scaler.pkl` in repository root. Do not assume a serialized pipeline includes column metadata.
  - Training script (`model.py`) is a linear script (reads CSV, trains, writes pickles). Expect quick, single-run retraining rather than a packaged CLI.

If any of these behavior assumptions are incorrect or you want me to implement one of the recommended fixes (save `feature_order.json`, move the template, or add a simple UI-to-feature mapping), tell me which change to make and I'll apply it.
