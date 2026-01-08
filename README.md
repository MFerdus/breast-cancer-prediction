# Breast Cancer Prediction (PyCaret) — End‑to‑End Project

This repository turns the provided Jupyter notebook into a clean, reproducible **end‑to‑end** ML project:
- ✅ Exploratory analysis + modeling in a notebook (PyCaret)
- ✅ A command‑line training script that saves a production model
- ✅ A **Streamlit UI** to run predictions on new CSV files
- ✅ A consistent folder structure, requirements, and run instructions

---

## 1) Project structure

```
breast-cancer-pycaret-e2e/
├─ notebooks/
│  └─ 01_breast_cancer_prediction_pycaret.ipynb
├─ src/
│  ├─ train.py          # train + compare + finalize + save model
│  ├─ predict.py        # batch predictions to CSV
│  └─ app.py            # Streamlit UI
├─ data/
│  └─ README.md         # where to place your dataset
├─ models/              # trained model artifacts (.pkl) will be saved here
├─ reports/             # predictions/reports outputs
├─ assets/              # screenshots, diagrams (optional)
└─ requirements.txt
```

---

## 2) Dataset requirements

The original notebook is titled **“Breast Cancer Prediction using PyCaret”** and loads a CSV named:

- `BRCA Data.csv`

Your dataset should be a **tabular CSV** and must include a **binary target column**.  
In the notebook, the target column is:

- `Patient_Status`

> **Important:** This repository does not include `BRCA Data.csv` (datasets are frequently private/licensed).  
Place it under: `data/BRCA Data.csv` and use the instructions below.

---

## 3) Setup environment

### Option A — create a virtual environment (recommended)

```bash
# from the project root
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Option B — use Conda

```bash
conda create -n brca-pycaret python=3.10 -y
conda activate brca-pycaret
pip install -r requirements.txt
```

> PyCaret works best on Python 3.9–3.10 in many environments.

---

## 4) Run the notebook end‑to‑end

Open the notebook:

```bash
jupyter notebook
# or
jupyter lab
```

Then run:

`notebooks/01_breast_cancer_prediction_pycaret.ipynb`

### What the notebook does (high‑level)

1. **Loads** `BRCA Data.csv`
2. Runs **EDA** (plots, class distribution, feature insights)
3. Uses **PyCaret Classification**:
   - `setup(...)` (with options like outlier removal, multicollinearity removal, median imputation, etc.)
   - trains models (e.g., Random Forest, Extra Trees, etc.)
   - evaluates
   - saves models (the notebook uses `save_model`)

If your CSV is in `data/`, update the notebook’s load line:

```python
df = pd.read_csv("data/BRCA Data.csv")
```

---

## 5) Train a model from the command line

Once your dataset is in `data/BRCA Data.csv`, run:

```bash
python -m src.train --data "data/BRCA Data.csv" --target Patient_Status --model_name best_model
```

Outputs:
- `models/best_model.pkl`

### Notes
- The script runs `setup()` + `compare_models()` to select a best baseline model.
- It then runs `finalize_model()` and saves the final model.
- If you want the training settings to match your notebook exactly, adjust `src/train.py` setup parameters.

---

## 6) Batch prediction (CSV → CSV)

After training:

```bash
python -m src.predict --model models/best_model --input data/new_patients.csv --output reports/predictions.csv
```

The output CSV includes:
- predicted label (PyCaret typically creates a `prediction_label` column)
- probability scores (often `prediction_score`)
- and the original features

---

## 7) Streamlit UI (interactive predictions)

### Start the app

```bash
streamlit run src/app.py
```

In the sidebar:
- Set **Model path** to `models/best_model` (default)

Then:
1. Upload a CSV (same feature columns used for training)
2. Click **Run prediction**
3. Download results as CSV

---

## 8) Common troubleshooting

### “Target column not found”
- Make sure your CSV includes the target column name you pass:
  - default: `Patient_Status`

### “Could not load model”
- Train the model first:
  - `python -m src.train ...`
- Confirm the file exists:
  - `models/best_model.pkl`

### Different columns at inference time
Your prediction CSV must contain the **same feature columns** as training (except the target).  
If the dataset has categorical columns, PyCaret expects the same column names and compatible categories.

---

## 9) How to customize this project

- **Model selection**: replace `compare_models()` with a fixed model like:
  ```python
  from pycaret.classification import create_model, tune_model
  m = create_model("rf")
  m = tune_model(m)
  ```
- **Explainability**: add SHAP (or PyCaret’s interpret tools) in the notebook
- **Deployment**: export the Streamlit app to HuggingFace Spaces / Streamlit Cloud / Docker
- **Evaluation**: add a `reports/metrics.md` and log metrics to MLflow

---

## 10) Reproducibility checklist

Before sharing or deploying:
- Pin `requirements.txt` versions
- Keep your dataset path consistent (`data/`)
- Save the final model under `models/`
- Add example input schema in `data/README.md`

---

## License

Use this code freely for learning and academic purposes.  
If you include a dataset, ensure you have rights to distribute it.
# breast-cancer-prediction
