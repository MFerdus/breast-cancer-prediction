



# Breast Cancer Prediction Using Machine Learning (PyCaret)

This project presents an **end-to-end machine learning pipeline** for **breast cancer prediction** using **PyCaret**, covering the full lifecycle from data preprocessing and model training to evaluation, deployment, and prediction.  
The goal is to build an **accurate, interpretable, and production-ready ML system** that assists in early breast cancer diagnosis.

---

## 🚀 Project Overview

Breast cancer is one of the leading causes of cancer-related deaths worldwide. Early and accurate diagnosis can significantly improve patient outcomes.  
This project leverages **automated machine learning (AutoML)** through PyCaret to efficiently compare multiple models and deploy the best-performing classifier.

---

## 🎯 Objectives

- Build an accurate breast cancer classification model
- Automate model selection and tuning using PyCaret
- Ensure reproducibility and explainability
- Provide inference-ready deployment artifacts
- Demonstrate an industry-style ML workflow

---

## 🧠 Machine Learning Workflow

The project follows a structured **end-to-end ML pipeline**:

### 1️⃣ Data Ingestion
- Breast cancer dataset loaded into a Pandas DataFrame
- Initial exploration and target inspection

### 2️⃣ Data Preprocessing
- Handling missing values
- Feature scaling and normalization
- Automatic encoding and transformation via PyCaret

### 3️⃣ Model Training & Comparison
- Multiple classifiers trained and compared automatically
- Evaluation using cross-validation
- Metrics include Accuracy, AUC, Precision, Recall, and F1-score

### 4️⃣ Model Selection & Tuning
- Best-performing model selected
- Hyperparameter tuning applied
- Final model finalized and saved

### 5️⃣ Model Evaluation
- Confusion matrix
- ROC-AUC curve
- Feature importance analysis

### 6️⃣ Deployment & Inference
- Trained model saved as a reusable artifact
- Batch prediction pipeline
- Optional Streamlit UI for real-time prediction

---




## 🛠️ Tech Stack

- **Python**
- **PyCaret**
- **Scikit-learn**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **Streamlit** (optional UI)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/breast-cancer-pycaret-e2e.git
cd breast-cancer-pycaret-e2e
````

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

* **Windows**

```bash
venv\Scripts\activate
```

* **Mac / Linux**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
notebooks/breast_cancer_prediction.ipynb
```

Run all cells sequentially to:

* Train models
* Compare algorithms
* Save the final model

---

## 🧪 Run Model Training via Script

```bash
python src/train.py
```

---

## 🔮 Run Predictions (Batch Inference)

```bash
python src/predict.py
```

---

## 🌐 Run Streamlit Web App (Optional)

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📊 Model Performance Highlights

* High classification accuracy
* Strong ROC-AUC score
* Robust generalization via cross-validation
* Automatically selected best algorithm

*(Exact metrics are reported inside the notebook)*

---

## 🏥 Real-World Impact

* Supports **early breast cancer detection**
* Demonstrates **cost-effective ML deployment**
* Reduces diagnostic time and manual effort
* Can assist clinicians as a **decision-support tool**

---





