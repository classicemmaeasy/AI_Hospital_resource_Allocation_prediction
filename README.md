# 🏥 AI Hospital Resource Allocation Prediction

A Streamlit-based web application that uses trained machine learning models to predict hospital resource allocation outcomes — helping healthcare administrators make proactive decisions.

---

## 🔍 Overview

This system provides three prediction tools powered by pre-trained models:

| Prediction | Model File | Description |
|---|---|---|
| 🚨 Resource Shortage | `best_classification_model.joblib` | Classifies whether a resource shortage will occur next week |
| 👥 Patients Refused | `best_regression_model.joblib` | Estimates the number of patients that may be refused admission |
| 🛏️ Avg Length of Stay | `secondary_regression_model.joblib` | Predicts the average patient length of stay next week |

---

## 🗂️ Project Structure

```
Resource Allocation/
├── main.py                          # Streamlit application entry point
├── requirements.txt                 # Python dependencies
├── Models/
│   ├── best_classification_model.joblib
│   ├── best_regression_model.joblib
│   ├── secondary_regression_model.joblib
│   ├── preprocessor.joblib
│   └── RESOURCE_ALLOCATION (1).ipynb   # Model training notebook
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/classicemmaeasy/AI_Hospital_resource_Allocation_prediction.git
cd AI_Hospital_resource_Allocation_prediction

# Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📦 Dependencies

```
streamlit
pandas
numpy
joblib
scikit-learn==1.6.1
xgboost
```

> **Note:** The models were trained with **scikit-learn 1.6.1**. Using a different version may cause compatibility warnings or loading errors.

---

## 🧠 Features

- **Separate input forms** for each prediction model — no cross-contamination of features
- **Live model status bar** showing which models loaded successfully
- **Model Loading Diagnostics** panel for troubleshooting
- **Model-to-Feature Mapping** panel showing exactly which features each model uses
- **Staff Status Summary** dashboard
- Fully responsive Streamlit layout with 3 tabbed prediction sections

---

## 📊 Input Features

### Resource Shortage Classification (9 features)
- `avg_length_of_stay_lag1`, `staff_required_lag1`, `capacity_gap_lag1`
- `staff_scheduled_lag1`, `staff_available_lag1`, `avg_satisfaction_score_lag1`
- `patients_requested_lag1`, `service`, `week_number`

### Patients Refused Regression (32 features)
Includes all classification features plus bed counts, equipment stats, rates, attendance data, weekly summaries, and event flags.

### Average Length of Stay Regression (9–13 features)
Core 9 features plus optional additional features resolved from the model at runtime.

---

## 📝 License

This project is for educational and research purposes.

---

## 👤 Author

Developed as part of an AI-powered hospital resource management system.
