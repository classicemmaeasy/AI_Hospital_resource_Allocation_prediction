# ============================================================
# HOSPITAL RESOURCE AND PATIENT FLOW PREDICTION
# Streamlit application with a Gradio-style interface
#
# IMPORTANT DESIGN:
# Each prediction has its OWN input form and its OWN button.
#
# 1. Resource Shortage Classification
#    -> 9 features -> best_classification_model.joblib
#
# 2. Patients Refused Regression
#    -> 32 features -> best_regression_model.joblib
#
# 3. Average Length of Stay Regression
#    -> 9 features -> secondary_regression_model.joblib
#
# This prevents features belonging to one model from being silently
# mixed with another model.
# ============================================================

import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# STREAMLIT CONFIGURATION
# MUST BE THE FIRST STREAMLIT COMMAND
# ============================================================
st.set_page_config(
    page_title="Hospital Resource & Patient Flow Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# COMPATIBILITY
# Models were trained with scikit-learn 1.6.1.
# ============================================================
try:
    import sklearn
    SKLEARN_VERSION = sklearn.__version__
except Exception:
    SKLEARN_VERSION = "Not installed"

try:
    import sklearn.compose._column_transformer as _column_transformer

    if not hasattr(_column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        _column_transformer._RemainderColsList = _RemainderColsList
except Exception:
    pass

try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings(
        "ignore",
        category=InconsistentVersionWarning,
    )
except Exception:
    pass


# ============================================================
# OPTIONAL XGBOOST
# ============================================================
try:
    import xgboost

    XGBOOST_AVAILABLE = True
    XGBOOST_VERSION = xgboost.__version__
except Exception:
    XGBOOST_AVAILABLE = False
    XGBOOST_VERSION = "Not installed"


# ============================================================
# OPTIONAL PLOTLY
# ============================================================
try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# ============================================================
# CSS - GRADIO-STYLE STREAMLIT UI
# ============================================================
st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .app-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .app-subtitle {
            text-align: center;
            font-size: 1rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }

        .model-card {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            background: rgba(248,250,252,0.7);
        }

        .prediction-card {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.25rem;
            min-height: 150px;
            background: rgba(248,250,252,0.7);
        }

        .model-title {
            font-size: 1.15rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .model-description {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }

        .success-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 999px;
            font-weight: 700;
            background: #dcfce7;
            color: #166534;
        }

        .danger-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 999px;
            font-weight: 700;
            background: #fee2e2;
            color: #991b1b;
        }

        div.stButton > button {
            width: 100%;
            min-height: 3rem;
            font-weight: 700;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# APPLICATION / MODEL DIRECTORY DISCOVERY
# ============================================================
if "__file__" in globals():
    APP_DIR = Path(__file__).resolve().parent
else:
    APP_DIR = Path.cwd()


def find_model_directory():
    candidates = [
        APP_DIR / "Models",
        APP_DIR / "models",
        APP_DIR.parent / "Models",
        APP_DIR.parent / "models",
        Path.cwd() / "Models",
        Path.cwd() / "models",
        APP_DIR / "trained_models_selected_features",
        APP_DIR / "trained_models",
    ]

    # Remove duplicates while preserving order.
    unique_candidates = []
    seen = set()

    for candidate in candidates:
        key = str(candidate.resolve())
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)

    required_files = [
        "best_classification_model.joblib",
        "best_regression_model.joblib",
        "preprocessor.joblib",
    ]

    # Prefer a directory containing the important model files.
    for directory in unique_candidates:
        if directory.exists() and all(
            (directory / filename).exists()
            for filename in required_files
        ):
            return directory

    # Otherwise return the first existing candidate so diagnostics
    # can clearly show what is missing.
    for directory in unique_candidates:
        if directory.exists():
            return directory

    return unique_candidates[0]


MODEL_DIR = find_model_directory()


# ============================================================
# EXACT MODEL FEATURE SCHEMAS
# ============================================================

# ------------------------------------------------------------
# CLASSIFICATION MODEL
# resource_shortage_next_week
# ------------------------------------------------------------
CLASSIFICATION_FEATURES = [
    "avg_length_of_stay_lag1",
    "staff_required_lag1",
    "capacity_gap_lag1",
    "staff_scheduled_lag1",
    "staff_available_lag1",
    "avg_satisfaction_score_lag1",
    "patients_requested_lag1",
    "service",
    "week_number",
]


# ------------------------------------------------------------
# PATIENTS REFUSED REGRESSION MODEL
# patients_refused_next_week
#
# This is the larger feature set used by the primary regression
# model in the supplied application.
# ------------------------------------------------------------
REFUSAL_REGRESSION_FEATURES = [
    "avg_length_of_stay_lag1",
    "staff_required_lag1",
    "capacity_gap_lag1",
    "staff_scheduled_lag1",
    "staff_available_lag1",
    "avg_satisfaction_score_lag1",
    "patients_requested_lag1",
    "service",
    "week_number",
    "available_beds_lag1",
    "occupied_beds_lag1",
    "vacant_beds_lag1",
    "patients_admitted_lag1",
    "patients_refused_lag1",
    "equipment_required_lag1",
    "equipment_available_lag1",
    "staff_morale_lag1",
    "bed_utilization_rate_lag1",
    "refusal_rate_lag1",
    "admission_rate_lag1",
    "vacancy_rate_lag1",
    "equipment_availability_rate_lag1",
    "staff_availability_rate_lag1",
    "staff_present_lag1",
    "staff_absent_y_lag1",
    "staff_attendance_rate_lag1",
    "patients_requested_weekly_summary_lag1",
    "patients_admitted_weekly_summary_lag1",
    "patients_refused_weekly_summary_lag1",
    "avg_waiting_time_lag1",
    "resource_shortage_lag1",
    "event_lag1",
]


# ------------------------------------------------------------
# SECONDARY REGRESSION MODEL
# avg_length_of_stay_next_week
#
# ALOS_FEATURES is resolved at runtime from the model's own
# feature_names_in_ attribute so it always matches what the
# model was actually trained with.  The list below is the
# known minimum; extra features exposed in the form supply
# values for whatever the model actually needs.
# ------------------------------------------------------------
ALOS_FEATURES_FALLBACK = [
    "avg_length_of_stay_lag1",
    "staff_required_lag1",
    "capacity_gap_lag1",
    "staff_scheduled_lag1",
    "staff_available_lag1",
    "avg_satisfaction_score_lag1",
    "patients_requested_lag1",
    "service",
    "week_number",
    "patients_admitted_lag1",
    "staff_present_lag1",
    "patients_requested_weekly_summary_lag1",
    "equipment_available_lag1",
]


# ============================================================
# MODEL FILES
# ============================================================
MODEL_FILES = {
    "classification": "best_classification_model.joblib",
    "refusal_regression": "best_regression_model.joblib",
    "alos_regression": "secondary_regression_model.joblib",
    "preprocessor": "preprocessor.joblib",
}


# ============================================================
# MODEL PATCHING
# ============================================================
def patch_loaded_object(obj):
    """
    Patch known sklearn serialization differences.

    The models were serialized with sklearn 1.6.1.
    The primary fix remains installing sklearn 1.6.1 in the
    environment used to run Streamlit.
    """
    if obj is None:
        return

    # Pipeline
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            patch_loaded_object(step)

    # ColumnTransformer after fitting
    if hasattr(obj, "transformers_"):
        for _, transformer, _ in obj.transformers_:
            patch_loaded_object(transformer)

    # ColumnTransformer before fitting / other structures
    if hasattr(obj, "transformers"):
        for _, transformer, _ in obj.transformers:
            patch_loaded_object(transformer)

    # OneHotEncoder compatibility
    if type(obj).__name__ == "OneHotEncoder":
        if not hasattr(obj, "sparse"):
            try:
                obj.sparse = obj.sparse_output
            except Exception:
                obj.sparse = True

        if not hasattr(obj, "sparse_output"):
            try:
                obj.sparse_output = obj.sparse
            except Exception:
                obj.sparse_output = True


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    obj = joblib.load(path)
    patch_loaded_object(obj)
    return obj


# ============================================================
# LOAD ALL MODELS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_models(model_directory_string):
    model_directory = Path(model_directory_string)

    result = {
        "classification": None,
        "refusal_regression": None,
        "alos_regression": None,
        "preprocessor": None,
        "errors": {},
        "paths": {},
    }

    for key, filename in MODEL_FILES.items():
        path = model_directory / filename
        result["paths"][key] = str(path)

        if not path.exists():
            result["errors"][key] = (
                f"File not found: {path}"
            )
            continue

        try:
            result[key] = load_model(path)
        except Exception as exc:
            result["errors"][key] = (
                f"{type(exc).__name__}: {exc}"
            )

    return result


models = load_models(str(MODEL_DIR))


def get_model(key):
    """
    Retrieve model from cached dictionary, attempting lazy load if missing.
    """
    model = models.get(key)
    if model is None:
        filename = MODEL_FILES.get(key)
        if filename:
            path = MODEL_DIR / filename
            if path.exists():
                try:
                    model = load_model(path)
                    models[key] = model
                    if "errors" in models and key in models["errors"]:
                        models["errors"].pop(key, None)
                except Exception as exc:
                    if "errors" in models:
                        models["errors"][key] = f"{type(exc).__name__}: {exc}"
    return model


# ============================================================
# HELPERS
# ============================================================
def model_expected_features(model):
    """
    Try to read the exact feature names stored in the model.
    """
    if model is None:
        return None

    names = getattr(model, "feature_names_in_", None)

    if names is not None:
        return list(names)

    if hasattr(model, "named_steps"):
        for step_name in ["preprocessor", "columntransformer"]:
            step = model.named_steps.get(step_name)
            if step is not None:
                names = getattr(step, "feature_names_in_", None)
                if names is not None:
                    return list(names)

    return None


def build_dataframe(values, feature_list):
    """
    Build a DataFrame using ONLY the feature list assigned to
    the selected prediction model.
    """
    row = {}

    for feature in feature_list:
        if feature not in values:
            raise KeyError(
                f"Missing required input feature: {feature}"
            )
        row[feature] = values[feature]

    return pd.DataFrame([row], columns=feature_list)


def predict_regression(model_or_key, values, feature_list):
    """
    Prediction function for a regression model.
    """
    if isinstance(model_or_key, str):
        key = model_or_key
        model = get_model(key)
    else:
        model = model_or_key
        key = None

    if model is None:
        err_detail = models.get("errors", {}).get(key, "Model is not loaded.") if key else "Model is not loaded."
        return None, (
            f"Model is not loaded ({err_detail}). "
            "Check Model Loading Diagnostics."
        ), None

    try:
        input_df = build_dataframe(values, feature_list)
        prediction = model.predict(input_df)

        return float(prediction[0]), None, input_df

    except Exception as exc:
        return None, (
            f"{type(exc).__name__}: {exc}"
        ), None


def predict_classification(model_or_key, values, feature_list):
    """
    Prediction function for the resource-shortage classifier.
    """
    if isinstance(model_or_key, str):
        key = model_or_key
        model = get_model(key)
    else:
        model = model_or_key
        key = None

    if model is None:
        err_detail = models.get("errors", {}).get(key, "Model is not loaded.") if key else "Model is not loaded."
        return None, None, (
            f"Classification model is not loaded ({err_detail}). "
            "Check Model Loading Diagnostics."
        ), None

    try:
        input_df = build_dataframe(values, feature_list)

        predicted_class = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            if len(probabilities) >= 2:
                probability = float(probabilities[1]) * 100.0
            else:
                probability = float(probabilities[0]) * 100.0
        else:
            probability = (
                100.0 if predicted_class == 1 else 0.0
            )

        return (
            predicted_class,
            probability,
            None,
            input_df,
        )

    except Exception as exc:
        return None, None, (
            f"{type(exc).__name__}: {exc}"
        ), None


# ============================================================
# DEFAULT VALUES
# ============================================================
DEFAULTS = {
    "service": "emergency",
    "week_number": 18,

    "avg_length_of_stay_lag1": 7.0,
    "staff_required_lag1": 24.0,
    "capacity_gap_lag1": 41.0,
    "staff_scheduled_lag1": 40.0,
    "staff_available_lag1": 22.0,
    "avg_satisfaction_score_lag1": 2.9,
    "patients_requested_lag1": 73.0,

    "available_beds_lag1": 30.0,
    "occupied_beds_lag1": 25.0,
    "vacant_beds_lag1": 5.0,
    "patients_admitted_lag1": 20.0,
    "patients_refused_lag1": 3.0,
    "equipment_required_lag1": 20.0,
    "equipment_available_lag1": 18.0,
    "staff_morale_lag1": 3.0,

    "bed_utilization_rate_lag1": 0.83,
    "refusal_rate_lag1": 0.04,
    "admission_rate_lag1": 0.27,
    "vacancy_rate_lag1": 0.17,
    "equipment_availability_rate_lag1": 0.90,
    "staff_availability_rate_lag1": 0.55,

    "staff_present_lag1": 22.0,
    "staff_absent_y_lag1": 18.0,
    "staff_attendance_rate_lag1": 0.55,

    "patients_requested_weekly_summary_lag1": 73.0,
    "patients_admitted_weekly_summary_lag1": 20.0,
    "patients_refused_weekly_summary_lag1": 3.0,
    "avg_waiting_time_lag1": 2.0,
    "resource_shortage_lag1": 0.0,
    "event_lag1": "none",
}


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="app-title">'
    '🏥 Hospital Resource and Patient Flow Prediction'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "Enter only the features required by the prediction you want "
    "to run. Each model has its own input section and prediction button."
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# MODEL STATUS BAR
# ============================================================
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if get_model("classification") is not None:
        st.success("🟢 Shortage Classification Model Ready")
    else:
        st.error("🔴 Shortage Classification Model Unavailable")

with status_col2:
    if get_model("refusal_regression") is not None:
        st.success("🟢 Patients Refused Model Ready")
    else:
        st.error("🔴 Patients Refused Model Unavailable")

with status_col3:
    if get_model("alos_regression") is not None:
        st.success("🟢 ALOS Model Ready")
    else:
        st.warning("🟡 ALOS Model Unavailable")


# ============================================================
# THREE MODEL TABS
# ============================================================
tab_shortage, tab_refusal, tab_alos = st.tabs(
    [
        "🚨 Resource Shortage",
        "👥 Patients Refused",
        "🛏️ Average Length of Stay",
    ]
)


# ============================================================
# 1. RESOURCE SHORTAGE CLASSIFICATION
# ============================================================
with tab_shortage:

    st.markdown(
        '<div class="model-title">'
        '🚨 Resource Shortage Prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="model-description">'
        'These inputs are mapped ONLY to '
        '<b>best_classification_model.joblib</b>. '
        'No patients-refused regression features are passed to this model.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("shortage_prediction_form"):

        c1, c2, c3 = st.columns(3)

        with c1:
            shortage_service = st.selectbox(
                "Service",
                [
                    "emergency",
                    "general_medicine",
                    "icu",
                    "surgery",
                ],
                index=0,
                key="shortage_service",
            )

            shortage_week = st.number_input(
                "Week Number",
                min_value=1,
                max_value=52,
                value=int(DEFAULTS["week_number"]),
                step=1,
                key="shortage_week",
            )

            shortage_los = st.number_input(
                "Avg Length of Stay (Lag 1)",
                min_value=0.0,
                max_value=60.0,
                value=float(DEFAULTS["avg_length_of_stay_lag1"]),
                step=0.1,
                key="shortage_los",
            )

        with c2:
            shortage_staff_required = st.number_input(
                "Staff Required (Lag 1)",
                min_value=0.0,
                max_value=300.0,
                value=float(DEFAULTS["staff_required_lag1"]),
                step=1.0,
                key="shortage_staff_required",
            )

            shortage_capacity_gap = st.number_input(
                "Capacity Gap (Lag 1)",
                min_value=-300.0,
                max_value=300.0,
                value=float(DEFAULTS["capacity_gap_lag1"]),
                step=1.0,
                key="shortage_capacity_gap",
            )

            shortage_staff_scheduled = st.number_input(
                "Staff Scheduled (Lag 1)",
                min_value=0.0,
                max_value=300.0,
                value=float(DEFAULTS["staff_scheduled_lag1"]),
                step=1.0,
                key="shortage_staff_scheduled",
            )

        with c3:
            shortage_staff_available = st.number_input(
                "Staff Available (Lag 1)",
                min_value=0.0,
                max_value=300.0,
                value=float(DEFAULTS["staff_available_lag1"]),
                step=1.0,
                key="shortage_staff_available",
            )

            shortage_satisfaction = st.number_input(
                "Avg Satisfaction Score (Lag 1)",
                min_value=0.0,
                max_value=5.0,
                value=float(DEFAULTS["avg_satisfaction_score_lag1"]),
                step=0.1,
                key="shortage_satisfaction",
            )

            shortage_patients = st.number_input(
                "Patients Requested (Lag 1)",
                min_value=0.0,
                max_value=1000.0,
                value=float(DEFAULTS["patients_requested_lag1"]),
                step=1.0,
                key="shortage_patients",
            )

        shortage_submit = st.form_submit_button(
            "🚨 Predict Resource Shortage",
            type="primary",
        )

    if shortage_submit:

        shortage_values = {
            "avg_length_of_stay_lag1": shortage_los,
            "staff_required_lag1": shortage_staff_required,
            "capacity_gap_lag1": shortage_capacity_gap,
            "staff_scheduled_lag1": shortage_staff_scheduled,
            "staff_available_lag1": shortage_staff_available,
            "avg_satisfaction_score_lag1": shortage_satisfaction,
            "patients_requested_lag1": shortage_patients,
            "service": shortage_service,
            "week_number": shortage_week,
        }

        (
            shortage_class,
            shortage_probability,
            shortage_error,
            shortage_input_df,
        ) = predict_classification(
            "classification",
            shortage_values,
            CLASSIFICATION_FEATURES,
        )

        st.markdown("---")
        st.subheader("Prediction Result")

        if shortage_class is not None:

            r1, r2 = st.columns(2)

            with r1:
                if shortage_class == 1:
                    st.markdown(
                        '<span class="danger-badge">'
                        '🚨 SHORTAGE PREDICTED — CLASS 1'
                        '</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="success-badge">'
                        '✅ NO SHORTAGE PREDICTED — CLASS 0'
                        '</span>',
                        unsafe_allow_html=True,
                    )

            with r2:
                st.metric(
                    "Shortage Probability",
                    f"{shortage_probability:.2f}%",
                )

            if shortage_class == 1:
                st.error(
                    f"Resource shortage is predicted for "
                    f"{shortage_service.replace('_', ' ').title()} "
                    f"with {shortage_probability:.2f}% probability."
                )
            else:
                st.success(
                    f"No resource shortage is predicted for "
                    f"{shortage_service.replace('_', ' ').title()}."
                )

            with st.expander("🔍 Classification Model Input"):
                st.dataframe(
                    shortage_input_df,
                    use_container_width=True,
                    hide_index=True,
                )

        else:
            st.error("Classification prediction failed.")
            st.code(shortage_error)


# ============================================================
# 2. PATIENTS REFUSED REGRESSION
# ============================================================
with tab_refusal:

    st.markdown(
        '<div class="model-title">'
        '👥 Patients Refused Prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="model-description">'
        'This section contains the complete feature set assigned to '
        '<b>best_regression_model.joblib</b>. '
        'These inputs are NOT sent to the shortage classifier unless '
        'you enter them again in the classification section.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("refusal_prediction_form"):

        st.markdown("#### 🏥 Core Operational Features")

        c1, c2, c3 = st.columns(3)

        with c1:
            refusal_service = st.selectbox(
                "Service",
                [
                    "emergency",
                    "general_medicine",
                    "icu",
                    "surgery",
                ],
                index=0,
                key="refusal_service",
            )

            refusal_week = st.number_input(
                "Week Number",
                min_value=1,
                max_value=52,
                value=int(DEFAULTS["week_number"]),
                step=1,
                key="refusal_week",
            )

            refusal_los = st.number_input(
                "Avg Length of Stay (Lag 1)",
                min_value=0.0,
                max_value=60.0,
                value=float(DEFAULTS["avg_length_of_stay_lag1"]),
                step=0.1,
                key="refusal_los",
            )

            refusal_staff_required = st.number_input(
                "Staff Required (Lag 1)",
                min_value=0.0,
                max_value=300.0,
                value=float(DEFAULTS["staff_required_lag1"]),
                step=1.0,
                key="refusal_staff_required",
            )

        with c2:
            refusal_capacity_gap = st.number_input(
                "Capacity Gap (Lag 1)",
                min_value=-300.0,
                max_value=300.0,
                value=float(DEFAULTS["capacity_gap_lag1"]),
                step=1.0,
                key="refusal_capacity_gap",
            )

            refusal_staff_scheduled = st.number_input(
                "Staff Scheduled (Lag 1)",
                min_value=0.0,
                max_value=300.0,
                value=float(DEFAULTS["staff_scheduled_lag1"]),
                step=1.0,
                key="refusal_staff_scheduled",
            )

            refusal_staff_available = st.number_input(
                "Staff Available (Lag 1)",
                min_value=0.0,
                max_value=300.0,
                value=float(DEFAULTS["staff_available_lag1"]),
                step=1.0,
                key="refusal_staff_available",
            )

            refusal_satisfaction = st.number_input(
                "Avg Satisfaction Score (Lag 1)",
                min_value=0.0,
                max_value=5.0,
                value=float(DEFAULTS["avg_satisfaction_score_lag1"]),
                step=0.1,
                key="refusal_satisfaction",
            )

        with c3:
            refusal_patients_requested = st.number_input(
                "Patients Requested (Lag 1)",
                min_value=0.0,
                max_value=1000.0,
                value=float(DEFAULTS["patients_requested_lag1"]),
                step=1.0,
                key="refusal_patients_requested",
            )

            refusal_available_beds = st.number_input(
                "Available Beds (Lag 1)",
                min_value=0.0,
                max_value=1000.0,
                value=float(DEFAULTS["available_beds_lag1"]),
                step=1.0,
                key="refusal_available_beds",
            )

            refusal_occupied_beds = st.number_input(
                "Occupied Beds (Lag 1)",
                min_value=0.0,
                max_value=1000.0,
                value=float(DEFAULTS["occupied_beds_lag1"]),
                step=1.0,
                key="refusal_occupied_beds",
            )

            refusal_vacant_beds = st.number_input(
                "Vacant Beds (Lag 1)",
                min_value=0.0,
                max_value=1000.0,
                value=float(DEFAULTS["vacant_beds_lag1"]),
                step=1.0,
                key="refusal_vacant_beds",
            )

        st.markdown("#### 🧑‍⚕️ Patient, Equipment & Staffing History")

        c1, c2, c3 = st.columns(3)

        with c1:
            refusal_patients_admitted = st.number_input(
                "Patients Admitted (Lag 1)",
                min_value=0.0,
                max_value=1000.0,
                value=float(DEFAULTS["patients_admitted_lag1"]),
                step=1.0,
                key="refusal_patients_admitted",
            )

            refusal_patients_refused = st.number_input(
                "Patients Refused (Lag 1)",
                min_value=0.0,
                max_value=1000.0,
                value=float(DEFAULTS["patients_refused_lag1"]),
                step=1.0,
                key="refusal_patients_refused",
            )

            refusal_equipment_required = st.number_input(
                "Equipment Required (Lag 1)",
                min_value=0.0,
                max_value=1000.0,
                value=float(DEFAULTS["equipment_required_lag1"]),
                step=1.0,
                key="refusal_equipment_required",
            )

            refusal_equipment_available = st.number_input(
                "Equipment Available (Lag 1)",
                min_value=0.0,
                max_value=1000.0,
                value=float(DEFAULTS["equipment_available_lag1"]),
                step=1.0,
                key="refusal_equipment_available",
            )

        with c2:
            refusal_morale = st.number_input(
                "Staff Morale (Lag 1)",
                min_value=0.0,
                max_value=5.0,
                value=float(DEFAULTS["staff_morale_lag1"]),
                step=0.1,
                key="refusal_morale",
            )

            refusal_bed_util = st.number_input(
                "Bed Utilization Rate (Lag 1)",
                min_value=0.0,
                max_value=1.0,
                value=float(DEFAULTS["bed_utilization_rate_lag1"]),
                step=0.01,
                key="refusal_bed_util",
            )

            refusal_rate = st.number_input(
                "Refusal Rate (Lag 1)",
                min_value=0.0,
                max_value=1.0,
                value=float(DEFAULTS["refusal_rate_lag1"]),
                step=0.01,
                key="refusal_rate",
            )

            admission_rate = st.number_input(
                "Admission Rate (Lag 1)",
                min_value=0.0,
                max_value=1.0,
                value=float(DEFAULTS["admission_rate_lag1"]),
                step=0.01,
                key="refusal_admission_rate",
            )

        with c3:
            vacancy_rate = st.number_input(
                "Vacancy Rate (Lag 1)",
                min_value=0.0,
                max_value=1.0,
                value=float(DEFAULTS["vacancy_rate_lag1"]),
                step=0.01,
                key="refusal_vacancy_rate",
            )

            equipment_availability_rate = st.number_input(
                "Equipment Availability Rate (Lag 1)",
                min_value=0.0,
                max_value=1.0,
                value=float(DEFAULTS["equipment_availability_rate_lag1"]),
                step=0.01,
                key="refusal_equipment_rate",
            )

            staff_availability_rate = st.number_input(
                "Staff Availability Rate (Lag 1)",
                min_value=0.0,
                max_value=1.0,
                value=float(DEFAULTS["staff_availability_rate_lag1"]),
                step=0.01,
                key="refusal_staff_rate",
            )

            staff_present = st.number_input(
                "Staff Present (Lag 1)",
                min_value=0.0,
                max_value=300.0,
                value=float(DEFAULTS["staff_present_lag1"]),
                step=1.0,
                key="refusal_staff_present",
            )

        st.markdown("#### 📊 Attendance & Weekly Summary")

        c1, c2, c3 = st.columns(3)

        with c1:
            staff_absent = st.number_input(
                "Staff Absent (Lag 1)",
                min_value=0.0,
                max_value=300.0,
                value=float(DEFAULTS["staff_absent_y_lag1"]),
                step=1.0,
                key="refusal_staff_absent",
                help=(
                    "Mapped to the training feature "
                    "'staff_absent_y_lag1'."
                ),
            )

            staff_attendance = st.number_input(
                "Staff Attendance Rate (Lag 1)",
                min_value=0.0,
                max_value=1.0,
                value=float(DEFAULTS["staff_attendance_rate_lag1"]),
                step=0.01,
                key="refusal_staff_attendance",
            )

        with c2:
            weekly_requested = st.number_input(
                "Patients Requested Weekly Summary (Lag 1)",
                min_value=0.0,
                max_value=5000.0,
                value=float(
                    DEFAULTS[
                        "patients_requested_weekly_summary_lag1"
                    ]
                ),
                step=1.0,
                key="refusal_weekly_requested",
            )

            weekly_admitted = st.number_input(
                "Patients Admitted Weekly Summary (Lag 1)",
                min_value=0.0,
                max_value=5000.0,
                value=float(
                    DEFAULTS[
                        "patients_admitted_weekly_summary_lag1"
                    ]
                ),
                step=1.0,
                key="refusal_weekly_admitted",
            )

            weekly_refused = st.number_input(
                "Patients Refused Weekly Summary (Lag 1)",
                min_value=0.0,
                max_value=5000.0,
                value=float(
                    DEFAULTS[
                        "patients_refused_weekly_summary_lag1"
                    ]
                ),
                step=1.0,
                key="refusal_weekly_refused",
            )

        with c3:
            waiting_time = st.number_input(
                "Avg Waiting Time (Lag 1)",
                min_value=0.0,
                max_value=1000.0,
                value=float(DEFAULTS["avg_waiting_time_lag1"]),
                step=0.1,
                key="refusal_waiting_time",
            )

            resource_shortage_lag1 = st.selectbox(
                "Resource Shortage (Lag 1)",
                [0, 1],
                index=int(DEFAULTS["resource_shortage_lag1"]),
                key="refusal_resource_shortage",
            )

            event_lag1 = st.selectbox(
                "Event (Lag 1)",
                ["none", "holiday", "strike", "outbreak", "other"],
                index=0,
                key="refusal_event",
            )

        refusal_submit = st.form_submit_button(
            "👥 Predict Patients Refused",
            type="primary",
        )

    if refusal_submit:

        refusal_values = {
            "avg_length_of_stay_lag1": refusal_los,
            "staff_required_lag1": refusal_staff_required,
            "capacity_gap_lag1": refusal_capacity_gap,
            "staff_scheduled_lag1": refusal_staff_scheduled,
            "staff_available_lag1": refusal_staff_available,
            "avg_satisfaction_score_lag1": refusal_satisfaction,
            "patients_requested_lag1": refusal_patients_requested,
            "service": refusal_service,
            "week_number": refusal_week,

            "available_beds_lag1": refusal_available_beds,
            "occupied_beds_lag1": refusal_occupied_beds,
            "vacant_beds_lag1": refusal_vacant_beds,
            "patients_admitted_lag1": refusal_patients_admitted,
            "patients_refused_lag1": refusal_patients_refused,
            "equipment_required_lag1": refusal_equipment_required,
            "equipment_available_lag1": refusal_equipment_available,
            "staff_morale_lag1": refusal_morale,
            "bed_utilization_rate_lag1": refusal_bed_util,
            "refusal_rate_lag1": refusal_rate,
            "admission_rate_lag1": admission_rate,
            "vacancy_rate_lag1": vacancy_rate,
            "equipment_availability_rate_lag1":
                equipment_availability_rate,
            "staff_availability_rate_lag1":
                staff_availability_rate,
            "staff_present_lag1": staff_present,
            "staff_absent_y_lag1": staff_absent,
            "staff_attendance_rate_lag1": staff_attendance,
            "patients_requested_weekly_summary_lag1":
                weekly_requested,
            "patients_admitted_weekly_summary_lag1":
                weekly_admitted,
            "patients_refused_weekly_summary_lag1":
                weekly_refused,
            "avg_waiting_time_lag1": waiting_time,
            "resource_shortage_lag1": resource_shortage_lag1,
            "event_lag1": event_lag1,
        }

        (
            refusal_prediction,
            refusal_error,
            refusal_input_df,
        ) = predict_regression(
            "refusal_regression",
            refusal_values,
            REFUSAL_REGRESSION_FEATURES,
        )

        st.markdown("---")
        st.subheader("Prediction Result")

        if refusal_prediction is not None:

            refusal_prediction = max(
                0.0,
                refusal_prediction,
            )

            st.metric(
                "Patients Refused Next Week",
                f"{refusal_prediction:.2f}",
            )

            if refusal_prediction > 0:
                st.warning(
                    f"The model predicts approximately "
                    f"{refusal_prediction:.2f} patient refusal(s) "
                    f"next week."
                )
            else:
                st.success(
                    "The model predicts no patient refusals "
                    "next week."
                )

            with st.expander(
                "🔍 Patients-Refused Model Input"
            ):
                st.dataframe(
                    refusal_input_df,
                    use_container_width=True,
                    hide_index=True,
                )

        else:
            st.error("Patients-refused prediction failed.")
            st.code(refusal_error)


# ============================================================
# 3. AVERAGE LENGTH OF STAY REGRESSION
# ============================================================
with tab_alos:

    st.markdown(
        '<div class="model-title">'
        '🛏️ Average Length of Stay Prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="model-description">'
        'These inputs are mapped ONLY to '
        '<b>secondary_regression_model.joblib</b>. '
        'This prediction is independent from the patients-refused model.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("alos_prediction_form"):

        c1, c2, c3 = st.columns(3)

        with c1:
            alos_service = st.selectbox(
                "Service",
                [
                    "emergency",
                    "general_medicine",
                    "icu",
                    "surgery",
                ],
                index=0,
                key="alos_service",
            )

            alos_week = st.number_input(
                "Week Number",
                min_value=1,
                max_value=52,
                value=int(DEFAULTS["week_number"]),
                step=1,
                key="alos_week",
            )

            alos_previous = st.number_input(
                "Avg Length of Stay (Lag 1)",
                min_value=0.0,
                max_value=60.0,
                value=float(DEFAULTS["avg_length_of_stay_lag1"]),
                step=0.1,
                key="alos_previous",
            )

        with c2:
            alos_staff_required = st.number_input(
                "Staff Required (Lag 1)",
                min_value=0.0,
                max_value=300.0,
                value=float(DEFAULTS["staff_required_lag1"]),
                step=1.0,
                key="alos_staff_required",
            )

            alos_capacity_gap = st.number_input(
                "Capacity Gap (Lag 1)",
                min_value=-300.0,
                max_value=300.0,
                value=float(DEFAULTS["capacity_gap_lag1"]),
                step=1.0,
                key="alos_capacity_gap",
            )

            alos_staff_scheduled = st.number_input(
                "Staff Scheduled (Lag 1)",
                min_value=0.0,
                max_value=300.0,
                value=float(DEFAULTS["staff_scheduled_lag1"]),
                step=1.0,
                key="alos_staff_scheduled",
            )

        with c3:
            alos_staff_available = st.number_input(
                "Staff Available (Lag 1)",
                min_value=0.0,
                max_value=300.0,
                value=float(DEFAULTS["staff_available_lag1"]),
                step=1.0,
                key="alos_staff_available",
            )

            alos_satisfaction = st.number_input(
                "Avg Satisfaction Score (Lag 1)",
                min_value=0.0,
                max_value=5.0,
                value=float(DEFAULTS["avg_satisfaction_score_lag1"]),
                step=0.1,
                key="alos_satisfaction",
            )

            alos_patients = st.number_input(
                "Patients Requested (Lag 1)",
                min_value=0.0,
                max_value=1000.0,
                value=float(DEFAULTS["patients_requested_lag1"]),
                step=1.0,
                key="alos_patients",
            )

        st.markdown("#### 📊 Additional Features (required by this model)")

        c1, c2 = st.columns(2)

        with c1:
            alos_patients_admitted = st.number_input(
                "Patients Admitted (Lag 1)",
                min_value=0.0,
                max_value=1000.0,
                value=float(DEFAULTS["patients_admitted_lag1"]),
                step=1.0,
                key="alos_patients_admitted",
            )

            alos_staff_present = st.number_input(
                "Staff Present (Lag 1)",
                min_value=0.0,
                max_value=300.0,
                value=float(DEFAULTS["staff_present_lag1"]),
                step=1.0,
                key="alos_staff_present",
            )

        with c2:
            alos_weekly_requested = st.number_input(
                "Patients Requested Weekly Summary (Lag 1)",
                min_value=0.0,
                max_value=5000.0,
                value=float(
                    DEFAULTS["patients_requested_weekly_summary_lag1"]
                ),
                step=1.0,
                key="alos_weekly_requested",
            )

            alos_equipment_available = st.number_input(
                "Equipment Available (Lag 1)",
                min_value=0.0,
                max_value=500.0,
                value=float(DEFAULTS["equipment_available_lag1"]),
                step=1.0,
                key="alos_equipment_available",
            )

        alos_submit = st.form_submit_button(
            "🛏️ Predict Average Length of Stay",
            type="primary",
        )

    if alos_submit:

        # Resolve the exact feature list from the model at runtime.
        _alos_model = get_model("alos_regression")
        _alos_known = model_expected_features(_alos_model)
        ALOS_FEATURES = (
            _alos_known if _alos_known is not None
            else ALOS_FEATURES_FALLBACK
        )

        alos_values = {
            "avg_length_of_stay_lag1": alos_previous,
            "staff_required_lag1": alos_staff_required,
            "capacity_gap_lag1": alos_capacity_gap,
            "staff_scheduled_lag1": alos_staff_scheduled,
            "staff_available_lag1": alos_staff_available,
            "avg_satisfaction_score_lag1": alos_satisfaction,
            "patients_requested_lag1": alos_patients,
            "service": alos_service,
            "week_number": alos_week,
            # Extra features exposed for models trained with wider schemas.
            "patients_admitted_lag1": alos_patients_admitted,
            "staff_present_lag1": alos_staff_present,
            "patients_requested_weekly_summary_lag1": alos_weekly_requested,
            "equipment_available_lag1": alos_equipment_available,
        }

        (
            alos_prediction,
            alos_error,
            alos_input_df,
        ) = predict_regression(
            _alos_model,
            alos_values,
            ALOS_FEATURES,
        )

        st.markdown("---")
        st.subheader("Prediction Result")

        if alos_prediction is not None:

            alos_prediction = max(
                0.0,
                alos_prediction,
            )

            st.metric(
                "Avg Length of Stay Next Week",
                f"{alos_prediction:.2f} Days",
            )

            with st.expander(
                "🔍 ALOS Model Input"
            ):
                st.dataframe(
                    alos_input_df,
                    use_container_width=True,
                    hide_index=True,
                )

        else:
            st.error(
                "Average Length of Stay prediction failed."
            )
            st.code(alos_error)

            st.info(
                "If secondary_regression_model.joblib is missing, "
                "the ALOS prediction cannot be generated until that "
                "model file is placed in the Models directory."
            )


# ============================================================
# MODEL LOADING DIAGNOSTICS
# ============================================================
st.markdown("---")

with st.expander(
    "🔍 Model Loading Diagnostics",
    expanded=False,
):

    st.write(
        f"**Application directory:** `{APP_DIR}`"
    )

    st.write(
        f"**Model directory selected:** `{MODEL_DIR}`"
    )

    st.write(
        f"**Model directory exists:** "
        f"`{MODEL_DIR.exists()}`"
    )

    st.write(
        f"**scikit-learn version:** `{SKLEARN_VERSION}`"
    )

    st.write(
        f"**XGBoost available:** `{XGBOOST_AVAILABLE}`"
    )

    st.write(
        f"**XGBoost version:** `{XGBOOST_VERSION}`"
    )

    if st.button("🔄 Clear Cache & Reload Models", key="reload_models_btn"):
        st.cache_resource.clear()
        st.rerun()

    diagnostic_rows = []

    role_names = {
        "classification": "Resource Shortage Classification",
        "refusal_regression": "Patients Refused Regression",
        "alos_regression": "Average Length of Stay Regression",
        "preprocessor": "Preprocessor",
    }

    for key, filename in MODEL_FILES.items():

        path = MODEL_DIR / filename
        model_obj = get_model(key)
        loaded = model_obj is not None

        diagnostic_rows.append(
            {
                "Component": role_names[key],
                "File": filename,
                "Exists": path.exists(),
                "Loaded": loaded,
                "Path": str(path),
                "Error": models.get("errors", {}).get(
                    key,
                    "",
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(diagnostic_rows),
        use_container_width=True,
        hide_index=True,
    )

    if SKLEARN_VERSION != "1.6.1":
        st.warning(
            "The supplied models were created with scikit-learn "
            "1.6.1. Your current environment is "
            f"scikit-learn {SKLEARN_VERSION}. "
            "For reliable model loading/prediction, run: "
            "`python -m pip install scikit-learn==1.6.1`"
        )

    if not XGBOOST_AVAILABLE:
        st.warning(
            "XGBoost is not installed. The patients-refused "
            "regression model may not load. Run: "
            "`python -m pip install xgboost`"
        )


# ============================================================
# MODEL FEATURE MAP
# ============================================================
with st.expander(
    "🧩 Model-to-Feature Mapping",
    expanded=False,
):

    st.markdown(
        """
        ### Resource Shortage Classification

        **Model:** `best_classification_model.joblib`

        **Inputs:** 9 features

        The following fields are passed to the classification model:
        """
    )

    st.code(
        "\n".join(
            f"{i + 1}. {feature}"
            for i, feature in enumerate(
                CLASSIFICATION_FEATURES
            )
        )
    )

    st.markdown(
        """
        ### Patients Refused Regression

        **Model:** `best_regression_model.joblib`

        **Inputs:** 32 features
        """
    )

    st.code(
        "\n".join(
            f"{i + 1}. {feature}"
            for i, feature in enumerate(
                REFUSAL_REGRESSION_FEATURES
            )
        )
    )

    st.markdown(
        """
        ### Average Length of Stay Regression

        **Model:** `secondary_regression_model.joblib`

        **Inputs:** 9 features
        """
    )

    st.code(
        "\n".join(
            f"{i + 1}. {feature}"
            for i, feature in enumerate(
                ALOS_FEATURES_FALLBACK
            )
        )
    )


# ============================================================
# STAFF STATUS SUMMARY
# ============================================================
st.markdown("---")

st.subheader("👨‍⚕️ Current Staff Status")

staff_col1, staff_col2, staff_col3 = st.columns(3)

staff_available_summary = float(
    DEFAULTS["staff_available_lag1"]
)
staff_required_summary = float(
    DEFAULTS["staff_required_lag1"]
)
staff_gap_summary = (
    staff_available_summary -
    staff_required_summary
)

with staff_col1:
    st.metric(
        "Staff Available",
        f"{staff_available_summary:.0f}",
    )

with staff_col2:
    st.metric(
        "Staff Required",
        f"{staff_required_summary:.0f}",
    )

with staff_col3:
    st.metric(
        "Current Staff Gap",
        f"{staff_gap_summary:+.0f}",
        delta=(
            "Surplus"
            if staff_gap_summary >= 0
            else "Deficit"
        ),
        delta_color=(
            "normal"
            if staff_gap_summary >= 0
            else "inverse"
        ),
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.caption(
    "🏥 Hospital Resource and Patient Flow Prediction System • "
    "Streamlit / Gradio-style interface • "
    "Each prediction button sends only its assigned feature set "
    "to its corresponding trained model."
)