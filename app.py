from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
METADATA_PATH = APP_DIR / "model_metadata.json"
METRICS_PATH = APP_DIR / "model_metrics.csv"
VALIDATION_PATH = APP_DIR / "validation_predictions.csv"

with METADATA_PATH.open("r", encoding="utf-8") as f:
    META = json.load(f)

FEATURES = META["features"]
FEATURE_COLUMNS = [f["column"] for f in FEATURES]
MODEL_FILENAME = META["model_filename"]
LOCAL_WINDOWS_MODEL = Path(
    r"C:\Users\24550372\OneDrive - UTS\UTS PhD\Photocatalyst\Pharmaceutical Review\Data and model\pharmamlv3\XGBPSOModel_success_seed605.pkl"
)

st.set_page_config(
    page_title="XGB–PSO Pharmaceutical Degradation Predictor",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,0.25); padding: 0.8rem; border-radius: 0.6rem;}
    .small-note {font-size: 0.9rem; opacity: 0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_serialized_model(source: Any):
    """Load a trusted joblib/pickle model from a path or in-memory bytes."""
    try:
        return joblib.load(source)
    except Exception as joblib_error:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
                return pickle.load(source)
            with open(source, "rb") as handle:
                return pickle.load(handle)
        except Exception as pickle_error:
            raise RuntimeError(
                f"The model could not be loaded. joblib error: {joblib_error}; "
                f"pickle error: {pickle_error}"
            ) from pickle_error


@st.cache_resource(show_spinner=False)
def load_model_from_path(path_string: str):
    return load_serialized_model(path_string)


@st.cache_resource(show_spinner=False)
def load_model_from_bytes(file_bytes: bytes):
    return load_serialized_model(io.BytesIO(file_bytes))


def find_local_model() -> Path | None:
    candidates = [APP_DIR / MODEL_FILENAME, LOCAL_WINDOWS_MODEL]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def expected_feature_count(model) -> int | None:
    count = getattr(model, "n_features_in_", None)
    if count is not None:
        return int(count)
    if model.__class__.__name__ == "Booster" and hasattr(model, "num_features"):
        return int(model.num_features())
    return None


def predict_degradation(model, ordered_frame: pd.DataFrame) -> np.ndarray:
    """Predict while preserving the exact 11-feature order used during training."""
    expected = expected_feature_count(model)
    if expected is not None and expected != len(FEATURE_COLUMNS):
        raise ValueError(
            f"The loaded model expects {expected} features, but this application provides "
            f"{len(FEATURE_COLUMNS)}. Confirm that the correct XGB–PSO .pkl file was loaded."
        )

    values = ordered_frame[FEATURE_COLUMNS].to_numpy(dtype=float)

    if model.__class__.__name__ == "Booster":
        import xgboost as xgb

        names = getattr(model, "feature_names", None)
        if names is not None and len(names) != len(FEATURE_COLUMNS):
            names = None
        matrix = xgb.DMatrix(values, feature_names=names)
        return np.asarray(model.predict(matrix), dtype=float).reshape(-1)

    feature_names_in = getattr(model, "feature_names_in_", None)
    if feature_names_in is not None and len(feature_names_in) == len(FEATURE_COLUMNS):
        model_input = pd.DataFrame(values, columns=list(feature_names_in))
        return np.asarray(model.predict(model_input), dtype=float).reshape(-1)

    return np.asarray(model.predict(values), dtype=float).reshape(-1)


def range_warnings(frame: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    for feature in FEATURES:
        col = feature["column"]
        lower = float(feature["min"])
        upper = float(feature["max"])
        bad = (frame[col] < lower) | (frame[col] > upper)
        if bad.any():
            rows = ", ".join(str(i + 2) for i in frame.index[bad][:8])
            suffix = "…" if int(bad.sum()) > 8 else ""
            warnings.append(
                f"{col}: {int(bad.sum())} value(s) outside the training range "
                f"[{lower:g}, {upper:g}] (spreadsheet row(s) {rows}{suffix})."
            )
    return warnings


def normalize_light_source(series: pd.Series) -> pd.Series:
    mapping = {
        "uv": 1,
        "ultraviolet": 1,
        "visible": 2,
        "visible light": 2,
        "solar": 3,
        "simulated solar": 3,
        "simulated solar light": 3,
    }

    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            text = value.strip().lower()
            if text in mapping:
                return mapping[text]
        return pd.to_numeric(value, errors="coerce")

    return series.map(convert)


def make_template() -> bytes:
    example = {f["column"]: f["default"] for f in FEATURES}
    example["Light source code"] = 2
    data = pd.DataFrame([example])
    notes = pd.DataFrame(
        {
            "Item": [
                "Feature order",
                "Light source coding",
                "No oxidant",
                "Model target",
                "Applicability warning",
            ],
            "Instruction": [
                "Do not rename or reorder the 11 input columns.",
                "UV = 1; visible light = 2; simulated solar light = 3.",
                "Enter oxidant concentration as 0 mM.",
                "The output is predicted pharmaceutical degradation (%).",
                "Predictions outside the model's training domain are extrapolations and require experimental confirmation.",
            ],
        }
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Prediction Inputs", index=False)
        notes.to_excel(writer, sheet_name="Instructions", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def make_results_workbook(results: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame(FEATURES).to_excel(writer, sheet_name="Feature Metadata", index=False)
    buffer.seek(0)
    return buffer.getvalue()


# Sidebar
st.sidebar.header("Model information")
st.sidebar.markdown(
    "**Model:** XGBoost optimized using Particle Swarm Optimization (XGB–PSO)"
)
st.sidebar.markdown("**Inputs:**")
for feature in FEATURES:
    unit = f" ({feature['unit']})" if feature["unit"] else ""
    st.sidebar.markdown(f"- {feature['label']}{unit}")
st.sidebar.markdown("**Output:** predicted degradation (%)")
st.sidebar.divider()
st.sidebar.markdown("**Developer:** Amir Dashti  ")
st.sidebar.markdown("School of Civil and Environmental Engineering, UTS")

model_upload = st.sidebar.file_uploader(
    "Load the trusted XGB–PSO model",
    type=["pkl", "joblib"],
    help="For Streamlit Cloud, place the model in the repository or upload it here. Only load a model file you trust.",
)

model = None
model_source = None
try:
    if model_upload is not None:
        model = load_model_from_bytes(model_upload.getvalue())
        model_source = f"Uploaded file: {model_upload.name}"
    else:
        model_path = find_local_model()
        if model_path is not None:
            model = load_model_from_path(str(model_path))
            model_source = str(model_path)
except Exception as exc:
    st.sidebar.error(f"Model loading failed: {exc}")

st.title("XGB–PSO Predictor for Pharmaceutical Degradation by Photocatalysts")
st.write(
    "This application estimates photocatalytic degradation efficiency using the final "
    "XGB–PSO model developed from literature-derived pharmaceutical degradation data."
)

if model is None:
    st.warning(
        f"The model is not currently loaded. Copy **{MODEL_FILENAME}** into the same folder as "
        "`app.py`, keep it at the local Windows path specified in the code, or upload it from the sidebar."
    )
else:
    st.success(f"Model loaded successfully. Source: {model_source}")

with st.expander("Input format, coding, and applicability notes"):
    st.markdown(
        "- The model requires **11 numeric inputs in the exact saved order**. The application enforces this order.\n"
        "- Light source coding: **UV = 1**, **visible light = 2**, and **simulated solar light = 3**.\n"
        "- Enter **0 mM** oxidant concentration when no persulfate/oxidant was used.\n"
        "- Inputs outside the training ranges are extrapolations. Such predictions should be interpreted cautiously and verified experimentally.\n"
        "- The predicted percentage is displayed without artificial clipping; a value outside 0–100% indicates model extrapolation or an incompatible input combination."
    )

tab_single, tab_batch, tab_performance, tab_domain = st.tabs(
    ["Single prediction", "Batch prediction", "Model performance", "Training domain"]
)

with tab_single:
    with st.form("single_prediction_form"):
        left, right = st.columns(2)
        values = {}
        for idx, feature in enumerate(FEATURES):
            target_col = left if idx % 2 == 0 else right
            with target_col:
                if feature["key"] == "Light":
                    label_to_code = META["light_source_codes"]
                    default_label = "Visible light"
                    selected = st.selectbox(
                        feature["label"],
                        options=list(label_to_code.keys()),
                        index=list(label_to_code.keys()).index(default_label),
                        help=feature["help"],
                    )
                    values[feature["column"]] = float(label_to_code[selected])
                else:
                    unit = f" ({feature['unit']})" if feature["unit"] else ""
                    values[feature["column"]] = st.number_input(
                        feature["label"] + unit,
                        value=float(feature["default"]),
                        format="%.4f",
                        help=(
                            f"{feature['help']} Training range: "
                            f"{feature['min']:g} to {feature['max']:g}."
                        ),
                    )
        submitted = st.form_submit_button("Predict degradation (%)", type="primary")

    if submitted:
        input_frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        warnings = range_warnings(input_frame)
        for warning in warnings:
            st.warning(warning)
        if model is None:
            st.error("Load the saved XGB–PSO model before making a prediction.")
        else:
            try:
                prediction = float(predict_degradation(model, input_frame)[0])
                st.metric("Predicted pharmaceutical degradation", f"{prediction:.2f}%")
                if prediction < 0 or prediction > 100:
                    st.warning(
                        "The raw model output is outside the physical 0–100% interval. "
                        "Review the inputs and training-domain warnings before interpretation."
                    )
                with st.expander("Show ordered model input"):
                    st.dataframe(input_frame, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab_batch:
    st.download_button(
        "Download Excel input template",
        data=make_template(),
        file_name="pharmaceutical_degradation_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    uploaded_data = st.file_uploader(
        "Upload a completed Excel or CSV file",
        type=["xlsx", "csv"],
        key="batch_file",
    )
    if uploaded_data is not None:
        try:
            if uploaded_data.name.lower().endswith(".csv"):
                batch = pd.read_csv(uploaded_data)
            else:
                batch = pd.read_excel(uploaded_data, sheet_name=0)

            missing = [c for c in FEATURE_COLUMNS if c not in batch.columns]
            if missing:
                st.error("Missing required columns: " + "; ".join(missing))
            else:
                ordered = batch[FEATURE_COLUMNS].copy()
                ordered["Light source code"] = normalize_light_source(ordered["Light source code"])
                for col in FEATURE_COLUMNS:
                    ordered[col] = pd.to_numeric(ordered[col], errors="coerce")

                invalid_mask = ordered.isna().any(axis=1)
                if invalid_mask.any():
                    invalid_rows = ", ".join(str(i + 2) for i in ordered.index[invalid_mask][:15])
                    st.error(
                        f"{int(invalid_mask.sum())} row(s) contain missing or nonnumeric values "
                        f"(spreadsheet rows {invalid_rows}). Correct them before prediction."
                    )
                else:
                    warnings = range_warnings(ordered)
                    for warning in warnings:
                        st.warning(warning)
                    st.dataframe(ordered.head(20), use_container_width=True, hide_index=True)
                    if st.button("Run batch prediction", type="primary"):
                        if model is None:
                            st.error("Load the saved XGB–PSO model before making predictions.")
                        else:
                            predictions = predict_degradation(model, ordered)
                            results = batch.copy()
                            results["Predicted degradation (%)"] = predictions
                            results["Outside 0–100%"] = (predictions < 0) | (predictions > 100)
                            st.success(f"Predictions generated for {len(results)} rows.")
                            st.dataframe(results.head(50), use_container_width=True, hide_index=True)
                            st.download_button(
                                "Download prediction results",
                                data=make_results_workbook(results),
                                file_name="pharmaceutical_degradation_predictions.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
        except Exception as exc:
            st.error(f"The uploaded data file could not be processed: {exc}")

with tab_performance:
    if METRICS_PATH.exists() and VALIDATION_PATH.exists():
        metric_df = pd.read_csv(METRICS_PATH)
        validation = pd.read_csv(VALIDATION_PATH)
        st.caption(
            f"Saved validation results: {META['data_points']} observations; "
            f"{META['training_points']} training and {META['testing_points']} testing; seed {META['seed']}."
        )
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        for dataset_name, subset in validation.groupby("Dataset"):
            ax.scatter(
                subset["Experimental degradation (%)"],
                subset["Predicted degradation (%)"],
                label=dataset_name,
                alpha=0.7,
                s=24,
            )
        min_value = float(
            min(
                validation["Experimental degradation (%)"].min(),
                validation["Predicted degradation (%)"].min(),
            )
        )
        max_value = float(
            max(
                validation["Experimental degradation (%)"].max(),
                validation["Predicted degradation (%)"].max(),
            )
        )
        ax.plot([min_value, max_value], [min_value, max_value], "--", label="45° line")
        ax.set_xlabel("Experimental degradation (%)")
        ax.set_ylabel("Predicted degradation (%)")
        ax.set_title("XGB–PSO experimental versus predicted degradation")
        ax.legend()
        ax.grid(False)
        st.pyplot(fig, use_container_width=False)
        st.caption(
            "These statistics and predictions are read from the supplied MATLAB results file; "
            "they are not recalculated from a newly uploaded model."
        )
    else:
        st.info("Validation result files were not found in the application folder.")

with tab_domain:
    domain_rows = []
    for feature in FEATURES:
        domain_rows.append(
            {
                "Feature": feature["label"],
                "Unit": feature["unit"],
                "Training minimum": feature["min"],
                "Training maximum": feature["max"],
                "Model-order position": len(domain_rows) + 1,
            }
        )
    st.dataframe(pd.DataFrame(domain_rows), use_container_width=True, hide_index=True)
    st.info(
        "The training range is a necessary but not sufficient definition of applicability. "
        "Predictions for unusual combinations of otherwise in-range inputs may still be unreliable."
    )
