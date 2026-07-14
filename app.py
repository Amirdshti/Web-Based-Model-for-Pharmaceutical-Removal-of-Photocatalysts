# -*- coding: utf-8 -*-
"""
Streamlit Web App for the saved XGB-PSO pharmaceutical photodegradation model.

Workflow:
1. Download an Excel input template.
2. Fill the experimental conditions and optionally the measured removal (%).
3. Upload the completed Excel/CSV file.
4. Apply the final saved XGB-PSO model to the uploaded rows as unseen data.
5. Display predictions, validation statistics, and an experimental-versus-predicted plot.
6. Download complete Excel results and the validation plot.

Important:
- The model is already trained. Uploaded rows are not used to retrain the model.
- The descriptive "Oxidant" column is retained in the spreadsheet but is not a model input.
- Light-source coding used by the saved model is UV = 1, visible light = 2,
  and simulated solar light = 3.
"""

from __future__ import annotations

import io
import pickle
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# Application and saved-model configuration
# ============================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "XGBPSOModel_success_seed605.pkl"
MODEL_PATH = APP_DIR / MODEL_FILENAME

# This Windows path is checked only when the app is run on the user's own PC.
# It is not accessible from Streamlit Community Cloud.
LOCAL_WINDOWS_MODEL = Path(
    r"C:\Users\24550372\OneDrive - UTS\UTS PhD\Photocatalyst\Pharmaceutical Review\Data and model\pharmamlv3\XGBPSOModel_success_seed605.pkl"
)

# Spreadsheet headers follow the user's SampleData.xlsx file.
BET_COL = "BET specific surface area (m2 g-1)"
OXIDANT_NAME_COL = "Oxidant"
OXIDANT_CONC_COL = "Oxidant concentration (mM)"
MW_COL = "Molecular Weight (g/mol)"
HBDC_COL = "HBDC"
HBAC_COL = "HBAC"
TPSA_COL = "Topological Polar Surface Area (Å²)"
INITIAL_CONC_COL = "Initial concentration of pollutant (mg/L)"
PH_COL = "pH"
LIGHT_COL = "Light source"
CATALYST_COL = "Catalyst dosage (mg/L)"
TIME_COL = "t (min)"
TARGET_COL = "Degradation or Removal (%)"
PREDICTION_COL = "Predicted Degradation or Removal (%)"
RESIDUAL_COL = "Residual: Experimental - Predicted"
ABS_ERROR_COL = "Absolute Error"
REL_ERROR_COL = "Absolute Relative Error (%)"
DOMAIN_COL = "Outside model training range"

TEMPLATE_COLUMNS = [
    BET_COL,
    OXIDANT_NAME_COL,
    OXIDANT_CONC_COL,
    MW_COL,
    HBDC_COL,
    HBAC_COL,
    TPSA_COL,
    INITIAL_CONC_COL,
    PH_COL,
    LIGHT_COL,
    CATALYST_COL,
    TIME_COL,
    TARGET_COL,
]

# Exact numerical feature order used in the final saved model.
MODEL_INPUT_COLUMNS = [
    BET_COL,
    OXIDANT_CONC_COL,
    MW_COL,
    HBDC_COL,
    HBAC_COL,
    TPSA_COL,
    INITIAL_CONC_COL,
    PH_COL,
    LIGHT_COL,
    CATALYST_COL,
    TIME_COL,
]

# Training-domain limits extracted from the supplied MATLAB model results.
TRAINING_DOMAIN = {
    BET_COL: (5.686, 733.03),
    OXIDANT_CONC_COL: (0.0, 8.0),
    MW_COL: (151.16, 480.9),
    HBDC_COL: (1.0, 7.0),
    HBAC_COL: (1.0, 12.0),
    TPSA_COL: (37.3, 235.0),
    INITIAL_CONC_COL: (0.5, 200.0),
    PH_COL: (1.0, 11.5),
    LIGHT_COL: (1.0, 3.0),
    CATALYST_COL: (50.0, 8000.0),
    TIME_COL: (0.5, 2880.0),
}

FINAL_MODEL_METRICS = pd.DataFrame(
    {
        "Metric": ["R²", "RMSE", "MAE", "AARD (%)"],
        "Training": [0.996338, 1.779452, 1.202669, 5.279755],
        "Testing": [0.968422, 4.947778, 3.371177, 11.857105],
        "Total": [0.992545, 2.520885, 1.527061, 6.263674],
    }
)

# Alternative column names accepted during upload.
COLUMN_ALIASES = {
    BET_COL: [BET_COL, "BET specific surface area (m²/g)", "Specific Surface Area (m2/g)"],
    OXIDANT_NAME_COL: [OXIDANT_NAME_COL, "Oxidant type"],
    OXIDANT_CONC_COL: [OXIDANT_CONC_COL, "Oxidant Concentration (mM)"],
    MW_COL: [MW_COL, "Molecular weight (g/mol)", "MW (g mol-1)"],
    HBDC_COL: [HBDC_COL],
    HBAC_COL: [HBAC_COL],
    TPSA_COL: [TPSA_COL, "Topological Polar Surface Area (Å²)", "TPSA (Å²)", "TPSA"],
    INITIAL_CONC_COL: [
        INITIAL_CONC_COL,
        "Initial pollutant concentration, C0 (mg/L)",
        "Pollutant dosage (mg/L)",
    ],
    PH_COL: [PH_COL, "Solution pH"],
    LIGHT_COL: [LIGHT_COL, "Light source code"],
    CATALYST_COL: [CATALYST_COL, "Photocatalyst dosage (mg/L)"],
    TIME_COL: [TIME_COL, "Reaction time (min)"],
    TARGET_COL: [TARGET_COL, "Degradation (%)", "Removal (%)"],
}

LIGHT_SOURCE_MAPPING = {
    "uv": 1.0,
    "ultraviolet": 1.0,
    "uv light": 1.0,
    "visible": 2.0,
    "visible light": 2.0,
    "vis": 2.0,
    "solar": 3.0,
    "solar light": 3.0,
    "simulated solar": 3.0,
    "simulated solar light": 3.0,
    "sunlight": 3.0,
}


# ============================================================
# Streamlit page
# ============================================================
st.set_page_config(
    page_title="XGB-PSO Pharmaceutical Removal Model",
    layout="wide",
)

st.title("Web-Based XGB-PSO Model for Pharmaceutical Removal by Photocatalysts")
st.write(
    "Download the Excel template, enter the photocatalyst and pharmaceutical data, "
    "upload the completed file, and apply the final XGB-PSO model to predict "
    "degradation or removal efficiency."
)


# ============================================================
# Utility functions
# ============================================================
def unwrap_model(model_object: Any) -> Any:
    """Extract an estimator when a trusted pickle stores it inside a dictionary."""
    if isinstance(model_object, dict):
        for key in ("model", "best_model", "xgb_model", "estimator", "regressor"):
            candidate = model_object.get(key)
            if candidate is not None and hasattr(candidate, "predict"):
                return candidate
    return model_object


def load_serialized_model(source: Any) -> Any:
    """Load a trusted joblib/pickle model from a file path or in-memory buffer."""
    try:
        return unwrap_model(joblib.load(source))
    except Exception as joblib_error:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
                return unwrap_model(pickle.load(source))
            with open(source, "rb") as handle:
                return unwrap_model(pickle.load(handle))
        except Exception as pickle_error:
            raise RuntimeError(
                "The model could not be loaded. "
                f"Joblib error: {joblib_error}; pickle error: {pickle_error}"
            ) from pickle_error


@st.cache_resource(show_spinner=False)
def load_model_from_path(path_string: str) -> Any:
    return load_serialized_model(path_string)


@st.cache_resource(show_spinner=False)
def load_model_from_bytes(file_bytes: bytes) -> Any:
    return load_serialized_model(io.BytesIO(file_bytes))


def find_model_path() -> Path | None:
    for candidate in (MODEL_PATH, LOCAL_WINDOWS_MODEL):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def expected_feature_count(model: Any) -> int | None:
    count = getattr(model, "n_features_in_", None)
    if count is not None:
        return int(count)
    if model.__class__.__name__ == "Booster" and hasattr(model, "num_features"):
        return int(model.num_features())
    return None


def predict_with_saved_model(model: Any, model_inputs: pd.DataFrame) -> np.ndarray:
    expected = expected_feature_count(model)
    supplied = model_inputs.shape[1]
    if expected is not None and expected != supplied:
        raise ValueError(
            f"The loaded model expects {expected} inputs, but the application supplies "
            f"{supplied}. Confirm that {MODEL_FILENAME} is the correct model file."
        )

    values = model_inputs.to_numpy(dtype=float)

    if model.__class__.__name__ == "Booster":
        import xgboost as xgb

        feature_names = getattr(model, "feature_names", None)
        if feature_names is not None and len(feature_names) != supplied:
            feature_names = None
        matrix = xgb.DMatrix(values, feature_names=feature_names)
        predictions = model.predict(matrix)
    else:
        saved_names = getattr(model, "feature_names_in_", None)
        if saved_names is not None and len(saved_names) == supplied:
            prediction_input = pd.DataFrame(values, columns=list(saved_names))
        else:
            prediction_input = values
        predictions = model.predict(prediction_input)

    return np.asarray(predictions, dtype=float).reshape(-1)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true)
    valid = denominator > 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid]) / denominator[valid]) * 100)


def compute_statistics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    statistics = {
        "Number of validation rows": float(len(y_true)),
        "R²": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": compute_mape(y_true, y_pred),
        "AARD (%)": compute_mape(y_true, y_pred),
    }
    return statistics


def normalize_uploaded_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename accepted aliases to the canonical SampleData.xlsx headers."""
    normalized = frame.copy()
    stripped_lookup = {str(col).strip(): col for col in normalized.columns}
    rename_map: dict[Any, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            source = stripped_lookup.get(alias.strip())
            if source is not None:
                rename_map[source] = canonical
                break

    return normalized.rename(columns=rename_map)


def normalize_light_source(series: pd.Series) -> pd.Series:
    def convert(value: Any) -> float:
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            text = value.strip().lower()
            if text in LIGHT_SOURCE_MAPPING:
                return LIGHT_SOURCE_MAPPING[text]
        numeric = pd.to_numeric(value, errors="coerce")
        return float(numeric) if not pd.isna(numeric) else np.nan

    return series.map(convert)


def prepare_uploaded_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned display data and ordered numeric model inputs."""
    frame = normalize_uploaded_columns(frame)

    missing_inputs = [col for col in MODEL_INPUT_COLUMNS if col not in frame.columns]
    if missing_inputs:
        raise ValueError("Missing required columns: " + "; ".join(missing_inputs))

    if OXIDANT_NAME_COL not in frame.columns:
        frame[OXIDANT_NAME_COL] = ""
    if TARGET_COL not in frame.columns:
        frame[TARGET_COL] = np.nan

    # Remove rows that have no model inputs at all.
    nonempty_mask = ~frame[MODEL_INPUT_COLUMNS].isna().all(axis=1)
    frame = frame.loc[nonempty_mask].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("The uploaded file does not contain any populated data rows.")

    ordered_inputs = frame[MODEL_INPUT_COLUMNS].copy()
    ordered_inputs[LIGHT_COL] = normalize_light_source(ordered_inputs[LIGHT_COL])

    for column in MODEL_INPUT_COLUMNS:
        if column != LIGHT_COL:
            ordered_inputs[column] = pd.to_numeric(ordered_inputs[column], errors="coerce")

    invalid_mask = ordered_inputs.isna().any(axis=1)
    if invalid_mask.any():
        spreadsheet_rows = [str(index + 2) for index in ordered_inputs.index[invalid_mask][:20]]
        raise ValueError(
            f"{int(invalid_mask.sum())} row(s) contain missing or nonnumeric model inputs. "
            "Correct spreadsheet row(s): " + ", ".join(spreadsheet_rows)
        )

    frame[TARGET_COL] = pd.to_numeric(frame[TARGET_COL], errors="coerce")
    frame[LIGHT_COL] = ordered_inputs[LIGHT_COL]

    return frame, ordered_inputs


def identify_domain_extrapolation(model_inputs: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    outside_any = pd.Series(False, index=model_inputs.index)
    messages: list[str] = []

    for column, (lower, upper) in TRAINING_DOMAIN.items():
        outside = (model_inputs[column] < lower) | (model_inputs[column] > upper)
        outside_any |= outside
        if outside.any():
            rows = ", ".join(str(index + 2) for index in model_inputs.index[outside][:12])
            suffix = "..." if int(outside.sum()) > 12 else ""
            messages.append(
                f"{column}: {int(outside.sum())} value(s) outside [{lower:g}, {upper:g}] "
                f"in spreadsheet row(s) {rows}{suffix}."
            )

    return outside_any, messages


def create_excel_template() -> bytes:
    blank_data = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    example_data = pd.DataFrame(
        [
            {
                BET_COL: 100.0,
                OXIDANT_NAME_COL: "None",
                OXIDANT_CONC_COL: 0.0,
                MW_COL: 444.4,
                HBDC_COL: 6,
                HBAC_COL: 9,
                TPSA_COL: 182.0,
                INITIAL_CONC_COL: 20.0,
                PH_COL: 7.0,
                LIGHT_COL: "Visible light",
                CATALYST_COL: 500.0,
                TIME_COL: 60.0,
                TARGET_COL: 85.0,
            }
        ],
        columns=TEMPLATE_COLUMNS,
    )

    instructions = pd.DataFrame(
        {
            "Item": [
                "Required input columns",
                "Oxidant column",
                "Light source",
                "Experimental target",
                "Catalyst dosage",
                "No oxidant",
                "Unseen-data validation",
                "Feature order",
            ],
            "Instruction": [
                "Do not rename the template columns. Add one experimental condition per row.",
                "Oxidant is descriptive only and is retained in the output; it is not sent to the model.",
                "Enter UV, Visible light, or Simulated solar light. Numeric codes 1, 2, and 3 are also accepted.",
                "The final column is optional. Fill it to calculate R², RMSE, MAE, MAPE, AARD, and residuals.",
                "Use mg/L, matching the model-development dataset.",
                "Enter 0 in Oxidant concentration (mM) when no oxidant is used.",
                "Uploaded rows are treated as new unseen observations. The saved final model is not retrained.",
                "The application automatically enforces the exact 11-variable order used during model training.",
            ],
        }
    )

    domain = pd.DataFrame(
        [
            {"Input": column, "Minimum": limits[0], "Maximum": limits[1]}
            for column, limits in TRAINING_DOMAIN.items()
        ]
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        blank_data.to_excel(writer, sheet_name="Data Template", index=False)
        example_data.to_excel(writer, sheet_name="Example Row", index=False)
        instructions.to_excel(writer, sheet_name="Instructions", index=False)
        domain.to_excel(writer, sheet_name="Training Domain", index=False)

        for sheet_name in writer.book.sheetnames:
            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                updated_font = copy(cell.font)
                updated_font.bold = True
                cell.font = updated_font
            for column_cells in worksheet.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                column_letter = column_cells[0].column_letter
                worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 42)

    buffer.seek(0)
    return buffer.getvalue()


def create_results_workbook(
    results: pd.DataFrame,
    statistics: dict[str, float] | None,
    model_source: str,
) -> bytes:
    if statistics:
        statistics_frame = pd.DataFrame(
            {"Metric": list(statistics.keys()), "Value": list(statistics.values())}
        )
    else:
        statistics_frame = pd.DataFrame(
            {
                "Metric": ["Validation statistics"],
                "Value": ["Not calculated because experimental target values were not supplied."],
            }
        )

    model_information = pd.DataFrame(
        {
            "Item": [
                "Model",
                "Model file",
                "Model source",
                "Prediction mode",
                "Model input count",
                "Light source coding",
                "Generated at (UTC)",
            ],
            "Value": [
                "XGBoost optimized using particle swarm optimization (XGB-PSO)",
                MODEL_FILENAME,
                model_source,
                "Fixed saved model applied to unseen uploaded data",
                len(MODEL_INPUT_COLUMNS),
                "UV = 1; Visible light = 2; Simulated solar light = 3",
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            ],
        }
    )

    domain = pd.DataFrame(
        [
            {"Input": column, "Minimum": limits[0], "Maximum": limits[1]}
            for column, limits in TRAINING_DOMAIN.items()
        ]
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Prediction Results", index=False)
        statistics_frame.to_excel(writer, sheet_name="Uploaded Data Statistics", index=False)
        FINAL_MODEL_METRICS.to_excel(writer, sheet_name="Final Model Performance", index=False)
        model_information.to_excel(writer, sheet_name="Model Information", index=False)
        domain.to_excel(writer, sheet_name="Training Domain", index=False)

        for sheet_name in writer.book.sheetnames:
            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                updated_font = copy(cell.font)
                updated_font.bold = True
                cell.font = updated_font
            for column_cells in worksheet.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                column_letter = column_cells[0].column_letter
                worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 42)

    buffer.seek(0)
    return buffer.getvalue()


def make_validation_plot(
    experimental: np.ndarray,
    predicted: np.ndarray,
    statistics: dict[str, float],
):
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(experimental, predicted, label="Uploaded validation data", s=38)

    minimum = float(min(np.min(experimental), np.min(predicted)))
    maximum = float(max(np.max(experimental), np.max(predicted)))
    margin = max((maximum - minimum) * 0.05, 1.0)
    lower = minimum - margin
    upper = maximum + margin

    axis.plot([lower, upper], [lower, upper], "--", label="45° line")

    if len(experimental) >= 2 and np.ptp(experimental) > 0:
        coefficients = np.polyfit(experimental, predicted, 1)
        x_line = np.linspace(lower, upper, 100)
        axis.plot(x_line, np.polyval(coefficients, x_line), label="Best-fit line")

    r2_value = statistics.get("R²", float("nan"))
    r2_text = "N/A" if np.isnan(r2_value) else f"{r2_value:.4f}"

    axis.text(
        0.05,
        0.95,
        f"R² = {r2_text}\n"
        f"RMSE = {statistics['RMSE']:.4f}\n"
        f"MAE = {statistics['MAE']:.4f}\n"
        f"MAPE = {statistics['MAPE (%)']:.2f}%",
        transform=axis.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Experimental degradation or removal (%)")
    axis.set_ylabel("Predicted degradation or removal (%)")
    axis.set_title("XGB-PSO Experimental versus Predicted Values")
    axis.legend()
    axis.grid(False)
    figure.tight_layout()
    return figure


def figure_to_png(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# Session state
# ============================================================
if "prediction_run" not in st.session_state:
    st.session_state.prediction_run = False
if "attempt" not in st.session_state:
    st.session_state.attempt = 0


# ============================================================
# Load model
# ============================================================
st.sidebar.header("Saved Model")
model_upload = st.sidebar.file_uploader(
    "Upload the trusted XGB-PSO model if it is not stored with app.py",
    type=["pkl", "joblib"],
    help="Only upload a model file that you trust.",
)

model = None
model_source = "Not loaded"
try:
    if model_upload is not None:
        model = load_model_from_bytes(model_upload.getvalue())
        model_source = f"Uploaded model: {model_upload.name}"
    else:
        available_path = find_model_path()
        if available_path is not None:
            model = load_model_from_path(str(available_path))
            model_source = str(available_path)
except Exception as error:
    st.sidebar.error(f"Model loading failed: {error}")

if model is not None:
    st.sidebar.success("XGB-PSO model loaded")
    expected = expected_feature_count(model)
    st.sidebar.write(f"Expected inputs: {expected if expected is not None else 'not reported'}")
else:
    st.sidebar.warning(
        f"Add {MODEL_FILENAME} beside app.py in GitHub, or upload it above."
    )

with st.sidebar.expander("Final model performance"):
    st.dataframe(FINAL_MODEL_METRICS, hide_index=True, use_container_width=True)


# ============================================================
# Step 1: Download Excel template
# ============================================================
st.subheader("Step 1: Download Excel Template")

st.download_button(
    label="Download Pharmaceutical Model Data Template",
    data=create_excel_template(),
    file_name="Pharmaceutical_XGB_PSO_Input_Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.info(
    "Fill the required input columns. The final experimental degradation/removal column is optional. "
    "When it is filled, the application calculates validation statistics and displays an R² plot."
)


# ============================================================
# Step 2: Upload completed file
# ============================================================
st.subheader("Step 2: Upload Completed Excel File")

uploaded_file = st.file_uploader(
    "Upload the completed Excel or CSV file",
    type=["xlsx", "csv"],
    key="experimental_data_upload",
)

if uploaded_file is None:
    st.warning("Please upload the completed Excel file after filling the template.")
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        raw_data = pd.read_csv(uploaded_file)
    else:
        raw_data = pd.read_excel(uploaded_file, sheet_name=0)

    display_data, model_inputs = prepare_uploaded_data(raw_data)
except Exception as error:
    st.error(f"The uploaded file could not be processed: {error}")
    st.stop()

st.success("Excel file uploaded and validated successfully.")
st.write(f"Valid uploaded rows: **{len(display_data)}**")
st.dataframe(display_data.head(25), use_container_width=True, hide_index=True)

outside_domain, domain_messages = identify_domain_extrapolation(model_inputs)
if domain_messages:
    with st.expander("Training-domain warnings", expanded=True):
        for message in domain_messages:
            st.warning(message)
else:
    st.success("All uploaded input values are within the individual model training ranges.")


# ============================================================
# Step 3: Model settings/information
# ============================================================
st.subheader("Step 3: Confirm Model Application")

left_info, middle_info, right_info = st.columns(3)
left_info.metric("Uploaded rows", len(display_data))
middle_info.metric("Model inputs", len(MODEL_INPUT_COLUMNS))
right_info.metric("Rows outside training ranges", int(outside_domain.sum()))

st.write(
    "The final saved XGB-PSO model will be applied directly to every uploaded row. "
    "The uploaded observations remain unseen validation/prediction data and are not used for retraining."
)

with st.expander("Show exact ordered model inputs"):
    ordered_preview = model_inputs.copy()
    ordered_preview.columns = [f"{index + 1}. {column}" for index, column in enumerate(ordered_preview.columns)]
    st.dataframe(ordered_preview.head(25), use_container_width=True, hide_index=True)


# ============================================================
# Step 4: Run/rerun prediction
# ============================================================
st.subheader("Step 4: Run or Rerun XGB-PSO Model")
run_model = st.button("Run / Rerun XGB-PSO Prediction", type="primary")

if run_model:
    if model is None:
        st.error(
            f"The model is not loaded. Upload {MODEL_FILENAME} in the sidebar or place it beside app.py."
        )
    else:
        try:
            predictions = predict_with_saved_model(model, model_inputs)

            results = display_data.copy()
            results[PREDICTION_COL] = predictions
            results[DOMAIN_COL] = outside_domain.to_numpy()

            actual_mask = results[TARGET_COL].notna()
            statistics = None
            validation_actual = None
            validation_predicted = None

            results[RESIDUAL_COL] = np.nan
            results[ABS_ERROR_COL] = np.nan
            results[REL_ERROR_COL] = np.nan

            if actual_mask.any():
                validation_actual = results.loc[actual_mask, TARGET_COL].to_numpy(dtype=float)
                validation_predicted = results.loc[actual_mask, PREDICTION_COL].to_numpy(dtype=float)
                statistics = compute_statistics(validation_actual, validation_predicted)

                residuals = validation_actual - validation_predicted
                absolute_errors = np.abs(residuals)
                relative_errors = np.divide(
                    absolute_errors,
                    np.abs(validation_actual),
                    out=np.full_like(absolute_errors, np.nan, dtype=float),
                    where=np.abs(validation_actual) > 0,
                ) * 100

                results.loc[actual_mask, RESIDUAL_COL] = residuals
                results.loc[actual_mask, ABS_ERROR_COL] = absolute_errors
                results.loc[actual_mask, REL_ERROR_COL] = relative_errors

            st.session_state.attempt += 1
            st.session_state.prediction_run = True
            st.session_state.results = results
            st.session_state.statistics = statistics
            st.session_state.validation_actual = validation_actual
            st.session_state.validation_predicted = validation_predicted
            st.session_state.model_source = model_source
        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ============================================================
# Display saved results after button/download reruns
# ============================================================
if st.session_state.prediction_run:
    results = st.session_state.results
    statistics = st.session_state.statistics

    st.subheader("Model Results")
    st.write(f"Prediction attempt number: **{st.session_state.attempt}**")

    if statistics is not None:
        r2_value = statistics["R²"]
        r2_display = "N/A" if np.isnan(r2_value) else f"{r2_value:.4f}"

        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Validation R²", r2_display)
        metric_2.metric("Validation RMSE", f"{statistics['RMSE']:.4f}")
        metric_3.metric("Validation MAE", f"{statistics['MAE']:.4f}")

        metric_4, metric_5, metric_6 = st.columns(3)
        metric_4.metric("Validation MAPE (%)", f"{statistics['MAPE (%)']:.2f}")
        metric_5.metric("Validation AARD (%)", f"{statistics['AARD (%)']:.2f}")
        metric_6.metric("Experimental rows", int(statistics["Number of validation rows"]))
    else:
        st.info(
            "Predictions were generated, but validation statistics were not calculated because "
            "the experimental degradation/removal column was empty."
        )

    st.subheader("Prediction Results")
    st.dataframe(results, use_container_width=True, hide_index=True)

    plot_png = None
    if statistics is not None:
        st.subheader("Experimental-versus-Predicted Plot")
        validation_figure = make_validation_plot(
            st.session_state.validation_actual,
            st.session_state.validation_predicted,
            statistics,
        )
        st.pyplot(validation_figure, use_container_width=False)
        plot_png = figure_to_png(validation_figure)
        plt.close(validation_figure)

    st.subheader("Download Results")
    excel_results = create_results_workbook(
        results,
        statistics,
        st.session_state.model_source,
    )

    download_col_1, download_col_2 = st.columns(2)
    with download_col_1:
        st.download_button(
            label="Download Excel Results",
            data=excel_results,
            file_name="XGB_PSO_Pharmaceutical_Prediction_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if plot_png is not None:
        with download_col_2:
            st.download_button(
                label="Download Validation Plot",
                data=plot_png,
                file_name="XGB_PSO_Experimental_vs_Predicted.png",
                mime="image/png",
            )

    if statistics is not None and not np.isnan(statistics["R²"]):
        if statistics["R²"] >= 0.80:
            st.success("The saved XGB-PSO model shows acceptable agreement for the uploaded validation data.")
        else:
            st.warning(
                "The validation R² is below 0.80. Check the input units, light-source coding, "
                "experimental values, and training-domain warnings."
            )
