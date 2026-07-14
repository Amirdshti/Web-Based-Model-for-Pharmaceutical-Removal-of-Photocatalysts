# -*- coding: utf-8 -*-
"""
Streamlit application for developing an XGBoost-PSO model from an uploaded
Excel dataset containing 11 inputs and 1 output.

Workflow
1. Download the Excel template.
2. Fill and upload the completed dataset.
3. Select the train/test ratio and model settings.
4. Develop or redevelop an XGBoost model, with optional PSO optimization.
5. Display R², RMSE, MAE and MAPE for train, test and total data.
6. Display an experimental-versus-predicted plot and optimized hyperparameters.
7. Download predictions, statistics, hyperparameters, plot and trained model.
"""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="XGBoost-PSO Pharmaceutical Photodegradation Model",
    layout="wide",
)

st.title("Web-Based XGBoost–PSO Model Development App")
st.write(
    "Download the Excel template, enter the complete modelling dataset, upload it, "
    "select the train/test ratio, and develop or redevelop an XGBoost–PSO model for "
    "pharmaceutical degradation or removal by photocatalysts."
)


# ============================================================
# Exact data structure: 11 inputs + 1 output
# ============================================================
INPUT_COLUMNS = [
    "BET specific surface area (m2 g-1)",
    "Oxidant concentration (mM)",
    "Molecular Weight (g/mol)",
    "HBDC",
    "HBAC",
    "Topological Polar Surface Area (Å²)",
    "Initial concentration of pollutant (mg/L)",
    "pH",
    "Light source",
    "Catalyst dosage (mg/L)",
    "t (min)",
]

OUTPUT_COLUMN = "Degradation or Removal (%)"
TEMPLATE_COLUMNS = INPUT_COLUMNS + [OUTPUT_COLUMN]

LIGHT_SOURCE_MAP = {
    "uv": 1.0,
    "ultraviolet": 1.0,
    "ultraviolet light": 1.0,
    "visible": 2.0,
    "visible light": 2.0,
    "vis": 2.0,
    "simulated solar": 3.0,
    "simulated solar light": 3.0,
    "solar": 3.0,
    "solar light": 3.0,
}


# ============================================================
# Session state
# ============================================================
DEFAULT_STATE: dict[str, Any] = {
    "attempt": 0,
    "model_trained": False,
    "dataset_signature": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_model_results() -> None:
    """Remove results when the uploaded dataset changes."""
    keys_to_remove = [
        "model",
        "results_df",
        "metrics_df",
        "params_df",
        "y_train",
        "y_test",
        "y_train_pred",
        "y_test_pred",
        "plot_png",
        "excel_results",
        "model_file",
        "best_cv_r2",
        "model_name",
    ]
    for key in keys_to_remove:
        st.session_state.pop(key, None)
    st.session_state.model_trained = False
    st.session_state.attempt = 0


# ============================================================
# Utility functions
# ============================================================
def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in %, excluding observations with a zero experimental value."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    nonzero = np.abs(y_true) > np.finfo(float).eps
    if not np.any(nonzero):
        return float("nan")
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
) -> dict[str, Any]:
    return {
        "Dataset": dataset_name,
        "Number of data points": int(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": compute_mape(y_true, y_pred),
    }


def encode_light_source(series: pd.Series) -> pd.Series:
    """Accept numeric codes or common light-source text labels."""
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower().map(LIGHT_SOURCE_MAP)
    encoded = numeric.fillna(text)
    return encoded


def create_fallback_template() -> bytes:
    """Generate a template only when SampleData.xlsx is absent."""
    output = io.BytesIO()
    template_df = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    instructions_df = pd.DataFrame(
        {
            "Item": ["Light-source coding", "Output", "Required structure"],
            "Instruction": [
                "Use 1 = UV, 2 = visible light, or 3 = simulated solar light.",
                "Enter the experimental degradation or removal percentage.",
                "Keep the 11 input columns and the final output column unchanged.",
            ],
        }
    )
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template_df.to_excel(writer, sheet_name="Data Template", index=False)
        instructions_df.to_excel(writer, sheet_name="Instructions", index=False)
    output.seek(0)
    return output.getvalue()


def get_template_bytes() -> bytes:
    template_path = Path(__file__).with_name("SampleData.xlsx")
    if template_path.exists():
        return template_path.read_bytes()
    return create_fallback_template()


def params_from_position(position: np.ndarray) -> dict[str, Any]:
    """Convert a seven-dimensional PSO particle into valid XGBoost settings."""
    return {
        "n_estimators": int(np.clip(np.rint(position[0]), 50, 500)),
        "learning_rate": float(np.clip(position[1], 0.01, 0.30)),
        "subsample": float(np.clip(position[2], 0.50, 1.00)),
        "max_depth": int(np.clip(np.rint(position[3]), 2, 15)),
        "min_child_weight": float(np.clip(position[4], 1.0, 20.0)),
        "gamma": float(np.clip(position[5], 0.0, 12.0)),
        "colsample_bytree": float(np.clip(position[6], 0.30, 1.00)),
    }


def make_xgb_model(params: dict[str, Any], random_seed: int, n_jobs: int = 1) -> XGBRegressor:
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        random_state=random_seed,
        n_jobs=n_jobs,
        tree_method="hist",
        verbosity=0,
    )


def optimize_xgb_with_pso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_particles: int,
    iterations: int,
    c1: float,
    c2: float,
    inertia: float,
    cv_folds: int,
    random_seed: int,
) -> tuple[dict[str, Any], float]:
    """A compact global-best PSO implementation that maximizes mean CV R²."""
    lower = np.array([50, 0.01, 0.50, 2, 1.0, 0.0, 0.30], dtype=float)
    upper = np.array([500, 0.30, 1.00, 15, 20.0, 12.0, 1.00], dtype=float)

    rng = np.random.default_rng(random_seed)
    positions = rng.uniform(lower, upper, size=(n_particles, len(lower)))
    velocities = rng.uniform(
        -0.10 * (upper - lower),
        0.10 * (upper - lower),
        size=positions.shape,
    )

    personal_best_positions = positions.copy()
    personal_best_costs = np.full(n_particles, np.inf)
    global_best_position = positions[0].copy()
    global_best_cost = np.inf

    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    progress_bar = st.progress(0)
    progress_message = st.empty()
    best_message = st.empty()

    def evaluate_particle(position: np.ndarray) -> float:
        params = params_from_position(position)
        model = make_xgb_model(params, random_seed=random_seed, n_jobs=1)
        try:
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=splitter,
                scoring="r2",
                n_jobs=1,
                error_score=np.nan,
            )
            mean_score = float(np.nanmean(scores))
            if not np.isfinite(mean_score):
                return 1.0e9
            return -mean_score
        except Exception:
            return 1.0e9

    for iteration in range(iterations):
        costs = np.array([evaluate_particle(position) for position in positions])

        improved = costs < personal_best_costs
        personal_best_costs[improved] = costs[improved]
        personal_best_positions[improved] = positions[improved]

        iteration_best_index = int(np.argmin(costs))
        if costs[iteration_best_index] < global_best_cost:
            global_best_cost = float(costs[iteration_best_index])
            global_best_position = positions[iteration_best_index].copy()

        current_best_r2 = -global_best_cost
        completed = int(100 * (iteration + 1) / iterations)
        progress_bar.progress(completed)
        progress_message.write(
            f"PSO progress: **{iteration + 1}/{iterations} iterations** ({completed}%)"
        )
        best_message.write(f"Best cross-validation R²: **{current_best_r2:.4f}**")

        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = (
            inertia * velocities
            + c1 * r1 * (personal_best_positions - positions)
            + c2 * r2 * (global_best_position - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)

    progress_message.success("PSO hyperparameter optimization completed.")
    return params_from_position(global_best_position), float(-global_best_cost)


def create_r2_plot(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    r2_train: float,
    r2_test: float,
    r2_all: float,
    model_name: str,
) -> tuple[plt.Figure, bytes]:
    y_all = np.concatenate([y_train, y_test])
    y_all_pred = np.concatenate([y_train_pred, y_test_pred])

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.scatter(
        y_train,
        y_train_pred,
        label="Train data",
        s=34,
        marker="o",
        facecolors="none",
        edgecolors="#E6A56A",
        linewidths=1.0,
        alpha=0.85,
    )
    ax.scatter(
        y_test,
        y_test_pred,
        label="Test data",
        s=36,
        marker="o",
        color="#423C9B",
        alpha=0.85,
    )

    minimum = float(min(np.min(y_all), np.min(y_all_pred)))
    maximum = float(max(np.max(y_all), np.max(y_all_pred)))
    span = maximum - minimum
    margin = 0.05 * span if span > 0 else 1.0
    low = minimum - margin
    high = maximum + margin

    ax.plot([low, high], [low, high], "--", color="#315BB5", linewidth=1.4, label="45° line")

    if len(np.unique(y_all)) >= 2:
        slope, intercept = np.polyfit(y_all, y_all_pred, 1)
        x_line = np.linspace(low, high, 200)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5, label="Best-fit line")

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Experimental Degradation or Removal (%)", fontsize=11)
    ax.set_ylabel("Predicted Degradation or Removal (%)", fontsize=11)
    ax.set_title(f"{model_name} Model", fontsize=14)
    ax.text(
        0.04,
        0.96,
        f"Train R² = {r2_train:.4f}\nTest R² = {r2_test:.4f}\nTotal R² = {r2_all:.4f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend(loc="lower right")
    ax.grid(False)
    fig.tight_layout()

    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
    png_buffer.seek(0)
    return fig, png_buffer.getvalue()


def create_excel_results(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    params_df: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Predictions", index=False)
        metrics_df.to_excel(writer, sheet_name="Statistical Results", index=False)
        params_df.to_excel(writer, sheet_name="Hyperparameters", index=False)
    output.seek(0)
    return output.getvalue()


def create_model_download(model: XGBRegressor) -> bytes:
    output = io.BytesIO()
    joblib.dump(model, output)
    output.seek(0)
    return output.getvalue()


# ============================================================
# Step 1: Template download
# ============================================================
st.subheader("Step 1: Download the Excel Data Template")

st.download_button(
    label="Download SampleData.xlsx",
    data=get_template_bytes(),
    file_name="SampleData.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.info(
    "The workbook contains exactly 11 model inputs and one output. Use the final column, "
    f"'{OUTPUT_COLUMN}', for the measured experimental result. Light-source coding: "
    "1 = UV, 2 = visible light, and 3 = simulated solar light. Text labels are also accepted."
)


# ============================================================
# Step 2: Upload and validate dataset
# ============================================================
st.subheader("Step 2: Upload the Completed Excel Dataset")

uploaded_file = st.file_uploader(
    "Upload the completed SampleData.xlsx file",
    type=["xlsx"],
)

if uploaded_file is None:
    st.warning("Download the template, enter the complete dataset, and upload the completed Excel file.")
    st.stop()

uploaded_bytes = uploaded_file.getvalue()
current_signature = hashlib.sha256(uploaded_bytes).hexdigest()
if st.session_state.dataset_signature != current_signature:
    clear_model_results()
    st.session_state.dataset_signature = current_signature

try:
    raw_df = pd.read_excel(io.BytesIO(uploaded_bytes))
except Exception as exc:
    st.error(f"The uploaded Excel file could not be read: {exc}")
    st.stop()

missing_columns = [column for column in TEMPLATE_COLUMNS if column not in raw_df.columns]
if missing_columns:
    st.error("The uploaded workbook does not match the required 11-input/1-output template.")
    st.write("Missing columns:")
    st.code("\n".join(missing_columns))
    st.write("Required columns in order:")
    st.dataframe(pd.DataFrame({"Required column": TEMPLATE_COLUMNS}), use_container_width=True)
    st.stop()

working_df = raw_df[TEMPLATE_COLUMNS].copy()
working_df["Light source"] = encode_light_source(working_df["Light source"])
for column in TEMPLATE_COLUMNS:
    if column != "Light source":
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

invalid_light = ~working_df["Light source"].isin([1.0, 2.0, 3.0])
working_df.loc[invalid_light, "Light source"] = np.nan

rows_before = len(working_df)
working_df = working_df.dropna(subset=TEMPLATE_COLUMNS).reset_index(drop=True)
removed_rows = rows_before - len(working_df)

if len(working_df) < 10:
    st.error(
        "At least 10 complete numerical rows are required to develop the model. "
        f"Only {len(working_df)} valid rows were detected."
    )
    st.stop()

st.success(f"Excel dataset uploaded successfully: {len(working_df)} valid rows.")
if removed_rows:
    st.warning(f"{removed_rows} incomplete or invalid rows were excluded before modelling.")

with st.expander("Preview the validated modelling dataset", expanded=True):
    st.dataframe(working_df.head(20), use_container_width=True)


# ============================================================
# Step 3: Model settings
# ============================================================
st.subheader("Step 3: Select the XGBoost–PSO Model Settings")

train_percent = st.slider(
    "Training data percentage (%)",
    min_value=50,
    max_value=90,
    value=85,
    step=5,
)
test_percent = 100 - train_percent
st.write(f"Selected data division: **{train_percent}% training / {test_percent}% testing**")

estimated_test_rows = int(np.ceil(len(working_df) * test_percent / 100.0))
if estimated_test_rows < 2:
    st.error("The selected split produces fewer than two test observations. Reduce the training percentage.")
    st.stop()

st.sidebar.header("Model Development Settings")
use_pso = st.sidebar.checkbox("Use PSO hyperparameter optimization", value=True)
if use_pso:
    st.sidebar.subheader("PSO Settings")
    n_particles = st.sidebar.slider("Number of particles", 5, 30, 5, 5)
    pso_iterations = st.sidebar.slider("PSO iterations", 5, 100, 30, 5)
    c1 = st.sidebar.number_input("Cognitive coefficient (c1)", 0.1, 5.0, 1.5, 0.1)
    c2 = st.sidebar.number_input("Social coefficient (c2)", 0.1, 5.0, 1.5, 0.1)
    inertia = st.sidebar.number_input("Inertia weight (w)", 0.1, 1.5, 0.7, 0.1)
    cv_folds_requested = st.sidebar.slider("Cross-validation folds", 2, 10, 5, 1)
    st.sidebar.caption(
        "The manuscript settings were 5 particles, 100 iterations, c1 = 1.5, "
        "c2 = 1.5 and w = 0.7. Fewer iterations run faster on Streamlit Cloud."
    )
else:
    st.sidebar.subheader("Manual XGBoost Hyperparameters")
    n_estimators = st.sidebar.slider("n_estimators", 50, 1000, 168, 10)
    learning_rate = st.sidebar.slider("learning_rate", 0.001, 0.300, 0.193, 0.001, format="%.3f")
    subsample = st.sidebar.slider("subsample", 0.50, 1.00, 0.718, 0.01)
    max_depth = st.sidebar.slider("max_depth", 1, 20, 9, 1)
    min_child_weight = st.sidebar.slider("min_child_weight", 0.1, 20.0, 4.7, 0.1)
    gamma = st.sidebar.slider("gamma", 0.0, 15.0, 8.1, 0.1)
    colsample_bytree = st.sidebar.slider("colsample_bytree", 0.30, 1.00, 0.945, 0.005)


# ============================================================
# Step 4: Develop model
# ============================================================
st.subheader("Step 4: Develop or Redevelop the XGBoost–PSO Model")

run_model = st.button("Develop / Redevelop XGBoost–PSO Model", type="primary")

if run_model:
    st.session_state.attempt += 1
    random_seed = int(time.time()) + st.session_state.attempt

    X = working_df[INPUT_COLUMNS].to_numpy(dtype=float)
    y = working_df[OUTPUT_COLUMN].to_numpy(dtype=float)
    row_indices = np.arange(len(working_df))

    try:
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X,
            y,
            row_indices,
            train_size=train_percent / 100.0,
            random_state=random_seed,
            shuffle=True,
        )
    except Exception as exc:
        st.error(f"The train/test split could not be created: {exc}")
        st.stop()

    if len(y_test) < 2:
        st.error("The test subset contains fewer than two observations; R² cannot be calculated.")
        st.stop()

    best_cv_r2 = float("nan")

    with st.spinner("Developing the model. This may take several minutes when PSO is enabled..."):
        if use_pso:
            # R² requires at least two observations in each validation fold.
            effective_cv = min(cv_folds_requested, max(2, len(y_train) // 2))
            if effective_cv < 2:
                st.error("Insufficient training observations for cross-validation.")
                st.stop()
            best_params, best_cv_r2 = optimize_xgb_with_pso(
                X_train,
                y_train,
                n_particles=n_particles,
                iterations=pso_iterations,
                c1=float(c1),
                c2=float(c2),
                inertia=float(inertia),
                cv_folds=effective_cv,
                random_seed=random_seed,
            )
            model_name = "XGBoost–PSO"
        else:
            best_params = {
                "n_estimators": int(n_estimators),
                "learning_rate": float(learning_rate),
                "subsample": float(subsample),
                "max_depth": int(max_depth),
                "min_child_weight": float(min_child_weight),
                "gamma": float(gamma),
                "colsample_bytree": float(colsample_bytree),
            }
            model_name = "XGBoost"

        model = make_xgb_model(best_params, random_seed=random_seed, n_jobs=-1)
        model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_all_pred = model.predict(X)

    train_metrics = calculate_metrics(y_train, y_train_pred, "Train")
    test_metrics = calculate_metrics(y_test, y_test_pred, "Test")
    total_metrics = calculate_metrics(y, y_all_pred, "Total")
    metrics_df = pd.DataFrame([train_metrics, test_metrics, total_metrics])

    results_df = working_df.copy()
    results_df["Predicted Degradation or Removal (%)"] = y_all_pred
    results_df["Residual (Experimental - Predicted)"] = y - y_all_pred
    results_df["Absolute Error"] = np.abs(y - y_all_pred)
    results_df["Data Set"] = ""
    results_df.loc[train_indices, "Data Set"] = "Train"
    results_df.loc[test_indices, "Data Set"] = "Test"

    pso_settings = {
        "Use PSO": use_pso,
        "Particles": n_particles if use_pso else None,
        "PSO iterations": pso_iterations if use_pso else None,
        "c1": float(c1) if use_pso else None,
        "c2": float(c2) if use_pso else None,
        "w": float(inertia) if use_pso else None,
        "CV folds": effective_cv if use_pso else None,
        "Best CV R2": best_cv_r2 if use_pso else None,
    }
    params_df = pd.DataFrame(
        [
            {
                **best_params,
                **pso_settings,
                "Training percentage": train_percent,
                "Testing percentage": test_percent,
                "Total valid rows": len(working_df),
            }
        ]
    )

    fig, plot_png = create_r2_plot(
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        train_metrics["R2"],
        test_metrics["R2"],
        total_metrics["R2"],
        model_name,
    )
    plt.close(fig)

    excel_results = create_excel_results(
        results_df,
        metrics_df,
        params_df,
    )
    model_file = create_model_download(model)

    st.session_state.model_trained = True
    st.session_state.model = model
    st.session_state.results_df = results_df
    st.session_state.metrics_df = metrics_df
    st.session_state.params_df = params_df
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.y_train_pred = y_train_pred
    st.session_state.y_test_pred = y_test_pred
    st.session_state.plot_png = plot_png
    st.session_state.excel_results = excel_results
    st.session_state.model_file = model_file
    st.session_state.best_cv_r2 = best_cv_r2
    st.session_state.model_name = model_name


# ============================================================
# Results remain visible after download-button reruns
# ============================================================
if st.session_state.model_trained:
    metrics_df = st.session_state.metrics_df
    train_row = metrics_df.loc[metrics_df["Dataset"] == "Train"].iloc[0]
    test_row = metrics_df.loc[metrics_df["Dataset"] == "Test"].iloc[0]
    total_row = metrics_df.loc[metrics_df["Dataset"] == "Total"].iloc[0]

    st.divider()
    st.header("Developed Model Results")
    st.write(
        f"Model: **{st.session_state.model_name}** | "
        f"Attempt: **{st.session_state.attempt}**"
    )

    if np.isfinite(st.session_state.best_cv_r2):
        st.metric("Best PSO cross-validation R²", f"{st.session_state.best_cv_r2:.4f}")

    st.subheader("Statistical Performance")
    r2_cols = st.columns(3)
    r2_cols[0].metric("Train R²", f"{train_row['R2']:.4f}")
    r2_cols[1].metric("Test R²", f"{test_row['R2']:.4f}")
    r2_cols[2].metric("Total R²", f"{total_row['R2']:.4f}")

    rmse_cols = st.columns(3)
    rmse_cols[0].metric("Train RMSE", f"{train_row['RMSE']:.4f}")
    rmse_cols[1].metric("Test RMSE", f"{test_row['RMSE']:.4f}")
    rmse_cols[2].metric("Total RMSE", f"{total_row['RMSE']:.4f}")

    mae_cols = st.columns(3)
    mae_cols[0].metric("Train MAE", f"{train_row['MAE']:.4f}")
    mae_cols[1].metric("Test MAE", f"{test_row['MAE']:.4f}")
    mae_cols[2].metric("Total MAE", f"{total_row['MAE']:.4f}")

    mape_cols = st.columns(3)
    mape_cols[0].metric("Train MAPE (%)", f"{train_row['MAPE (%)']:.2f}")
    mape_cols[1].metric("Test MAPE (%)", f"{test_row['MAPE (%)']:.2f}")
    mape_cols[2].metric("Total MAPE (%)", f"{total_row['MAPE (%)']:.2f}")

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.subheader("Experimental versus Predicted Plot")
    st.image(st.session_state.plot_png, use_container_width=False)

    st.subheader("Optimized Model Hyperparameters")
    st.dataframe(st.session_state.params_df, use_container_width=True, hide_index=True)

    with st.expander("Prediction results", expanded=False):
        st.dataframe(st.session_state.results_df, use_container_width=True)

    st.subheader("Download the Developed Model and Results")
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download Excel Results",
        data=st.session_state.excel_results,
        file_name="XGBoost_PSO_Modeling_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    download_cols[1].download_button(
        "Download Trained Model",
        data=st.session_state.model_file,
        file_name="Trained_XGBoost_PSO_Model.pkl",
        mime="application/octet-stream",
    )
    download_cols[2].download_button(
        "Download R² Plot",
        data=st.session_state.plot_png,
        file_name="XGBoost_PSO_R2_Plot.png",
        mime="image/png",
    )

    if not np.isfinite(float(test_row["R2"])):
        st.warning("Test R² could not be calculated for the current test subset.")
    elif float(test_row["R2"]) < 0.80:
        st.warning(
            "The test R² is below 0.80. Review the dataset and PSO settings, or click "
            "Develop / Redevelop XGBoost–PSO Model to evaluate another train/test split."
        )
    else:
        st.success("The developed model shows acceptable test-set performance for the current split.")
