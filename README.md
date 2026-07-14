# XGB-PSO Pharmaceutical Removal Streamlit App

This version follows the Excel workflow described in Section 7.4:

1. Download the Excel template.
2. Fill the photocatalyst/pharmaceutical data.
3. Upload the completed file.
4. Run the saved XGB-PSO model.
5. Review predictions, optional validation metrics, and the R2 plot.
6. Download the Excel results and plot.

## Required repository files

- `app.py`
- `requirements.txt`
- `runtime.txt`
- `SampleData.xlsx` (the exact downloadable Excel template)
- `XGBPSOModel_success_seed605.pkl`

The model file must be stored beside `app.py` for automatic loading on Streamlit Community Cloud. Alternatively, it can be uploaded through the app sidebar during each session.

## Important input rules

- The descriptive `Oxidant` column is not a numerical model input.
- The model uses oxidant concentration in mM.
- Light source: UV = 1, visible light = 2, simulated solar light = 3. Text labels are accepted.
- Catalyst dosage must be entered in mg/L.
- The experimental degradation/removal column is optional. When supplied, the app calculates R2, RMSE, MAE, MAPE, AARD, residuals, and the experimental-versus-predicted plot.
- Uploaded rows are treated as unseen data; the final model is not retrained.


The Step 1 download button serves `SampleData.xlsx` directly, without recreating or modifying the workbook.
