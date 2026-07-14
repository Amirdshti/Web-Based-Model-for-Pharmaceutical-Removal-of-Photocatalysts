# XGB–PSO Pharmaceutical Degradation Predictor

This Streamlit project loads the saved XGB–PSO model and predicts removal/degradation of pharmaceutical pollutants by photocatalysts.

## Required model file

Copy this file into the project folder, beside `app.py`:

`XGBPSOModel_success_seed605.pkl`

For local use, the app also checks your current Windows path:

`C:\Users\24550372\OneDrive - UTS\UTS PhD\Photocatalyst\Pharmaceutical Review\Data and model\pharmamlv3\XGBPSOModel_success_seed605.pkl`

The `.pkl` file is not included in this package because it was not uploaded to the chat.

## Exact model input order

1. BET specific surface area (m²/g)
2. Oxidant concentration (mM)
3. Molecular weight (g/mol)
4. HBDC
5. HBAC
6. TPSA (Å²)
7. Initial pollutant concentration, C0 (mg/L)
8. Solution pH
9. Light source code: UV = 1, visible light = 2, simulated solar light = 3
10. Photocatalyst dosage (mg/L)
11. Reaction time (min)

This order was reconstructed from the supplied `pharmamlv20.mat` matrix and the manuscript SHAP/PDP figures. Do not change it.

## Run locally

1. Put the model beside `app.py`, or leave it at the Windows path above.
2. Open Command Prompt in this project folder.
3. Install packages:

```bash
python -m pip install -r requirements.txt
```

4. Start the app:

```bash
python -m streamlit run app.py
```

You can also double-click `run_local.bat` after installing the requirements.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all files in this folder, including the trusted `.pkl` model.
3. In Streamlit Community Cloud, select the repository and set the main file to `app.py`.
4. Deploy and copy the public app URL into the manuscript.

## Model compatibility

Pickled models can depend on the versions of `xgboost`, `scikit-learn`, and Python used during training. If loading fails, install the same package versions used when `XGBPSOModel_success_seed605.pkl` was created, then pin those versions in `requirements.txt`.

## Included functions

- Single-condition prediction
- Exact light-source coding
- Training-range warnings
- Excel template download
- Batch Excel/CSV prediction
- Downloadable prediction results
- Saved train/test performance table
- Experimental-versus-predicted validation plot
