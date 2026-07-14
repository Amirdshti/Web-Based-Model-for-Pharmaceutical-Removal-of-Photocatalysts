# XGBoost–PSO Pharmaceutical Photodegradation Model Development App

This Streamlit application develops a new XGBoost model from a user-uploaded Excel dataset. It does **not** load the previously saved `.pkl` model.

## Excel structure

The workbook contains exactly 11 inputs and 1 output:

1. BET specific surface area (m2 g-1)
2. Oxidant concentration (mM)
3. Molecular Weight (g/mol)
4. HBDC
5. HBAC
6. Topological Polar Surface Area (Å²)
7. Initial concentration of pollutant (mg/L)
8. pH
9. Light source
10. Catalyst dosage (mg/L)
11. t (min)
12. Degradation or Removal (%) — output

Light source coding: `1 = UV`, `2 = visible light`, `3 = simulated solar light`.

## Main workflow

1. Download `SampleData.xlsx` from the app.
2. Enter the full modelling dataset and upload it.
3. Choose the train/test split.
4. Enable PSO and select particles, iterations and CV folds.
5. Develop/redevelop the XGBoost–PSO model.
6. Review train/test/total R², RMSE, MAE and MAPE, the R² plot, feature importance and optimized hyperparameters.
7. Download the Excel results, trained `.pkl` model and R² plot.

## Streamlit deployment files

Upload these files to the same GitHub repository:

- `app.py`
- `SampleData.xlsx`
- `requirements.txt`
- `runtime.txt`

No pre-trained model file is required because the model is trained from the uploaded Excel dataset.

## Local run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Computational note

PSO runtime is approximately proportional to:

`particles × iterations × cross-validation folds`

The manuscript used 5 particles and 100 iterations. The app defaults to 30 iterations so it is more practical on Streamlit Community Cloud; users can select 100 iterations when needed.
