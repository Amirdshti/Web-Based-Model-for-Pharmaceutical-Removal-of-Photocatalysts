# XGB-PSO Pharmaceutical Removal Streamlit App

This app applies the saved `XGBPSOModel_success_seed605.pkl` model to an uploaded Excel workbook.

## Required Excel structure

The first worksheet must contain exactly 11 model inputs followed by 1 experimental output:

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
12. Degradation or Removal (%) — experimental output

Light-source coding: UV = 1, visible light = 2, simulated solar light = 3. Text labels are also accepted.

## Repository files

- `app.py`
- `SampleData.xlsx`
- `requirements.txt`
- `runtime.txt`
- `XGBPSOModel_success_seed605.pkl`

The app downloads `SampleData.xlsx`, accepts the completed workbook, generates XGB-PSO predictions, calculates validation statistics, plots experimental versus predicted values, and exports the results to Excel.
