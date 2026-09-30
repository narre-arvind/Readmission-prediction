# 30-Day Hospital Readmission in Diabetes

## Problem
This project estimates whether a patient with diabetes will be readmitted within 30 days. The target is `readmitted == "<30"`; all other outcomes are negative. The intended users are hospital care-management and pharma analytics teams prioritizing follow-up resources.

## Data
The project uses the UCI Diabetes 130-US Hospitals dataset covering 1999-2008. Raw inputs are kept under `data/` and ignored by Git. The pipeline replaces `?` with missing values, drops columns with more than 40% missingness, excludes death or hospice discharge codes, keeps the first encounter per patient, and removes identifiers before modeling. Social factors are not available in this dataset.

## Approach
- Stratified 80/20 patient-level split with `random_state=42`.
- Diagnosis groups, age buckets, total prior visits, and medication-change count.
- Train-only imputation, scaling, one-hot encoding, and SMOTE where used.
- Logistic regression, balanced random forest, and XGBoost.
- Five-fold randomized XGBoost tuning by ROC-AUC.
- Cost threshold selection with miss-to-false-alarm cost ratio 5.
- SHAP global importance and two individual patient explanations.

## Results
The values below are copied from the generated CSV reports. Precision, recall, F1, accuracy, and balanced accuracy are calculated at the chosen cost threshold `0.6301169991493225`.

| Model | ROC-AUC | PR-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgboost_tuned | 0.6555308793634541 | 0.18249211347824104 | 0.8747678239748535 | 0.5723187259891815 | 0.25396825396825395 | 0.20365950676213207 | 0.2260485651214128 |
| xgboost | 0.6502900919906213 | 0.1788552350218356 | 0.856408058294042 | 0.5747816934374004 | 0.21797752808988763 | 0.2315035799522673 | 0.22453703703703703 |
| random_forest | 0.6284207230811123 | 0.13988832792031197 | 0.9102014573510502 | 0.5 | 0.0 | 0.0 | 0.0 |
| logistic_regression | 0.6213568196734188 | 0.14860980473239352 | 0.8344763537648235 | 0.5552048626584464 | 0.16875 | 0.21479713603818615 | 0.18900945047252363 |

Accuracy is misleading here because the positive class is uncommon, so a model can look accurate while missing many readmissions; balanced accuracy, PR-AUC, precision, and recall are more informative.

### Cost analysis
For miss cost 5, the chosen threshold has total cost `5757`, compared with `6285` for flag nobody and `12741` for flag everybody. Savings versus flag nobody are `528` and `-6456`, respectively. See `reports/cost_summary.csv`.

| Miss-to-false-alarm ratio | Best threshold | Total cost | Flag-nobody cost | Savings vs flag nobody |
|---:|---:|---:|---:|---:|
| 5 | 0.6301169991493225 | 5757 | 6285 | 528 |
| 10 | 0.49468448758125305 | 9829 | 12570 | 2741 |
| 20 | 0.33033502101898193 | 12356 | 25140 | 12784 |

These values are from `reports/cost_sensitivity.csv`.

### Top-risk groups
| Group | Patients | Readmissions | Precision | Recall | Lift |
|---|---:|---:|---:|---:|---:|
| top_10_percent | 1399 | 307 | 0.2194424588992137 | 0.24423229912490055 | 2.4437196019659457 |
| top_20_percent | 2799 | 474 | 0.16934619506966775 | 0.37708830548926014 | 1.8858456949763 |

These values are from `reports/top_risk_groups.csv`.

### Top SHAP drivers
The top measured drivers, copied from `reports/top_drivers.csv`, are discharge disposition (`0.05838786475360394`), prior inpatient visits (`0.017191477085153263`), time in hospital (`0.01600374645160304`), total prior visits (`0.013628679337600867`), circulatory first diagnosis (`0.011649767172005436`), other first diagnosis (`0.011226466479400793`), age 50-60 (`0.010741801715559433`), number of diagnoses (`0.00961609705206421`), number of medications (`0.009310362363027206`), and laboratory procedures (`0.009119597856369285`). These are predictive associations, not causal effects.

## Key Insights
The model separates risk modestly, which is expected for noisy administrative data. The top-risk groups have higher observed readmission concentration than the full test set, but they still contain many patients who are not readmitted. Threshold choice changes materially as the cost of a missed readmission increases.

## Limitations
A modest AUC is expected because the data is noisy and administrative. It is old US data from 1999-2008, lacks social factors and current clinical context, and does not establish causality. Before deployment, validate calibration, temporal and external performance, subgroup fairness, drift, and workflow impact.

## Business Recommendations
Use the score to prioritize human review, medication reconciliation, early follow-up scheduling, and discharge planning. Estimate costs with clinical and operations stakeholders, monitor performance after launch, and never use the score as an autonomous clinical decision.

## How to Run
```powershell
cd c:\project
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/train.py
python src/evaluate.py
python src/explain.py
streamlit run app/app.py
```

Run the notebook from VS Code or execute `notebooks/01_eda.ipynb`. Charts are saved under `reports/figures/`. The deployable model bundle is `models/model_bundle.joblib`; it is `461382` bytes and does not require the raw CSV at app runtime.

## Live App
`[Live Streamlit app link placeholder]`
