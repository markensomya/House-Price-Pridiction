# House Price Prediction

A machine learning project that predicts house prices based on property features using Linear Regression and Random Forest models.

## Dataset

**Source:** [Housing Prices Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)  
**File:** `Housing.csv` — 545 rows, 13 columns

**Features:** area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus  
**Target:** price

## Project Structure

```
HousePricePridiction_Somya_Marken/
├── analysis.ipynb       # Main Jupyter Notebook (all 5 tasks)
├── Housing.csv          # Dataset
├── run_analysis.py      # Standalone Python script
├── summary.txt          # Written summary of findings
└── charts/
    ├── chart1_price_distribution.png
    ├── chart2_correlation_heatmap.png
    └── chart3_actual_vs_predicted.png
```

## Tasks Completed

- **Task 1** — Data loading and exploration
- **Task 2** — Data cleaning, encoding, duplicate removal
- **Task 3** — Linear Regression and Random Forest models with MAE, RMSE, R²
- **Task 4** — 3 visualizations (price distribution, correlation heatmap, actual vs predicted)
- **Task 5** — Insights and business recommendations

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 970,043 | 1,324,507 | 0.6529 |
| Random Forest | 1,021,546 | 1,400,566 | 0.6119 |

## Key Findings

- **Area** is the strongest predictor of house price (46.8% feature importance)
- **Bathrooms** and **air conditioning** are the next most influential features
- Bedroom count alone is a weak predictor — total space and amenities matter more

## Tools Used

- Python 3.x, Jupyter Notebook
- Pandas, Scikit-learn, Matplotlib, Seaborn

## How to Run

1. Clone the repo and place `Housing.csv` in the root folder
2. Open `analysis.ipynb` in Jupyter Notebook and run all cells
3. Or run `python run_analysis.py` directly
