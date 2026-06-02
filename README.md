# Intercity Service Demand Forecasting using Ensemble Machine Learning

## Overview

This project predicts intercity service demand by estimating the number of service units required between origin and destination hubs. Accurate demand forecasting helps transportation and logistics providers optimize resources, improve operational efficiency, reduce shortages, and support better decision-making.

The project uses feature engineering and an ensemble of machine learning models including LightGBM, XGBoost, and CatBoost to forecast demand patterns based on historical service data.

---

## Problem Statement

Transportation systems often experience fluctuations in demand due to factors such as weekdays, weekends, seasonal trends, route popularity, and travel behavior. Traditional estimation methods may fail to capture these complex patterns.

The goal of this project is to build a machine learning model capable of accurately predicting future intercity service demand using historical data and route information.

---

## Dataset Features

The dataset contains historical service information with the following key fields:

- `service_date`
- `origin_hub_id`
- `destination_hub_id`
- `final_service_units` *(target variable)*

---

## Feature Engineering

Several additional features were generated to improve prediction performance:

### Date Features
- Day of week (`dow`)
- Month
- Day
- Week of year
- Weekend indicator (`is_weekend`)

### Route-Based Features
- Origin average demand (`origin_avg`)
- Destination average demand (`dest_avg`)
- Route average demand (`route_avg`)
- Route-day average demand (`route_dow_avg`)

---

## Machine Learning Models Used

The following gradient boosting models were implemented:

- LightGBM
- XGBoost
- CatBoost

A weighted ensemble approach combines predictions from all models:

```python
Final Prediction =
0.4 × LightGBM +
0.3 × XGBoost +
0.3 × CatBoost
```

---

## Cross Validation Strategy

The models are trained using:

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

Cross-validation helps improve generalization and reduces overfitting.

---

## Evaluation Metrics

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- R² Score
- Prediction accuracy within a specified tolerance range

---

## Experimental Results

| Model | OOF MAE | OOF R² |
|---------|---------|---------|
| LightGBM | 305.50 | 0.8445 |
| XGBoost | 307.86 | 0.8417 |
| CatBoost | 339.98 | 0.7799 |
| Ensemble | 307.16 | 0.8366 |

### Best Performing Model

**LightGBM**

- MAE: **305.50**
- R²: **0.8445**

---

## Project Structure

```bash
Intercity-Service-Demand-Forecasting/
│
├── train.csv
├── test.csv
├── sample_submission_file.csv
│
├── demand_forecasting.py
│
├── my_submission_day2_lgb_xgb_cat.csv
│
├── README.md
│
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Intercity-Service-Demand-Forecasting.git
```

Move to the project directory:

```bash
cd Intercity-Service-Demand-Forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Required Libraries

```txt
pandas
numpy
scikit-learn
lightgbm
xgboost
catboost
```

---

## Run the Project

Execute:

```bash
python demand_forecasting.py
```

Predictions will automatically be saved as:

```bash
my_submission_day2_lgb_xgb_cat.csv
```

---

## Future Improvements

Possible enhancements:

- Replace KFold with TimeSeriesSplit
- Add lag features
- Add rolling averages
- Include holiday information
- Hyperparameter optimization
- Improve ensemble weighting

---

## Applications

- Bus demand forecasting
- Railway capacity planning
- Airline route optimization
- Logistics demand prediction
- Smart transportation systems

---

## Conclusion

This project demonstrates how feature engineering and ensemble machine learning methods can effectively forecast intercity service demand. The developed model can assist transportation providers in making data-driven decisions and improve operational efficiency.
