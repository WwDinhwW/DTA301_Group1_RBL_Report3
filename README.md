# DTA301 - Group 1 - Report 3

## Project Overview

This project is part of the DTA301 Data Analysis course. The research topic is:

**Battery Consumption Prediction for Surveillance Drones in Smart City**

The objective of this report is to develop and evaluate a baseline Linear Regression model capable of predicting drone energy consumption based on operational flight characteristics.

The dataset used in this project is derived from the CMU DJI Matrice 100 Flight Dataset and was preprocessed into a flight-level dataset, where each row represents a single flight.

---

## Dataset

### Input Dataset

Location:

```text
data/cleaned_flight_summary.xlsx
```

### Dataset Summary

* Total Flights: 207
* Total Features: 9
* Target Variable: `total_energy_wh`

### Selected Features

The following features were selected based on the findings from Report 2 (EDA):

* `flight_duration_s`
* `altitude`
* `payload`
* `avg_wind_speed`

### Target Variable

* `total_energy_wh`

This variable represents the total energy consumption of a drone flight measured in Watt-hours (Wh).

---

## Model Development Workflow

The baseline model follows the workflow below:

```text
Load Dataset
    ↓
Feature Selection
    ↓
Train/Test Split (80/20)
    ↓
StandardScaler
    ↓
Linear Regression
    ↓
Performance Evaluation
    ↓
5-Fold Cross Validation
```

### Train/Test Split

* Training Set: 80%
* Testing Set: 20%
* Random State: 42

### Feature Scaling

StandardScaler was applied to the training data and then used to transform the testing data to prevent data leakage.

### Machine Learning Model

* Linear Regression (Scikit-Learn)

---

## Evaluation Metrics

The model is evaluated using:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* MAPE (Mean Absolute Percentage Error)
* R² Score (Coefficient of Determination)

Additionally, 5-Fold Cross Validation is performed to assess model stability and generalization performance.

---

## Installation

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

Execute the baseline model:

```bash
python notebooks/03_baseline_model.py
```

The script will:

1. Load the cleaned dataset
2. Perform train/test splitting
3. Apply feature scaling
4. Train the Linear Regression model
5. Generate predictions
6. Calculate evaluation metrics
7. Perform 5-Fold Cross Validation
8. Display model coefficients and performance results

---

## Results Summary

### Test Set Performance

| Metric   |    Value |
| -------- | -------: |
| RMSE     |   2.7452 |
| MAE      |   1.9420 |
| MAPE (%) | 150.0568 |
| R²       |   0.8565 |

### 5-Fold Cross Validation

| Metric    |  Value |
| --------- | -----: |
| Mean RMSE | 2.5842 |
| RMSE Std  | 1.0458 |
| Mean R²   | 0.7953 |
| R² Std    | 0.0569 |

### Feature Importance (Linear Regression Coefficients)

| Feature           | Coefficient |
| ----------------- | ----------: |
| flight_duration_s |      4.0697 |
| payload           |      1.7794 |
| altitude          |      1.1298 |
| avg_wind_speed    |      1.0044 |

---

## Conclusion

The baseline Linear Regression model demonstrated strong predictive performance for estimating drone energy consumption. On the test dataset, the model achieved an R² score of 0.8565, indicating that approximately 85.65% of the variation in flight energy consumption can be explained by flight duration, altitude, payload, and average wind speed.

The model achieved a test RMSE of 2.7452 Wh and a test MAE of 1.9420 Wh, showing that prediction errors remained relatively small compared to the overall energy consumption range observed in the dataset.

The 5-Fold Cross Validation results produced a mean R² score of 0.7953 with low variability across folds, suggesting that the model generalizes reasonably well and is not strongly dependent on a single train-test split.

Analysis of the model coefficients indicates that flight duration is the most influential factor affecting energy consumption, followed by payload, altitude, and average wind speed. These findings are consistent with domain knowledge and support the research objective of identifying operational factors that influence drone battery usage.

Although MAPE values appear relatively high, this metric is heavily affected by a small number of valid flights with extremely low energy consumption values. Because MAPE expresses error as a percentage of the actual value, observations close to zero can disproportionately inflate the metric. Therefore, RMSE, MAE, and R² are considered more representative indicators of model performance for this dataset.

Overall, the baseline Linear Regression model provides a solid foundation for subsequent comparison with more advanced machine learning models in later stages of the project.
