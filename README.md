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

- Total Flights: **207**
- Total Features: **9**
- Target Variable: `total_energy_wh`

### Selected Features

The following features were selected based on the findings from Report 2 (EDA):

- `flight_duration_s`
- `altitude`
- `payload`
- `avg_wind_speed`

### Target Variable

- `total_energy_wh`

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

- Training Set: 80%
- Testing Set: 20%
- Random State: 42

### Feature Scaling

StandardScaler was fitted only on the training dataset and then applied to both the training and testing datasets to prevent data leakage.

### Machine Learning Model

- Linear Regression (Scikit-Learn)

---

## Evaluation Metrics

The model is evaluated using the following metrics:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score (Coefficient of Determination)

Additionally, **5-Fold Cross Validation** is performed to evaluate model stability and generalization performance.

---

## Installation

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment (Windows)

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

### Baseline Model

Execute the baseline Linear Regression model:

```bash
python notebooks/03_baseline_model.py
```

This script will:

1. Load the cleaned dataset
2. Perform feature selection
3. Split the dataset into training and testing sets
4. Apply StandardScaler
5. Train the Linear Regression model
6. Generate predictions
7. Calculate RMSE, MAE, MAPE and R²
8. Perform 5-Fold Cross Validation
9. Display standardized feature coefficients

---

### Analyst Visualizations

Generate the evaluation figures:

```bash
python notebooks/Report3_analyst.py
```

This script generates the following figures inside the `output/` directory:

- `plot6_actual_vs_predicted.png`
- `plot7_residual_plot.png`
- `plot8_feature_coefficients.png`

---

## Results Summary

### Training Set Performance

| Metric | Value |
|--------|------:|
| RMSE | 2.3874 Wh |
| MAE | 1.6696 Wh |
| R² | 0.8477 |

### Test Set Performance

| Metric | Value |
|--------|------:|
| RMSE | 2.7452 Wh |
| MAE | 1.9420 Wh |
| MAPE | 150.0568 % |
| R² | 0.8565 |

### 5-Fold Cross Validation

| Metric | Value |
|--------|------:|
| Mean RMSE | 2.5842 Wh |
| RMSE Std | 1.0458 Wh |
| Mean R² | 0.7953 |
| R² Std | 0.0569 |

### Standardized Feature Coefficients

| Feature | Coefficient |
|--------|------------:|
| Flight Duration | 4.0697 |
| Payload | 1.7794 |
| Altitude | 1.1298 |
| Average Wind Speed | 1.0044 |

---

## Model Interpretation

### Actual vs Predicted Analysis

The Actual vs Predicted plot shows that most test samples are distributed close to the ideal prediction line (**y = x**), indicating that the Linear Regression model predicts typical drone flights with good accuracy.

The largest prediction errors occur on a small number of flights with extremely low energy consumption values.

---

### Residual Analysis

Residuals are centered close to zero (**mean = -0.1475 Wh**), indicating that the model does not systematically overestimate or underestimate battery consumption.

The residual plot shows no obvious funnel-shaped pattern, suggesting that the assumption of approximately constant error variance (homoscedasticity) is reasonably satisfied.

Only **three** observations have residuals greater than **±5 Wh**, all corresponding to near-zero energy flights.

---

### Feature Importance

Among the selected operational features:

1. Flight Duration
2. Payload
3. Altitude
4. Average Wind Speed

Flight Duration has the strongest influence on predicted energy consumption.

All standardized coefficients are positive, indicating that increasing any of these operational factors generally increases battery consumption.

The ranking of feature importance is consistent with the findings from the Exploratory Data Analysis (Report 2).

---

## Project Structure

```text
DTA301_Group1_RBL_Report3
│
├── data/
│   └── cleaned_flight_summary.xlsx
│
├── notebooks/
│   ├── 03_baseline_model.py
│   └── Report3_analyst.py
│
├── output/
│   ├── plot6_actual_vs_predicted.png
│   ├── plot7_residual_plot.png
│   └── plot8_feature_coefficients.png
│
├── requirements.txt
└── README.md
```

---

## Conclusion

This project successfully developed a baseline Linear Regression model for predicting drone energy consumption using four operational features: flight duration, altitude, payload, and average wind speed.

The model achieved a **Test R² score of 0.8565**, indicating that approximately **85.65%** of the variation in drone energy consumption can be explained by the selected features. Prediction accuracy remained strong, with a **Test RMSE of 2.7452 Wh** and a **Test MAE of 1.9420 Wh**.

Five-Fold Cross Validation produced a **mean R² score of 0.7953** with a **standard deviation of 0.0569**, demonstrating that the model generalizes consistently across different subsets of the dataset and is not overly dependent on a single train-test split.

Visualization-based analysis further supports these findings. The Actual vs Predicted plot shows that most predictions closely follow the ideal prediction line, while residual analysis indicates no significant systematic prediction bias. Feature coefficient analysis identifies **Flight Duration** as the most influential predictor, followed by **Payload**, **Altitude**, and **Average Wind Speed**, consistent with both the exploratory data analysis and expected drone flight behavior.

Although the MAPE value is relatively high, this is primarily caused by several valid flights with extremely low energy consumption values. Since MAPE expresses error as a percentage of the true value, observations close to zero disproportionately inflate the metric. Therefore, **RMSE, MAE, and R²** provide a more representative evaluation of model performance for this dataset.

Overall, the baseline Linear Regression model demonstrates strong predictive capability, provides interpretable results, and establishes a reliable benchmark for comparison with the Random Forest Regression model in subsequent stages of the project.