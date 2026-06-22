import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_excel(
    "data/cleaned_flight_summary.xlsx"
)

print("\n=== Dataset Shape ===")
print(df.shape)

# =============================================================================
# FEATURE SELECTION
# =============================================================================

features = [
    "flight_duration_s",
    "altitude",
    "payload",
    "avg_wind_speed"
]

target = "total_energy_wh"

X = df[features]
y = df[target]

# =============================================================================
# TRAIN TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n=== Train/Test Split ===")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# =============================================================================
# FEATURE SCALING
# =============================================================================

scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)

X_test_sc = scaler.transform(X_test)

print("\n=== Scaling Complete ===")
print("X_train_sc:", X_train_sc.shape)
print("X_test_sc :", X_test_sc.shape)

# =============================================================================
# LINEAR REGRESSION MODEL
# =============================================================================

model = LinearRegression()

model.fit(X_train_sc, y_train)

print("\n=== Model Training Complete ===")

# =============================================================================
# PREDICTIONS
# =============================================================================

y_pred_train = model.predict(X_train_sc)

y_pred_test = model.predict(X_test_sc)

print("\n=== Prediction Complete ===")

print("\n=== Sample Predictions ===")

comparison = pd.DataFrame(
    {
        "Actual": y_test.values[:5],
        "Predicted": y_pred_test[:5]
    }
)

print(comparison.round(2))

print("\n=== Target Statistics ===")

print("Min Energy:", y.min())
print("Max Energy:", y.max())

# =============================================================================
# MODEL EVALUATION
# =============================================================================

train_rmse = mean_squared_error(
    y_train,
    y_pred_train
) ** 0.5

test_rmse = mean_squared_error(
    y_test,
    y_pred_test
) ** 0.5

train_mae = mean_absolute_error(
    y_train,
    y_pred_train
)

test_mae = mean_absolute_error(
    y_test,
    y_pred_test
)

train_mape = (
    mean_absolute_percentage_error(
        y_train,
        y_pred_train
    ) * 100
)

test_mape = (
    mean_absolute_percentage_error(
        y_test,
        y_pred_test
    ) * 100
)

train_r2 = r2_score(
    y_train,
    y_pred_train
)

test_r2 = r2_score(
    y_test,
    y_pred_test
)

results = pd.DataFrame(
    {
        "Dataset": ["Train", "Test"],
        "RMSE": [train_rmse, test_rmse],
        "MAE": [train_mae, test_mae],
        "MAPE (%)": [train_mape, test_mape],
        "R2": [train_r2, test_r2]
    }
)

print("\n=== Linear Regression Results ===")
print(results.round(4))

# =============================================================================
# 5-FOLD CROSS VALIDATION
# =============================================================================

pipeline = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

# -------------------------------------------------------------------------
# RMSE
# -------------------------------------------------------------------------

cv_rmse = (
    -cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )
)

print("\n=== 5-Fold Cross Validation (RMSE) ===")

print("Fold Scores:")
print(cv_rmse.round(4))

print(
    f"\nMean RMSE: {cv_rmse.mean():.4f}"
)

print(
    f"Std RMSE : {cv_rmse.std():.4f}"
)

# -------------------------------------------------------------------------
# R²
# -------------------------------------------------------------------------

cv_r2 = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\n=== 5-Fold Cross Validation (R²) ===")

print("Fold Scores:")
print(cv_r2.round(4))

print(
    f"\nMean R²: {cv_r2.mean():.4f}"
)

print(
    f"Std R² : {cv_r2.std():.4f}"
)

# -------------------------------------------------------------------------
# The coefficient analysis
# -------------------------------------------------------------------------

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print("\n=== Feature Coefficients ===")
print(coef_df.sort_values(
    by="Coefficient",
    ascending=False
))

print("\nIntercept:")
print(model.intercept_)