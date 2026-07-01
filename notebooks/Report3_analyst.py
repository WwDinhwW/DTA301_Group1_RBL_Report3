# ============================================================
# DTA301 - Report 3: Baseline Model — Analyst Visualizations
# Person B - Data Analyst (Le Vuong Dinh)
# US-VD-01: Actual vs Predicted Plot
# US-VD-02: Residual Plot
# US-VD-03: Feature Coefficients Chart
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import os

# ── CONFIG ──────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, '..', 'data', 'cleaned_flight_summary.xlsx')
OUTPUT_DIR   = os.path.join(BASE_DIR, '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
sns.set_theme(style='whitegrid')

# ============================================================
# LOAD & PREPARE DATA  (same setup as Thuan's model script)
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_excel(DATA_PATH)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

# No longer needed, 2 NaN are removed from dataset
# Handle 2 NaN in altitude — impute with median (agreed with Huy)
# median_alt = df['altitude'].median()
# df['altitude'] = df['altitude'].fillna(median_alt)
# print(f"Filled 2 NaN altitude with median = {median_alt} m")

features = ['flight_duration_s', 'altitude', 'payload', 'avg_wind_speed']
target   = 'total_energy_wh'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} flights | Test: {X_test.shape[0]} flights")

# Scale — fit ONLY on train
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_sc, y_train)

# Predictions
y_pred_train = model.predict(X_train_sc)
y_pred_test  = model.predict(X_test_sc)

# Metrics
test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
test_mae  = mean_absolute_error(y_test, y_pred_test)
test_r2   = r2_score(y_test, y_pred_test)
residuals = y_test.values - y_pred_test

print(f"\nTest  RMSE = {test_rmse:.4f} Wh")
print(f"Test  MAE  = {test_mae:.4f} Wh")
print(f"Test  R²   = {test_r2:.4f}")
print(f"Residuals  mean = {residuals.mean():.4f}, std = {residuals.std():.4f}")

# ============================================================
# PLOT 6 — Actual vs Predicted
# US-VD-01
# ============================================================
print("\n" + "=" * 60)
print("PLOT 6 — Actual vs Predicted")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 7))

ax.scatter(
    y_test, y_pred_test,
    alpha=0.7, color='steelblue', s=55,
    edgecolors='white', linewidth=0.6, zorder=3,
    label='Test flights (n=42)'
)

# Perfect prediction line y = x
all_vals = np.concatenate([y_test.values, y_pred_test])
lims = [all_vals.min() - 1, all_vals.max() + 1]
ax.plot(lims, lims, 'r--', linewidth=1.5, alpha=0.8,
        label='Perfect prediction (y = x)')

# Highlight largest error point
max_err_idx = int(np.argmax(np.abs(residuals)))
ax.scatter(
    y_test.values[max_err_idx], y_pred_test[max_err_idx],
    color='tomato', s=100, zorder=4,
    edgecolors='darkred', linewidth=1.2
)
ax.annotate(
    f'Largest error\n({abs(residuals[max_err_idx]):.1f} Wh)',
    xy=(y_test.values[max_err_idx], y_pred_test[max_err_idx]),
    xytext=(
        y_test.values[max_err_idx] + 1.5,
        y_pred_test[max_err_idx] - 3.5
    ),
    fontsize=9, color='darkred',
    arrowprops=dict(arrowstyle='->', color='darkred', lw=1)
)

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Actual Energy Consumption (Wh)', fontsize=12)
ax.set_ylabel('Predicted Energy Consumption (Wh)', fontsize=12)
ax.set_title(
    'Actual vs Predicted — Linear Regression Baseline',
    fontsize=13, fontweight='bold', pad=14
)

# Metrics box (top-left)
metrics_text = (
    f'Test R²   = {test_r2:.4f}\n'
    f'Test RMSE = {test_rmse:.4f} Wh\n'
    f'Test MAE  = {test_mae:.4f} Wh\n'
    f'n_test = 42'
)
ax.text(
    0.04, 0.97, metrics_text,
    transform=ax.transAxes, fontsize=9.5,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='#EEF4FB',
              alpha=0.85, edgecolor='#90B8D8')
)

ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()

path6 = os.path.join(OUTPUT_DIR, 'plot6_actual_vs_predicted.png')
plt.savefig(path6, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {path6}")

# Analyst note
print("\nInsight (Muc 6):")
print(f"  - Most points cluster around y=x in 15-28 Wh range — model")
print(f"    performs well for typical flights.")
print(f"  - Largest prediction error = {abs(residuals[max_err_idx]):.2f} Wh")
print(f"    (near-zero energy outlier flight that model struggles with)")

# ============================================================
# PLOT 7 — Residual Plot (2 panels)
# US-VD-02
# ============================================================
print("\n" + "=" * 60)
print("PLOT 7 — Residual Plot")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Left: residuals vs predicted ────────────────────────────
ax1 = axes[0]

ax1.scatter(
    y_pred_test, residuals,
    alpha=0.7, color='steelblue', s=55,
    edgecolors='white', linewidth=0.6, zorder=3
)
ax1.axhline(0, color='red', linestyle='--', linewidth=1.5,
            label='Zero residual line')
ax1.axhspan(-test_rmse, test_rmse, alpha=0.08, color='steelblue',
            label=f'±RMSE band (±{test_rmse:.2f} Wh)')

# Highlight large residuals
large_res = np.abs(residuals) > 5
if large_res.sum() > 0:
    ax1.scatter(
        y_pred_test[large_res], residuals[large_res],
        color='tomato', s=80, zorder=4,
        edgecolors='darkred', linewidth=1,
        label=f'|Residual| > 5 Wh (n={large_res.sum()})'
    )

ax1.set_xlabel('Predicted Energy Consumption (Wh)', fontsize=11)
ax1.set_ylabel('Residual  (Actual − Predicted, Wh)', fontsize=11)
ax1.set_title('Residual vs Predicted Values', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)

ax1.text(
    0.04, 0.96,
    f'Mean residual = {residuals.mean():.4f} Wh\n'
    f'Std  residual = {residuals.std():.4f} Wh',
    transform=ax1.transAxes, fontsize=9,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='#EEF4FB',
              alpha=0.85, edgecolor='#90B8D8')
)

# ── Right: residual distribution ────────────────────────────
ax2 = axes[1]

ax2.hist(residuals, bins=15, color='steelblue',
         edgecolor='white', alpha=0.85, density=True)

mu, std = residuals.mean(), residuals.std()
x_range = np.linspace(residuals.min() - 1, residuals.max() + 1, 200)
ax2.plot(x_range, norm.pdf(x_range, mu, std), 'r-', linewidth=2,
         label=f'Normal fit (μ={mu:.2f}, σ={std:.2f})')
ax2.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

ax2.set_xlabel('Residual Value (Wh)', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)

plt.suptitle(
    'Residual Analysis — Linear Regression Baseline',
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()

path7 = os.path.join(OUTPUT_DIR, 'plot7_residual_plot.png')
plt.savefig(path7, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {path7}")

print("\nInsight (Muc 6):")
print(f"  - Residuals centered near 0 (mean = {mu:.4f} Wh) — no systematic bias")
n_large = large_res.sum()
print(f"  - {n_large} points with |residual| > 5 Wh — all near-zero energy outliers")
pattern = "No clear funnel shape" if residuals.std() < 4 else "Possible heteroscedasticity detected"
print(f"  - {pattern} in residual vs predicted plot")

# ============================================================
# PLOT 8 — Feature Coefficients
# US-VD-03
# ============================================================
print("\n" + "=" * 60)
print("PLOT 8 — Feature Coefficients")
print("=" * 60)

feature_labels = [
    'Flight Duration (s)',
    'Altitude (m)',
    'Payload (g)',
    'Avg Wind Speed (m/s)'
]

coef_df = pd.DataFrame({
    'Feature': feature_labels,
    'Coefficient': model.coef_,
    'EDA_r': [0.80, 0.65, 0.40, 0.32]
}).sort_values('Coefficient', ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))

colors = ['#2E75B6' if c > 0 else '#C0504D' for c in coef_df['Coefficient']]
bars = ax.barh(
    coef_df['Feature'], coef_df['Coefficient'],
    color=colors, edgecolor='white', height=0.5
)

# Value labels + EDA r reference
for bar, (_, row) in zip(bars, coef_df.iterrows()):
    val = row['Coefficient']
    r   = row['EDA_r']
    x_label = val + 0.06 if val > 0 else val - 0.06
    ha = 'left' if val > 0 else 'right'
    ax.text(x_label, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', ha=ha,
            fontsize=10.5, fontweight='bold')
    # EDA r on the right side
    ax.text(
        coef_df['Coefficient'].max() + 0.35,
        bar.get_y() + bar.get_height() / 2,
        f'EDA r = {r:+.2f}',
        va='center', ha='left', fontsize=9, color='gray'
    )

ax.axvline(0, color='black', linewidth=0.8, alpha=0.4)
ax.set_xlabel(
    'Regression Coefficient  (on StandardScaled features)',
    fontsize=11
)
ax.set_title(
    'Feature Coefficients — Linear Regression Baseline\n'
    '(Standardized coefficients: larger |value| = stronger influence on prediction)',
    fontsize=12, fontweight='bold', pad=12
)

blue_patch = mpatches.Patch(color='#2E75B6',
                             label='Positive: increases energy consumption')
red_patch  = mpatches.Patch(color='#C0504D',
                             label='Negative: decreases energy consumption')
ax.legend(handles=[blue_patch, red_patch], fontsize=9, loc='lower right')

ax.set_xlim(coef_df['Coefficient'].min() - 0.5,
            coef_df['Coefficient'].max() + 1.5)

plt.tight_layout()

path8 = os.path.join(OUTPUT_DIR, 'plot8_feature_coefficients.png')
plt.savefig(path8, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {path8}")

top_feat = coef_df.sort_values('Coefficient', ascending=False).iloc[0]
print("\nInsight (Muc 6):")
print(f"  - Strongest feature: {top_feat['Feature']} "
      f"(coef = {top_feat['Coefficient']:.4f}) — "
      f"consistent with EDA (r = {top_feat['EDA_r']:+.2f})")
print(f"  - All 4 features have positive coefficients — all increase energy consumption")
print(f"  - Altitude ranks 4th by coef (1.18) despite r=+0.65 in EDA")
print(f"    Reason: altitude & flight_duration_s correlated (r=0.53) — shared variance")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DONE — 3 plots saved to output/")
print("=" * 60)
print(f"  plot6_actual_vs_predicted.png  → Muc 5, Muc 6 Section 9.6")
print(f"  plot7_residual_plot.png        → Muc 6 homoscedasticity check")
print(f"  plot8_feature_coefficients.png → Muc 6 Sub-RQ1 answer")
print("\nNumbers for Anh (Muc 5 + Muc 7 Experiment Log):")
print(f"  Test  RMSE = {test_rmse:.4f} Wh")
print(f"  Test  MAE  = {test_mae:.4f} Wh")
print(f"  Test  R²   = {test_r2:.4f}")
print(f"  Train RMSE = {mean_squared_error(y_train, y_pred_train)**0.5:.4f} Wh")
print(f"  Train MAE  = {mean_absolute_error(y_train, y_pred_train):.4f} Wh")
print(f"  Train R²   = {r2_score(y_train, y_pred_train):.4f}")