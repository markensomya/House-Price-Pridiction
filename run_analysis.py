"""
Standalone script version of analysis.ipynb
Run this after placing Housing.csv in the same folder.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs('charts', exist_ok=True)

# ── Task 1: Load & Explore ────────────────────────────────────────────────────
df = pd.read_csv('Housing.csv')
print('=== First 10 Rows ===')
print(df.head(10).to_string())
print(f'\nShape: {df.shape[0]} rows, {df.shape[1]} columns')
print(f'\nTarget column : price')
print(f'Feature columns: {[c for c in df.columns if c != "price"]}')
print('\n=== Missing Values ===')
print(df.isnull().sum())

# ── Task 2: Clean ─────────────────────────────────────────────────────────────
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

before = len(df)
df.drop_duplicates(inplace=True)
print(f'\nDuplicates removed: {before - len(df)}')

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f'Categorical columns encoded: {categorical_cols}')
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)
print(f'Shape after encoding: {df.shape}')

# ── Task 3: Model Building ────────────────────────────────────────────────────
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'\nTrain: {len(X_train)} | Test: {len(X_test)}')

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)
print(f'\n=== Linear Regression ===\nMAE: {mae_lr:,.0f} | RMSE: {rmse_lr:,.0f} | R2: {r2_lr:.4f}')

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)
print(f'\n=== Random Forest ===\nMAE: {mae_rf:,.0f} | RMSE: {rmse_rf:,.0f} | R2: {r2_rf:.4f}')

# Comparison
comp = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [mae_lr, mae_rf], 'RMSE': [rmse_lr, rmse_rf], 'R2': [r2_lr, r2_rf]
})
print('\n=== Model Comparison ===')
print(comp.to_string(index=False))

# ── Task 4: Charts ────────────────────────────────────────────────────────────

# Chart 1: Price Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['price'], bins=40, color='steelblue', edgecolor='white', linewidth=0.6)
ax.set_title('Distribution of House Prices', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Price', fontsize=13)
ax.set_ylabel('Number of Houses', fontsize=13)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/chart1_price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nChart 1 saved: charts/chart1_price_distribution.png')

# Chart 2: Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, annot_kws={'size': 8}, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('charts/chart2_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 2 saved: charts/chart2_correlation_heatmap.png')

# Chart 3: Actual vs Predicted
fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(y_test, y_pred_rf, alpha=0.6, color='darkorange', edgecolors='white', linewidths=0.4, s=60)
min_val = min(y_test.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred_rf.max())
ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=1.8, label='Perfect Prediction')
ax.set_title('Actual vs Predicted House Prices\n(Random Forest)', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('Actual Price', fontsize=13)
ax.set_ylabel('Predicted Price', fontsize=13)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.text(0.05, 0.93, f'R² = {r2_rf:.3f}', transform=ax.transAxes,
        fontsize=12, color='navy', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
plt.tight_layout()
plt.savefig('charts/chart3_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 3 saved: charts/chart3_actual_vs_predicted.png')

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print('\n=== Top 10 Important Features ===')
print(feat_imp.head(10).to_string())

print('\nAll tasks completed successfully.')
