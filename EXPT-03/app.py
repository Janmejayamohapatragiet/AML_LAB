import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = {
    'primary': '#0077b6',
    'secondary': '#00b4d8',
    'accent': '#ffb703',
    'danger': '#e63946',
    'dark': '#2b2d42'
}


def fit_and_report(X, y, title):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    regressor = LinearRegression()
    regressor.fit(X_tr, y_tr)
    preds = regressor.predict(X_te)
    
    metrics = {
        'MAE': mean_absolute_error(y_te, preds),
        'MSE': mean_squared_error(y_te, preds),
        'RMSE': np.sqrt(mean_squared_error(y_te, preds)),
        'R2': r2_score(y_te, preds)
    }
    
    terms = [f"{coef:+.4f} * {col}" for coef, col in zip(regressor.coef_, X.columns)]
    eq_str = f"y = {regressor.intercept_:.4f} " + " ".join(terms)
    
    print(f"\n==================== {title.upper()} ====================")
    print(f"Model Formula: {eq_str}")
    print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    
    return regressor, X_te, y_te, preds, metrics


def render_feature_diagnostics(dataframe):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    sns.heatmap(
        dataframe.corr(), 
        annot=True, 
        cmap='YlGnBu', 
        fmt=".2f", 
        cbar=True, 
        ax=axes[0]
    )
    axes[0].set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
    
    sns.boxplot(
        data=dataframe, 
        palette='Blues', 
        orient='h', 
        ax=axes[1]
    )
    axes[1].set_title('Distribution & Outlier Inspection Across Semesters', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def render_regression_comparison(y_test_s, pred_s, y_test_m, pred_m, x_simple, feat_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(x_simple.values.ravel(), y_test_s, color=PALETTE['primary'], alpha=0.7, label='Observed Data')
    axes[0].plot(x_simple.values.ravel(), pred_s, color=PALETTE['danger'], linewidth=2, label='Regression Trend')
    axes[0].set_title(f'Simple LR Fit: {feat_name} vs SEM 5', fontweight='bold')
    axes[0].set_xlabel(feat_name)
    axes[0].set_ylabel('SEM 5')
    axes[0].legend()
    
    axes[1].scatter(y_test_m, pred_m, color=PALETTE['secondary'], edgecolors=PALETTE['dark'], alpha=0.8, label='Predictions')
    min_val = min(y_test_m.min(), pred_m.min())
    max_val = max(y_test_m.max(), pred_m.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], color=PALETTE['accent'], linestyle='--', linewidth=2, label='1:1 Perfect Fit')
    axes[1].set_title('Multiple LR: Actual vs Predicted', fontweight='bold')
    axes[1].set_xlabel('Actual SEM 5 Marks')
    axes[1].set_ylabel('Predicted SEM 5 Marks')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def render_metrics_comparison(metrics_simple, metrics_multi):
    df_metrics = pd.DataFrame([metrics_simple, metrics_multi], index=['Simple LR', 'Multiple LR']).T
    
    ax = df_metrics.plot(kind='bar', figsize=(9, 5), color=[PALETTE['primary'], PALETTE['accent']], width=0.6)
    plt.title('Performance Metrics Comparison', fontsize=13, fontweight='bold')
    plt.ylabel('Score Value')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
    plt.tight_layout()
    plt.show()


url = 'https://raw.githubusercontent.com/Ayushman2005-cmyk/AML_EXPERIMENTS/main/EXPT-3/studentGradeDataSet.csv'
data = pd.read_csv(url)

print("=== DATASET OVERVIEW ===")
print(f"Dimensions: {data.shape[0]} rows x {data.shape[1]} columns")
print(f"Missing Values total: {data.isnull().sum().sum()}")
print("\nDescriptive Statistics:\n", data.describe().T[['mean', 'std', 'min', '50%', 'max']])

render_feature_diagnostics(data)

features = ['SEM 1', 'SEM 2', 'SEM 3', 'SEM 4']
X_all = data[features]
y = data['SEM 5']

top_predictor = X_all.corrwith(y).abs().idxmax()
print(f"\nHighest Correlated Predictor for SEM 5: '{top_predictor}'")

X_simple = data[[top_predictor]]
simple_model, X_test_s, y_test_s, pred_s, metrics_s = fit_and_report(X_simple, y, 'Simple Linear Regression')

multi_model, X_test_m, y_test_m, pred_m, metrics_m = fit_and_report(X_all, y, 'Multiple Linear Regression')

render_regression_comparison(y_test_s, pred_s, y_test_m, pred_m, X_test_s, top_predictor)
render_metrics_comparison(metrics_s, metrics_m)

print("\n" + "="*40)
print("     STUDENT SEMESTER 5 PREDICTOR     ")
print("="*40)

user_scores = {}
for sem in features:
    user_scores[sem] = float(input(f"Enter score for {sem}: "))

input_df = pd.DataFrame([user_scores])
estimated_grade = multi_model.predict(input_df)[0]

print(f"\n>>> Estimated SEM 5 Score: {estimated_grade:.2f} <<<")