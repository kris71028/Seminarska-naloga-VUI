from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

important_features = [
    "Age_Category", "Sex", "Exercise", "Diabetes", "Arthritis", "General_Health"
]

all_features = [
    "Age_Category", "Sex", "Smoking_History", "Exercise",
    "Alcohol_Consumption", "Fruit_Consumption",
    "Green_Vegetables_Consumption", "FriedPotato_Consumption",
    "Diabetes", "Depression", "Arthritis",
    "Checkup", "Heart_Disease"
]

target_feature = "BMI"

def run_linear(df, features=all_features):
    X = df[features]
    y = df[target_feature]

    X = X.dropna()
    y = y.loc[X.index]

    # Keep y in raw units
    # Optionally scale X if you want comparable coefficients
    scaler_X = StandardScaler()
    X_std = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns, index=X.index)

    X_std = sm.add_constant(X_std)

    model = sm.OLS(y, X_std).fit()  # fit on raw y
    y_pred = model.predict(X_std)

    rmse = mean_squared_error(y, y_pred)  # raw units
    r2 = model.rsquared

    summary_df = pd.DataFrame({
        "Feature": model.params.index,
        "Std_Coefficient": model.params.values,
        "t_value": model.tvalues,
        "p_value": model.pvalues
    })

    return model, highlight_linear_importance(summary_df), rmse, None, r2



def highlight_linear_importance(summary_df,
                                beta_threshold=0.1,
                                t_threshold=2.0,
                                p_threshold=0.05):

    def highlight_row(row):
        color = [''] * len(row)

        # Strong effect size
        if row["Feature"] != "const" and abs(row["Std_Coefficient"]) >= beta_threshold:
            color[row.index.get_loc("Std_Coefficient")] = \
                'background-color: #90EE90; font-weight:bold'

        # Insignificant
        if abs(row["t_value"]) < t_threshold:
            color[row.index.get_loc("t_value")] = \
                'background-color: #ff4d4d; color:white'

        if row["p_value"] > p_threshold:
            color[row.index.get_loc("p_value")] = \
                'background-color: #ff4d4d; color:white'

        return color

    return summary_df.style.apply(highlight_row, axis=1)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def run_xgboost(df, test_size=0.2,
                           n_estimators=100, max_depth=5, features=all_features):

    X = df[features].dropna()
    y = df.loc[X.index, target_feature]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size
    )

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    model.fit(X_train, y_train)

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    importance_df = importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    importance_df = highlight_xgb_importance(importance_df)
    # Evaluation
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rmse_mean, rmse_std = cv_rmse(model, X, y)
    return model, importance_df, rmse_mean, rmse_std, r2

def highlight_xgb_importance(importance_df, top_n=5):
    def highlight_row(row):
        color = [''] * len(row)
        if row.name < top_n:
            color[importance_df.columns.get_loc("Importance")] = \
                'background-color: #90EE90; font-weight:bold'
        return color

    return importance_df.style.apply(highlight_row, axis=1)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# RANDOM FOREST
# -----------------------------
def run_random_forest(df, test_size=0.2, n_estimators=50, max_depth=None, features = all_features):

    X = df[features].dropna()
    y = df.loc[X.index, target_feature]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    importance_df = highlight_xgb_importance(importance_df)  # reuse highlight function

    # Evaluation
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rmse_mean, rmse_std = cv_rmse(model, X, y)
    return model, importance_df, rmse_mean, rmse_std, r2

# -----------------------------
# GRADIENT BOOSTING
# -----------------------------
def run_gradient_boosting(df, test_size=0.2, n_estimators=50, max_depth=3, features=all_features):

    X = df[features].dropna()
    y = df.loc[X.index, target_feature]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size
    )

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    model.fit(X_train, y_train)

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    importance_df = highlight_xgb_importance(importance_df)

    # Evaluation
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rmse_mean, rmse_std = cv_rmse(model, X, y)
    return model, importance_df, rmse_mean, rmse_std, r2

# -----------------------------
# RIDGE REGRESSION
# -----------------------------
def run_ridge(df, alpha=1.0, features=all_features):
    X = df[features]
    y = df[target_feature]

    X = X.dropna()
    y = y.loc[X.index]

    # Scale X only
    scaler_X = StandardScaler()
    X_std = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns, index=X.index)

    model = Ridge(alpha=alpha)
    model.fit(X_std, y)  # fit on raw y
    y_pred = model.predict(X_std)

    rmse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    summary_df = pd.DataFrame({
        "Feature": X_std.columns,
        "Std_Coefficient": model.coef_
    })

    # Highlight strong coefficients (>0.1)
    def highlight_ridge(row, beta_threshold=0.1):
        color = [''] * len(row)
        if abs(row["Std_Coefficient"]) >= beta_threshold:
            color[row.index.get_loc("Std_Coefficient")] = 'background-color: #90EE90; font-weight:bold'
        return color

    rmse_mean, rmse_std = cv_rmse(model, X, y)
    return model, summary_df.style.apply(highlight_ridge, axis=1), rmse_mean, rmse_std, r2

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

rmse_scorer = make_scorer(
    mean_squared_error,
    greater_is_better=False
)

def cv_rmse(model, X, y, n_splits=2, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf, scoring=rmse_scorer)
    rmse_scores = -scores  # sklearn returns negative
    return rmse_scores.mean(), rmse_scores.std()

def cv_rmse_linear(df, features, n_splits=3):
    X = df[features].dropna()
    y = df.loc[X.index, target_feature]

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses = []

    for train_idx, test_idx in kf.split(X_std):
        X_train, X_test = X_std[train_idx], X_std[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        model = sm.OLS(y_train, X_train).fit()
        y_pred = model.predict(X_test)

        rmses.append(mean_squared_error(y_test, y_pred))

    return np.mean(rmses), np.std(rmses)

from joblib import dump
from sklearn.pipeline import Pipeline

def compare_models(df, features=None):
    save_dir="models"
    import os
    os.makedirs(save_dir, exist_ok=True)

    comparison = []

    # ---------------- Linear Regression (no saving) ----------------
    rmse_mean, rmse_std = cv_rmse_linear(df, features)
    linear_model, _, _, _, r2 = run_linear(df, features)
    comparison.append([
        "Linear Regression",
        f"{rmse_mean:.3f} ± {rmse_std:.3f}",
        r2,
        "-"
    ])

    # ---------------- Ridge Regression (SAVE) ----------------
    X = df[features].dropna()
    y = df.loc[X.index, target_feature]

    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    ridge_pipeline.fit(X, y)

    rmse_mean, rmse_std = cv_rmse(ridge_pipeline, X, y)
    r2 = ridge_pipeline.score(X, y)

    dump(ridge_pipeline, f"{save_dir}/ridge_model.joblib")

    comparison.append([
        "Ridge Regression",
        f"{rmse_mean:.3f} ± {rmse_std:.3f}",
        r2,
        "alpha=1.0 [SAVED]"
    ])

    # ---------------- Random Forest (not saved) ----------------
    rf_model, _, rmse_mean, rmse_std, r2 = run_random_forest(
        df, n_estimators=50, max_depth=3, features=features
    )

    comparison.append([
        "Random Forest",
        f"{rmse_mean:.3f} ± {rmse_std:.3f}",
        r2,
        "n_estimators=50, max_depth=3"
    ])

    # ---------------- Gradient Boosting (SAVE) ----------------
    gb_model, _, rmse_mean, rmse_std, r2 = run_gradient_boosting(
        df, n_estimators=50, max_depth=3, features=features
    )

    dump(gb_model, f"{save_dir}/gradient_boosting_model.joblib")

    comparison.append([
        "Gradient Boosting",
        f"{rmse_mean:.3f} ± {rmse_std:.3f}",
        r2,
        "n_estimators=50, max_depth=3 [SAVED]"
    ])

    # ---------------- XGBoost (SAVE) ----------------
    xgb_model, _, rmse_mean, rmse_std, r2 = run_xgboost(
        df, n_estimators=100, max_depth=5, features=features
    )

    dump(xgb_model, f"{save_dir}/xgboost_model.joblib")

    comparison.append([
        "XGBoost",
        f"{rmse_mean:.3f} ± {rmse_std:.3f}",
        r2,
        "n_estimators=100, max_depth=5 [SAVED]"
    ])

    # ---------------- Create table ----------------
    df_comparison = pd.DataFrame(
        comparison,
        columns=["Model", "RMSE (CV mean ± std)", "R²", "Parameters"]
    )

    df_comparison = df_comparison.sort_values(
        by="RMSE (CV mean ± std)",
        key=lambda s: s.str.split(" ± ").str[0].astype(float)
    )

    models = {
        "linear": linear_model,
        "ridge": ridge_pipeline,
        "random_forest": rf_model,
        "gradient_boosting": gb_model,
        "xgboost": xgb_model
    }

    return df_comparison, models


def spearman_correlation(df, features):
    data = df[features + ["BMI"]].dropna()

    corr = data.corr(method="spearman")["BMI"].drop("BMI")
    pvals = {}

    from scipy.stats import spearmanr
    for f in features:
        rho, p = spearmanr(data[f], data["BMI"])
        pvals[f] = p

    corr_df = pd.DataFrame({
        "Feature": corr.index,
        "Spearman_rho": corr.values,
        "p_value": [pvals[f] for f in corr.index]
    }).sort_values("Spearman_rho", key=abs, ascending=False)

    return corr_df

def pearson_correlation(df, features):
    data = df[features + ["BMI"]].dropna()

    corr = data.corr(method="pearson")["BMI"].drop("BMI")
    pvals = {}

    from scipy.stats import pearsonr
    for f in features:
        r, p = pearsonr(data[f], data["BMI"])
        pvals[f] = p

    corr_df = pd.DataFrame({
        "Feature": corr.index,
        "Pearson_r": corr.values,
        "p_value": [pvals[f] for f in corr.index]
    }).sort_values("Pearson_r", key=abs, ascending=False)

    return corr_df

def mann_whitney_u_correlation(df, features):
    from scipy.stats import mannwhitneyu

    results = []

    for f in features:
        subset = df[[f, "BMI"]].dropna()

        group0 = subset[subset[f] == 0]["BMI"]
        group1 = subset[subset[f] == 1]["BMI"]

        # Skip if one group is empty
        if len(group0) == 0 or len(group1) == 0:
            continue

        u_stat, p_value = mannwhitneyu(
            group0,
            group1,
            alternative="two-sided"
        )

        results.append({
            "Feature": f,
            "U_statistic": u_stat,
            "p_value": p_value,
            "Median_BMI_0": group0.median(),
            "Median_BMI_1": group1.median()
        })

    result_df = pd.DataFrame(results).sort_values("p_value")

    return result_df


import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(df):
    features = all_features
    target = "BMI"
    binary_threshold = 2
    n_rows = 4
    fig_width = 18
    fig_height = 8

    n_features = len(features)
    n_cols = math.ceil(n_features / n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False
    )

    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        data = df[[feature, target]].dropna()
        unique_vals = data[feature].nunique()

        if unique_vals <= binary_threshold:
            sns.boxplot(
                x=feature,
                y=target,
                data=data,
                ax=ax
            )
            sns.stripplot(
                x=feature,
                y=target,
                data=data,
                color="black",
                alpha=0.4,
                ax=ax
            )
        else:
            sns.regplot(
                x=feature,
                y=target,
                data=data,
                lowess=True,
                scatter_kws={"alpha": 0.4},
                line_kws={"color": "red"},
                ax=ax
            )

        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import scipy.stats as stats
def check_normality(df, columns=None, plot=False, alpha=0.05):
    """
    Checks normality for selected columns (numeric or categorical with >2 levels).
    Ignores binary columns automatically.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with preprocessed columns.
    columns : list, optional
        List of columns to check. If None, all columns are checked.
    plot : bool
        Whether to show Q–Q plots.
    alpha : float
        Significance level for Shapiro-Wilk.

    Returns
    -------
    pd.DataFrame
        Table with Variable, Shapiro_p, Normal (YES/NO), N_unique
    """
    results = []

    if columns is None:
        columns = df.columns

    for col in columns:
        data = df[col].dropna()

        # Skip binary columns
        if data.nunique() <= 2:
            continue

        # Ensure numeric type
        try:
            data_numeric = pd.to_numeric(data)
            data_numeric = data_numeric.sample(n=5000)
        except:
            print(f"Skipping column {col}: not numeric")
            continue

        if len(data_numeric) < 3:
            normality = "Insufficient data"
            p_val = None
        else:
            stat, p_val = stats.shapiro(data_numeric)
            normality = "YES" if float(p_val) > alpha else "NO"

            if plot:
                plt.figure(figsize=(4, 4))
                stats.probplot(data_numeric, dist="norm", plot=plt)
                plt.title(f"Normality: {col} (p={p_val:.3f})")
                plt.tight_layout()
                plt.show()

        results.append({
            "Variable": col,
            "Shapiro_p": p_val,
            "Normal": normality,
            "N_unique": data_numeric.nunique()
        })

    return pd.DataFrame(results).sort_values("Variable")

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

def make_everyone_exercise(df, model, important_features, tolerance=2.0):
    df_test = df[model.get_booster().feature_names].copy()
    y_true = df["BMI"].copy()

    # Copy and apply global change
    df_test_modified = df_test.copy()
    df_test_modified["Exercise"] = 1

    # Predictions
    y_pred_before = model.predict(df_test)
    y_pred_after  = model.predict(df_test_modified)

    # --- Regression metrics ---
    mae_before = mean_absolute_error(y_true, y_pred_before)
    mae_after  = mean_absolute_error(y_true, y_pred_after)
    rmse_before = np.sqrt(mean_squared_error(y_true, y_pred_before))
    rmse_after  = np.sqrt(mean_squared_error(y_true, y_pred_after))
    mean_before = np.mean(y_pred_before)
    mean_after  = np.mean(y_pred_after)

    metrics_table = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "Mean BMI"],
        "Before": [mae_before, rmse_before, mean_before],
        "After":  [mae_after, rmse_after, mean_after],
        "Difference": [mae_after - mae_before, rmse_after - rmse_before, mean_after - mean_before]
    })

    mean_bmi_diff = metrics_table.loc[metrics_table["Metric"] == "Mean BMI", "Difference"].values[0]
    if mean_bmi_diff < 0:
        print(f"\nPredicted BMI decreased by {abs(mean_bmi_diff):.2f}.")
    else:
        print(f"\nPredicted BMI increased by {mean_bmi_diff:.2f}.")

    # --- Six Sigma Analysis ---
    errors_before = np.abs(y_pred_before - y_true)
    errors_after  = np.abs(y_pred_after - y_true)

    defects_before = np.sum(errors_before > tolerance)
    defects_after  = np.sum(errors_after > tolerance)

    N = len(y_true)
    DPMO_before = defects_before / N * 1e6
    DPMO_after  = defects_after / N * 1e6

    def dpmo_to_sigma(dpmo):
        return norm.ppf(1 - dpmo / 1e6)

    sigma_before = dpmo_to_sigma(DPMO_before)
    sigma_after  = dpmo_to_sigma(DPMO_after)

    six_sigma_table = pd.DataFrame({
        "Metric": ["Defects", "DPMO", "Sigma Level"],
        "Before": [defects_before, DPMO_before, sigma_before],
        "After": [defects_after, DPMO_after, sigma_after],
        "Difference": [defects_after - defects_before, DPMO_after - DPMO_before, sigma_after - sigma_before]
    })

    # Wilcoxon test
    stat, p_value = wilcoxon(errors_before, errors_after)
    return metrics_table, six_sigma_table, p_value
