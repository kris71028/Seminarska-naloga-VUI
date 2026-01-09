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

    return model, highlight_linear_importance(summary_df), rmse, r2



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
                           n_estimators=200, max_depth=5, features=all_features):

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

    return model, importance_df, rmse, r2

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
def run_random_forest(df, test_size=0.2, n_estimators=100, max_depth=None, features = all_features):

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

    return model, importance_df, rmse, r2

# -----------------------------
# GRADIENT BOOSTING
# -----------------------------
def run_gradient_boosting(df, test_size=0.2, n_estimators=100, max_depth=3, features=all_features):

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

    return model, importance_df, rmse, r2

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

    return model, summary_df.style.apply(highlight_ridge, axis=1), rmse, r2

def compare_models(df, features=None):
    """
    Runs all models on the given features and returns a comparison table with RMSE and R².
    """
    comparison = []

    # Linear
    _, _, rmse, r2 = run_linear(df, features=features)
    comparison.append(["Linear Regression", rmse, r2])

    # Ridge
    _, _, rmse, r2 = run_ridge(df, features=features)
    comparison.append(["Ridge Regression", rmse, r2])

    # Random Forest
    _, _, rmse, r2 = run_random_forest(df, features=features)
    comparison.append(["Random Forest", rmse, r2])

    # Gradient Boosting
    _, _, rmse, r2 = run_gradient_boosting(df, features=features)
    comparison.append(["Gradient Boosting", rmse, r2])

    # XGBoost
    _, _, rmse, r2 = run_xgboost(df, features=features)
    comparison.append(["XGBoost", rmse, r2])

    return pd.DataFrame(comparison, columns=["Model", "RMSE", "R²"]).sort_values("RMSE")
