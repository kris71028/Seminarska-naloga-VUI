import statsmodels.api as sm
import pandas as pd
import numpy as np

def run_logistic_regression(df):
    """
    Run logistic regression using statsmodels with full summary.
    """
    features = [
        # Demographics
        "Age_Category",
        "Sex",

        # Lifestyle
        "Smoking_History",
        "Exercise",
        "Alcohol_Consumption",

        # Diet
        "Fruit_Consumption",
        "Green_Vegetables_Consumption",
        "FriedPotato_Consumption",

        # Clinical risk factors
        "BMI",
        "Diabetes",
        "Depression",
        "Arthritis",

        # Healthcare / self-report
        "Checkup",
        "General_Health"
    ]

    X = df[features]
    y = df["Heart_Disease"]

    X = X.dropna()
    y = y.loc[X.index]

    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False)

    summary_df = pd.DataFrame({
        "Feature": model.params.index,
        "Coefficient": model.params.values,
        "Std_Error": model.bse.values,
        "Odds_Ratio": np.exp(model.params.values),
        "CI_2.5%": np.exp(model.conf_int()[0]),
        "CI_97.5%": np.exp(model.conf_int()[1]),
        "z_value": model.tvalues,
        "p_value": model.pvalues
    }).sort_values("Odds_Ratio", ascending=False)
    
    return model, X, highlight_discardable_vars(summary_df)

def highlight_discardable_vars(summary_df, or_threshold=0.05, z_threshold=2.0, p_threshold=0.05):
    def highlight_row(row):
        color = [''] * len(row)
        
        or_value = row['Odds_Ratio']
        if abs(or_value - 1.0) <= or_threshold:
            color[row.index.get_loc('Feature')] = 'background-color: #ffcccc'
            color[row.index.get_loc('Odds_Ratio')] = 'background-color: #ff4d4d; color:white'  
        
        if abs(row['z_value']) < z_threshold:
            color[row.index.get_loc('z_value')] = 'background-color: #ff4d4d; color:white' 
        if row['p_value'] > p_threshold:
            color[row.index.get_loc('p_value')] = 'background-color: #ff4d4d; color:white'

        return color

    return summary_df.style.apply(highlight_row, axis=1)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def run_xgboost(df, test_size=0.2, random_state=42, n_estimators=200, max_depth=5):
    features = [
        "Age_Category", "Sex", "Smoking_History", "Exercise", "Alcohol_Consumption",
        "Fruit_Consumption", "Green_Vegetables_Consumption", "FriedPotato_Consumption",
        "BMI", "Diabetes", "Depression", "Arthritis", "Checkup", "General_Health"
    ]

    X = df[features].dropna()
    y = df.loc[X.index, "Heart_Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        eval_metric='logloss',
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    return model, importance_df, auc

