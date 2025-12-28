from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def run_logistic_regression(df):
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

    # Drop rows with missing values
    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X

def show_odds_ratios(model, X):
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0],
        "Odds_Ratio": np.exp(model.coef_[0])
    }).sort_values(by="Odds_Ratio", ascending=False)

    print("\n📊 Odds Ratios ( >1 increases risk, <1 decreases risk )")
    print(coef_df)
