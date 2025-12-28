import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_categorical_distributions(df):
    columns = [
        "Heart_Disease",
        "Smoking_History",
        "Exercise",
        "Sex",
        "Diabetes",
        "Depression",
        "Arthritis",
        "General_Health"
    ]

    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        counts = df[col].value_counts(dropna=False).sort_index()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[len(columns):]:
        ax.axis("off")

    plt.suptitle("Distribucije kategoričnih spremenljivk", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_normality_qq(df):
    columns = [
        "BMI",
        "Alcohol_Consumption",
        "Fruit_Consumption",
        "Green_Vegetables_Consumption",
        "Height_(cm)",
        "Weight_(kg)",
        "FriedPotato_Consumption"
    ]

    data = df[columns].dropna()

    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        stats.probplot(data[col], dist="norm", plot=ax)
        ax.set_title(col)

    for ax in axes[len(columns):]:
        ax.axis("off")

    plt.suptitle("Q–Q grafi", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(df):
    columns = [
        "BMI",
        "Alcohol_Consumption",
        "Fruit_Consumption",
        "Green_Vegetables_Consumption",
        "Height_(cm)",
        "Weight_(kg)",
        "FriedPotato_Consumption"
    ]


    data = df[columns].dropna()

    plt.figure(figsize=(10, 5))
    plt.boxplot(
        data.values,
        labels=columns,
        showfliers=True
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title("Distribucije numeričnih spremenljivk")
    plt.tight_layout()
    plt.show()


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def dataset_info(df: pd.DataFrame) -> None:
    print(df.info())

def summary_statistics(df: pd.DataFrame) -> None:
    print(df.describe())


def general_health_to_numeric(df):
    mapping = {
        "Poor": 1,
        "Fair": 2,
        "Good": 3,
        "Very Good": 4,
        "Excellent": 5
    }
    df = df.copy()
    df["General_Health"] = df["General_Health"].map(mapping)
    return df


def age_category_to_numeric(df):
    age_map = {
        "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37,
        "40-44": 42, "45-49": 47, "50-54": 52, "55-59": 57,
        "60-64": 62, "65-69": 67, "70-74": 72,
        "75-79": 77, "80+": 82
    }
    df = df.copy()
    df["Age_Category"] = df["Age_Category"].map(age_map)
    return df


def binary_columns_to_numeric(df):
    df = df.copy()

    binary_cols = [
        "Heart_Disease",
        "Smoking_History",
        "Exercise",
        "Skin_Cancer",
        "Other_Cancer",
        "Depression",
        "Diabetes",
        "Arthritis"
    ]

    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0})

    return df


def checkup_to_numeric(df):
    df = df.copy()
    df["Checkup"] = df["Checkup"].map({
        "Within the past year": 1,
        "Within the past 2 years": 0,
        "Within the past 5 years": 0,
        "5 or more years ago": 0,
        "Never": 0
    })
    return df

def preprocess(df):
    df = general_health_to_numeric(df)
    df = age_category_to_numeric(df)
    df = binary_columns_to_numeric(df)
    df = checkup_to_numeric(df)
    return df




