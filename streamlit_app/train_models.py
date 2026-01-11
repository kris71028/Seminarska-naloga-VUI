\
import argparse
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


def train_and_save_top3(csv_path: str, target: str, out_dir: str = "models/classification", seed: int = 42):
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' ni v CSV.")

    # y to 0/1
    y_raw = df[target]
    if y_raw.dtype == "object":
        y = y_raw.astype(str).str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
        if y.isna().any():
            y = pd.Series(pd.factorize(y_raw)[0], index=df.index)
    else:
        y = y_raw.astype(int)

    X = df.drop(columns=[target]).copy()

    cat_cols = [c for c in X.columns if (X[c].dtype == "object" or str(X[c].dtype).startswith("category") or X[c].dtype == "bool")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop"
    )

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=14, class_weight="balanced",
            n_jobs=-1, random_state=seed
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, max_depth=16, class_weight="balanced",
            n_jobs=-1, random_state=seed
        ),
        "LogisticRegression": LogisticRegression(
            solver="lbfgs", penalty="l2", C=0.8, max_iter=500,
            class_weight="balanced", n_jobs=-1
        )
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, est in models.items():
        pipe = Pipeline([("preprocess", pre), ("model", est)])
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, out / f"{name}.joblib")

    print("Saved classification models to:", out.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="Heart_Disease")
    ap.add_argument("--out_dir", default="models/classification")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_and_save_top3(args.csv, args.target, args.out_dir, args.seed)
