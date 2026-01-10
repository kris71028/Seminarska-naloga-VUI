\
import os
from pathlib import Path
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Step 4: Simulacija in optimizacija", layout="wide")

CLASS_TARGET = "Heart_Disease"
REG_TARGET = "BMI"

TOP3_CLASS = ["RandomForest", "ExtraTrees", "LogisticRegression"]
TOP3_REG = ["Ridge", "GradientBoosting", "XGBoost"]

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Regression preprocessing (BMI)
# -----------------------------
AGE_MAP = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
    "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
}
HEALTH_MAP = {"Poor": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5}
CHECKUP_MAP = {
    "Never": 0,
    "5 or more years ago": 1,
    "Within the past 5 years": 2,
    "Within the past 2 years": 3,
    "Within the past year": 4
}
DIABETES_MAP = {
    "No": 0,
    "No, pre-diabetes or borderline diabetes": 1,
    "Yes, but female told only during pregnancy": 2,
    "Yes": 3
}
SEX_MAP = {"Female": 0, "Male": 1}
YESNO_MAP = {"No": 0, "Yes": 1, "False": 0, "True": 1, "0": 0, "1": 1}

# Default regression feature list (matches the error you saw)
REG_DEFAULT_FEATURES = [
    "Age_Category", "Sex", "Exercise", "Diabetes", "Arthritis", "General_Health",
    "FriedPotato_Consumption", "Depression"
]


def preprocess_regression_X(X: pd.DataFrame) -> pd.DataFrame:
    """Map categorical strings to numeric codes to match regression models."""
    X = X.copy()

    def map_col(col, mapper):
        if col in X.columns:
            X[col] = X[col].astype(str).map(mapper)

    def map_yesno(col):
        if col in X.columns:
            X[col] = X[col].astype(str).map(YESNO_MAP)

    map_col("Age_Category", AGE_MAP)
    map_col("General_Health", HEALTH_MAP)
    map_col("Checkup", CHECKUP_MAP)
    map_col("Diabetes", DIABETES_MAP)
    map_col("Sex", SEX_MAP)

    for c in ["Exercise", "Smoking_History", "Heart_Disease", "Skin_Cancer", "Other_Cancer", "Depression", "Arthritis"]:
        map_yesno(c)

    # Ensure numeric types where possible
    for c in X.columns:
        if X[c].dtype == "object":
            # try convert remaining objects to numeric
            X[c] = pd.to_numeric(X[c], errors="ignore")

    return X


def get_expected_reg_features(model, df: pd.DataFrame) -> list[str]:
    """
    Get the exact feature names the regression model was fitted with.
    This avoids 'feature_names mismatch' errors.
    """
    # sklearn estimators and pipelines often expose feature_names_in_
    if hasattr(model, "feature_names_in_"):
        feats = list(model.feature_names_in_)
        return [f for f in feats if f in df.columns]

    # some pipelines expose it too
    if hasattr(model, "named_steps") and hasattr(model, "feature_names_in_"):
        feats = list(model.feature_names_in_)
        return [f for f in feats if f in df.columns]

    # xgboost wrapper / booster
    if hasattr(model, "get_booster"):
        try:
            feats = model.get_booster().feature_names
            if feats:
                return [f for f in feats if f in df.columns]
        except Exception:
            pass

    # fallback: what we know from your training (and error message)
    return [f for f in REG_DEFAULT_FEATURES if f in df.columns]


# -----------------------------
# Helpers
# -----------------------------
def load_model_from_bytes(uploaded_file):
    data = uploaded_file.getvalue()
    return joblib.load(BytesIO(data))


def ensure_dataset(uploaded_csv, csv_path_text):
    if uploaded_csv is not None:
        return pd.read_csv(uploaded_csv)
    p = Path(csv_path_text)
    if p.exists():
        return pd.read_csv(p)
    return None


def scan_models(task: str):
    folder = MODELS_DIR / task
    folder.mkdir(parents=True, exist_ok=True)
    files = {p.stem: p for p in folder.glob("*.joblib")}
    return files


def infer_feature_types(df: pd.DataFrame, features: list[str]):
    X = df[features]
    cat_cols = [c for c in X.columns if (X[c].dtype == "object" or str(X[c].dtype).startswith("category") or X[c].dtype == "bool")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def single_input_ui(df: pd.DataFrame, features: list[str], num_cols: list[str], cat_cols: list[str], ncols=3):
    row = {}
    cols = st.columns(ncols)

    num_stats = {}
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            num_stats[c] = (float(s.min()), float(s.max()), float(s.median()))
        else:
            num_stats[c] = (0.0, 1.0, 0.0)

    cat_values = {}
    for c in cat_cols:
        vals = sorted(df[c].dropna().astype(str).unique().tolist())
        if len(vals) == 0:
            vals = [""]
        cat_values[c] = vals

    i = 0
    for f in features:
        with cols[i % ncols]:
            if f in num_cols:
                mn, mx, med = num_stats[f]
                if mn == mx:
                    row[f] = st.number_input(f, value=float(mn))
                else:
                    row[f] = st.slider(f, min_value=float(mn), max_value=float(mx), value=float(med))
            else:
                row[f] = st.selectbox(f, options=cat_values.get(f, [""]), index=0)
        i += 1
    return row


def predict_classification(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        p1 = float(model.predict_proba(X)[0, 1])
        pred = int(p1 >= 0.5)
        return pred, p1
    pred = int(model.predict(X)[0])
    return pred, float("nan")


def predict_regression(model, X: pd.DataFrame):
    yhat = float(model.predict(X)[0])
    return yhat


def simulate_regression(model, base_row: dict, feature: str, grid_vals, reg_features: list[str]):
    rows = []
    for v in grid_vals:
        r = dict(base_row)
        r[feature] = v
        rows.append(r)
    Xs = pd.DataFrame(rows)
    Xs_pp = preprocess_regression_X(Xs)
    Xs_pp = Xs_pp[reg_features]
    y = model.predict(Xs_pp)
    return pd.DataFrame({feature: grid_vals, "yhat": y})


def simulate_classification(model, base_row: dict, feature: str, grid_vals):
    rows = []
    for v in grid_vals:
        r = dict(base_row)
        r[feature] = v
        rows.append(r)
    Xs = pd.DataFrame(rows)
    if hasattr(model, "predict_proba"):
        y = model.predict_proba(Xs)[:, 1]
        return pd.DataFrame({feature: grid_vals, "proba_1": y})
    y = model.predict(Xs)
    return pd.DataFrame({feature: grid_vals, "score": y})


# -----------------------------
# UI
# -----------------------------
st.title("Step 4 – Interaktivna aplikacija (Simulacija in optimizacija)")

with st.sidebar:
    st.header("Nastavitve")

    task = st.radio(
        "Izberi primer",
        ["classification", "regression"],
        format_func=lambda x: "Klasifikacija (Heart_Disease)" if x == "classification" else "Regresija (BMI)"
    )

    st.subheader("Podatki")
    uploaded_csv = st.file_uploader("Naloži očiščen CSV", type=["csv"])
    csv_path = st.text_input("...ali pot do CSV", value="CVD_cleaned.csv")

    df = ensure_dataset(uploaded_csv, csv_path)
    if df is None:
        st.info("Najprej naloži CSV ali nastavi pravilno pot do datoteke.")
        st.stop()

    target = CLASS_TARGET if task == "classification" else REG_TARGET
    if target not in df.columns:
        st.error(f"V CSV ni stolpca '{target}'.")
        st.stop()

    st.subheader("Modeli (TOP 3)")
    model_files = scan_models(task)

    # Upload model (optional)
    up = st.file_uploader("Naloži model (.joblib) – opcijsko", type=["joblib"], key=f"up_{task}")
    if up is not None:
        out = MODELS_DIR / task / up.name
        out.write_bytes(up.getvalue())
        st.success(f"Shranjeno: {out}")
        model_files = scan_models(task)

    # Classification: train if missing
    if task == "classification":
        missing = [m for m in TOP3_CLASS if m not in model_files]
        if missing:
            st.warning("Klasifikacijski modeli niso najdeni. Lahko jih naučiš in shraniš (gumb).")

        if missing and st.button("Nauči & shrani TOP3 klasifikacijske modele"):
            from sklearn.model_selection import train_test_split
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
            from sklearn.linear_model import LogisticRegression

            y_raw = df[CLASS_TARGET]
            y = y_raw.astype(str).str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
            if y.isna().any():
                y = pd.Series(pd.factorize(y_raw)[0], index=df.index)

            features_cls = [c for c in df.columns if c != CLASS_TARGET]
            X = df[features_cls].copy()

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

            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            models = {
                "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=14, class_weight="balanced", n_jobs=-1, random_state=42),
                "ExtraTrees": ExtraTreesClassifier(n_estimators=200, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=42),
                "LogisticRegression": LogisticRegression(solver="lbfgs", penalty="l2", C=0.8, max_iter=500, class_weight="balanced", n_jobs=-1),
            }

            outdir = MODELS_DIR / "classification"
            outdir.mkdir(parents=True, exist_ok=True)
            for name, est in models.items():
                pipe = Pipeline([("preprocess", pre), ("model", est)])
                pipe.fit(X_train, y_train)
                joblib.dump(pipe, outdir / f"{name}.joblib")

            st.success("Klasifikacijski modeli so shranjeni v models/classification ✅")
            model_files = scan_models(task)

        available = [m for m in TOP3_CLASS if m in model_files]
        if not available:
            st.error("Ni najdenih TOP3 klasifikacijskih modelov. Dodaj/ustvari modele in poskusi znova.")
            st.stop()

        model_name = st.selectbox("Izberi model", options=available)
        model_path = model_files[model_name]
    else:
        # regression
        available = [m for m in TOP3_REG if m in model_files]
        if not available:
            st.error("Ni najdenih TOP3 regression modelov. Preveri models/regression.")
            st.stop()

        model_name = st.selectbox("Izberi model", options=available)
        model_path = model_files[model_name]

    st.caption(f"Datoteka: {model_path}")

    try:
        model = joblib.load(model_path)
        st.success("Model naložen ✅")
    except Exception as e:
        st.error("Model se ni naložil (verzije paketov se ne ujemajo). Namesti requirements.txt in poskusi znova.")
        st.exception(e)
        st.stop()

    # Determine features AFTER model load (fixes feature mismatch)
    if task == "classification":
        features = [c for c in df.columns if c != CLASS_TARGET]
        top3 = TOP3_CLASS
        reg_features = None
    else:
        reg_features = get_expected_reg_features(model, df)
        if len(reg_features) == 0:
            st.error("Ne morem določiti regression featurejev iz modela. (feature_names_in_ manjkajo)")
            st.stop()
        features = reg_features
        top3 = TOP3_REG
        st.caption(f"Regression featureji (iz modela): {', '.join(reg_features)}")

    num_cols, cat_cols = infer_feature_types(df, features)

tab1, tab2, tab3 = st.tabs(["4.1 Izbira modela", "4.2 Napoved", "4.3 Simulacija sprememb"])

with tab1:
    st.subheader("4.1 Izbira modela")
    st.write(f"**Primer:** {'Klasifikacija (Heart_Disease)' if task=='classification' else 'Regresija (BMI)'}")
    st.write(f"**Izbran TOP3 model:** `{model_name}`")

with tab2:
    st.subheader("4.2 Napoved za posamezen ali skupinski vzorec")

    st.markdown("### Posamezen vzorec (ročni vnos)")
    base_row = single_input_ui(df, features, num_cols, cat_cols, ncols=3)

    if st.button("Izračunaj napoved"):
        X_one = pd.DataFrame([base_row])

        if task == "regression":
            X_one_pp = preprocess_regression_X(X_one)[reg_features]
            yhat = predict_regression(model, X_one_pp)
            st.write(f"**Napoved BMI (Ŷ):** `{yhat:.3f}`")
        else:
            pred, p1 = predict_classification(model, X_one)
            st.write(f"**Napoved razreda (0/1):** `{pred}`")
            if not np.isnan(p1):
                st.write(f"**P(Heart_Disease=Yes):** `{p1:.3f}`")
                fig = plt.figure()
                plt.bar(["P(No)", "P(Yes)"], [1 - p1, p1])
                plt.ylim(0, 1)
                plt.title("Verjetnosti razredov")
                st.pyplot(fig, clear_figure=True)

    st.markdown("---")
    st.markdown("### Skupinski vzorec (CSV upload)")
    up_batch = st.file_uploader("Naloži CSV za batch napoved", type=["csv"], key=f"batch_{task}")
    if up_batch is not None:
        batch = pd.read_csv(up_batch)

        missing_cols = [c for c in features if c not in batch.columns]
        if missing_cols:
            st.error(f"Manjkajo stolpci: {missing_cols[:20]}{'...' if len(missing_cols)>20 else ''}")
        else:
            Xb = batch[features].copy()
            out = batch.copy()

            if task == "regression":
                Xb_pp = preprocess_regression_X(Xb)[reg_features]
                yhat = model.predict(Xb_pp)
                out["BMI_pred"] = yhat

                st.dataframe(out.head(20), use_container_width=True)
                fig = plt.figure()
                plt.hist(yhat, bins=30)
                plt.title("Porazdelitev napovedi BMI – batch")
                st.pyplot(fig, clear_figure=True)
            else:
                if hasattr(model, "predict_proba"):
                    p1 = model.predict_proba(Xb)[:, 1]
                    pred = (p1 >= 0.5).astype(int)
                    out["pred"] = pred
                    out["proba_yes"] = p1

                    st.dataframe(out.head(20), use_container_width=True)
                    fig = plt.figure()
                    plt.hist(p1, bins=30)
                    plt.title("Porazdelitev P(Yes) – batch")
                    st.pyplot(fig, clear_figure=True)
                else:
                    pred = model.predict(Xb)
                    out["pred"] = pred
                    st.dataframe(out.head(20), use_container_width=True)

            st.download_button(
                "Prenesi rezultate (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{task}_{model_name}.csv",
                mime="text/csv"
            )

with tab3:
    st.subheader("4.3 Simulacija sprememb")
    st.caption("Izberi eno neodvisno spremenljivko, spreminjaj njeno vrednost in opazuj vpliv na napoved.")

    sim_feature = st.selectbox("Spremenljivka za simulacijo", options=features)

    # base_row comes from tab2 UI; ensure it's present
    try:
        base_row
    except NameError:
        base_row = {f: (df[f].median() if f in num_cols else str(df[f].dropna().iloc[0])) for f in features}

    if sim_feature in num_cols:
        s = pd.to_numeric(df[sim_feature], errors="coerce")
        mn = float(s.min()) if s.notna().any() else 0.0
        mx = float(s.max()) if s.notna().any() else 1.0
        if mn == mx:
            mn, mx = mn - 1, mx + 1

        start, end = st.slider("Obseg vrednosti", min_value=float(mn), max_value=float(mx), value=(float(mn), float(mx)))
        steps = st.slider("Število korakov", 10, 200, 50)
        grid = np.linspace(start, end, steps).tolist()

        if task == "regression":
            sim = simulate_regression(model, base_row, sim_feature, grid, reg_features)
            fig = plt.figure()
            plt.plot(sim[sim_feature], sim["yhat"])
            plt.title(f"Vpliv {sim_feature} na napoved BMI")
            plt.xlabel(sim_feature)
            plt.ylabel("BMÎ")
            st.pyplot(fig, clear_figure=True)
        else:
            sim = simulate_classification(model, base_row, sim_feature, grid)
            ycol = "proba_1" if "proba_1" in sim.columns else "score"
            fig = plt.figure()
            plt.plot(sim[sim_feature], sim[ycol])
            if ycol == "proba_1":
                plt.ylim(0, 1)
            plt.title(f"Vpliv {sim_feature} na napoved (score)")
            plt.xlabel(sim_feature)
            plt.ylabel(ycol)
            st.pyplot(fig, clear_figure=True)

    else:
        cats = sorted(df[sim_feature].dropna().astype(str).unique().tolist())
        if task == "regression":
            # simulate categorical by using original category strings; preprocessing maps them
            sim = simulate_regression(model, base_row, sim_feature, cats, reg_features)
            fig = plt.figure()
            plt.bar(sim[sim_feature], sim["yhat"])
            plt.title(f"Vpliv {sim_feature} na napoved BMI")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig, clear_figure=True)
        else:
            sim = simulate_classification(model, base_row, sim_feature, cats)
            ycol = "proba_1" if "proba_1" in sim.columns else "score"
            fig = plt.figure()
            plt.bar(sim[sim_feature], sim[ycol])
            if ycol == "proba_1":
                plt.ylim(0, 1)
            plt.title(f"Vpliv {sim_feature} na napoved (score)")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig, clear_figure=True)

    st.download_button(
        "Izvozi simulacijo (CSV)",
        data=sim.to_csv(index=False).encode("utf-8"),
        file_name=f"simulation_{task}_{model_name}_{sim_feature}.csv",
        mime="text/csv"
    )
