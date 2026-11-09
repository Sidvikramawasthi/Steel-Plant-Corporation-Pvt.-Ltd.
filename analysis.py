"""Reproducible analysis pipeline for Steel Corporation Pvt. Ltd.
Dataset: UCI Steel Plates Faults (1941 observations; 27 numeric predictors; 7 fault classes).
All findings are observational; this dataset contains detected faults only, not production denominators.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, balanced_accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "steel_plates_faults.csv"
TARGET = "Fault"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Standardise field names and boolean steel-grade flags.
    df.columns = [c.strip() for c in df.columns]
    for c in ["TypeOfSteel_A300", "TypeOfSteel_A400"]:
        if c in df:
            df[c] = df[c].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)
    return df

def clean_data(df):
    out = df.copy()
    out = out.drop_duplicates()
    out[TARGET] = out[TARGET].astype(str).str.strip().str.replace(" ", "_")
    out = out.replace([np.inf, -np.inf], np.nan)
    # Dataset source reports no missing values; median imputation is retained as a
    # robust pipeline safeguard and is fitted only on training data for modelling.
    return out

def quality_audit(df):
    return pd.DataFrame({
        "check": ["rows", "columns", "duplicate_rows", "missing_cells", "fault_classes",
                  "non_numeric_predictor_columns"],
        "result": [len(df), df.shape[1], int(df.duplicated().sum()), int(df.isna().sum().sum()),
                   df[TARGET].nunique(), int((~df.drop(columns=[TARGET]).dtypes.apply(
                       lambda x: pd.api.types.is_numeric_dtype(x))).sum())]
    })

def class_summary(df):
    s = df[TARGET].value_counts().rename_axis("fault_type").reset_index(name="plates")
    s["share_pct"] = (s["plates"] / len(df) * 100).round(2)
    return s

def model_comparison(df, tune=True):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "Decision Tree": Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                             random_state=42, n_jobs=-1))]),
        "K-Nearest Neighbours": Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7))]),
        "Support Vector Machine": Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()), ("model", SVC(class_weight="balanced", probability=True,
                                                       random_state=42))]),
        "Gradient Boosting": Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=42))])
    }
    rows, reports, fitted = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({"model": name, "accuracy": accuracy_score(y_test, pred),
                     "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                     "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
                     "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
                     "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
                     "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0)})
        reports[name] = {"report": classification_report(y_test, pred, output_dict=True, zero_division=0),
                         "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(y.unique())).tolist(),
                         "labels": sorted(y.unique())}
        fitted[name] = model
    scores = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    tuning_result = None
    if tune:
        # Tune the strongest tree ensemble on stratified CV; optimise macro-F1 to
        # give minority fault classes a voice in model selection.
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", RandomForestClassifier(class_weight="balanced_subsample",
                                                          random_state=42, n_jobs=-1))])
        grid = {"model__n_estimators": [250, 400],
                "model__max_depth": [None, 12, 20],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.7]}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=0)
        search.fit(X_train, y_train)
        p = search.predict(X_test)
        tuning_result = {"best_params": search.best_params_, "cv_macro_f1": search.best_score_,
                         "test_accuracy": accuracy_score(y_test, p),
                         "test_balanced_accuracy": balanced_accuracy_score(y_test, p),
                         "test_macro_f1": f1_score(y_test, p, average="macro"),
                         "test_weighted_f1": f1_score(y_test, p, average="weighted"),
                         "classification_report": classification_report(y_test, p, output_dict=True, zero_division=0)}
    return scores, reports, fitted, tuning_result

def cluster_analysis(df, k=4):
    X = df.drop(columns=[TARGET]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(Xs)
    result = df.copy()
    result["cluster"] = labels
    profile = result.groupby("cluster").agg(
        plates=(TARGET, "size"),
        average_defect_area=("Pixels_Areas", "mean"),
        average_thickness=("Steel_Plate_Thickness", "mean"),
        average_conveyor_length=("Length_of_Conveyer", "mean")
    ).round(2)
    profile["dominant_fault"] = result.groupby("cluster")[TARGET].agg(lambda x: x.value_counts().index[0])
    return result, profile, silhouette_score(Xs, labels)

def association_rules(df):
    # Compact, interpretable transaction set. Rules are descriptive associations,
    # not evidence that steel grade or thickness causes a defect.
    from mlxtend.frequent_patterns import apriori, association_rules as make_rules
    tx = pd.DataFrame(index=df.index)
    tx["Steel_A300"] = df["TypeOfSteel_A300"].astype(bool)
    tx["Steel_A400"] = df["TypeOfSteel_A400"].astype(bool)
    tx["Thin_plate_le_50mm"] = df["Steel_Plate_Thickness"].le(50)
    tx["Mid_plate_51_100mm"] = df["Steel_Plate_Thickness"].between(51, 100)
    tx["Thick_plate_gt_100mm"] = df["Steel_Plate_Thickness"].gt(100)
    for fault in sorted(df[TARGET].unique()):
        tx["Fault_" + fault] = df[TARGET].eq(fault)
    frequent = apriori(tx, min_support=0.02, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame()
    rules = make_rules(frequent, metric="confidence", min_threshold=0.35)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return rules.sort_values(["lift", "confidence"], ascending=False).head(30)

if __name__ == "__main__":
    data = clean_data(load_data())
    print("QUALITY AUDIT\n", quality_audit(data).to_string(index=False))
    print("\nFAULT DISTRIBUTION\n", class_summary(data).to_string(index=False))
    scores, reports, _, tuned = model_comparison(data, tune=False)
    print("\nMODEL COMPARISON\n", scores.to_string(index=False))
    clustered, profile, sil = cluster_analysis(data)
    print("\nCLUSTER PROFILE\n", profile.to_string())
    print("Silhouette score:", round(sil, 4))
    print("\nASSOCIATION RULES\n", association_rules(data).to_string(index=False))
