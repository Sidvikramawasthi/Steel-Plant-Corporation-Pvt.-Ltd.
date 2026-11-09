import io
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             hamming_loss, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Steel Corporation | Production Intelligence", page_icon="🏭",
                   layout="wide", initial_sidebar_state="expanded")

# Friendly Field Guide visual language: calm paper background, teal/coral/violet accents,
# plain-English explanations, and business-first navigation.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#1e2a37; --muted:#66758a; --teal:#138f88; --coral:#e97850; --violet:#786be8; }
html, body, [class*="css"] { font-family:Inter,sans-serif; color:var(--ink); }
.stApp { background:#f5f7fa; }
.block-container { padding-top:1.4rem; max-width:1500px; }
h1,h2,h3 { font-family:Manrope,Inter,sans-serif; letter-spacing:-.03em; }
.hero { padding:1.5rem 1.8rem; border:1px solid #e2e8f0; border-radius:22px;
        background:linear-gradient(120deg,#fff 55%,#e6f4f2); margin-bottom:1.2rem; }
.eyebrow { color:#10817a; font-size:.75rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.hero h1 { font-size:2.4rem; line-height:1.1; margin:.5rem 0; }
.hero p { color:#56657a; max-width:850px; margin-bottom:0; }
.metric { background:white; border:1px solid #e2e8f0; border-radius:16px; padding:1rem 1.1rem; min-height:110px; }
.metric .label { color:#10817a; font-size:.72rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
.metric .value { font-size:1.8rem; font-weight:800; color:#1e2a37; margin:.25rem 0; }
.metric .hint { color:#66758a; font-size:.8rem; }
.note { background:#e6f4f2; border-left:4px solid #138f88; padding:.8rem 1rem; border-radius:9px; }
div[data-testid="stMetric"] { background:white; border:1px solid #e2e8f0; padding:14px 16px; border-radius:14px; }
section[data-testid="stSidebar"] { background:#f0f3f8; }
</style>
""", unsafe_allow_html=True)

FAULTS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
UCI_URL = "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"

@st.cache_data(show_spinner=False)
def load_uci():
    import requests
    response = requests.get(UCI_URL, timeout=30)
    response.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    candidates = [n for n in zf.namelist() if n.lower().endswith((".csv", ".data", ".txt"))]
    # UCI's original file is a headerless whitespace/tabular file; locate the 34-column data.
    for name in candidates:
        raw = zf.read(name)
        for sep in [r"\s+", ",", ";", "\t"]:
            try:
                frame = pd.read_csv(io.BytesIO(raw), sep=sep, header=None, engine="python")
                if frame.shape[1] >= 34 and frame.shape[0] > 100:
                    return frame.iloc[:, :34]
            except Exception:
                pass
    raise ValueError("Could not identify the Steel Plates Faults data file in the UCI archive.")

def prepare_frame(df):
    df = df.copy()
    # UCI dataset contains 27 numeric predictors followed by seven one-vs-rest fault labels.
    if all(c in df.columns for c in FAULTS):
        target_cols = FAULTS
        X = df.drop(columns=target_cols)
        Y = df[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    elif df.shape[1] >= 34:
        X = df.iloc[:, :27].copy()
        X.columns = [f"feature_{i+1:02d}" for i in range(X.shape[1])]
        Y = df.iloc[:, 27:34].copy()
        Y.columns = FAULTS
        Y = Y.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    else:
        raise ValueError("Expected the 27 predictor columns and seven fault-label columns.")
    X = X.apply(pd.to_numeric, errors="coerce")
    # Drop fully empty columns and rows with no usable target.
    X = X.dropna(axis=1, how="all")
    valid = Y.sum(axis=1) > 0
    X, Y = X.loc[valid], Y.loc[valid]
    return X, Y

def metric_card(label, value, hint):
    st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="hint">{hint}</div></div>', unsafe_allow_html=True)

st.sidebar.title("🏭 Steel Corporation")
st.sidebar.caption("Production intelligence workspace")
source = st.sidebar.radio("Data source", ["UCI Steel Plates Faults", "Upload my CSV"])
st.sidebar.divider()
section = st.sidebar.radio("Explore", ["Executive overview", "Data explorer", "Fault prediction", "Production patterns", "Methodology & reference"])

try:
    if source == "Upload my CSV":
        uploaded = st.sidebar.file_uploader("Upload a CSV (27 predictors + 7 fault labels)", type=["csv"])
        if uploaded is None:
            st.info("Upload your Steel Plates Faults CSV to begin. The built-in UCI source is available from the other data-source option.")
            st.stop()
        raw = pd.read_csv(uploaded)
    else:
        st.sidebar.caption("Source: UCI Machine Learning Repository · Dataset 198")
        raw = load_uci()
    X, Y = prepare_frame(raw)
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()

# Reusable, leakage-safe numeric preprocessing.
X = X.replace([np.inf, -np.inf], np.nan)
X = X.loc[:, X.notna().any()]
if X.empty:
    st.error("No usable numeric predictor columns were found.")
    st.stop()
fault_counts = Y.sum().sort_values(ascending=False)
fault_total = int(Y.sum().sum())
rows, cols = X.shape

st.markdown("""
<div class="hero">
  <div class="eyebrow">Steel Corporation Pvt. Ltd. · Manufacturing analytics</div>
  <h1>From quality data to smarter production decisions.</h1>
  <p>Explore steel plate defect patterns, compare predictive models, and discover groups of similar production observations. Every result is calculated from the selected dataset — not invented plant KPIs.</p>
</div>
""", unsafe_allow_html=True)

if section == "Executive overview":
    st.subheader("Executive overview")
    a,b,c,d = st.columns(4)
    with a: metric_card("Production records", f"{rows:,}", "Rows available for analysis")
    with b: metric_card("Process variables", f"{cols}", "Numeric predictor columns")
    with c: metric_card("Fault labels", f"{len(FAULTS)}", "Seven defect categories")
    with d: metric_card("Positive fault tags", f"{fault_total:,}", "Sum across multi-label targets")
    st.write("")
    left,right = st.columns([1.15, .85])
    with left:
        st.markdown("#### Fault-tag distribution")
        chart = fault_counts.rename_axis("Fault category").reset_index(name="Tagged records")
        st.plotly_chart(px.bar(chart, x="Fault category", y="Tagged records", color="Fault category",
                               color_discrete_sequence=["#138f88","#e97850","#786be8","#46b886","#e6a234","#3e8ede","#e66a8c"]),
                        use_container_width=True)
    with right:
        st.markdown("#### Data quality snapshot")
        quality = pd.DataFrame({"Measure":["Missing predictor cells","Duplicate rows","Rows with at least one fault"],
                                "Count":[int(X.isna().sum().sum()), int(raw.duplicated().sum()), int((Y.sum(axis=1)>0).sum())]})
        st.dataframe(quality, hide_index=True, use_container_width=True)
        st.markdown('<div class="note">A fault record can carry more than one positive label. Treat the targets as multi-label outcomes rather than forcing every plate into a single defect class.</div>', unsafe_allow_html=True)
    st.markdown("#### What should a plant manager ask?")
    st.markdown("- Which defect categories account for the largest share of recorded fault tags?\n- Can the available process measurements identify defect patterns early?\n- Do observations form natural operating groups that deserve separate investigation?")
    st.caption("This is a public benchmark dataset, not internal Steel Corporation production data. Findings should be validated against plant records before operational use.")

elif section == "Data explorer":
    st.subheader("Data explorer")
    st.write("Inspect the source records and the seven target labels before modelling.")
    selected = st.multiselect("Fault labels to inspect", FAULTS, default=FAULTS)
    st.dataframe(pd.concat([X.reset_index(drop=True), Y[selected].reset_index(drop=True)], axis=1).head(100),
                 use_container_width=True, height=420)
    st.markdown("#### Predictor distributions")
    feature = st.selectbox("Choose a process variable", X.columns)
    st.plotly_chart(px.histogram(X, x=feature, nbins=40, marginal="box",
                                 color_discrete_sequence=["#138f88"]), use_container_width=True)
    st.markdown("#### Correlation map")
    corr = X.corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, color_continuous_scale="Tealrose", aspect="auto",
                              title="Pearson correlation between numeric predictors"),
                    use_container_width=True)

elif section == "Fault prediction":
    st.subheader("Fault prediction · supervised learning")
    st.write("The guide's supervised-learning logic applies because historical examples include known defect labels. This is a multi-label classification task.")
    model_name = st.selectbox("Choose an algorithm", ["Random Forest", "Decision Tree", "K-Nearest Neighbours"])
    test_size = st.slider("Test data share", 0.15, 0.40, 0.25, 0.05)
    seed = st.number_input("Random seed", 1, 999, 42)
    if st.button("Train and evaluate model", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=int(seed))
        base = {"Random Forest":RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample", random_state=int(seed), n_jobs=-1),
                "Decision Tree":DecisionTreeClassifier(max_depth=12, min_samples_leaf=3, class_weight="balanced", random_state=int(seed)),
                "K-Nearest Neighbours":KNeighborsClassifier(n_neighbors=7, weights="distance")}
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler() if model_name == "K-Nearest Neighbours" else "passthrough"),
                             ("model", MultiOutputClassifier(base[model_name]))])
        with st.spinner("Training model and evaluating holdout records..."):
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1_micro = f1_score(y_test, pred, average="micro", zero_division=0)
        f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
        hamming = hamming_loss(y_test, pred)
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.metric("Exact-match accuracy", f"{acc:.1%}")
        with m2: st.metric("Micro F1", f"{f1_micro:.3f}")
        with m3: st.metric("Macro F1", f"{f1_macro:.3f}")
        with m4: st.metric("Hamming loss", f"{hamming:.3f}")
        st.markdown("#### Fault-wise performance")
        report = classification_report(y_test, pred, target_names=FAULTS, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
        st.caption("Exact-match accuracy requires all seven labels for a record to match. Micro/macro F1 and Hamming loss provide complementary multi-label views.")
    else:
        st.info("Select a model and click **Train and evaluate model** to run a holdout evaluation.")

elif section == "Production patterns":
    st.subheader("Production patterns · unsupervised learning")
    st.write("Clustering does not use the defect labels. It groups records by similarity in their measured process variables.")
    k = st.slider("Number of clusters (K)", 2, 10, 4)
    sample_n = min(3000, len(X))
    sample = X.sample(sample_n, random_state=42) if len(X) > sample_n else X.copy()
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=42))])
    labels = pipe.fit_predict(sample)
    sil = silhouette_score(pipe[:-1].transform(sample), labels) if len(np.unique(labels)) > 1 and len(sample) > k else np.nan
    l,r = st.columns([1,1])
    with l:
        st.metric("Silhouette score", f"{sil:.3f}" if np.isfinite(sil) else "N/A",
                  help="Higher values generally indicate better-separated clusters.")
        st.plotly_chart(px.bar(pd.Series(labels).value_counts().sort_index().rename_axis("Cluster").reset_index(name="Records"),
                               x="Cluster", y="Records", color="Cluster",
                               color_discrete_sequence=["#138f88","#786be8","#e97850","#e6a234"]),
                        use_container_width=True)
    with r:
        st.markdown("#### Defect mix by cluster")
        mix = Y.loc[sample.index].copy()
        mix["Cluster"] = labels
        profile = mix.groupby("Cluster")[FAULTS].mean().T
        st.plotly_chart(px.imshow(profile, labels={"x":"Cluster","y":"Fault","color":"Share"},
                                  color_continuous_scale="Tealrose", aspect="auto"),
                        use_container_width=True)
    st.caption("Cluster profiles are exploratory signals, not proof of root cause. Investigate process settings and operational context before acting.")

else:
    st.subheader("Methodology & reference")
    st.markdown("""
    **Analytics framework adapted from the supplied “Friendly Field Guide to Data Analytics”.**
    
    - **Supervised learning:** known historical fault labels are used to train a model that predicts defect categories.
    - **Classification:** the output is a category/label. This project uses multi-label classification because one record may have multiple fault flags.
    - **Model evaluation:** compare exact-match accuracy, micro F1, macro F1, Hamming loss, and class-wise precision/recall.
    - **Unsupervised learning:** K-Means searches for natural groups without using the fault labels.
    - **Business interpretation:** translate model output into quality-control questions; do not treat correlation or cluster membership as causal evidence.
    
    **Dataset:** UCI Machine Learning Repository, Steel Plates Faults (Dataset 198). The public benchmark contains 27 input variables and seven binary fault indicators.
    
    **Important distinction about the second supplied reference (Claude.pdf):** its four-page executive dashboard explicitly labels its 5,000 respondents as *synthetic demonstration data*, not actual Steel Corporation customers or survey participants. It describes a separate job-search / AI-product-interest survey. Its visual hierarchy and dashboard layout are treated as design inspiration only; its survey figures are not mixed into the steel-plate dataset or presented as plant facts.
    """)
    st.markdown("#### Suggested next steps for a real plant deployment")
    st.markdown("1. Replace the public benchmark with approved, de-identified plant quality and process records.\n2. Add timestamps, line/machine identifiers, grade, shift, and process settings.\n3. Validate predictions on a time-based holdout and monitor drift.\n4. Confirm false-negative cost with quality and production leaders before setting alerts.")

st.divider()
st.caption("Steel Corporation Pvt. Ltd. · Academic analytics prototype · Built with Streamlit, scikit-learn, pandas and Plotly")
