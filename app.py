import streamlit as st
import pandas as pd
import plotly.express as px
from analysis import load_data, clean_data, class_summary, quality_audit, model_comparison, cluster_analysis, association_rules

st.set_page_config(page_title="Steel Corporation | Manufacturing Intelligence", page_icon="🏭", layout="wide")
st.title("Steel Corporation Pvt. Ltd.")
st.caption("Manufacturing Intelligence | Steel plate surface-defect analytics")
st.warning("Research demonstration using the public UCI Steel Plates Faults dataset (1,941 detected-fault records). It is not internal Steel Corporation production data. The dataset contains faulty plates only, so a plant-wide defect rate or yield cannot be calculated.")
df = clean_data(load_data())
summary = class_summary(df)
total = len(df)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Inspected fault records", f"{total:,}")
c2.metric("Fault categories", f"{df['Fault'].nunique()}")
c3.metric("Largest category", summary.iloc[0]["fault_type"])
c4.metric("Largest-category share", f"{summary.iloc[0]['share_pct']:.1f}%")
tab1,tab2,tab3,tab4,tab5 = st.tabs(["Executive overview","Quality & EDA","Predictive models","Process segmentation","Data & methodology"])
with tab1:
    st.subheader("Where should quality-improvement attention start?")
    left,right = st.columns([1.2,1])
    with left:
        st.plotly_chart(px.bar(summary.sort_values("plates",ascending=True), x="plates", y="fault_type", orientation="h",
                               text="plates", title="Recorded fault mix", color="fault_type"), use_container_width=True)
    with right:
        st.dataframe(summary, hide_index=True, use_container_width=True)
    st.info("Prioritisation logic: combine fault frequency with model recall and process context. High frequency alone does not establish high scrap cost or severity; cost and disposition data are not available in this dataset.")
with tab2:
    st.subheader("Exploratory data analysis")
    a,b = st.columns(2)
    with a:
        st.plotly_chart(px.histogram(df, x="Steel_Plate_Thickness", color="Fault", marginal="box",
                                     title="Plate thickness by recorded fault"), use_container_width=True)
    with b:
        st.plotly_chart(px.box(df, x="Fault", y="Pixels_Areas", color="Fault",
                               title="Defect pixel area by fault category"), use_container_width=True)
    st.plotly_chart(px.scatter(df, x="Pixels_Areas", y="Sum_of_Luminosity", color="Fault",
                               hover_data=["Steel_Plate_Thickness","Length_of_Conveyer"],
                               title="Defect area and luminosity"), use_container_width=True)
    st.subheader("Data-quality audit")
    st.dataframe(quality_audit(df), hide_index=True, use_container_width=True)
with tab3:
    st.subheader("Seven-class defect prediction")
    st.write("Models use a stratified 80/20 train-test split. Report macro-F1 and balanced accuracy alongside accuracy because fault classes are imbalanced.")
    if st.button("Run model comparison and hyperparameter tuning", type="primary"):
        with st.spinner("Training six classifiers and tuning the Random Forest..."):
            scores, reports, fitted, tuned = model_comparison(df, tune=True)
        st.session_state["scores"] = scores
        st.session_state["tuned"] = tuned
    if "scores" in st.session_state:
        st.dataframe(st.session_state["scores"].style.format(precision=3), use_container_width=True)
        st.json(st.session_state["tuned"])
    else:
        st.caption("Select the button to train models. Results are calculated from the dataset at runtime.")
with tab4:
    st.subheader("Unsupervised plate-pattern segmentation")
    k = st.slider("Number of clusters", 2, 8, 4)
    clustered, profile, silhouette = cluster_analysis(df, k)
    st.metric("Silhouette score", f"{silhouette:.3f}")
    st.dataframe(profile, use_container_width=True)
    st.caption("Clusters summarise similarity in measured defect geometry and plate/process descriptors. They are not production shifts or operator groups.")
    rules = association_rules(df)
    st.subheader("Association rules")
    if len(rules):
        st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]].head(15).style.format(precision=3), use_container_width=True)
    else:
        st.info("No rules met the configured support and confidence thresholds.")
with tab5:
    st.subheader("Dataset, scope and limitations")
    st.markdown("""
    **Source:** Buscema, Terzi and Tastle (2010), UCI Machine Learning Repository, Steel Plates Faults, DOI: 10.24432/C5J88N.

    **Observational unit:** one recorded steel-plate fault case. The file contains 1,941 rows, 27 numeric predictor fields and a seven-category fault label.

    **Business objective:** use defect-pattern analytics to support inspection prioritisation, defect recognition and process-investigation decisions at a steel manufacturer.

    **Important limitations:** this is a historical public benchmark dataset, not company-owned data; it contains detected faults rather than all inspected plates; it has no timestamps, batch IDs, operator IDs, machine IDs, costs, scrap/rework disposition or non-defective production denominator. Do not infer causal process drivers, plant defect rate, OEE, downtime or workforce performance from it.
    """)
    st.download_button("Download analysed data", df.to_csv(index=False).encode("utf-8"), "steel_plates_faults_clean.csv", "text/csv")
