# Steel Corporation Pvt. Ltd. — Production Analytics

Streamlit prototype using the UCI Steel Plates Faults benchmark.

## Features
- Executive overview, fault distribution and data quality
- Data explorer, predictor distributions and correlations
- Multi-label classification: Random Forest, Decision Tree and K-Nearest Neighbours
- Exact-match accuracy, micro/macro F1, Hamming loss and class-level metrics
- K-Means clustering with silhouette score and cluster fault profiles
- Methodology aligned with the supplied Friendly Field Guide to Data Analytics

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app downloads UCI dataset 198 or accepts a CSV with 27 numeric predictors and seven binary target columns: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults.

## Supplied references
- `references/data-analytics-guide.html`: supplied Friendly Field Guide, used for conceptual framing and design inspiration.
- `references/survey-dashboard-reference.pdf`: separate four-page synthetic survey dashboard. Its own caveat says the 5,000 respondents are not actual Steel Corporation customers or real survey participants.
- `data/survey_reference_summary.json`: extracted summary values and source caveat.

The survey concerns employment/job-search, AI awareness/adoption, digital product interest and willingness to pay. It is not steel-plant production data and is not merged with UCI fault records.

Dataset: https://archive.ics.uci.edu/dataset/198/steel+plates+faults
