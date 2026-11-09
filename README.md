# Steel Corporation Pvt. Ltd. — Production Analytics

Streamlit prototype using the UCI Steel Plates Faults benchmark.

## Features
- Executive overview, fault distribution and data quality
- Data explorer, predictor distributions and correlations
- Multi-label classification: Random Forest, Decision Tree and K-Nearest Neighbours
- Exact-match accuracy, micro/macro F1, Hamming loss and class-level metrics
- K-Means clustering with silhouette score and fault profiles
- Methodology aligned with the supplied Friendly Field Guide to Data Analytics

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app downloads UCI dataset 198, or accepts a CSV with 27 numeric predictors and seven binary target columns: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults.

## Supplied reference analysis
- The supplied HTML *Friendly Field Guide to Data Analytics* is used as the conceptual and visual reference. The application follows its supervised/unsupervised learning structure.
- The supplied four-page PDF dashboard is a separate synthetic survey demonstration. Its source caveat states that the 5,000 respondents are not actual Steel Corporation customers or real survey participants. Key aggregate values and the caveat are transcribed in `data/survey_reference_summary.json`; the interpretation is documented in `docs/analytics-methodology-reference.md`.

**Important:** The PDF and HTML originals have not been committed as binary/source attachments in this repository. The PDF's selected text-based summary and methodology notes are included. The survey data is not steel-plant production data and is not merged with UCI fault records.

Dataset: https://archive.ics.uci.edu/dataset/198/steel+plates+faults
