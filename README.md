# Steel Corporation Pvt. Ltd. — Manufacturing Intelligence

**Individual Data Analytics Project | Master of Global Business**

Live dashboard: https://sidvikramawasthi.github.io/Steel-Plant-Corporation-Pvt.-Ltd/

## Objective
Analyse recorded steel-plate surface defects to support quality inspection, defect recognition and process-investigation decisions.

## Dataset
- UCI Machine Learning Repository: Steel Plates Faults (Buscema, Terzi, & Tastle, 2010)
- 1,941 fault records; 27 predictors; seven fault classes
- DOI: https://doi.org/10.24432/C5J88N
- Dataset is public benchmark data, not internal Steel Corporation data.

## Critical limitation
The dataset contains detected fault records only. It has no all-inspected-plate denominator, timestamps, plant/machine/operator identifiers, scrap/rework costs or non-defective plates. Do not infer plant-wide defect rate, yield, OEE, downtime, scrap cost or workforce productivity from this file. Associations are not causation.

## Repository contents
- `index.html`: GitHub Pages dashboard
- `analysis.py`: cleaning, EDA, classification, tuning, clustering and association rules
- `app.py`: Streamlit dashboard
- `data/steel_plates_faults.csv`: dataset
- `notebooks/steel_manufacturing_analytics.ipynb`: reproducible notebook
- `reports/individual_project_report.md`: academic report draft
- `data_dictionary.md`: field guide
- `requirements.txt`: dependencies

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Run analysis: `python analysis.py`. Open notebook with Jupyter.

## Methods
EDA; Logistic Regression, Decision Tree, Random Forest, KNN, SVM and Gradient Boosting; stratified 80/20 split; accuracy, balanced accuracy, weighted precision/recall/F1 and macro-F1; Random Forest GridSearchCV (5-fold stratified, macro-F1); K-Means; silhouette score; association rules (support, confidence, lift).

## Attribution
Buscema, M., Terzi, S., & Tastle, W. (2010). *Steel Plates Faults* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5J88N
