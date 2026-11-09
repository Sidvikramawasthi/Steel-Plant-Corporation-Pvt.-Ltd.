# Steel Corporation Pvt Ltd — AI-Powered Workforce Intelligence Dashboard

> **SYNTHETIC DEMONSTRATION DATA – 5,000 RESPONDENTS.** These are not actual Steel Corporation Pvt Ltd customers or real survey participants.

**Live dashboard:** https://sidvikramawasthi.github.io/Steel-Plant-Corporation-Pvt.-Ltd./

An interactive business-intelligence dashboard built on a 25-question survey of 5,000 synthetic job seekers. It turns survey responses into decisions on target customers, product features, pricing and go-to-market for an AI-powered employment and career-readiness platform.

## Modules
| Group | Modules |
|---|---|
| Command | Executive Overview, CEO Decision Centre |
| Customer & market | Respondent Intelligence (cross-tabs), Employment Analytics, Skill Gap Analysis (heatmap, training matrix), AI Adoption |
| Product & revenue | Product-Market Fit, Revenue Intelligence, Customer Segmentation, NPS & Sentiment |
| Strategy | Strategic Recommendations (Data-to-Decision engine), Market Opportunity, What-if Simulator |
| Analytics lab | Predictive Models, Statistical Tests |
| Data | Data Explorer (search, sort, CSV/Excel export), Data Quality, Settings |

## Analytics methods used
- **Classification:** K-Nearest Neighbours, Decision Tree, Random Forest, Gradient Boosted Trees
- **Model evaluation:** confusion matrix, accuracy, precision, recall, F1, ROC curve and AUC
- **Regression:** Linear, Ridge (L2), Lasso (L1)
- **Clustering:** K-Means (with elbow method), Latent Class Analysis
- **Association rules:** support, confidence, lift
- **Statistical tests:** chi-square, one-way ANOVA, Welch t-test, Pearson correlation, Wilson confidence intervals

## How it works
- Single self-contained `index.html`; no server or install needed.
- The dataset is generated in the browser from a fixed random seed (20261109), so every visit shows the identical 5,000 records.
- Raw (5,012 rows incl. 12 duplicates), cleaned (5,000) and analytical layers are kept separate; missing values, invalid codes and inconsistent answers are flagged, not silently removed.
- Charts: Chart.js 4.4.1 · Excel export: SheetJS 0.18.5 (both loaded from cdnjs).

## Run locally
Download `index.html` and open it in any modern browser (internet connection needed for the chart library and fonts).

## Limitations
- All data is synthetic and illustrative.
- Employer skill demand is *respondent-perceived* (Q12), not an employer survey.
- Stated willingness to pay overstates real purchasing; revenue and market figures are hypothetical projections from editable assumptions.
