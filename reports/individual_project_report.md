# Data-Driven Manufacturing Intelligence for Steel Corporation Pvt. Ltd.
## Individual Data Analytics Project

**Programme:** Master of Global Business  
**Domain:** Manufacturing and Production Analytics  
**Dataset:** UCI Steel Plates Faults

## Abstract
This project applies descriptive, diagnostic and predictive analytics to a public steel-plate surface-defect dataset. The business problem is how a steel manufacturer can use recorded defect characteristics to support quality inspection, identify recurring fault patterns and prioritise process investigation. The dataset contains 1,941 records, 27 numeric predictors and seven fault categories. The workflow includes data-quality checks, exploratory analysis, six classification algorithms, class-sensitive model evaluation, Random Forest hyperparameter tuning, K-Means clustering and association-rule mining. This is a public benchmark, not internal company data. It contains detected fault cases only and therefore cannot support a plant-wide defect rate, yield, OEE, downtime or employee productivity calculation.

## 1. Introduction
Manufacturing quality affects material utilisation, rework, customer satisfaction and operating cost. Steel plate surface defects differ in geometry, area, luminosity and spatial characteristics. Structured analysis can help quality teams understand recorded fault patterns and assess whether measured descriptors distinguish defect categories.

For Steel Corporation Pvt. Ltd., this project positions analytics as decision support for inspection and process investigation—not as a replacement for engineering judgement.

## 2. Business problem and objectives
**Problem:** Steel manufacturers need consistent methods to classify surface defects and understand patterns in inspection data. Manual inspection may be inconsistent at scale, while counts alone do not explain measurable characteristics of fault categories.

**Primary objective:** Build a reproducible analytics workflow that describes recorded steel-plate faults and evaluates whether measured defect and plate characteristics can predict fault category.

**Specific objectives**
1. Audit and prepare the dataset.
2. Describe the distribution of seven fault categories.
3. Explore relationships among defect geometry, luminosity, thickness and steel grade.
4. Compare classifiers using metrics suitable for imbalanced classes.
5. Tune Random Forest using stratified cross-validation and macro-F1.
6. Segment similar defect records with K-Means.
7. Explore co-occurrence patterns with association rules.
8. Translate findings into qualified management recommendations.

**Research questions:** Which fault types occur most often? How do defect area, luminosity and thickness vary by class? Which model best balances performance across classes? Do clusters reveal distinct record profiles? Which steel-grade/thickness/fault combinations co-occur?

## 3. Dataset and source
The UCI Steel Plates Faults dataset was donated in 2010. It has 1,941 instances, 27 independent variables and seven target classes: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps and Other_Faults.

**Unit of analysis:** one recorded steel-plate fault case. Predictors describe defect geometry/location, luminosity, conveyor and plate characteristics.

**Important:** this is not a complete production history. It does not provide all inspected plates, non-defective plates, timestamps, plant lines, machine IDs, operators, disposition or cost.

## 4. Analytical framework
1. Data understanding: dimensions, types, labels and source documentation.
2. Preparation: standardise field names and Boolean grade flags; check duplicates and missingness.
3. Descriptive analytics: fault counts, proportions and numeric summaries.
4. Diagnostic exploration: compare area, thickness and luminosity by fault; inspect correlations and cross-tabs.
5. Predictive analytics: Logistic Regression, Decision Tree, Random Forest, KNN, SVM and Gradient Boosting.
6. Evaluation: stratified 80/20 train-test split; accuracy, balanced accuracy, weighted precision/recall/F1 and macro-F1.
7. Tuning: Random Forest GridSearchCV with stratified five-fold CV optimising macro-F1.
8. Unsupervised analysis: scaled K-Means and cluster profiles.
9. Association analysis: support, confidence and lift.
10. Business interpretation: inspection and process-investigation recommendations.

## 5. Data cleaning and transformation
The Python pipeline audits row/column counts, duplicates, missing cells and target classes. Steel-grade flags are converted to binary values. The target remains a seven-class categorical variable. Infinite values are treated as missing. Median imputation is included inside model pipelines so imputation is learned from training data only. Standardisation is included for scale-sensitive models. The source documentation reports no missing values; confirm this with the local audit.

## 6. Descriptive findings
The supplied CSV contains the following observed class distribution:

| Fault | Records | Share |
|---|---:|---:|
| Other_Faults | 673 | 34.7% |
| Bumps | 402 | 20.7% |
| K_Scatch | 391 | 20.1% |
| Z_Scratch | 190 | 9.8% |
| Pastry | 158 | 8.1% |
| Stains | 72 | 3.7% |
| Dirtiness | 55 | 2.8% |

Other_Faults, Bumps and K_Scatch together account for approximately 75.5% of this fault-only sample. This provides a starting point for quality review, but frequency does not equal economic severity: no scrap, rework or customer-claim cost is available.

K_Scatch is associated with the A400 indicator in 390 of 391 records in this sample. This is a sample association, not evidence that grade causes the defect. Grade, thickness and process conditions may be interrelated.

## 7. Predictive analytics
Six classifiers are implemented in `analysis.py`. Stratification preserves class proportions. Macro-F1 gives equal weight to each category; balanced accuracy and per-class recall help identify poor minority-class recognition. Weighted metrics describe performance under the sample class mix.

Run the notebook or Streamlit model comparison to populate the actual performance table, per-class classification report and confusion matrix. Do not report model scores before running the code; results depend on preprocessing, split and hyperparameters.

Random Forest tuning varies tree count, depth, minimum leaf size and feature sampling with five-fold stratified GridSearchCV. Macro-F1 is used to avoid selecting a model based only on majority-class performance.

## 8. Clustering and association rules
K-Means is applied to scaled predictors. The number of clusters is configurable and silhouette score is reported. Cluster profiles include record counts, average defect pixel area, average thickness, average conveyor descriptor and dominant fault class. Clusters are statistical groupings, not actual production shifts or operator groups.

Association rules explore selected combinations of steel grade, thickness bands and fault category. Support, confidence and lift describe co-occurrence; they do not prove causation.

## 9. Management implications
1. Start quality review with the three most frequent categories, then add severity and cost data before deciding resource allocation.
2. Select predictive models using macro-F1, balanced accuracy and per-class recall, not accuracy alone.
3. Keep human review in the loop and validate predictions on recent plant-specific samples before deployment.
4. Investigate the observed K_Scatch/A400 association with process engineers; check grade, thickness, supplier, rolling conditions and inspection setup before drawing conclusions.
5. Improve future data capture: conforming plates, timestamps, line/machine IDs, batch/coil IDs, process settings, operator/shift, disposition, rework/scrap cost and severity.
6. Pilot the model and monitor class-wise drift, false negatives and operator overrides.

## 10. Limitations
- Public benchmark data, not Steel Corporation internal data.
- Fault records only; no inspected-plate denominator.
- No timestamps or sequence IDs for time-series or SPC analysis.
- No cost or disposition fields for severity or financial impact.
- Associations do not demonstrate causation.
- Benchmark model scores do not guarantee performance at another plant or inspection setup.

## 11. Conclusion
The workflow demonstrates how manufacturing analytics can turn defect records into structured evidence. Descriptive analysis identifies common fault categories; classification compares defect-recognition methods; clustering explores recurring profiles; association rules highlight combinations for further investigation. Operational deployment requires representative plant data, process context, conforming plates and cost/severity information.

## References
Buscema, M., Terzi, S., & Tastle, W. (2010). *Steel Plates Faults* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5J88N

UCI Machine Learning Repository. (n.d.). *Steel Plates Faults*. https://archive.ics.uci.edu/dataset/198/steel+plates+faults

## Appendix: execution evidence to add
Insert the audit output, EDA charts, correlation heatmap, model comparison, tuned parameters, confusion matrix, per-class report, cluster profile, silhouette score, top association rules and dashboard screenshots after executing the notebook.
