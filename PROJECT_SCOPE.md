# Project scope and reference alignment

**Steel Corporation Pvt. Ltd. — Manufacturing & Production Analytics**

This Streamlit application explores steel-plate fault patterns and measurement variation to demonstrate how analytics can support quality investigation and production decision-making.

## Uploaded references

- `references/dashboard-reference.pdf`: visual direction for executive overview, filters, KPI cards, charts and insights. Its survey data is explicitly synthetic, not actual company data.
- `references/data-analytics-guide.html`: conceptual reference for supervised/unsupervised learning, classification, clustering and model evaluation.
- `references/project-based-learning-brief.docx`: academic requirements for cleaning, EDA, correlation, classification evaluation, clustering or regression, and report/presentation.

## Dataset

Primary benchmark: UCI Steel Plates Faults — https://archive.ics.uci.edu/dataset/198/steel+plates+faults

This is public benchmark data, not verified production data from Steel Corporation Pvt. Ltd. Synthetic fallback records must remain clearly labelled.

## Workflow

1. Load UCI benchmark, a user-uploaded CSV, or synthetic demonstration data.
2. Clean headers, remove exact duplicates, coerce numeric fields where appropriate and inspect missingness.
3. Perform descriptive and diagnostic EDA with distributions, grouped comparisons and correlations.
4. Compare classifiers with accuracy, macro precision, macro recall, macro F1 and confusion matrices.
5. Apply K-Means clustering and interpret profiles cautiously.
6. Present operational implications as hypotheses, not causal findings.

## Limitations

The benchmark does not represent company-specific conditions. The dominant-fault target derived from multi-label flags is a simplifying assumption. Correlation does not prove causation; clusters are not verified operating modes; hold-out metrics do not guarantee future performance. Operational use requires validated internal data, domain review and assessment of false-positive/false-negative costs.
