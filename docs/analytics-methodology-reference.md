# Analytics methodology reference

The supplied *Friendly Field Guide to Data Analytics* organizes methods into supervised learning (classification and regression) and unsupervised learning (clustering and association rules). It explains K-Nearest Neighbours, Decision Trees, Random Forest, Gradient Boosted Trees, K-Means, Latent Class and association-rule measures such as support, confidence and lift.

This steel project uses supervised multi-label classification because the UCI dataset supplies seven binary fault indicators. K-Means separately explores process-variable similarity without using fault labels. Correlations and clusters are exploratory signals, not causal evidence.

The supplied PDF is a separate synthetic survey dashboard of 5,000 demonstration respondents. Its source caveat and key figures are recorded in data/survey_reference_summary.json; these are not steel-plant observations.
