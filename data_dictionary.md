# Data dictionary

The dataset contains 1,941 records and 28 columns: 27 predictors plus target `Fault`.

| Field | Meaning |
|---|---|
| X_Minimum, X_Maximum | Horizontal bounding coordinates of the defect |
| Y_Minimum, Y_Maximum | Vertical bounding coordinates of the defect |
| Pixels_Areas | Defect area in image pixels |
| X_Perimeter, Y_Perimeter | Perimeter measurements along image axes |
| Sum_of_Luminosity | Sum of pixel luminosity in the defect |
| Minimum_of_Luminosity, Maximum_of_Luminosity | Minimum and maximum pixel luminosity |
| Length_of_Conveyer | Conveyor length descriptor |
| TypeOfSteel_A300, TypeOfSteel_A400 | Binary steel-grade indicators |
| Steel_Plate_Thickness | Plate thickness measurement |
| Edges_Index, Empty_Index, Square_Index | Shape descriptors |
| Outside_X_Index, Edges_X_Index, Edges_Y_Index, Outside_Global_Index | Spatial and edge descriptors |
| LogOfAreas, Log_X_Index, Log_Y_Index | Log-transformed geometry |
| Orientation_Index | Orientation descriptor |
| Luminosity_Index | Relative luminosity descriptor |
| SigmoidOfAreas | Nonlinear area descriptor |
| Fault | Seven-class target: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults |

## Cleaning
- Audit dimensions, types, duplicate rows, missing cells and target labels.
- Convert Boolean grade flags to 0/1 for modelling.
- Keep `Fault` as the categorical target.
- Use median imputation and scaling inside modelling pipelines to avoid test leakage.
- Keep the committed CSV in source units; do not overwrite it with scaled values.

## Scope limitations
This is a fault-case dataset, not a complete production log. It does not contain all inspected plates, timestamps, line/machine/operator IDs, disposition, scrap costs or production totals. Do not infer defect rates, yield, OEE, downtime or productivity.
