# Verification methods and data dictionary

All data files are UTF-8 CSV, with decimal numbers using a full stop. Units for turbidity are NTU. `val` denotes validation. The scripts intentionally accept the fixed released schema rather than silently inferring or dropping columns/rows.

## Mainline prediction data

`data/mainline_predictions.csv` contains 8,985 rows, one per paired side/top observation. It retains exported prediction and reference-label values without retraining or recomputing inference. It includes every retained frame from all 15 groups, not a favorable selection of test rows.

| Field | Meaning |
|---|---|
| `experiment_group` | Recorded experimental-group identifier |
| `frame_index` | Integer frame index within group, 1–599; do not assume this is elapsed time in seconds |
| `split` | Explicit `train`, `val` or `test` assignment |
| `pred_ntu` | Saved predicted turbidity in NTU |
| `target_ntu` | Saved evaluation reference label in NTU; not a claim of an independent instrument measurement at every frame |

Source `timestamp` was renamed `frame_index` to avoid implying a time unit. Source blank split entries were assigned to `train` after checking the actual group manifest. No numeric prediction/target values were changed. Attention weights, interface predictions, initial-condition columns and unrelated fields were omitted because they are not required for the selected metric checks.

`data/group_manifest.csv` contains `experiment_group`, `split`, and `samples`, with 599 samples per group. Validation is `10000NTU_100`; test groups are `1000NTU_150` and `20000NTU_200`. The remaining twelve groups are training groups. The split checker verifies all group assignments, every frame index, unique `(group,frame_index)` pairs and all pairwise group intersections. This is a record-consistency check, not a universal data-leakage assessment.

## Evaluation formulas and historical references

For prediction p and reference y, and N retained records:

```text
MAE  = sum(abs(p-y)) / N
RMSE = sqrt(sum((p-y)^2) / N)
R2   = 1 - sum((p-y)^2) / sum((y-mean(y))^2)
split MAPE = mean(min(abs(p-y)/(y+1)*100, 1000))
group MAPE in the current-method calculation = mean(abs(p-y)/(y+1)*100)
```

The archived convention returns R² = 1 when the target variance is zero; this behavior is retained for consistency, not presented as a generally preferred convention. Targets in this release must be finite and nonnegative. The metrics are not computed with a library's default MAPE definition.

The original overall evaluator used float32 log predictions and labels, clipped log values to [-1,12], converted them using `expm1`, and reduced the resulting tensor values. This package starts from the saved NTU columns and uses Python binary64 arithmetic with `math.fsum`. Historical validation/test values are external anchors, with absolute tolerances of 0.000001 NTU for MAE, 0.00001 NTU for RMSE, 0.000002 percentage points for MAPE, and 0.00000002 for R². These small tolerances address measured arithmetic differences; they are not allowances for changing records or formulas.

`expected_results/metrics_reference.json` distinguishes historical float32 split metrics, archived summary-table anchors, and newly frozen binary64 calculations on the supplied export. New frozen values test release consistency; they are not independent historical observations. Training/all-group recomputations are explicitly distinguished from historical validation/test anchors.

### Two historical group-MAPE conventions

During package preparation, the archived validation/test group-summary MAPE values did not match the current evaluator's `y+1` denominator. They can instead be numerically reproduced, to floating-point precision, with `mean(abs(p-y)/y*100)` on the same saved rows. For example, the validation group yields 16.1964008698% with `y+1`, versus the archived 16.8593324554% reproduced with `y`.

The package preserves this distinction instead of changing source data or widening tolerances. Current-method group values and the separate archived-table diagnostic are labelled with their formulas in the verification output/reference notes. The numerical match identifies a compatible alternative formula; it does not by itself establish which historical code revision generated that table. Historical overall validation/test metrics retain the documented `y+1` convention and pass their separate checks. Do not replace a historical group table with a newly calculated column without reporting the convention.

## Attention–interface coordinate data

`data/attention_distance_samples.csv` contains only nine side-view representative records.

| Field | Meaning |
|---|---|
| `sample_id` | Stable `group_side_frame` identifier |
| `group`, `split` | Experimental group and original data split |
| `phase`, `frame_index` | Representative early/middle/late selection and source frame index |
| `view` | Always `side` for this distance calculation |
| `peak_row` | Zero-based row on the saved 224-pixel attention-map grid, recovered exactly from saved normalized peak row × 223 |
| `image_height` | Attention-map grid height, 224 |
| `interface_position_pixels` | Saved interface-label y coordinate in the cropped ROI |
| `roi_height` | ROI height from the corresponding archived crop configuration |
| `saved_peak_y_norm` | Saved peak row divided by 223, retained for comparison |
| `saved_interface_y_norm` | Saved interface coordinate divided by ROI height, retained for comparison |
| `saved_distance` | Saved dimensionless distance retained as a comparison value |

The implemented distance is the absolute difference of two normalized vertical coordinates:

```text
d = abs(peak_row / (image_height - 1) - interface_position_pixels / roi_height)
  = abs(peak_row / 223 - interface_position_pixels / roi_height)
```

It is not a nearest-point Euclidean distance or an unnormalized pixel distance. The two denominators reflect the original definitions and are intentionally different. The selected side-view ROI heights are 743, 760 and 788 pixels for `1000NTU_150`, `10000NTU_100` and `20000NTU_200`, respectively.

Within each group, the original 599 label records were sorted by integer frame index. Zero-based indices `round((599-1)*f)` for f = 0.1, 0.5, 0.9 give indices 60, 299, 538, hence frame IDs 61, 300, 539. These are representative positions in the frame sequence, not three equally spaced measurements in physical time. The complete label file is not redistributed; source selection was checked during curation, and the released script verifies the selected identities.

Each group mean is an unweighted mean of three distances. The overall mean is an unweighted mean of all nine distances; it is also the mean of the three equally sized group means. Six representatives are test records and three are validation records. Top-view records are excluded. Frozen references are:

| Group | Split | Number | Mean | Rounded |
|---|---|---:|---:|---:|
| `1000NTU_150` | test | 3 | 0.10090591409206402 | 0.101 |
| `10000NTU_100` | val | 3 | 0.07192195735976714 | 0.072 |
| `20000NTU_200` | test | 3 | 0.17504154241879308 | 0.175 |
| All representatives | mixed | 9 | 0.11595647129020809 | 0.116 |

Coordinate and arithmetic comparisons use absolute tolerance 1e-12. The script does not regenerate attention maps, locate peaks anew, establish image-registration consistency, or verify the label-generation algorithm.
