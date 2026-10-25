# Model Comparison and Evaluation

## 1. Objective

The objective is to compare the forecasting models and select a model using validation performance, followed by one final evaluation on the test dataset.

## 2. Evaluation Metric

The models were evaluated using mean pinball loss across seven quantiles: q10, q25, q50, q75, q90, q95, and q99. Lower loss indicates better predictive performance.

## 3. Model Comparison Results

| Model | Validation Mean Pinball Loss | Test Mean Pinball Loss |
|---|---:|---:|
| Deep Learning | **1.199770** | 1.174463 |
| LightGBM | 1.210501 | **1.168874** |
| Ridge Linear Regression | 1.501997 | 1.464766 |

## 4. Discussion

Deep Learning achieved the lowest validation loss of 1.199770 and was selected based on validation performance. LightGBM ranked second with a validation loss of 1.210501, while Ridge Linear Regression had the highest loss at 1.501997.

On the final test dataset, LightGBM achieved the lowest loss of 1.168874, followed by Deep Learning at 1.174463. Ridge Linear Regression recorded a test loss of 1.464766.

The test results are reported for final evaluation only. They were not used to change the selected model or tune hyperparameters.

## 5. Conclusion

Deep Learning was selected according to the validation results. LightGBM achieved a slightly lower loss on the held-out test dataset. Both models performed better than Ridge Linear Regression according to mean pinball loss.

The selected model and final test results should be retained for reporting. Further business evaluation could examine forecast accuracy by demand type and the potential effects on inventory planning and stockout reduction.

## 6. Reproducibility

The evaluation script is located at `src/evaluate.py`.

The predictions are stored in:

- `predictions/deep.parquet`
- `predictions/lgbm.parquet`
- `predictions/linear.parquet`

The evaluation was run using:

```bash
python -m src.evaluate
```
