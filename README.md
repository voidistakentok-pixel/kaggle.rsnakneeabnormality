# RSNA knee MRI notebook

[Download the notebook](rsna_knee_dinov2_nested_cv.ipynb) and import it into Kaggle. Attach the competition data and your prior experiment outputs, enable GPU and Internet, then run all cells. Configuration is in the first cell. Outputs, resumable feature caches, fold assignments, models, metrics, and `submission.csv` are written to `/kaggle/working/`.

The notebook implements frozen DINOv2 Small/Base features, multiple MRI planes, slice-pooling and regularization experiments, study/patient-grouped nested validation, and ensemble selection. Prior artifacts are audited; compatible feature caches are reused. External predictions are compared only when aligned and their provenance is supplied. Reports are excluded from image features.

The official evaluation metric could not be verified from the development environment because Kaggle requests returned HTTP 403. The notebook attempts official metadata discovery on Kaggle. If the metric or dataset schema cannot be established safely, it stops and requests explicit configuration instead of assuming a metric or inventing predictions. Supported built-in metrics are macro AUC, macro log loss, and explicitly weighted log loss; an official custom scorer can also be supplied.

The notebook has passed schema and Python compilation checks. Synthetic execution tests cover training/tuning, saved fold reuse, nested ensembling, bootstrap evaluation, both submission layouts, and single-class training folds. Real MRI feature extraction and Kaggle GPU training have not been executed here. There are no measured competition results or score guarantees.

To rerun the CPU validation:

```sh
python -m pip install -r requirements-validation.txt
python scripts/validate.py
```

Synthetic scores printed by this check are software-test outputs, not competition performance. Validation artifacts go to a temporary directory.

Use Kaggle **Save Version → Run All** and preserve the output dataset to keep caches after a session ends. Reattach prior outputs as inputs for later sessions. A kernel restart cannot preserve files from a destroyed session unless outputs were saved.
