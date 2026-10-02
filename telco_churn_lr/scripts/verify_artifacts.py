"""Kiểm tra độc lập các invariant quan trọng từ artifact đã xuất."""

from pathlib import Path
import json
import math
import zipfile

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
selected = json.loads((RESULTS / "selected_model.json").read_text(encoding="utf-8"))
manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
predictions = pd.read_csv(RESULTS / "test_predictions.csv")
split = pd.read_csv(RESULTS / "split_manifest.csv")
cv = pd.read_csv(RESULTS / "cv_results.csv", keep_default_na=False)

assert manifest["execution_environment"] == "local-python"
assert metrics["run_id"] == manifest["run_id"]
assert selected["selected_before_test"] is True
assert selected["threshold"] == 0.5
assert len(split) == 7043
train_ids = set(split.loc[split["split"] == "train", "customerID"])
test_ids = set(split.loc[split["split"] == "test", "customerID"])
assert train_ids.isdisjoint(test_ids)
assert len(train_ids) == 5634 and len(test_ids) == 1409

fold_rows = cv.loc[cv["record_type"] == "fold"]
summary_rows = cv.loc[cv["record_type"] == "summary"]
assert len(fold_rows) == 30 and len(summary_rows) == 6
assert int(pd.to_numeric(fold_rows["convergence_warnings"]).sum()) == 0

counts = predictions["error_group"].value_counts().to_dict()
test_metrics = metrics["test"]["logistic_regression"]
assert counts == {
    "TN": test_metrics["tn"],
    "TP": test_metrics["tp"],
    "FP": test_metrics["fp"],
    "FN": test_metrics["fn"],
}
tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
assert math.isclose(precision, test_metrics["precision_churn"], rel_tol=0, abs_tol=1e-12)
assert math.isclose(recall, test_metrics["recall_churn"], rel_tol=0, abs_tol=1e-12)
assert math.isclose(f1, test_metrics["f1_churn"], rel_tol=0, abs_tol=1e-12)

notebook_path = ROOT / "notebooks" / "telco_churn_lr.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
nbformat.validate(notebook)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in code_cells)
error_outputs = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
assert not error_outputs

zip_path = ROOT / "exports" / f"telco_churn_lr_results_{metrics['run_id']}.zip"
assert zip_path.exists()
with zipfile.ZipFile(zip_path) as archive:
    assert archive.testzip() is None
    members = {name.replace("\\", "/") for name in archive.namelist()}
assert {
    "README.md",
    "results/metrics.json",
    "results/run_manifest.json",
    "reports/report.md",
    "reports/verification.md",
    "reports/colab_vscode_setup.md",
}.issubset(members)

print(json.dumps({
    "run_id": metrics["run_id"],
    "notebook_cells": len(notebook.cells),
    "executed_code_cells": len(code_cells),
    "error_outputs": len(error_outputs),
    "fold_rows": len(fold_rows),
    "test_rows": len(predictions),
    "zip": str(zip_path),
    "status": "PASS",
}, ensure_ascii=False, indent=2))
