"""Đóng gói lại artifact của run hiện tại sau các kiểm tra ngoài notebook."""

from pathlib import Path
import json
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
EXPORTS = ROOT / "exports"
metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
zip_path = EXPORTS / f"telco_churn_lr_results_{metrics['run_id']}.zip"

paths = [
    ROOT / "README.md",
    ROOT / "PLAN.md",
    ROOT / "config.json",
    ROOT / "requirements.txt",
    ROOT / "requirements-lock.txt",
    ROOT / "data" / "README.md",
]
paths += [path for path in RESULTS.rglob("*") if path.is_file()]
paths += [path for path in REPORTS.rglob("*") if path.is_file()]

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, path.relative_to(ROOT))

with zipfile.ZipFile(zip_path) as archive:
    assert archive.testzip() is None
    assert "PLAN.md" in archive.namelist()
    print(f"Packaged {len(archive.namelist())} files: {zip_path}")
