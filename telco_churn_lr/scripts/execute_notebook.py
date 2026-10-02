"""Chạy notebook tuần tự bằng kernel của môi trường hiện tại và lưu output."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "telco_churn_lr.ipynb"

nb = nbformat.read(NOTEBOOK, as_version=4)
client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
    allow_errors=False,
)
client.execute()
nbformat.write(nb, NOTEBOOK)
print(f"Executed and saved: {NOTEBOOK}")
