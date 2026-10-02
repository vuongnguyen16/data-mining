"""Sinh notebook chính bằng nbformat.

Script này chỉ là công cụ authoring. Toàn bộ logic phân tích vẫn nằm trực tiếp
trong các cell của notebook được tạo ra.
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "telco_churn_lr.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"name": "telco_churn_lr.ipynb", "provenance": []},
}

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# Phân tích và dự đoán khách hàng rời bỏ dịch vụ viễn thông bằng Logistic Regression

Notebook này là sản phẩm chính của đồ án Data Mining. Mục tiêu là xây dựng một quy trình có thể chạy lại, tránh rò rỉ dữ liệu và trả lời ba câu hỏi:

1. Tỷ lệ churn khác nhau thế nào theo loại hợp đồng và thời gian sử dụng?
2. Logistic Regression phát hiện được bao nhiêu khách hàng churn và tạo ra bao nhiêu cảnh báo nhầm?
3. Những thuộc tính nào có liên hệ đáng chú ý với churn trong mô hình, và chúng gợi ý giả thuyết giữ chân nào cần kiểm chứng?

> **Phạm vi:** đây là đánh giá phân loại trên một bộ dữ liệu mẫu công khai. Kết quả không chứng minh quan hệ nhân quả, không chứng minh hiệu quả chiến dịch và không đại diện cho dự báo churn trong một khoảng thời gian tương lai cụ thể.
""")

md(r"""
## Tóm tắt trạng thái

- Thiết kế được cố định trước khi xem test: split 80/20 có stratify, seed 42, 5-fold CV trên train, ngưỡng 0.5.
- Mô hình chính: Logistic Regression L2; baseline: dự đoán lớp đa số.
- Toàn bộ preprocessing nằm trong `Pipeline` và được fit lại trong từng fold.
- Các kết quả số ở dưới được sinh trực tiếp từ lần chạy hiện tại. Không dùng điểm số từ tài liệu tham khảo.
- Biến `CONFIRM_COLAB_VSCODE` ở phần setup chỉ được bật sau khi người dùng thực sự chọn kernel Colab trong VS Code. Nếu để `False`, manifest sẽ ghi môi trường là local/chưa xác nhận.

### Mục lục

0. Giới thiệu và phương pháp  
1. Setup VS Code–Colab và môi trường  
2. Đọc và kiểm tra dữ liệu  
3. Làm sạch xác định trước và chia train/test  
4. EDA trên train  
5. Pipeline và baseline  
6. Cross-validation và chốt cấu hình  
7. Đánh giá cuối trên test  
8. Phân tích lỗi  
9. Diễn giải hệ số Logistic Regression  
10. Kết luận và giả thuyết hành động  
11. Xuất kết quả và kiểm chứng  
12. Tài liệu tham khảo
""")

md(r"""
## 0. Bối cảnh và phương pháp

### Đơn vị quan sát và nhãn

Mỗi dòng dự kiến là một khách hàng, khóa truy vết là `customerID`. Nhãn `Churn` được ánh xạ `Yes → 1`, `No → 0`; lớp dương luôn là churn.

### Các giả định quan trọng

- Không có chi phí tương đối của false negative/false positive hay ngân sách chăm sóc. Vì vậy mean F1 của lớp churn trên train-CV là tiêu chí chọn chính.
- Nếu mean F1 bằng nhau trong `1e-6`, ưu tiên mean Recall cao hơn, sau đó `C` nhỏ hơn, rồi `class_weight=None`.
- `SeniorCitizen` là categorical. `No internet service` và `No phone service` là trạng thái hợp lệ.
- `TotalCharges` được chuẩn hóa chuỗi và chuyển số trước split; việc điền thiếu, scale và one-hot chỉ được học bên trong pipeline.
- Test không được dùng để chọn biến, cấu hình, cách điền thiếu hoặc ngưỡng.
""")

md(r"""
## 1. Setup VS Code–Colab và môi trường

Trong VS Code, cài extension **Colab** của publisher **Google** (`google.colab`) và Jupyter. Mở notebook rồi chọn **Select Kernel → Colab → Auto Connect**, hoàn tất đăng nhập trên giao diện Google. Không đặt token hoặc thông tin đăng nhập trong cell.

Notebook không giả định file Windows tự xuất hiện trong runtime. Hãy đưa cả thư mục dự án vào runtime (Upload to Colab trong Explorer) hoặc mount Drive, sao cho `config.json` và `data/raw/...csv` cùng nằm dưới một project root. Có thể đặt biến môi trường `TELCO_PROJECT_ROOT`, hoặc để cell dưới tự tìm trong thư mục hiện tại và `/content/telco_churn_lr`.

Nếu vừa kết nối bằng extension chính thức, đặt `CONFIRM_COLAB_VSCODE = True` trước khi chạy toàn bộ. Việc import được `google.colab` một mình không đủ để notebook tự nhận đã chạy từ VS Code–Colab.
""")

code(r"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import platform
import sys

ANALYSIS_CODE_VERSION = "telco-churn-lr-notebook-v1"
RUN_STARTED_AT = datetime.now(timezone.utc)
RUN_ID = RUN_STARTED_AT.strftime("%Y%m%dT%H%M%SZ")

# Chỉ đổi thành True sau khi chính bạn đã chọn kernel Colab trong VS Code.
CONFIRM_COLAB_VSCODE = False

def find_project_root() -> Path:
    candidates = []
    env_root = os.environ.get("TELCO_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([
        Path.cwd(),
        Path.cwd().parent,
        Path("/content/telco_churn_lr"),
    ])
    for candidate in candidates:
        if (candidate / "config.json").exists() and (candidate / "data" / "raw").exists():
            return candidate.resolve()
    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Không tìm thấy project root chứa config.json và data/raw. "
        "Hãy upload thư mục telco_churn_lr, mount Drive hoặc đặt TELCO_PROJECT_ROOT.\n"
        f"Đã thử:\n{searched}"
    )

PROJECT_ROOT = find_project_root()
CONFIG_PATH = PROJECT_ROOT / "config.json"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPORTS_DIR = PROJECT_ROOT / "exports"
for directory in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR, EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG = json.load(file)

def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False

runtime_markers = {
    "google_colab_module_available": module_available("google.colab"),
    "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
    "cwd": str(Path.cwd()),
}
if CONFIRM_COLAB_VSCODE:
    EXECUTION_ENVIRONMENT = "google-colab-vscode-user-confirmed"
elif not runtime_markers["google_colab_module_available"] and not runtime_markers["colab_release_tag"]:
    EXECUTION_ENVIRONMENT = "local-python"
else:
    EXECUTION_ENVIRONMENT = "colab-runtime-vscode-unconfirmed"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_PATH:", DATA_PATH)
print("RUN_ID:", RUN_ID)
print("EXECUTION_ENVIRONMENT:", EXECUTION_ENVIRONMENT)
print("Runtime markers:", runtime_markers)
print("Python:", sys.version.replace("\n", " "))
print("Executable:", sys.executable)
print("Platform:", platform.platform())
print("Config:", json.dumps(CONFIG, ensure_ascii=False, indent=2))
""")

md(r"""
### 1.1. Cài phụ thuộc trong đúng kernel

Colab thường đã có phần lớn thư viện. Chỉ đặt `INSTALL_REQUIREMENTS=True` khi cell import báo thiếu hoặc phiên bản không tương thích. Lệnh dưới dùng IPython `%pip`, nên cài vào đúng kernel notebook; terminal VS Code thông thường không được xem là terminal của Colab.
""")

code(r"""
INSTALL_REQUIREMENTS = False
if INSTALL_REQUIREMENTS:
    requirements_path = PROJECT_ROOT / "requirements.txt"
    if requirements_path.exists():
        get_ipython().run_line_magic("pip", f'install -r "{requirements_path}"')
    else:
        get_ipython().run_line_magic(
            "pip",
            "install numpy pandas scikit-learn matplotlib seaborn nbformat nbclient ipykernel",
        )
    print("Nếu thư viện lõi đã được thay đổi, restart kernel rồi chạy lại từ đầu.")
else:
    print("Không cài lại gói. Đặt INSTALL_REQUIREMENTS=True nếu runtime thiếu phụ thuộc.")
""")

code(r"""
import warnings
import zipfile

import matplotlib
import matplotlib.pyplot as plt
import jinja2
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from IPython.display import Markdown, display
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PACKAGE_VERSIONS = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": sns.__version__,
    "jinja2": jinja2.__version__,
}
print(json.dumps(PACKAGE_VERSIONS, indent=2))

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (9, 5.4),
    "figure.dpi": 110,
    "savefig.dpi": 180,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
COLORS = {"no": "#4C78A8", "yes": "#F28E2B", "negative": "#4C78A8", "positive": "#F28E2B"}
""")

md(r"""
## 2. Đọc và kiểm tra dữ liệu

Audit toàn bộ dữ liệu ở bước này chỉ nhằm xác minh schema, grain, nhãn và chất lượng đầu vào đã định trước. Các phân tích có nhãn phục vụ phát triển nội dung/mô hình ở phần sau chỉ dùng train.
""")

code(r"""
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Thiếu CSV tại {DATA_PATH}. Hãy dùng Upload to Colab hoặc mount Drive, "
        "sau đó giữ đúng cấu trúc data/raw."
    )

data_hash = sha256_file(DATA_PATH)
raw_df = pd.read_csv(DATA_PATH, dtype={"customerID": "string"})
schema_missing = sorted(set(EXPECTED_COLUMNS) - set(raw_df.columns))
schema_extra = sorted(set(raw_df.columns) - set(EXPECTED_COLUMNS))

total_charges_text = raw_df["TotalCharges"].astype("string").str.strip()
total_charges_blank = int(total_charges_text.eq("").sum())
total_charges_numeric = pd.to_numeric(total_charges_text.replace("", pd.NA), errors="coerce")
total_charges_coercion_failures = int(
    (total_charges_text.ne("") & total_charges_numeric.isna()).sum()
)

object_blank_counts = {}
for column in raw_df.select_dtypes(include=["object", "string"]).columns:
    object_blank_counts[column] = int(raw_df[column].astype("string").str.strip().eq("").sum())

data_audit = {
    "data_path": str(DATA_PATH),
    "sha256": data_hash,
    "shape": {"rows": int(raw_df.shape[0]), "columns": int(raw_df.shape[1])},
    "columns": raw_df.columns.tolist(),
    "schema_missing_columns": schema_missing,
    "schema_extra_columns": schema_extra,
    "dtypes_on_read": {column: str(dtype) for column, dtype in raw_df.dtypes.items()},
    "customer_id": {
        "missing": int(raw_df["customerID"].isna().sum()),
        "unique": int(raw_df["customerID"].nunique(dropna=True)),
        "duplicate_rows_by_id": int(raw_df.duplicated("customerID", keep=False).sum()),
    },
    "exact_duplicate_rows": int(raw_df.duplicated().sum()),
    "missing_before_conversion": {column: int(value) for column, value in raw_df.isna().sum().items()},
    "blank_strings": object_blank_counts,
    "target_counts": {str(key): int(value) for key, value in raw_df["Churn"].value_counts(dropna=False).items()},
    "target_unique_non_null": sorted(raw_df["Churn"].dropna().astype(str).unique().tolist()),
    "total_charges": {
        "blank_before_conversion": total_charges_blank,
        "missing_after_conversion": int(total_charges_numeric.isna().sum()),
        "nonblank_coercion_failures": total_charges_coercion_failures,
    },
    "domain_checks": {
        "negative_tenure": int((raw_df["tenure"] < 0).sum()),
        "negative_monthly_charges": int((raw_df["MonthlyCharges"] < 0).sum()),
        "negative_total_charges_after_conversion": int((total_charges_numeric < 0).sum()),
    },
    "categorical_values": {
        column: sorted(raw_df[column].dropna().astype(str).unique().tolist())
        for column in raw_df.columns
        if column not in {"customerID", "tenure", "MonthlyCharges", "TotalCharges"}
    },
}

with (RESULTS_DIR / "data_audit.json").open("w", encoding="utf-8") as file:
    json.dump(data_audit, file, ensure_ascii=False, indent=2)

display(pd.DataFrame({
    "thuộc tính": ["Số dòng", "Số cột", "customerID duy nhất", "Dòng trùng hoàn toàn", "TotalCharges trống", "Lỗi chuyển số không phải chuỗi trống"],
    "giá trị": [
        raw_df.shape[0], raw_df.shape[1], raw_df["customerID"].nunique(),
        raw_df.duplicated().sum(), total_charges_blank, total_charges_coercion_failures,
    ],
}))
display(raw_df.head(3))
display(raw_df["Churn"].value_counts(dropna=False).rename_axis("Churn").to_frame("count"))

assert not schema_missing and not schema_extra, f"Schema lệch: thiếu={schema_missing}, thừa={schema_extra}"
assert raw_df["customerID"].notna().all(), "customerID có missing"
assert raw_df["customerID"].is_unique, "customerID không duy nhất; phải xử lý trước split"
assert raw_df.duplicated().sum() == 0, "Có dòng trùng hoàn toàn; phải xử lý trước split"
assert set(raw_df["Churn"].dropna().unique()) == {"Yes", "No"}, "Nhãn lạ hoặc thiếu"
assert raw_df["Churn"].notna().all(), "Không được tự impute nhãn thiếu"
assert data_audit["domain_checks"]["negative_tenure"] == 0
assert data_audit["domain_checks"]["negative_monthly_charges"] == 0
assert data_audit["domain_checks"]["negative_total_charges_after_conversion"] == 0
print("Audit schema/grain/nhãn đạt các điều kiện chặn. SHA-256:", data_hash)
""")

md(r"""
### Nhận xét audit

Các chuỗi trống ở `TotalCharges` được chuyển thành missing số và sẽ được median imputation **bên trong pipeline**. Không xóa dòng chỉ vì đây là missing hoặc vì một giá trị trông giống outlier. Nếu một assertion ở trên lỗi, dừng quy trình thay vì tự ép dữ liệu cho khớp con số tham khảo.
""")

md(r"""
## 3. Làm sạch xác định trước và chia train/test

Những thao tác không học thống kê gồm strip tên cột/chuỗi `TotalCharges`, chuyển kiểu số và ánh xạ nhãn. Sau đó tạo split cố định 80/20 với stratify. `customerID` chỉ dùng để truy vết, không đi vào feature.
""")

code(r"""
clean_df = raw_df.copy()
clean_df["TotalCharges"] = total_charges_numeric
clean_df["ChurnBinary"] = clean_df["Churn"].map({"No": 0, "Yes": 1})
assert clean_df["ChurnBinary"].notna().all()

all_indices = np.arange(len(clean_df))
y_all = clean_df["ChurnBinary"].astype(int)
train_indices, test_indices = train_test_split(
    all_indices,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["random_state"],
    stratify=y_all,
)

feature_columns = [column for column in EXPECTED_COLUMNS if column not in {"customerID", "Churn"}]
X_all = clean_df[feature_columns]

X_train = X_all.iloc[train_indices].reset_index(drop=True)
X_test = X_all.iloc[test_indices].reset_index(drop=True)
y_train = y_all.iloc[train_indices].reset_index(drop=True)
y_test = y_all.iloc[test_indices].reset_index(drop=True)
id_train = clean_df["customerID"].iloc[train_indices].reset_index(drop=True)
id_test = clean_df["customerID"].iloc[test_indices].reset_index(drop=True)

assert set(id_train).isdisjoint(set(id_test))
assert len(X_train) + len(X_test) == len(clean_df)
assert "customerID" not in X_train.columns and "Churn" not in X_train.columns

split_manifest = pd.concat([
    pd.DataFrame({
        "row_index": train_indices,
        "customerID": id_train,
        "split": "train",
        "ChurnBinary": y_train,
    }),
    pd.DataFrame({
        "row_index": test_indices,
        "customerID": id_test,
        "split": "test",
        "ChurnBinary": y_test,
    }),
], ignore_index=True).sort_values("row_index")
split_manifest.to_csv(RESULTS_DIR / "split_manifest.csv", index=False)

split_summary = (
    split_manifest.groupby(["split", "ChurnBinary"], observed=True)
    .size().rename("n").reset_index()
)
display(split_summary)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("ID giao nhau:", len(set(id_train) & set(id_test)))
""")

md(r"""
## 4. EDA trên train

Tất cả biểu đồ có nhãn dưới đây chỉ dùng train. Mỗi hình có bảng nguồn được lưu tại `results/tables/`; tỷ lệ churn của một nhóm luôn là `số churn / tổng số khách hàng trong nhóm`.
""")

code(r"""
train_eda = X_train.copy()
train_eda.insert(0, "customerID", id_train)
train_eda["ChurnBinary"] = y_train
train_eda["ChurnLabel"] = train_eda["ChurnBinary"].map({0: "Không churn", 1: "Churn"})

churn_distribution = (
    train_eda.groupby(["ChurnBinary", "ChurnLabel"], observed=True)
    .size().rename("n").reset_index()
)
churn_distribution["tỷ_lệ"] = churn_distribution["n"] / len(train_eda)
churn_distribution.to_csv(TABLES_DIR / "eda_churn_distribution_train.csv", index=False)
display(churn_distribution.style.format({"tỷ_lệ": "{:.2%}"}))

fig, ax = plt.subplots()
ax.bar(churn_distribution["ChurnLabel"], churn_distribution["n"], color=[COLORS["no"], COLORS["yes"]])
ax.set(title="Phân bố Churn trên tập train", xlabel="Nhãn", ylabel="Số khách hàng")
ax.set_ylim(0, churn_distribution["n"].max() * 1.16)
for index, row in churn_distribution.iterrows():
    ax.text(index, row["n"] + len(train_eda) * 0.01, f'{int(row["n"]):,}\n({row["tỷ_lệ"]:.1%})', ha="center")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_churn_distribution_train.png", bbox_inches="tight")
plt.show()

train_churn_rate = float(y_train.mean())
display(Markdown(
    f"**Nhận xét.** Tập train có **{int(y_train.sum()):,}/{len(y_train):,}** khách hàng churn "
    f"(**{train_churn_rate:.1%}**). Baseline dự đoán lớp đa số có thể đạt Accuracy khá cao nhưng Recall/F1 "
    "cho churn bằng 0 vì không phát hiện ca dương nào. **Giới hạn:** tỷ lệ lớp chỉ mô tả mẫu train, không tự chứng minh hiệu năng mô hình."
))
""")

code(r"""
contract_table = (
    train_eda.groupby("Contract", observed=True)["ChurnBinary"]
    .agg(n="size", churn_count="sum", churn_rate="mean")
    .reset_index().sort_values("churn_rate", ascending=False)
)
contract_table.to_csv(TABLES_DIR / "eda_contract_churn_train.csv", index=False)
display(contract_table.style.format({"churn_rate": "{:.2%}"}))

fig, ax = plt.subplots()
bars = ax.barh(contract_table["Contract"], contract_table["churn_rate"], color=COLORS["yes"])
ax.invert_yaxis()
ax.set(title="Tỷ lệ churn theo loại hợp đồng — train", xlabel="Tỷ lệ churn trong nhóm", ylabel="Loại hợp đồng")
ax.set_xlim(0, contract_table["churn_rate"].max() * 1.22)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
for bar, (_, row) in zip(bars, contract_table.iterrows()):
    ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2, f'{row["churn_rate"]:.1%} (n={int(row["n"]):,})', va="center")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_contract_churn_train.png", bbox_inches="tight")
plt.show()

highest_contract = contract_table.iloc[0]
lowest_contract = contract_table.iloc[-1]
display(Markdown(
    f"**Nhận xét.** Nhóm `{highest_contract['Contract']}` có tỷ lệ churn cao nhất trong train "
    f"(**{highest_contract['churn_rate']:.1%}**, n={int(highest_contract['n']):,}); nhóm "
    f"`{lowest_contract['Contract']}` thấp nhất (**{lowest_contract['churn_rate']:.1%}**, n={int(lowest_contract['n']):,}). "
    "**Giới hạn:** đây là liên hệ mô tả; loại hợp đồng có thể đi cùng tenure, dịch vụ và cơ chế tự chọn của khách hàng."
))
""")

code(r"""
tenure_group = pd.cut(
    train_eda["tenure"],
    bins=CONFIG["tenure_bins"],
    labels=CONFIG["tenure_labels"],
    include_lowest=True,
    right=True,
)
tenure_table = (
    train_eda.assign(tenure_group=tenure_group)
    .groupby("tenure_group", observed=False)["ChurnBinary"]
    .agg(n="size", churn_count="sum", churn_rate="mean")
    .reset_index()
)
tenure_table.to_csv(TABLES_DIR / "eda_tenure_churn_train.csv", index=False)
display(tenure_table.style.format({"churn_rate": "{:.2%}"}))

fig, ax = plt.subplots()
bars = ax.bar(tenure_table["tenure_group"].astype(str), tenure_table["churn_rate"], color=COLORS["yes"])
ax.set(title="Tỷ lệ churn theo nhóm thời gian sử dụng — train", xlabel="Nhóm tenure", ylabel="Tỷ lệ churn trong nhóm")
ax.set_ylim(0, tenure_table["churn_rate"].max() * 1.17)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
for bar, (_, row) in zip(bars, tenure_table.iterrows()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008, f'{row["churn_rate"]:.1%}\nn={int(row["n"]):,}', ha="center")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_tenure_churn_train.png", bbox_inches="tight")
plt.show()

highest_tenure = tenure_table.sort_values("churn_rate", ascending=False).iloc[0]
lowest_tenure = tenure_table.sort_values("churn_rate", ascending=True).iloc[0]
display(Markdown(
    f"**Nhận xét.** Nhóm `{highest_tenure['tenure_group']}` có tỷ lệ churn cao nhất "
    f"(**{highest_tenure['churn_rate']:.1%}**, n={int(highest_tenure['n']):,}), trong khi nhóm "
    f"`{lowest_tenure['tenure_group']}` là **{lowest_tenure['churn_rate']:.1%}** "
    f"(n={int(lowest_tenure['n']):,}). **Giới hạn:** các nhóm tenure là bin cố định để mô tả; không phải ngưỡng nhân quả hay ngưỡng tối ưu."
))
""")

code(r"""
monthly_table = (
    train_eda.groupby("ChurnLabel", observed=True)["MonthlyCharges"]
    .agg(n="size", mean="mean", std="std", median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75))
    .reset_index()
)
monthly_table.to_csv(TABLES_DIR / "eda_monthly_charges_train.csv", index=False)
display(monthly_table.style.format({column: "{:.2f}" for column in ["mean", "std", "median", "q1", "q3"]}))

fig, ax = plt.subplots()
sns.boxplot(data=train_eda, x="ChurnLabel", y="MonthlyCharges", order=["Không churn", "Churn"], palette=[COLORS["no"], COLORS["yes"]], ax=ax)
ax.set(title="Phân bố MonthlyCharges theo Churn — train", xlabel="Nhãn", ylabel="MonthlyCharges")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_monthly_charges_train.png", bbox_inches="tight")
plt.show()

median_no = float(monthly_table.loc[monthly_table["ChurnLabel"] == "Không churn", "median"].iloc[0])
median_yes = float(monthly_table.loc[monthly_table["ChurnLabel"] == "Churn", "median"].iloc[0])
display(Markdown(
    f"**Nhận xét.** Median `MonthlyCharges` là **{median_yes:.2f}** ở nhóm churn và **{median_no:.2f}** "
    "ở nhóm không churn trong train; boxplot cho thấy cả vị trí trung tâm lẫn độ phân tán. "
    "**Giới hạn:** khác biệt phân bố đơn biến chưa kiểm soát loại dịch vụ, hợp đồng hoặc tenure."
))
""")

md(r"""
## 5. Thiết kế mô hình

Logistic Regression mô hình hóa log-odds của lớp churn như một tổ hợp tuyến tính của feature. Numeric được median-impute rồi chuẩn hóa; categorical được most-frequent-impute rồi one-hot với `drop='first'`. Việc bỏ một mức tạo nhóm tham chiếu để tránh dư thừa tuyến tính và phục vụ diễn giải.

`Pipeline` bao quanh `ColumnTransformer` và mô hình, vì vậy median, scaler và encoder được fit lại trên phần train của từng fold. Baseline `most_frequent` cho biết Accuracy có thể đánh lừa thế nào khi lớp churn ít hơn.
""")

code(r"""
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [column for column in feature_columns if column not in NUMERIC_FEATURES]
assert "SeniorCitizen" in CATEGORICAL_FEATURES

def build_pipeline(C: float, class_weight):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=True)),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        penalty=CONFIG["model"]["penalty"],
        solver=CONFIG["model"]["solver"],
        max_iter=CONFIG["model"]["max_iter"],
        random_state=CONFIG["random_state"],
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])

def positive_scores(estimator, X):
    probabilities = estimator.predict_proba(X)
    classes = list(estimator.classes_ if hasattr(estimator, "classes_") else estimator.named_steps["model"].classes_)
    return probabilities[:, classes.index(1)]

def binary_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_churn": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_churn": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_churn": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }
    assert tn + fp + fn + tp == len(y_true)
    assert np.isclose(result["precision_churn"], tp / (tp + fp) if tp + fp else 0.0)
    assert np.isclose(result["recall_churn"], tp / (tp + fn) if tp + fn else 0.0)
    recomputed_f1 = (
        2 * result["precision_churn"] * result["recall_churn"]
        / (result["precision_churn"] + result["recall_churn"])
        if result["precision_churn"] + result["recall_churn"] else 0.0
    )
    assert np.isclose(result["f1_churn"], recomputed_f1)
    return result

print("Numeric:", NUMERIC_FEATURES)
print("Categorical:", CATEGORICAL_FEATURES)
""")

md(r"""
## 6. Cross-validation trên train và chốt cấu hình

Sáu cấu hình (`C ∈ {0.1, 1, 10}` × `class_weight ∈ {None, balanced}`) dùng đúng cùng 5 fold. Ngưỡng là 0.5 cho mọi cấu hình. Độ lệch chuẩn giữa các fold chỉ mô tả biến thiên fold, không phải khoảng tin cậy 95%.
""")

code(r"""
cv = StratifiedKFold(
    n_splits=CONFIG["cv"]["n_splits"],
    shuffle=CONFIG["cv"]["shuffle"],
    random_state=CONFIG["cv"]["random_state"],
)
fold_indices = list(cv.split(X_train, y_train))
threshold = float(CONFIG["classification_threshold"])
cv_rows = []

for C in CONFIG["model"]["C_grid"]:
    for class_weight in CONFIG["model"]["class_weight_grid"]:
        config_id = f"C={C}|class_weight={class_weight}"
        for fold_number, (fold_train_idx, fold_valid_idx) in enumerate(fold_indices, start=1):
            estimator = build_pipeline(C=C, class_weight=class_weight)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(X_train.iloc[fold_train_idx], y_train.iloc[fold_train_idx])
            convergence_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
            y_score = positive_scores(estimator, X_train.iloc[fold_valid_idx])
            y_pred = (y_score >= threshold).astype(int)
            metrics = binary_metrics(y_train.iloc[fold_valid_idx], y_pred, y_score)
            cv_rows.append({
                "record_type": "fold",
                "config_id": config_id,
                "fold": fold_number,
                "C": float(C),
                "class_weight": "None" if class_weight is None else str(class_weight),
                "threshold": threshold,
                "convergence_warnings": convergence_warnings,
                **metrics,
            })

cv_folds = pd.DataFrame(cv_rows)
metric_columns = ["accuracy", "precision_churn", "recall_churn", "f1_churn", "roc_auc"]
summary_rows = []
for (config_id, C, class_weight), group in cv_folds.groupby(["config_id", "C", "class_weight"], sort=False):
    row = {
        "record_type": "summary",
        "config_id": config_id,
        "fold": pd.NA,
        "C": C,
        "class_weight": class_weight,
        "threshold": threshold,
        "convergence_warnings": int(group["convergence_warnings"].sum()),
    }
    for metric in metric_columns:
        row[f"mean_{metric}"] = float(group[metric].mean())
        row[f"std_{metric}"] = float(group[metric].std(ddof=1))
    summary_rows.append(row)

cv_summary = pd.DataFrame(summary_rows)
cv_results = pd.concat([cv_folds, cv_summary], ignore_index=True, sort=False)
cv_results.to_csv(RESULTS_DIR / "cv_results.csv", index=False)

display(
    cv_summary[["config_id", "mean_f1_churn", "std_f1_churn", "mean_precision_churn", "mean_recall_churn", "mean_roc_auc", "convergence_warnings"]]
    .sort_values("mean_f1_churn", ascending=False)
    .style.format({column: "{:.4f}" for column in ["mean_f1_churn", "std_f1_churn", "mean_precision_churn", "mean_recall_churn", "mean_roc_auc"]})
)
assert cv_folds.shape[0] == 6 * CONFIG["cv"]["n_splits"]
assert cv_folds["convergence_warnings"].sum() == 0, "Có ConvergenceWarning; cần xử lý trước khi mở test"
""")

code(r"""
best_f1 = cv_summary["mean_f1_churn"].max()
tie_tolerance = float(CONFIG["selection"]["tie_tolerance"])
candidates = cv_summary.loc[(best_f1 - cv_summary["mean_f1_churn"]).abs() <= tie_tolerance].copy()
candidates["class_weight_rank"] = candidates["class_weight"].ne("None").astype(int)
candidates = candidates.sort_values(
    ["mean_recall_churn", "C", "class_weight_rank"],
    ascending=[False, True, True],
)
selected_row = candidates.iloc[0]
selected_C = float(selected_row["C"])
selected_class_weight = None if selected_row["class_weight"] == "None" else selected_row["class_weight"]

selected_model_record = {
    "selected_before_test": True,
    "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    "selection_data": "train_only_5_fold_stratified_cv",
    "primary_metric": CONFIG["selection"]["primary_metric"],
    "tie_tolerance": tie_tolerance,
    "tie_breakers": CONFIG["selection"]["tie_breakers"],
    "threshold": threshold,
    "parameters": {
        "C": selected_C,
        "class_weight": selected_class_weight,
        "penalty": CONFIG["model"]["penalty"],
        "solver": CONFIG["model"]["solver"],
        "max_iter": CONFIG["model"]["max_iter"],
    },
    "cv_summary": {
        key: float(selected_row[key])
        for key in ["mean_f1_churn", "std_f1_churn", "mean_precision_churn", "mean_recall_churn", "mean_roc_auc"]
    },
    "data_sha256": data_hash,
    "config_sha256": sha256_file(CONFIG_PATH),
}
with (RESULTS_DIR / "selected_model.json").open("w", encoding="utf-8") as file:
    json.dump(selected_model_record, file, ensure_ascii=False, indent=2)

display(Markdown(
    f"**Đã chốt trước khi mở test:** `C={selected_C}`, `class_weight={selected_class_weight}`, "
    f"threshold={threshold:.1f}. Mean F1 CV = **{selected_row['mean_f1_churn']:.4f}**, "
    f"Recall = **{selected_row['mean_recall_churn']:.4f}**, Precision = **{selected_row['mean_precision_churn']:.4f}**. "
    "Quyết định đã được ghi vào `results/selected_model.json`."
))
""")

md(r"""
## 7. Đánh giá cuối trên test

Từ cell này test mới được dùng. Cấu hình và threshold đã được lưu trước đó; không có vòng lặp quay lại để cải thiện điểm test.
""")

code(r"""
final_pipeline = build_pipeline(C=selected_C, class_weight=selected_class_weight)
with warnings.catch_warnings(record=True) as final_warnings:
    warnings.simplefilter("always", ConvergenceWarning)
    final_pipeline.fit(X_train, y_train)
final_convergence_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in final_warnings)
assert final_convergence_warnings == 0

lr_score = positive_scores(final_pipeline, X_test)
lr_pred = (lr_score >= threshold).astype(int)
lr_metrics = binary_metrics(y_test, lr_pred, lr_score)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_score = positive_scores(baseline, X_test)
baseline_pred = baseline.predict(X_test)
baseline_metrics = binary_metrics(y_test, baseline_pred, baseline_score)

metrics_table = pd.DataFrame([
    {"model": "Dummy most_frequent", **baseline_metrics},
    {"model": "Logistic Regression", **lr_metrics},
])
metrics_table.to_csv(TABLES_DIR / "test_metrics_comparison.csv", index=False)
display(metrics_table[["model", "accuracy", "precision_churn", "recall_churn", "f1_churn", "roc_auc", "tn", "fp", "fn", "tp"]].style.format({
    "accuracy": "{:.4f}", "precision_churn": "{:.4f}", "recall_churn": "{:.4f}", "f1_churn": "{:.4f}", "roc_auc": "{:.4f}",
}))

prediction_df = pd.DataFrame({
    "customerID": id_test,
    "y_true": y_test,
    "y_pred": lr_pred,
    "score_churn": lr_score,
})
prediction_df["error_group"] = np.select(
    [
        (prediction_df["y_true"] == 1) & (prediction_df["y_pred"] == 1),
        (prediction_df["y_true"] == 0) & (prediction_df["y_pred"] == 0),
        (prediction_df["y_true"] == 0) & (prediction_df["y_pred"] == 1),
        (prediction_df["y_true"] == 1) & (prediction_df["y_pred"] == 0),
    ],
    ["TP", "TN", "FP", "FN"],
    default="UNKNOWN",
)
assert not prediction_df["error_group"].eq("UNKNOWN").any()
prediction_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

metrics_payload = {
    "run_id": RUN_ID,
    "run_started_at_utc": RUN_STARTED_AT.isoformat(),
    "execution_environment": EXECUTION_ENVIRONMENT,
    "data": {
        "sha256": data_hash,
        "n_total": int(len(clean_df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
    },
    "selection": selected_model_record,
    "test": {"logistic_regression": lr_metrics, "baseline_most_frequent": baseline_metrics},
    "package_versions": PACKAGE_VERSIONS,
}
with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as file:
    json.dump(metrics_payload, file, ensure_ascii=False, indent=2)

alerts = lr_metrics["tp"] + lr_metrics["fp"]
display(Markdown(
    f"**Kết quả test.** LR phát hiện đúng **{lr_metrics['tp']}/{lr_metrics['tp'] + lr_metrics['fn']}** khách hàng churn "
    f"(Recall **{lr_metrics['recall_churn']:.1%}**). Mô hình tạo **{alerts}** cảnh báo, trong đó "
    f"**{lr_metrics['fp']}** là cảnh báo nhầm (Precision **{lr_metrics['precision_churn']:.1%}**). "
    f"F1 = **{lr_metrics['f1_churn']:.4f}**, ROC-AUC = **{lr_metrics['roc_auc']:.4f}**. "
    "False negative là khách hàng churn bị bỏ sót; false positive là khách hàng không churn nhưng bị đưa vào danh sách can thiệp."
))
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, title, values in [
    (axes[0], "Baseline — test", baseline_metrics),
    (axes[1], "Logistic Regression — test", lr_metrics),
]:
    matrix = np.array([[values["tn"], values["fp"]], [values["fn"], values["tp"]]])
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Dự đoán No", "Dự đoán Yes"], yticklabels=["Thực tế No", "Thực tế Yes"])
    ax.set_title(title)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
fig.suptitle("Confusion matrix trên tập test", y=1.03)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "test_confusion_matrices.png", bbox_inches="tight")
plt.show()
""")

code(r"""
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_score)
fpr_dummy, tpr_dummy, _ = roc_curve(y_test, baseline_score)
roc_table = pd.DataFrame({"fpr_lr": pd.Series(fpr_lr), "tpr_lr": pd.Series(tpr_lr)})
roc_table.to_csv(TABLES_DIR / "test_roc_curve_lr.csv", index=False)

fig, ax = plt.subplots()
ax.plot(fpr_lr, tpr_lr, color=COLORS["yes"], linewidth=2, label=f"LR (AUC={lr_metrics['roc_auc']:.3f})")
ax.plot(fpr_dummy, tpr_dummy, color="#777777", linestyle="--", label=f"Baseline (AUC={baseline_metrics['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], color="#BBBBBB", linestyle=":")
ax.set(title="ROC curve trên tập test", xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=(0, 1), ylim=(0, 1))
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "test_roc_curve.png", bbox_inches="tight")
plt.show()
""")

md(r"""
## 8. Phân tích lỗi sau thực nghiệm

Các ví dụ dưới đây được chọn theo quy tắc cố định: FP có `score_churn` cao nhất và FN có `score_churn` thấp nhất (mô hình tự tin nhất nhưng sai). Đây là phân tích sau khi đánh giá; không dùng các phát hiện này để sửa mô hình rồi tiếp tục gọi cùng test là đánh giá độc lập.
""")

code(r"""
test_context = X_test.copy()
test_context.insert(0, "customerID", id_test)
test_context["tenure_group"] = pd.cut(
    test_context["tenure"], bins=CONFIG["tenure_bins"], labels=CONFIG["tenure_labels"], include_lowest=True, right=True
)
error_detail = prediction_df.merge(test_context, on="customerID", how="left", validate="one_to_one")

fp_examples = error_detail.query("error_group == 'FP'").sort_values("score_churn", ascending=False).head(10)
fn_examples = error_detail.query("error_group == 'FN'").sort_values("score_churn", ascending=True).head(10)
example_columns = ["customerID", "error_group", "score_churn", "Contract", "tenure", "tenure_group", "MonthlyCharges", "InternetService", "PaymentMethod"]
fp_examples[example_columns].to_csv(TABLES_DIR / "error_examples_fp.csv", index=False)
fn_examples[example_columns].to_csv(TABLES_DIR / "error_examples_fn.csv", index=False)

error_by_contract = pd.crosstab(error_detail["Contract"], error_detail["error_group"]).reindex(columns=["TN", "FP", "FN", "TP"], fill_value=0)
error_by_tenure = pd.crosstab(error_detail["tenure_group"], error_detail["error_group"], dropna=False).reindex(columns=["TN", "FP", "FN", "TP"], fill_value=0)
error_by_contract.to_csv(TABLES_DIR / "error_by_contract_test.csv")
error_by_tenure.to_csv(TABLES_DIR / "error_by_tenure_test.csv")

display(Markdown("### False positive tự tin nhất"))
display(fp_examples[example_columns])
display(Markdown("### False negative tự tin nhất"))
display(fn_examples[example_columns])
display(Markdown("### Phân bố lỗi theo Contract (test, mô tả sau thực nghiệm)"))
display(error_by_contract)
""")

md(r"""
## 9. Diễn giải Logistic Regression

`exp(coef)` là **odds ratio**, không phải tỷ số xác suất. Với numeric đã chuẩn hóa, một hệ số tương ứng tăng một độ lệch chuẩn ước lượng từ train sau median imputation. Với categorical, hệ số so sánh một mức với mức tham chiếu thực tế do encoder đã fit và bỏ đi. Regularization, tương quan giữa biến và `class_weight` có thể làm thay đổi hệ số; không xem đây là xếp hạng nhân quả.
""")

code(r"""
preprocess_fitted = final_pipeline.named_steps["preprocess"]
model_fitted = final_pipeline.named_steps["model"]
numeric_fitted = preprocess_fitted.named_transformers_["num"]
categorical_fitted = preprocess_fitted.named_transformers_["cat"]
scaler_fitted = numeric_fitted.named_steps["scaler"]
onehot_fitted = categorical_fitted.named_steps["onehot"]

coefficient_rows = []
for feature, coefficient, scale in zip(NUMERIC_FEATURES, model_fitted.coef_[0][:len(NUMERIC_FEATURES)], scaler_fitted.scale_):
    coefficient_rows.append({
        "transformed_feature": f"num__{feature}",
        "source_feature": feature,
        "level": "",
        "reference_category": "",
        "coefficient": float(coefficient),
        "odds_ratio": float(np.exp(coefficient)),
        "unit_or_comparison": f"Tăng 1 SD trên train (scale={scale:.4f})",
    })

coefficient_offset = len(NUMERIC_FEATURES)
cat_position = 0
for feature, categories, drop_index in zip(CATEGORICAL_FEATURES, onehot_fitted.categories_, onehot_fitted.drop_idx_):
    reference = categories[int(drop_index)] if drop_index is not None else "(không bỏ mức)"
    for category_index, category in enumerate(categories):
        if drop_index is not None and category_index == int(drop_index):
            continue
        coefficient = float(model_fitted.coef_[0][coefficient_offset + cat_position])
        coefficient_rows.append({
            "transformed_feature": f"cat__{feature}_{category}",
            "source_feature": feature,
            "level": str(category),
            "reference_category": str(reference),
            "coefficient": coefficient,
            "odds_ratio": float(np.exp(coefficient)),
            "unit_or_comparison": f"{feature}={category} so với {feature}={reference}",
        })
        cat_position += 1

coefficients = pd.DataFrame(coefficient_rows)
assert len(coefficients) == model_fitted.coef_.shape[1]
coefficients["direction"] = np.where(coefficients["coefficient"] >= 0, "Liên hệ odds churn cao hơn", "Liên hệ odds churn thấp hơn")
coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
coefficients.to_csv(RESULTS_DIR / "coefficients.csv", index=False)

positive_top = coefficients.nlargest(4, "coefficient")
negative_top = coefficients.nsmallest(4, "coefficient")
coefficient_plot = pd.concat([negative_top, positive_top]).drop_duplicates("transformed_feature").sort_values("coefficient")
coefficient_plot.to_csv(TABLES_DIR / "top_coefficients.csv", index=False)
display(coefficients.sort_values("abs_coefficient", ascending=False).head(15).style.format({"coefficient": "{:.4f}", "odds_ratio": "{:.4f}"}))
print("Intercept:", float(model_fitted.intercept_[0]))
""")

code(r"""
def coefficient_label(row):
    if row["level"]:
        return f'{row["source_feature"]}={row["level"]}'
    return f'{row["source_feature"]} (+1 SD)'

plot_labels = coefficient_plot.apply(coefficient_label, axis=1)
plot_colors = [COLORS["positive"] if value >= 0 else COLORS["negative"] for value in coefficient_plot["coefficient"]]
fig, ax = plt.subplots(figsize=(9, 5.8))
bars = ax.barh(plot_labels, coefficient_plot["coefficient"], color=plot_colors)
ax.axvline(0, color="#333333", linewidth=1)
ax.set(title="Các hệ số có độ lớn nổi bật trong Logistic Regression", xlabel="Hệ số log-odds (không phải tác động nhân quả)", ylabel="Feature sau biến đổi")
coefficient_span = coefficient_plot["coefficient"].max() - coefficient_plot["coefficient"].min()
ax.set_xlim(
    coefficient_plot["coefficient"].min() - 0.14 * coefficient_span,
    coefficient_plot["coefficient"].max() + 0.14 * coefficient_span,
)
for bar, value in zip(bars, coefficient_plot["coefficient"]):
    ax.text(value + (0.03 if value >= 0 else -0.03), bar.get_y()+bar.get_height()/2, f"{value:.2f}", va="center", ha="left" if value >= 0 else "right")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "logistic_coefficients_top.png", bbox_inches="tight")
plt.show()

top_positive = coefficients.nlargest(1, "coefficient").iloc[0]
top_negative = coefficients.nsmallest(1, "coefficient").iloc[0]
display(Markdown(
    f"**Diễn giải có điều kiện.** Hệ số dương lớn nhất trong mô hình là `{top_positive['unit_or_comparison']}` "
    f"(coef={top_positive['coefficient']:.3f}, odds ratio={top_positive['odds_ratio']:.3f}); hệ số âm lớn nhất về độ lớn là "
    f"`{top_negative['unit_or_comparison']}` (coef={top_negative['coefficient']:.3f}, odds ratio={top_negative['odds_ratio']:.3f}). "
    "Các giá trị này mô tả liên hệ có điều kiện trong mô hình đã regularize, không chứng minh rằng thay đổi thuộc tính sẽ làm churn thay đổi tương ứng."
))
""")

md(r"""
## 10. Kết luận và giả thuyết cần kiểm chứng

Cell sau tạo kết luận từ đúng bảng của lần chạy hiện tại. Các hành động được viết theo cấu trúc bằng chứng quan sát → giả thuyết → thử nghiệm → chỉ số theo dõi; không gán hiệu quả nhân quả cho dữ liệu quan sát.
""")

code(r"""
conclusion_md = f'''
### Trả lời ba câu hỏi khai phá

1. **Contract và tenure.** Trong train, `{highest_contract['Contract']}` có churn rate cao nhất ({highest_contract['churn_rate']:.1%}, n={int(highest_contract['n']):,}); nhóm tenure `{highest_tenure['tenure_group']}` cao nhất ({highest_tenure['churn_rate']:.1%}, n={int(highest_tenure['n']):,}). Đây là liên hệ mô tả, không phải bằng chứng nhân quả.
2. **Phát hiện và cảnh báo nhầm.** Trên test, LR phát hiện {lr_metrics['tp']}/{lr_metrics['tp'] + lr_metrics['fn']} ca churn (Recall {lr_metrics['recall_churn']:.1%}) và tạo {lr_metrics['fp']} cảnh báo nhầm trong tổng {alerts} cảnh báo (Precision {lr_metrics['precision_churn']:.1%}). F1={lr_metrics['f1_churn']:.4f}; AUC={lr_metrics['roc_auc']:.4f}.
3. **Liên hệ trong mô hình.** Feature dương nổi bật là `{top_positive['unit_or_comparison']}`; feature âm nổi bật là `{top_negative['unit_or_comparison']}`. Đây là odds ratio có điều kiện sau preprocessing/regularization, không phải tỷ số xác suất hay tác động can thiệp.

### Giả thuyết hành động

- **Bằng chứng:** churn rate cao ở nhóm `{highest_contract['Contract']}` trong train. **Giả thuyết:** hỗ trợ/ưu đãi phù hợp với nhóm hợp đồng này có thể cải thiện giữ chân. **Thử nghiệm:** A/B test một đề nghị rõ ràng trên đúng nhóm đủ điều kiện. **Theo dõi:** churn theo cửa sổ định trước, uptake, chi phí trên khách hàng và guardrail khiếu nại.
- **Bằng chứng:** nhóm tenure `{highest_tenure['tenure_group']}` có churn rate cao nhất. **Giả thuyết:** trải nghiệm onboarding/chăm sóc sớm có thể là điểm can thiệp. **Thử nghiệm:** chương trình hỗ trợ có nhóm đối chứng. **Theo dõi:** activation, liên hệ hỗ trợ, churn và chi phí.
- **Bằng chứng:** mô hình tạo danh sách {alerts} cảnh báo với Precision {lr_metrics['precision_churn']:.1%}. **Giả thuyết:** dùng score để ưu tiên nguồn lực có thể hữu ích hơn liên hệ đại trà. **Thử nghiệm:** đánh giá nhiều ngưỡng bằng validation/cost data mới, không bằng test hiện tại. **Theo dõi:** precision/recall theo capacity, uplift và chi phí can thiệp.

### Giới hạn

- Bộ dữ liệu công khai, không có mốc quan sát/nhãn tương lai và chi phí kinh doanh; một split không chứng minh khả năng tổng quát production.
- Dữ liệu có thể không đại diện khách hàng hiện tại; probability từ mô hình có `class_weight` (nếu được chọn) không mặc nhiên được hiệu chỉnh.
- EDA và hệ số là liên hệ. Hiệu quả giữ chân phải được kiểm chứng bằng thiết kế can thiệp phù hợp.
'''
display(Markdown(conclusion_md))
""")

md(r"""
## 11. Xuất kết quả và kiểm chứng

Cell này ghi report, data README, version lock, run manifest và ZIP. ZIP không chứa chính thư mục `exports/`, không chứa `.venv`, token hay cache. Trên Colab, file trong `/content` là tạm thời: cần sao chép ZIP sang Drive đã mount hoặc tải về máy sau khi chạy.
""")

code(r"""
report_text = f'''# Báo cáo kết quả — Telco Customer Churn với Logistic Regression

Run ID: `{RUN_ID}`  
Môi trường ghi nhận: `{EXECUTION_ENVIRONMENT}`  
SHA-256 dữ liệu: `{data_hash}`

## Phương pháp

Split 80/20 stratified, seed {CONFIG['random_state']}; chọn trong 6 cấu hình LR bằng {CONFIG['cv']['n_splits']}-fold CV trên train với mean F1 churn; threshold cố định {threshold}. Preprocessing nằm trong pipeline ở từng fold. Test chỉ dùng sau khi đã ghi `selected_model.json`.

## EDA trên train

- Churn: {int(y_train.sum()):,}/{len(y_train):,} ({train_churn_rate:.1%}).
- Contract churn cao nhất: {highest_contract['Contract']} — {highest_contract['churn_rate']:.1%} (n={int(highest_contract['n']):,}).
- Tenure churn cao nhất: {highest_tenure['tenure_group']} — {highest_tenure['churn_rate']:.1%} (n={int(highest_tenure['n']):,}).
- Median MonthlyCharges: churn {median_yes:.2f}; không churn {median_no:.2f}.

## Chọn mô hình bằng train-CV

Chọn `C={selected_C}`, `class_weight={selected_class_weight}`. Mean F1={selected_row['mean_f1_churn']:.4f}, Recall={selected_row['mean_recall_churn']:.4f}, Precision={selected_row['mean_precision_churn']:.4f}, AUC={selected_row['mean_roc_auc']:.4f}.

## Đánh giá test

| Mô hình | Accuracy | Precision churn | Recall churn | F1 churn | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | {baseline_metrics['accuracy']:.4f} | {baseline_metrics['precision_churn']:.4f} | {baseline_metrics['recall_churn']:.4f} | {baseline_metrics['f1_churn']:.4f} | {baseline_metrics['roc_auc']:.4f} | {baseline_metrics['tn']} | {baseline_metrics['fp']} | {baseline_metrics['fn']} | {baseline_metrics['tp']} |
| Logistic Regression | {lr_metrics['accuracy']:.4f} | {lr_metrics['precision_churn']:.4f} | {lr_metrics['recall_churn']:.4f} | {lr_metrics['f1_churn']:.4f} | {lr_metrics['roc_auc']:.4f} | {lr_metrics['tn']} | {lr_metrics['fp']} | {lr_metrics['fn']} | {lr_metrics['tp']} |

LR phát hiện {lr_metrics['tp']}/{lr_metrics['tp'] + lr_metrics['fn']} ca churn và tạo {lr_metrics['fp']} cảnh báo nhầm trong {alerts} cảnh báo.

## Diễn giải và đề xuất

{conclusion_md}

## Truy vết

Chi tiết fold ở `results/cv_results.csv`; predictions ở `results/test_predictions.csv`; hệ số/nhóm tham chiếu ở `results/coefficients.csv`; bảng nguồn EDA và đánh giá ở `results/tables/`; hình ở `results/figures/`.
'''
(REPORTS_DIR / "report.md").write_text(report_text, encoding="utf-8")

data_readme = f'''# Dữ liệu

- File: `data/raw/{DATA_PATH.name}`
- Nơi nhận bản sao: file `Dataset/{DATA_PATH.name}` do người dùng cung cấp trong workspace; không có URL/commit đi kèm.
- Ngày kiểm tra: {RUN_STARTED_AT.date().isoformat()}
- SHA-256: `{data_hash}`
- Kích thước kiểm tra: {raw_df.shape[0]:,} dòng × {raw_df.shape[1]} cột
- Nhãn: {json.dumps(data_audit['target_counts'], ensure_ascii=False)}
- Grain đã xác minh: `customerID` duy nhất, {raw_df['customerID'].nunique():,} ID.
- Schema: {', '.join(raw_df.columns)}

Đây là bản CSV 21 cột thường được gọi là IBM Telco Customer Churn. Workspace không kèm metadata gốc hoặc giấy phép, vì vậy dự án phân biệt nơi nhận bản sao với nguồn mô tả. Không trộn file này với bản Telco mở rộng có `Churn Score`, `Churn Reason` hoặc `CLTV`.
'''
(PROJECT_ROOT / "data" / "README.md").write_text(data_readme, encoding="utf-8")

lock_lines = [
    f"numpy=={np.__version__}",
    f"pandas=={pd.__version__}",
    f"scikit-learn=={sklearn.__version__}",
    f"matplotlib=={matplotlib.__version__}",
    f"seaborn=={sns.__version__}",
    f"jinja2=={jinja2.__version__}",
]
(PROJECT_ROOT / "requirements-lock.txt").write_text("\n".join(lock_lines) + "\n", encoding="utf-8")

notebook_path = PROJECT_ROOT / "notebooks" / "telco_churn_lr.ipynb"
if notebook_path.exists():
    code_hash = sha256_file(notebook_path)
    code_hash_kind = "notebook_file_before_current_save"
else:
    code_hash = hashlib.sha256(ANALYSIS_CODE_VERSION.encode()).hexdigest()
    code_hash_kind = "analysis_code_version_fallback"

run_manifest = {
    "run_id": RUN_ID,
    "started_at_utc": RUN_STARTED_AT.isoformat(),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "execution_environment": EXECUTION_ENVIRONMENT,
    "runtime_markers": runtime_markers,
    "data_sha256": data_hash,
    "config_sha256": sha256_file(CONFIG_PATH),
    "code_sha256": code_hash,
    "code_hash_kind": code_hash_kind,
    "analysis_code_version": ANALYSIS_CODE_VERSION,
    "package_versions": PACKAGE_VERSIONS,
    "artifacts": sorted(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in RESULTS_DIR.rglob("*") if path.is_file()),
    "checks": {
        "schema_exact": not schema_missing and not schema_extra,
        "customer_id_unique": bool(raw_df["customerID"].is_unique),
        "train_test_ids_disjoint": set(id_train).isdisjoint(set(id_test)),
        "cv_fold_rows": int(len(cv_folds)),
        "cv_convergence_warnings": int(cv_folds["convergence_warnings"].sum()),
        "final_convergence_warnings": int(final_convergence_warnings),
        "confusion_matrix_total_matches_test": int(lr_metrics["tn"] + lr_metrics["fp"] + lr_metrics["fn"] + lr_metrics["tp"]) == len(y_test),
        "selected_before_test_recorded": selected_model_record["selected_before_test"],
    },
}
with (RESULTS_DIR / "run_manifest.json").open("w", encoding="utf-8") as file:
    json.dump(run_manifest, file, ensure_ascii=False, indent=2)

verification_text = f'''# Kiểm chứng lần chạy {RUN_ID}

## Trạng thái

- Notebook đã tạo: **Có**.
- Đã chạy/kiểm tra trong môi trường hiện tại: **Có** — `{EXECUTION_ENVIRONMENT}`.
- Đã xác minh Colab từ VS Code: **{'Có' if CONFIRM_COLAB_VSCODE else 'Chưa; cần người dùng kết nối kernel Colab và chạy lại với CONFIRM_COLAB_VSCODE=True'}**.

## Kiểm tra tự động đã đạt

- Schema đúng 21 cột kỳ vọng; `customerID` không thiếu và duy nhất; không có dòng trùng hoàn toàn.
- Nhãn chỉ gồm `Yes/No`; không impute nhãn.
- Train/test không giao ID; ID và target không đi vào feature.
- Có 30 dòng kết quả fold (6 cấu hình × 5 fold), không có `ConvergenceWarning`.
- `selected_model.json` được ghi trước cell đánh giá test; threshold giữ ở {threshold}.
- Bốn ô confusion matrix cộng lại bằng {len(y_test)}; Precision/Recall/F1 đã được tính lại từ TP/FP/FN và khớp thư viện.
- AUC dùng `predict_proba` của lớp 1.
- Các file kết quả dùng cùng run ID/config/hash dữ liệu.

## Giới hạn còn lại

{'Không có bằng chứng điều khiển runtime Colab/đăng nhập Google trong môi trường tác giả. Người dùng cần làm theo reports/colab_vscode_setup.md, chạy lại toàn bộ, lưu notebook output và đưa ZIP ra khỏi /content.' if not CONFIRM_COLAB_VSCODE else 'Cần sao chép ZIP và notebook đã lưu ra nơi bền vững, rồi kiểm tra lại file trên máy cá nhân.'}
'''
(REPORTS_DIR / "verification.md").write_text(verification_text, encoding="utf-8")

zip_path = EXPORTS_DIR / f"telco_churn_lr_results_{RUN_ID}.zip"
include_paths = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "PLAN.md",
    PROJECT_ROOT / "config.json",
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "requirements-lock.txt",
    PROJECT_ROOT / "data" / "README.md",
]
include_paths += [path for path in RESULTS_DIR.rglob("*") if path.is_file()]
include_paths += [path for path in REPORTS_DIR.rglob("*") if path.is_file()]
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in include_paths:
        archive.write(path, path.relative_to(PROJECT_ROOT))

with zipfile.ZipFile(zip_path, "r") as archive:
    corrupt_member = archive.testzip()
    zip_members = archive.namelist()
assert corrupt_member is None
assert "results/metrics.json" in [name.replace("\\", "/") for name in zip_members]
print("Đã ghi run manifest:", RESULTS_DIR / "run_manifest.json")
print("Đã tạo và kiểm tra ZIP:", zip_path)
print("Số file trong ZIP:", len(zip_members))
""")

md(r"""
## 12. Tài liệu tham khảo

- [scikit-learn — Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn — Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn — LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn — F-beta score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)
- [Google Colab VS Code extension](https://github.com/googlecolab/colab-vscode)
- [Google Colab extension — Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=Google.colab)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

Các nguồn đồ án tham khảo và phạm vi đã/ chưa đọc được ghi riêng trong `reports/source_audit.md`. Không có metric nào trong notebook được chép từ các nguồn đó.
""")

nb["cells"] = cells
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(f"Wrote {OUTPUT} with {len(cells)} cells")

