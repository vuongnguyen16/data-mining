# Telco Customer Churn — Logistic Regression

Sản phẩm chính là notebook [notebooks/telco_churn_lr.ipynb](notebooks/telco_churn_lr.ipynb). Notebook thực hiện audit dữ liệu, split cố định, EDA trên train, baseline, 5-fold cross-validation, chọn Logistic Regression trước khi mở test, đánh giá cuối, phân tích lỗi, diễn giải hệ số và xuất kết quả.

## Phạm vi repository

Repository lưu mã nguồn, cấu hình, tài liệu hướng dẫn và notebook đã xóa output. CSV, kết quả dự đoán, biểu đồ, báo cáo thực nghiệm và các bản xuất chỉ được giữ cục bộ, được bỏ qua bằng `.gitignore`.

## Chuẩn bị dữ liệu

Đặt file `WA_Fn-UseC_-Telco-Customer-Churn.csv` tại `data/raw/` trước khi chạy. Bộ dữ liệu mẫu công khai có thể tải từ [repository IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv).

PowerShell, từ thư mục `telco_churn_lr`:

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv' -OutFile 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
```

## Chạy bằng VS Code + Google Colab

Đọc [reports/colab_vscode_setup.md](reports/colab_vscode_setup.md) rồi:

1. Cài extension `google.colab` của Google và `ms-toolsai.jupyter`.
2. Mở notebook, chọn **Select Kernel → Colab → Auto Connect** và đăng nhập.
3. Upload cả thư mục dự án vào runtime hoặc mount Drive; bảo đảm runtime thấy `config.json` và `data/raw/...csv`.
4. Đặt `CONFIRM_COLAB_VSCODE=True` chỉ sau khi đã chọn đúng kernel Colab.
5. Chạy tuần tự toàn bộ notebook; lưu notebook trong VS Code và đưa ZIP trong `exports/` ra khỏi `/content`.

## Chạy kiểm tra cục bộ

PowerShell, từ thư mục dự án:

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe scripts\build_notebook.py
.venv\Scripts\python.exe scripts\execute_notebook.py
.venv\Scripts\python.exe scripts\verify_artifacts.py
```

Lần chạy cục bộ hiện tại dùng Python 3.11.15. Phiên bản trực tiếp đã chạy được ghi trong `requirements-lock.txt` và `results/run_manifest.json`.

## File được sinh sau khi chạy

Các file dưới đây được tạo cục bộ sau khi chạy notebook và không được đưa lên Git:

- `results/data_audit.json`: audit schema, nhãn, missing, duplicate và miền giá trị.
- `results/split_manifest.csv`: ID và train/test.
- `results/cv_results.csv`: kết quả từng fold và thống kê từng cấu hình.
- `results/selected_model.json`: quyết định chốt trước test.
- `results/metrics.json`, `results/test_predictions.csv`: đánh giá test.
- `results/coefficients.csv`: hệ số, odds ratio, nhóm tham chiếu/đơn vị.
- `reports/report.md`: báo cáo sinh từ cùng lần chạy.
- `reports/verification.md`: kiểm tra đã thực hiện và giới hạn môi trường.
- `exports/telco_churn_lr_results_<run_id>.zip`: gói kết quả đã kiểm tra tính toàn vẹn.

## Giới hạn

Dữ liệu không có mốc quan sát và nhãn tương lai, chi phí FP/FN hoặc dữ liệu thử nghiệm can thiệp. EDA và hệ số chỉ là liên hệ; không suy ra tác động nhân quả hay lợi nhuận. Test không được dùng để chọn mô hình hoặc threshold.
