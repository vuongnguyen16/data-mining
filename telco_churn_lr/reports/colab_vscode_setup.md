# Hướng dẫn chạy notebook bằng Google Colab trong VS Code

Tài liệu chính thức: [Google Colab VS Code extension](https://github.com/googlecolab/colab-vscode), [Marketplace](https://marketplace.visualstudio.com/items?itemName=Google.colab), [User Guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide), [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## 1. Chuẩn bị VS Code

1. Mở thư mục `telco_churn_lr` trong VS Code.
2. Cài **Colab** của publisher **Google**, ID `google.colab`.
3. Cài/bật extension **Jupyter**, ID `ms-toolsai.jupyter`.
4. Mở `notebooks/telco_churn_lr.ipynb`.

File `.vscode/extensions.json` đã khai báo hai extension trên. Không cần tạo `.venv` trên Windows để dùng kernel Colab.

## 2. Kết nối đúng kernel Colab

1. Trong góc phải notebook, chọn **Select Kernel**.
2. Chọn **Colab → Auto Connect** (hoặc server CPU Colab nếu giao diện phiên bản hiện tại hiển thị khác).
3. Hoàn tất đăng nhập Google trên cửa sổ do extension mở. Không dán token/mật khẩu vào notebook hay gửi cho Codex.
4. Chạy cell môi trường đầu tiên. Cell phải in `sys.version`, `sys.executable`, `platform`, `Path.cwd()`, marker Colab và đường dẫn project.
5. Sau khi đã xác nhận kernel đang chọn là Colab trong VS Code, đổi `CONFIRM_COLAB_VSCODE = True`. Đây là xác nhận thủ công có chủ đích; việc import được `google.colab` riêng lẻ không đủ.

Nếu UI không có lựa chọn như trên, mở Command Palette và kiểm tra extension Colab đã được bật/đăng nhập. Không đổi sang kernel local rồi vẫn ghi trạng thái là Colab.

## 3. Đưa project và CSV vào runtime

Notebook cần runtime nhìn thấy đồng thời:

```text
telco_churn_lr/
├── config.json
├── requirements.txt
└── data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Cách A — Upload to Colab trong VS Code

Trong Explorer, dùng thao tác **Upload to Colab** cho thư mục/file cần thiết. Kiểm tra vị trí thực trong Colab Contents. Nếu thư mục đích là `/content/telco_churn_lr`, notebook sẽ tự tìm thấy.

Nếu upload vào vị trí khác, đặt biến môi trường trong cell đầu trước khi gọi `find_project_root()`:

```python
os.environ["TELCO_PROJECT_ROOT"] = "/content/duong-dan-thuc-te/telco_churn_lr"
```

### Cách B — Google Drive

Mở Command Palette và chạy **Colab: Mount Google Drive to Server...**. Chạy cell mount do extension thêm, rồi đặt:

```python
os.environ["TELCO_PROJECT_ROOT"] = "/content/drive/MyDrive/duong-dan/telco_churn_lr"
```

Không hard-code đường dẫn `D:\...` trong kernel Colab. File ở Explorer máy cá nhân không mặc nhiên tồn tại trong `/content`.

## 4. Phụ thuộc

Chạy notebook từ đầu. Mặc định `INSTALL_REQUIREMENTS=False` để dùng gói tương thích có sẵn. Nếu cell import báo thiếu hoặc API không tương thích:

1. Đặt `INSTALL_REQUIREMENTS=True`.
2. Chạy cell `%pip`.
3. Restart kernel nếu cell thông báo đã thay thư viện lõi.
4. Chạy lại toàn bộ notebook từ đầu.

`%pip` trong notebook cài vào kernel đang dùng; `pip` ở terminal PowerShell của VS Code có thể chỉ cài trên máy local.

## 5. Chạy tuần tự và kiểm tra

1. Chọn **Restart Kernel** để loại trạng thái ẩn.
2. Chạy **Run All** từ cell đầu.
3. Không thay grid, threshold hoặc preprocessing sau khi xem test.
4. Kiểm tra cell cuối báo đã tạo ZIP và `archive.testzip()` không tìm thấy file hỏng.
5. Mở `results/metrics.json`, `results/run_manifest.json` và `reports/verification.md`. Manifest phải ghi `google-colab-vscode-user-confirmed` sau khi bạn đã đặt biến xác nhận đúng lúc.
6. Lưu notebook trong VS Code để output của lần chạy Colab được ghi vào file `.ipynb` local.

## 6. Giữ kết quả sau khi runtime ngắt

Lưu notebook và filesystem runtime là hai việc khác nhau.

- **Notebook:** dùng Save trong VS Code, rồi đóng/mở lại để xác nhận output còn hiện.
- **Kết quả runtime:** ZIP nằm trong `exports/telco_churn_lr_results_<run_id>.zip` của project root trên runtime.

Nếu project nằm trong Drive, ZIP đã ở nơi bền vững. Nếu project nằm trong `/content`, dùng Colab Explorer để tải ZIP về hoặc mount Drive rồi sao chép, ví dụ trong một cell sau khi notebook chạy xong:

```python
from pathlib import Path
from shutil import copy2

zip_file = max(EXPORTS_DIR.glob("telco_churn_lr_results_*.zip"), key=lambda p: p.stat().st_mtime)
drive_target = Path("/content/drive/MyDrive/telco_churn_lr_exports")
drive_target.mkdir(parents=True, exist_ok=True)
saved_zip = copy2(zip_file, drive_target / zip_file.name)
print("Đã sao chép:", saved_zip)
```

Sau đó tải ZIP về workspace local, mở thử và đối chiếu phải có `results/metrics.json`, `reports/report.md`, `reports/verification.md`. Không coi file trong `/content` là đã lưu lâu dài.

## 7. Bằng chứng cần giữ để đóng checklist Colab

- Notebook local đã save với output mới.
- Cell môi trường in kernel/executable/platform và trạng thái `google-colab-vscode-user-confirmed`.
- `results/run_manifest.json` của cùng run ID.
- ZIP đã ra khỏi `/content` và mở đọc được.
- `reports/verification.md` được tạo lại trong run đó.

Cho đến khi đủ các bằng chứng trên, Giai đoạn 7 trong `PLAN.md` phải giữ trạng thái chưa hoàn thành.
