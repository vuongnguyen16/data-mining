# Nhật ký quyết định

## 2026-10-01 — Khởi tạo

- Không tìm thấy rubric/đề bài riêng ngoài `PLAN.md`; triển khai theo các giả định mặc định của kế hoạch và không tuyên bố đã đáp ứng một rubric chưa được cung cấp.
- Dùng đúng CSV do người dùng đặt trong workspace; giữ một bản sao bất biến tại `data/raw/`.
- Đơn vị phân tích giả định là một khách hàng trên một dòng và sẽ được xác minh bằng `customerID` trước khi mô hình hóa.
- Giữ bài toán phân loại nhị phân, mô hình chính Logistic Regression L2 và baseline `DummyClassifier(strategy="most_frequent")`.
- Cố định split 80/20 có stratify, seed 42; chọn cấu hình chỉ bằng 5-fold CV trên train; ngưỡng 0.5 không được điều chỉnh bằng test.
- `SeniorCitizen` được xử lý như biến phân loại. `No internet service` và `No phone service` là trạng thái hợp lệ, không phải missing.
- EDA có nhãn phục vụ phát triển mô hình chỉ dùng train; toàn bộ dữ liệu chỉ dùng cho audit/schema đã định trước.
- Notebook là nguồn triển khai chính. Script trong `scripts/` chỉ dùng để sinh/kiểm tra notebook, không chứa pipeline huấn luyện độc lập.
- Môi trường hiện có không cung cấp quyền điều khiển runtime Colab/đăng nhập Google. Kết quả có thể được chạy kiểm tra cục bộ, nhưng trạng thái Colab chỉ được đánh dấu sau khi người dùng kết nối và chạy bằng extension chính thức.

## Điểm còn chưa biết

- Không có thông tin chi phí tương đối của false negative/false positive hoặc ngân sách can thiệp; vì vậy dùng F1 của lớp churn làm tiêu chí chọn chính như kế hoạch.
- Không có metadata do giảng viên cung cấp để xác nhận bản quyền/nguồn gốc bản CSV cụ thể; `data/README.md` ghi rõ đây là bản sao được cung cấp trong workspace và tách biệt nơi tải với nguồn mô tả bộ dữ liệu.
- Phiên bản Colab và phiên bản extension thực tế chỉ có thể được ghi nhận sau khi người dùng kết nối runtime.
