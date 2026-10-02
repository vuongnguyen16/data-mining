# Kiểm kê nguồn

Ngày kiểm kê: 2026-10-01.

| Nguồn | Mục đích | Trạng thái trong lần triển khai này | Giới hạn |
|---|---|---|---|
| `PLAN.md` trong workspace | Yêu cầu, thiết kế thực nghiệm, tiêu chí nghiệm thu | Đã đọc toàn bộ 495 dòng | Là kế hoạch, không phải bằng chứng kết quả |
| `Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dữ liệu thực nghiệm | Đã đọc trực tiếp; bản sao giữ tại `data/raw/` | Không có metadata giảng viên đi kèm |
| [Google Colab VS Code extension](https://github.com/googlecolab/colab-vscode) | Quy trình kết nối Colab trong VS Code | Đã đối chiếu Quick Start và các lệnh chính thức | Chưa thể xác nhận đăng nhập/runtime trên máy người dùng |
| [Visual Studio Marketplace — Google Colab](https://marketplace.visualstudio.com/items?itemName=Google.colab) | Xác nhận extension/publisher và thao tác kết nối | Đã đối chiếu | Giao diện có thể đổi theo phiên bản |
| [scikit-learn — Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Tránh leakage | Được dùng làm chuẩn phương pháp | Không thay thế kiểm tra pipeline của dự án |
| [scikit-learn — Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) | Thiết kế CV | Được dùng làm chuẩn phương pháp | Kết quả phải đến từ lần chạy của dự án |
| [scikit-learn — LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | API và diễn giải mô hình | Được dùng làm chuẩn API | Phiên bản thực tế được ghi trong manifest |
| [Adhang — Telco project](https://adhang.github.io/pages/project-telco-customer-churn.html) | Bối cảnh CRISP-DM | Chỉ ghi nhận theo kế hoạch; không chạy mã | Không lấy metric làm kết quả của dự án |
| [Shaimaa — repository](https://github.com/ShaimaaMohammed10/Telco-Customer-Churn-Prediction) | Tham khảo tiền xử lý | Chỉ ghi nhận theo kế hoạch; không chạy notebook | Không sao chép điểm số/kết quả |
| [Team 7 — repository](https://github.com/anhnhatcs/UMA_Data_Mining_Project) | Tham khảo bố cục và phân tích lỗi | Chỉ ghi nhận theo kế hoạch; không chạy notebook | Không sao chép PCA/lấy mẫu trước split |
| [Suhas — repository](https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction) | Tham khảo EDA | Chỉ ghi nhận theo kế hoạch; không chạy notebook | Không suy mất cân bằng từ giới tính |
| [Mittal — MSc report](https://norma.ncirl.ie/6230/1/manishkumarmittal.pdf) | Tham khảo tổ chức báo cáo | Chưa dùng trực tiếp | Không dùng làm benchmark |

Không có notebook bên ngoài nào được chạy trong lần triển khai này. Mọi metric trong `results/` phải do notebook của dự án tạo từ CSV đang lưu tại `data/raw/`.
