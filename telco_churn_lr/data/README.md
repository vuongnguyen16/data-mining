# Dữ liệu

- File: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- CSV không được lưu trong Git; đặt bản sao cục bộ tại `data/raw/` trước khi chạy.
- Nguồn công khai để tải: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv).
- Ngày kiểm tra: 2026-10-01
- SHA-256: `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`
- Kích thước kiểm tra: 7,043 dòng × 21 cột
- Nhãn: {"No": 5174, "Yes": 1869}
- Grain đã xác minh: `customerID` duy nhất, 7,043 ID.
- Schema: customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn

Đây là bản CSV 21 cột thường được gọi là IBM Telco Customer Churn. Workspace không kèm metadata gốc hoặc giấy phép, vì vậy dự án phân biệt nơi nhận bản sao với nguồn mô tả. Không trộn file này với bản Telco mở rộng có `Churn Score`, `Churn Reason` hoặc `CLTV`.
