# PLAN — Đồ án Data Mining: IBM Telco Customer Churn với Logistic Regression

Ngày cập nhật: 01/10/2026 · Ngôn ngữ: tiếng Việt · Đầu ra trọng tâm: notebook và kết quả thực nghiệm trên Google Colab thông qua VS Code.

**Ưu tiên hiện tại: làm bài phân tích và có kết quả được kiểm chứng trước.** Chưa triển khai bài thuyết trình; không đặt giới hạn số slide hoặc thời lượng. Việc lựa chọn nội dung thuyết trình sẽ được quyết định sau khi xem kết quả.

## 0. Cách sử dụng và chỉ dẫn cho Codex

Đây là kế hoạch triển khai, chưa phải báo cáo kết quả. Những checkbox chưa đánh dấu phải được thực hiện và có bằng chứng trước khi chuyển thành hoàn thành.

Đọc toàn bộ file này trước khi bắt đầu. Sau đó kiểm tra workspace, tài liệu đề bài và `AGENTS.md` nếu có; lập danh sách công việc và thực hiện lần lượt các giai đoạn bên dưới. Không dừng ở việc đề xuất cách làm nếu đã có đủ dữ liệu và công cụ để triển khai.

Quy tắc làm việc:

- Giữ phạm vi chính là một bài toán phân loại nhị phân với Logistic Regression (LR). Có baseline đơn giản để so sánh.
- Tự xử lý các lựa chọn kỹ thuật thông thường theo kế hoạch; ghi quyết định vào `reports/decision_log.md`.
- Nếu có rubric/đề bài chính thức, ưu tiên yêu cầu của đề bài và ghi rõ các điều chỉnh so với kế hoạch. Không tự nhận rằng đã đáp ứng rubric chưa được cung cấp.
- Nếu chưa có rubric, tiếp tục theo các giả định tại mục 1. Không cần hỏi lại chỉ để xác nhận các giả định mặc định này.
- Chỉ hỏi khi thiếu đầu vào thực sự ngăn việc tiếp tục, ví dụ không có dữ liệu và không tải được từ nguồn hợp lệ, hoặc có hai đề bài mâu thuẫn.
- Đọc mã nguồn bên ngoài trước khi chạy. Không chạy toàn bộ notebook của người khác chỉ vì notebook được liệt kê trong tài liệu tham khảo.
- Không bịa kết quả, không điền số tham khảo vào chỗ dành cho kết quả thực nghiệm của bài làm.
- Khi một công cụ không khả dụng, hoàn thành phần có thể làm và ghi chính xác phần bị chặn. Không đánh dấu sản phẩm chưa tạo là đã hoàn thành.
- Sản phẩm chính là notebook `.ipynb`, có mã phân tích đầy đủ, giải thích và output thực tế. Môi trường đích: mở/sửa notebook trong VS Code và thực thi bằng kernel Google Colab qua extension chính thức của Google.
- Chưa tạo slide, PowerPoint, lời thuyết trình hoặc kịch bản thời gian trong phạm vi này. Không cắt bớt phân tích để vừa một số slide định trước.
- Phân biệt rõ “đã chạy trên Colab” với “đã kiểm tra cục bộ”. Nếu chưa kết nối được runtime Colab, vẫn hoàn thiện notebook, kiểm tra phần có thể chạy và ghi đúng trạng thái còn chờ xác minh trên Colab.
- Lưu sản phẩm theo quy định của môi trường Codex đang sử dụng. Giữ các file dự án liên quan trong cùng một thư mục; nếu có Git repository thì dùng quy trình lưu của repository đó.

## 1. Mục tiêu, phạm vi và giả định

### 1.1. Mục tiêu bài làm

Xây dựng một quy trình khai phá dữ liệu có thể tái lập để:

1. Hiểu những đặc điểm có liên hệ với việc khách hàng rời bỏ dịch vụ.
2. Huấn luyện LR để phân loại `Churn` trên bộ dữ liệu Telco Customer Churn.
3. Đánh giá khả năng phát hiện khách hàng churn và số lượng cảnh báo nhầm.
4. Diễn giải kết quả thành các giả thuyết giữ chân khách hàng cần kiểm chứng.
5. Bàn giao notebook có thể chạy lại, các kết quả xuất ra và bản tổng hợp đủ rõ để người dùng xem xét trước khi lên kế hoạch thuyết trình.

Tên đề tài mặc định: **Phân tích và dự đoán khách hàng rời bỏ dịch vụ viễn thông bằng Logistic Regression**.

### 1.2. Ba câu hỏi khai phá

- Tỷ lệ churn khác nhau như thế nào giữa các loại hợp đồng và nhóm thời gian sử dụng?
- LR phát hiện được bao nhiêu khách hàng churn, đồng thời tạo ra bao nhiêu cảnh báo nhầm?
- Những thuộc tính nào có liên hệ đáng chú ý với churn trong mô hình, và chúng gợi ý những hướng thử nghiệm giữ chân nào?

### 1.3. Giả định mặc định

| Nội dung | Mặc định triển khai |
|---|---|
| Dữ liệu | Bản CSV kinh điển `WA_Fn-UseC_-Telco-Customer-Churn.csv`, dự kiến 7.043 dòng và 21 cột |
| Đơn vị quan sát | Một khách hàng trên một dòng; kiểm tra lại tính duy nhất của `customerID` |
| Nhãn | `Churn`: `Yes` → 1, `No` → 0; lớp dương là churn |
| Thuật toán chính | Logistic Regression có regularization L2 |
| Người đọc notebook | Người học môn Data Mining, biết ML ở mức nhập môn |
| Trình bày thuyết trình | Để giai đoạn sau khi có kết quả; chưa quy định số slide hoặc thời lượng |
| Ngôn ngữ | Markdown giải thích, nhận xét kết quả và báo cáo tiếng Việt; giữ tên cột, API và thuật ngữ cần thiết bằng tiếng Anh |
| Phần mềm | VS Code + extension Google Colab và Jupyter; Python, pandas, NumPy, scikit-learn, matplotlib/seaborn trong runtime Colab |
| Tài nguyên tính toán | CPU mặc định, phù hợp với LR trên dữ liệu này; không yêu cầu GPU hoặc gói trả phí |
| Hướng triển khai | Notebook phân tích chạy trên Colab từ VS Code; không cần website, API hay hệ thống production |

Nếu dữ liệu thực nhận khác bản kinh điển, phải đối chiếu schema và nguồn trước khi tiếp tục. Không ép số dòng hoặc sửa dữ liệu chỉ để khớp các con số dự kiến.

### 1.4. Ngoài phạm vi mặc định

Không mở rộng sang KNN, Random Forest, boosting, stacking, neural network, PCA, SMOTE/SMOTEENN, SHAP, triển khai web hoặc tối ưu hàng trăm cấu hình. Chỉ bổ sung nếu đề bài chính thức yêu cầu hoặc người dùng đổi phạm vi.

Không tuyên bố mô hình đã dự đoán được churn trong một khoảng thời gian tương lai cụ thể nếu dữ liệu không có mốc quan sát và nhãn tương lai tương ứng. Phải mô tả đây là đánh giá phân loại trên bộ dữ liệu mẫu, đồng thời nêu giới hạn khi ứng dụng thực tế.

## 2. Tài liệu tham khảo và cách sử dụng

### 2.1. Nguồn chính

| ID | Nguồn | Dùng cho phần nào | Giới hạn cần giữ |
|---|---|---|---|
| S1 | [Adhang — Project](https://adhang.github.io/pages/project-telco-customer-churn.html) | CRISP-DM, đặt vấn đề, nối phân tích với kinh doanh | Không sao chép cách gọi F-beta từ Accuracy và Recall; không lấy kết quả LightGBM làm kết quả LR |
| S2 | [Shaimaa — Repository](https://github.com/ShaimaaMohammed10/Telco-Customer-Churn-Prediction) và [Notebook](https://github.com/ShaimaaMohammed10/Telco-Customer-Churn-Prediction/blob/main/telco_churn_project.ipynb) | Tham khảo tiền xử lý, LR, hệ số và cách tính metric | Không chọn mô hình bằng F1 trên test; cần đưa toàn bộ preprocessing vào pipeline khi cross-validation |
| S3 | [Team 7 — Repository](https://github.com/anhnhatcs/UMA_Data_Mining_Project) và [Notebook](https://github.com/anhnhatcs/UMA_Data_Mining_Project/blob/main/data_mining_project_team_7.ipynb) | Bố cục đồ án và phân tích lỗi | Không sao chép phần lấy mẫu/PCA trước khi chia fold |
| S4 | [Suhas — Repository](https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction) | Ý tưởng EDA, lựa chọn biểu đồ | Mất cân bằng lớp phải xét theo `Churn`, không suy từ tỷ lệ nam/nữ |
| S5 | [Mittal — MSc report](https://norma.ncirl.ie/6230/1/manishkumarmittal.pdf) | Chỉ tham khảo thêm cách tổ chức báo cáo nếu cần | Không dùng làm benchmark mặc định do bất nhất số mẫu trong phần đánh giá |

Phạm vi kiểm tra đã có trong cuộc trao đổi trước khi lập kế hoạch: trang dự án của các nguồn; các đoạn mã quan trọng của S2/S3; và chạy lại riêng preprocessing + LR của S2. Không có xác nhận rằng toàn bộ notebook của cả năm nguồn đã được kiểm định.

Thông tin này là bối cảnh lựa chọn tài liệu. Khi triển khai, Codex phải tạo bằng chứng chạy của chính dự án, không chuyển các kiểm tra từ cuộc trao đổi thành log thực thi mới.

### 2.2. Tài liệu chuẩn cho phương pháp

- [scikit-learn — Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [scikit-learn — Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
- [scikit-learn — LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- [scikit-learn — F-beta](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html).
- [imbalanced-learn — Common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html), chỉ cần nếu thảo luận lỗi lấy mẫu trong nguồn tham khảo.

Tài liệu chính thức cho môi trường chạy, đã đối chiếu khi cập nhật kế hoạch:

- [Google Colab extension — Marketplace, publisher Google](https://marketplace.visualstudio.com/items?itemName=Google.colab).
- [Google Colab for VS Code — Repository](https://github.com/googlecolab/colab-vscode).
- [Google Colab for VS Code — User Guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).
- [Google Colab — FAQ](https://research.google.com/colaboratory/faq.html).

Đối chiếu API với phiên bản thực tế đã cài; không sao chép tham số cũ khi phiên bản hiện tại đã thay đổi. Ghi phiên bản môi trường sau khi chạy thành công.

### 2.3. Lấy và ghi nhận nguồn dữ liệu

Ưu tiên CSV người dùng/giảng viên cung cấp. Nếu chưa có, có thể dùng [CSV trong repository S2](https://github.com/ShaimaaMohammed10/Telco-Customer-Churn-Prediction/blob/main/WA_Fn-UseC_-Telco-Customer-Churn.csv), sau khi kiểm tra schema.

Lưu `data/README.md` gồm tên file, URL tải, ngày truy cập, SHA-256, số dòng/cột và nguồn mô tả dữ liệu. Nếu tải từ GitHub, ưu tiên ghi commit SHA để cố định phiên bản. Phân biệt nguồn gốc IBM được tài liệu xác nhận với nơi tải bản sao; không gọi repository S2 là nguồn dữ liệu chính thức của IBM.

Xác minh mô tả dữ liệu mẫu/giả lập và ý nghĩa các cột từ tài liệu gốc phù hợp. Không nhầm bản CSV 21 cột với những bản Telco mở rộng có các cột như Churn Score, Churn Reason, CLTV. Nếu không xác minh được một chi tiết nguồn gốc, ghi giới hạn thay vì khẳng định.

## 3. Thiết kế thực nghiệm cố định trước khi xem kết quả test

### 3.1. Chia dữ liệu

- Giữ nguyên file raw.
- Các thao tác xác định trước, không học thống kê như bỏ khoảng trắng, chuyển kiểu `TotalCharges`, ánh xạ nhãn và tách ID có thể thực hiện trước khi chia.
- Chia 80% train và 20% test với `stratify=y`, `random_state=42`; lưu danh sách ID của hai tập.
- Với bản 7.043 dòng, dự kiến có 5.634 dòng train và 1.409 dòng test. Kiểm tra thực tế, không hard-code kích thước vào phép tính metric.
- Chỉ dùng train cho EDA có nhãn để phát triển nội dung/mô hình, chọn cấu hình và kiểm tra giả thuyết. Thống kê toàn bộ dữ liệu trước chia chỉ phục vụ kiểm tra chất lượng/schema và mô tả tổng quan đã định trước.
- Test được giữ riêng cho đánh giá cuối. Không dùng test để chọn biến, cấu hình, cách điền thiếu hoặc ngưỡng.
- Bộ dữ liệu công khai đã có nhiều kết quả được công bố, và split seed 42 đã được thảo luận trước đó. Không mô tả đây là một benchmark hoàn toàn mù; chỉ khẳng định quy trình của dự án không dùng test để điều chỉnh mô hình.

### 3.2. Pipeline tiền xử lý

`Dữ liệu đầu vào → ColumnTransformer → LogisticRegression`.

| Nhóm | Cột/cách xác định | Xử lý |
|---|---|---|
| ID | `customerID` | Giữ để truy vết và kiểm tra tập; loại khỏi feature |
| Nhãn | `Churn` | Tách khỏi X; không bao giờ impute nhãn thiếu |
| Số liên tục | `tenure`, `MonthlyCharges`, `TotalCharges` | `SimpleImputer(strategy='median')` rồi `StandardScaler()` |
| Phân loại | 16 thuộc tính còn lại, gồm `SeniorCitizen` | Chuẩn hóa kiểu dữ liệu nhất quán, `SimpleImputer(strategy='most_frequent')`, `OneHotEncoder(drop='first', handle_unknown='ignore')` |

`SeniorCitizen` là biến nhị phân; xử lý như categorical. Những giá trị `No internet service`/`No phone service` là trạng thái dịch vụ có ý nghĩa, không tự coi là missing.

Fit toàn bộ pipeline riêng trong từng fold. Không tính median, fit encoder hoặc fit scaler trên toàn train rồi mới gọi CV cho LR đã nhận dữ liệu biến đổi. Test chỉ được `transform` qua bộ biến đổi đã fit trên train.

Nếu đầu ra có cảnh báo category chưa gặp, ghi lại và kiểm tra nguyên nhân; không âm thầm xóa dòng hoặc fit lại encoder bằng test.

### 3.3. Baseline và lựa chọn LR

Baseline bắt buộc: `DummyClassifier(strategy='most_frequent')`, học lớp đa số từ train. Với dữ liệu dự kiến, baseline này không phát hiện được churn; giải thích vì sao Accuracy một mình chưa phản ánh mục tiêu bài toán.

Tìm cấu hình LR bằng 5-fold `StratifiedKFold(shuffle=True, random_state=42)` trên train, với một grid nhỏ được định trước:

- `C`: `[0.1, 1.0, 10.0]`.
- `class_weight`: `[None, 'balanced']`.
- Regularization L2, solver phù hợp như `lbfgs`, `max_iter=2000`.
- Ngưỡng phân loại cố định `0.5` cho tất cả cấu hình.
- Tổng cộng 6 cấu hình × 5 fold, ngoài baseline; không mở rộng grid để đuổi theo điểm test.

**Tiêu chí chọn chính: mean F1 của lớp churn trên các fold.** Đồng thời lưu Precision, Recall, ROC-AUC và Accuracy theo fold. F1 được dùng vì đề bài chưa cung cấp chi phí FN/FP hoặc ngân sách chăm sóc; đây là lựa chọn phương pháp mặc định, không phải chứng minh tối ưu lợi nhuận.

Nếu bằng nhau trong sai số `1e-6`, ưu tiên mean Recall cao hơn, sau đó C nhỏ hơn; nếu vẫn bằng nhau, ưu tiên `class_weight=None`. Ghi quy tắc trước khi chạy và thực hiện đúng quy tắc. Không mặc định `'balanced'` tốt hơn cấu hình không trọng số.

Sau khi chọn, lưu `selected_model.json` chứa tham số, metric chọn, threshold và lý do. Refit pipeline được chọn trên toàn train, rồi đánh giá test. Không chọn lại mô hình sau khi thấy bảng test.

### 3.4. Quy định về metric

- Precision, Recall, F1 phải tính cho lớp `Churn=1`, không vô tình thay bằng weighted/macro average.
- `Precision = TP / (TP + FP)`; `Recall = TP / (TP + FN)`; `F1 = 2PR / (P + R)`.
- ROC-AUC lấy từ `predict_proba(X)[:, index_of_class_1]`, không lấy từ nhãn dự đoán 0/1.
- Dùng `zero_division=0` cho metric cần thiết; giải thích Precision/F1 bằng 0 của baseline khi không dự đoán ca dương nào.
- Confusion matrix dùng `labels=[0,1]`; hàng là thực tế, cột là dự đoán: `[[TN, FP], [FN, TP]]`.
- Tổng bốn ô phải bằng số mẫu được đánh giá. Các metric tính lại từ bốn ô phải khớp kết quả thư viện trong sai số số thực.
- Độ lệch chuẩn giữa các fold không phải khoảng tin cậy của hiệu năng thực tế. Không gọi mean ± std là “95% CI”.

### 3.5. Giới hạn diễn giải

- EDA cho thấy liên hệ; hệ số LR cho thấy liên hệ có điều kiện trong mô hình. Cả hai không chứng minh quan hệ nhân quả.
- `exp(coef)` là odds ratio theo cách mã hóa đã dùng, không phải tỷ số xác suất.
- Với biến số đã chuẩn hóa, hệ số tương ứng với tăng một độ lệch chuẩn trên train, không phải mặc định tăng một tháng hoặc một đơn vị tiền.
- Với one-hot, phải ghi nhóm tham chiếu lấy từ encoder đã fit. Không suy đoán nhóm tham chiếu từ thứ tự tự đặt trên biểu đồ.
- LR có regularization và các biến có tương quan; không diễn giải độ lớn hệ số là bảng xếp hạng nhân quả hoặc độ quan trọng tuyệt đối giữa mọi thuộc tính.
- `class_weight='balanced'` thay đổi hàm mục tiêu; điểm `predict_proba` không tự động là xác suất churn đã được hiệu chỉnh cho ứng dụng thực tế.
- Không dùng dữ liệu này để khẳng định lợi nhuận tăng, churn giảm hoặc hiệu quả chiến dịch nếu chưa có dữ liệu chi phí và thử nghiệm can thiệp.

## 4. Cấu trúc sản phẩm bàn giao

Tạo thư mục dự án `telco_churn_lr/`. Có thể điều chỉnh cấu trúc nhỏ theo workspace hiện có nhưng giữ chức năng của từng sản phẩm.

| Đường dẫn tương đối | Nội dung |
|---|---|
| `PLAN.md` | Kế hoạch này và trạng thái các giai đoạn |
| `README.md` | Mục tiêu, mở notebook, kết nối Colab từ VS Code, chạy lại, kết quả chính và giới hạn |
| `requirements.txt` | Các phụ thuộc phân tích trực tiếp cần dùng trong kernel notebook |
| `requirements-lock.txt` | Phiên bản đã chạy thành công của các gói cần cho bài; không đóng băng toàn bộ các gói không liên quan trong Colab |
| `config.json` | Seed, split, CV, grid, metric chọn, threshold cố định |
| `.vscode/extensions.json` | Gợi ý extension `google.colab` và `ms-toolsai.jupyter`; không chứa thông tin tài khoản |
| `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dữ liệu raw, không sửa trực tiếp |
| `data/README.md` | Nguồn dữ liệu, checksum, schema và điều kiện sử dụng xác minh được |
| `notebooks/telco_churn_lr.ipynb` | Sản phẩm chính: mã phân tích, Markdown và output thật; chạy tuần tự trên Colab từ VS Code |
| `results/data_audit.json` | Kiểm tra chất lượng, nhãn, missing, duplicates |
| `results/split_manifest.csv` | ID khách hàng và tập train/test |
| `results/cv_results.csv` | Kết quả từng cấu hình, từng fold, mean và std |
| `results/selected_model.json` | Quyết định chốt cấu hình trước test |
| `results/metrics.json` | Kết quả cuối của LR và baseline, thông tin run |
| `results/test_predictions.csv` | ID, y_true, y_pred, score_churn, nhóm TP/TN/FP/FN |
| `results/coefficients.csv` | Feature sau mã hóa, hệ số, odds ratio, nhóm tham chiếu/đơn vị |
| `results/run_manifest.json` | Hash dữ liệu/config/code, phiên bản môi trường, thời điểm chạy |
| `results/figures/` | Biểu đồ PNG rõ nét để đọc và tái sử dụng; SVG nếu công cụ hỗ trợ |
| `results/tables/` | Số liệu gốc của các biểu đồ EDA và bảng đánh giá |
| `reports/report.md` | Tổng hợp phương pháp, kết quả, diễn giải và giới hạn; độ dài theo nội dung cần giải thích |
| `reports/source_audit.md` | Nguồn, phần sử dụng, phạm vi kiểm tra, hạn chế |
| `reports/decision_log.md` | Giả định và các quyết định ảnh hưởng phương pháp |
| `reports/verification.md` | Lệnh đã chạy, kết quả kiểm tra, lỗi đã sửa, giới hạn còn lại |
| `reports/colab_vscode_setup.md` | Các bước kết nối, đưa dữ liệu vào runtime, cài gói, chạy và lấy kết quả về |
| `exports/telco_churn_lr_results_<run_id>.zip` | Gói kết quả của một lần chạy để lưu bền vững/tải về; không nén chính thư mục exports vào nó |

Notebook là nguồn triển khai chính: định nghĩa các hàm tái sử dụng ngay trong các cell phù hợp và gọi chúng ở từng bước. Người dùng phải xem được logic tiền xử lý, CV, đánh giá và diễn giải trong notebook, không chỉ thấy một lệnh gọi script hoặc đọc kết quả có sẵn.

Không bắt buộc CLI, `src/run_analysis.py` hay package Python riêng. Chỉ tách helper khi cần, và bảo đảm helper cũng có trong runtime Colab; không duy trì hai bản huấn luyện độc lập. Bản notebook lưu trong VS Code là bản mã nguồn chính; các CSV/JSON/hình sinh trên Colab phải được xuất về nơi lưu bền vững riêng, không mặc định đã đồng bộ về máy.

Không tạo thư mục `presentation/` trong giai đoạn này. Khi người dùng yêu cầu bước thuyết trình sau, dùng các kết quả đã kiểm chứng để quyết định bố cục và mức độ chi tiết.

## 5. Các giai đoạn triển khai và điều kiện hoàn thành

### Giai đoạn 0 — Xác định đầu vào và khởi tạo

- [x] Đọc đề bài/rubric nếu có; kiểm tra file đang tồn tại để không ghi đè công việc của người dùng.
- [x] Lập `decision_log.md`, ghi các giả định ở mục 1 và điểm còn chưa biết.
- [x] Tạo cấu trúc dự án, notebook, config và hướng dẫn VS Code + Colab; dùng `pathlib` và một cell cấu hình đường dẫn tập trung.
- [x] Thiết lập hoặc hướng dẫn kết nối kernel Colab chính thức; chạy cell kiểm tra môi trường nếu có quyền truy cập runtime.
- [x] Chọn cách đưa CSV vào Colab và lưu kết quả; phân biệt đường dẫn máy cá nhân, `/content` trên runtime và Google Drive.
- [x] Ghi lại cách cài gói trong kernel bằng `%pip`, đường dẫn project/data/output và bước lưu kết quả. Không giả định terminal VS Code đang chạy trên Colab.
- [x] Tạo source audit với URL, ngày truy cập, nội dung dự định dùng và trạng thái “đã đọc/chưa đọc/đã chạy phần nào”.

**Hoàn thành khi:** Có dữ liệu hoặc đường tải hợp lệ, cấu hình thực nghiệm và hướng dẫn khởi chạy. Kết nối Colab được xác nhận bằng cell chạy thực tế, hoặc được đánh dấu rõ là còn chờ thao tác đăng nhập/kết nối của người dùng.

### Giai đoạn 1 — Kiểm tra dữ liệu và cố định split

- [x] Kiểm tra số dòng/cột, tên cột, kiểu dữ liệu, tính duy nhất của ID, dòng trùng hoàn toàn và missing.
- [x] Đếm giá trị `Churn`; xác nhận chỉ có nhãn được hỗ trợ. Không tự coi nhãn lạ/missing là `No`.
- [x] Kiểm tra `TotalCharges`: chuỗi trắng và lỗi chuyển số; ghi lượng missing trước/sau chuyển kiểu.
- [x] Kiểm tra miền giá trị như tenure/charges không âm và giá trị category bất thường; không xóa tự động chỉ vì “outlier”.
- [x] Nếu có khách hàng xuất hiện nhiều lần hoặc bản ghi nhân bản, giải quyết trước split và ghi lý do để tránh cùng một khách hàng ở cả hai tập.
- [x] Tách ID/target và tạo split cố định theo mục 3; lưu manifest.
- [x] Lưu data audit, checksum và mô tả schema.

Con số tham chiếu từ lần kiểm tra dữ liệu trong S2 trước khi lập kế hoạch: 7.043 dòng, 21 cột, 1.869 `Yes`, 5.174 `No`, 11 giá trị `TotalCharges` thiếu sau chuyển số. Đây là dấu hiệu đối chiếu phiên bản, không phải số liệu được phép sao chép thay cho kiểm tra hiện tại.

**Hoàn thành khi:** Có audit tái lập được, lý do cho mọi thay đổi dữ liệu và hai tập ID không giao nhau.

### Giai đoạn 2 — EDA phục vụ ba câu hỏi khai phá

Thực hiện phân tích có nhãn trên train và ghi rõ tập dữ liệu trong caption. Tạo:

1. Phân bố `Churn`: count và tỷ lệ phần trăm, dùng để giải thích baseline.
2. Tỷ lệ churn theo `Contract`: mỗi nhóm có số mẫu `n` và số churn.
3. Tỷ lệ churn theo nhóm `tenure` cố định: 0–12, 13–24, 25–48, trên 48 tháng.
4. Phân bố `MonthlyCharges` theo churn, dùng boxplot hoặc biểu đồ phù hợp; giải thích ngay sau biểu đồ trong notebook.

Yêu cầu:

- [x] Tỷ lệ churn trong một nhóm = số churn của nhóm / tổng khách hàng trong nhóm.
- [x] Không nhầm countplot với tỷ lệ churn; không nhầm tỷ trọng một nhóm trong toàn bộ churn với rủi ro churn của nhóm đó.
- [x] Mọi biểu đồ đều có bảng số liệu nguồn; nhóm ít mẫu phải được chỉ rõ.
- [x] Viết một nhận xét mô tả và một giới hạn cho mỗi biểu đồ.
- [x] Giữ các biểu đồ cần để trả lời câu hỏi khai phá trong notebook; có thể bổ sung EDA trên train khi có câu hỏi cụ thể. Không hạn chế số biểu đồ vì bố cục thuyết trình, nhưng tránh biểu đồ trùng thông tin.

**Hoàn thành khi:** Có biểu đồ dễ đọc, số mẫu và câu kết luận được số liệu hỗ trợ; không dự đoán sẵn hợp đồng nào chắc chắn quan trọng nhất trước khi chạy.

### Giai đoạn 3 — Xây pipeline và cross-validation

- [x] Triển khai pipeline theo mục 3.2 và baseline.
- [x] Đánh giá 6 cấu hình LR trên cùng các fold; fit preprocessing bên trong mỗi fold.
- [x] Lưu từng fold và thống kê tổng hợp vào `cv_results.csv`.
- [x] Chọn theo quy tắc F1 đã chốt; ghi thêm đánh đổi Precision/Recall của các cấu hình đáng chú ý.
- [x] Kiểm tra hội tụ. Nếu chưa hội tụ, xử lý bằng cấu hình số học hợp lý và ghi thay đổi; không tắt toàn bộ warnings để che lỗi.
- [x] Lưu cấu hình được chọn, danh sách feature và ngưỡng trước khi chạy test.

**Hoàn thành khi:** Có thể giải thích mô hình được chọn bằng kết quả CV trên train mà không cần viện dẫn điểm test.

### Giai đoạn 4 — Đánh giá cuối và phân tích lỗi

- [x] Fit pipeline được chọn và baseline trên train; đánh giá cùng tập test.
- [x] Lưu metric, predictions và confusion matrix; mọi phép tính dùng kích thước tập thực tế.
- [x] Tạo ROC curve từ score, hiển thị AUC; giữ ở phụ lục nếu bài chính quá tải.
- [x] Giải thích FN là bỏ sót người churn, FP là cảnh báo nhầm người không churn; cả hai đều có thể gây chi phí.
- [x] Phân tích vài trường hợp FP/FN theo quy tắc rõ ràng, ví dụ các lỗi có score tự tin nhất; không chọn vài ví dụ thuận lợi để khẳng định mô hình tốt.
- [x] Có thể mô tả lỗi theo hợp đồng/tenure sau đánh giá, nhưng xem đây là phân tích sau thực nghiệm; không dùng phát hiện này để sửa mô hình rồi tiếp tục gọi cùng test là đánh giá độc lập.

**Hoàn thành khi:** Có bảng LR–baseline, bốn ô confusion matrix tự nhất quán và giải thích cụ thể về đánh đổi phát hiện/báo động nhầm.

Không có ngưỡng điểm tối thiểu được đặt trước để ép bài phải “đẹp”. Nếu kết quả không tốt, kiểm tra lỗi kỹ thuật; nếu quy trình đúng thì báo cáo trung thực và thảo luận giới hạn.

### Giai đoạn 5 — Diễn giải LR và liên hệ kinh doanh

- [x] Trích hệ số từ pipeline cuối; ghép đúng tên feature sau biến đổi.
- [x] Xuất bảng đầy đủ kèm category tham chiếu và đơn vị của biến số.
- [x] Tạo biểu đồ khoảng 6–8 hệ số nổi bật, giữ dấu dương/âm, tiêu đề không khẳng định nhân quả.
- [x] Chọn 2–3 phát hiện nhất quán để giải thích trong bài; khi EDA và hệ số khác hướng, kiểm tra điều kiện hóa, tương quan và mã hóa trước khi diễn giải.
- [x] Viết 2–3 đề xuất theo cấu trúc: bằng chứng quan sát → giả thuyết → hành động thử nghiệm → chỉ số cần theo dõi.

Ví dụ cách viết, chỉ dùng nếu số liệu thực hỗ trợ: “Nhóm hợp đồng theo tháng có tỷ lệ churn cao hơn trong train; có thể thử chương trình hỗ trợ/ưu đãi phù hợp cho nhóm này và đánh giá bằng thử nghiệm có nhóm đối chứng.” Không viết “đổi hợp đồng chắc chắn làm giảm churn X%”.

**Hoàn thành khi:** Người đọc phân biệt được dự đoán, liên hệ thống kê và hiệu quả can thiệp; không có con số lợi nhuận/churn giảm bị suy diễn.

### Giai đoạn 6 — Hoàn thiện notebook và tổng hợp kết quả

- [x] Báo cáo gồm: vấn đề → dữ liệu → chuẩn bị → EDA → LR → đánh giá → diễn giải → đề xuất → giới hạn → nguồn.
- [x] Hoàn thiện các phần notebook tại mục 6, đan xen Markdown, code, output và nhận xét.
- [x] Giải thích LR, pipeline, CV, class weight, Precision/Recall, hệ số và odds ratio bằng tiếng Việt dễ hiểu.
- [x] Lấy toàn bộ số trên notebook/báo cáo từ cùng lần chạy; không gõ lại theo trí nhớ hoặc chép điểm từ nguồn tham khảo.
- [x] Kết luận trả lời trực tiếp ba câu hỏi khai phá, tách điều đã quan sát khỏi giả thuyết cần kiểm chứng.
- [x] Đính kèm nguồn gần phần sử dụng và danh mục đầy đủ cuối notebook/báo cáo.
- [x] Xuất bảng, hình, metric và gói ZIP; kiểm tra font tiếng Việt, nhãn trục và khả năng đọc hình.
- [x] Lưu notebook kèm output và hướng dẫn tiếp tục khi Colab ngắt kết nối.

**Hoàn thành khi:** Người dùng có thể đọc notebook để hiểu toàn bộ quá trình, xem kết quả thật và truy vết kết luận về bảng/biểu đồ tương ứng. Chưa yêu cầu bất kỳ sản phẩm thuyết trình nào.

### Giai đoạn 7 — Chạy lại và nghiệm thu

- [ ] Restart kernel và chạy notebook tuần tự trên Colab từ VS Code sau khi hoàn tất các cell setup cần tương tác; không phụ thuộc biến còn sót.
- [ ] Xác nhận đã dùng kernel Colab và ghi thông tin môi trường thực tế; chạy cục bộ chỉ là kiểm tra bổ sung.
- [x] Kiểm tra số liệu trong notebook/báo cáo đối chiếu với metrics và bảng nguồn.
- [x] Ghi lệnh đã chạy, kết quả, thời điểm, cảnh báo và hạn chế còn lại vào verification.
- [ ] Kiểm tra notebook đã lưu output ở máy cá nhân và các file kết quả đã được xuất khỏi ổ tạm runtime; thử mở/đọc gói ZIP đã lưu.
- [ ] Kiểm tra phần phân tích có thể chạy lại sau reset runtime bằng các cell setup và dữ liệu đã ghi nhận.
- [x] Cập nhật README, checklist và danh sách bàn giao.

Chạy lại để kiểm tra khả năng tái lập với cấu hình cố định là hợp lệ. “Test chỉ đánh giá cuối” nghĩa là không dùng kết quả test để thử đi thử lại và chọn phương pháp tốt hơn.

## 6. Cấu trúc notebook bắt buộc

Đây là các phần phân tích trong notebook, không phải số slide. Chia cell đủ nhỏ để đọc, chạy và gỡ lỗi; không gom cả dự án vào một cell dài.

| Phần | Markdown giải thích | Code/output cần có |
|---|---|---|
| 0. Giới thiệu | Mục tiêu, câu hỏi, phạm vi, nguồn | Mục lục và trạng thái thực thi |
| 1. Setup Colab từ VS Code | Cách chọn kernel, đường dẫn, cách đưa dữ liệu vào | Cell kiểm tra môi trường, cài gói, import, config và tạo thư mục |
| 2. Đọc và kiểm tra dữ liệu | Đơn vị quan sát, nhãn, ý nghĩa các nhóm cột | Shape, schema, missing, duplicates, label counts, checksum |
| 3. Làm sạch và chia tập | Thao tác xác định trước, giữ test riêng | Chuyển kiểu, tách ID/target, split, kiểm tra ID và lưu manifest |
| 4. EDA trên train | Câu hỏi trước biểu đồ, nhận xét sau biểu đồ | Tỷ lệ churn, hợp đồng, tenure, MonthlyCharges và bảng số liệu nguồn |
| 5. Thiết kế mô hình | LR hoạt động thế nào, tại sao chuẩn hóa và mã hóa | Định nghĩa pipeline và baseline, giải thích các tham số |
| 6. Cross-validation | Vì sao chọn cấu hình bằng train-CV | Các fold, grid nhỏ, bảng kết quả và cấu hình chốt trước test |
| 7. Đánh giá cuối | Precision/Recall/F1/AUC và ý nghĩa lỗi | So sánh baseline, confusion matrix, ROC và predictions |
| 8. Phân tích lỗi | Giới hạn của các trường hợp minh họa | Một số FP/FN theo quy tắc đã ghi, phân tích sau thực nghiệm |
| 9. Diễn giải LR | Hệ số, nhóm tham chiếu, odds và giới hạn | Bảng hệ số đầy đủ, biểu đồ và diễn giải theo đúng đơn vị |
| 10. Kết luận | Trả lời ba câu hỏi, đề xuất cần thử nghiệm, hạn chế | Bản tổng hợp từ kết quả vừa chạy; không có chỉ tiêu hiệu quả bịa ra |
| 11. Xuất và kiểm chứng | Cách lưu notebook/output và chạy lại | Export JSON/CSV/hình, run manifest, ZIP và kiểm tra nhất quán |
| 12. Tài liệu tham khảo | Nguồn đã dùng và phạm vi đã xác minh | Link nguồn, phiên bản dữ liệu/phần mềm liên quan |

Yêu cầu chất lượng notebook:

- Các cell chạy từ trên xuống theo thứ tự; hàm/biến được định nghĩa trước khi dùng.
- Ghi cấu hình thí nghiệm tập trung; không để tham số khác nhau ẩn trong nhiều cell.
- Mỗi bảng/biểu đồ quan trọng đi kèm lời giải thích dựa trên output thực tế. Không viết kết luận cố định trước khi chạy dữ liệu.
- Kết quả tóm tắt bằng số nên được tạo trực tiếp từ biến/bảng kết quả. Khi chạy lại, rà lại phần diễn giải Markdown để tránh kết luận cũ.
- Các thao tác cần người dùng, như đăng nhập/mount Drive, đặt trong nhóm setup riêng và đánh dấu rõ. Sau setup, phần phân tích phải chạy tuần tự không cần nhập tay số liệu.
- Không bắt buộc `google.colab.files.upload()` hoặc `files.download()` trong luồng chính của VS Code; chọn cách truyền file đã kiểm tra phù hợp với extension và tài khoản đang dùng.
- Không giữ output lỗi đã sửa, dữ liệu xác thực, token hoặc link có chữ ký truy cập trong notebook bàn giao.
- Cho phép notebook dài theo nhu cầu giải thích. Không rút nội dung chỉ để đáp ứng thời lượng trình bày chưa được chốt.

## 7. Thiết lập Google Colab trên VS Code và chạy lại

### 7.1. Kết nối runtime

Quy trình theo extension Google Colab chính thức; tên nút có thể thay đổi theo phiên bản, đối chiếu hướng dẫn nếu giao diện khác:

1. Mở thư mục dự án trong VS Code, cài **Colab** của publisher **Google** (`google.colab`) và bảo đảm extension Jupyter có sẵn.
2. Mở `notebooks/telco_churn_lr.ipynb` → **Select Kernel → Colab → Auto Connect**; hoặc chọn server CPU nếu cần.
3. Người dùng hoàn tất đăng nhập Google khi được yêu cầu. Không đưa mật khẩu/token vào code hoặc yêu cầu gửi chúng cho Codex.
4. Chạy cell kiểm tra và ghi nhận kernel đang kết nối Colab. Lưu phiên bản extension trong hướng dẫn thiết lập nếu truy xuất được.

Nguồn thao tác kết nối: [Google Colab extension](https://marketplace.visualstudio.com/items?itemName=Google.colab) và [User Guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).

Cell kiểm tra môi trường nên in `sys.version`, `sys.executable`, `platform.platform()`, `Path.cwd()` và phiên bản các gói phân tích. Dùng thông tin này cùng kernel/server đang chọn để ghi bằng chứng runtime; riêng việc import được `google.colab` chưa đủ chứng minh đang chạy trên hạ tầng Colab.

Không yêu cầu `.venv` trên máy cá nhân để chạy kernel Colab. Terminal VS Code thông thường có thể là terminal máy cá nhân; cài gói ở đó không đồng nghĩa đã cài vào runtime notebook.

### 7.2. Đường dẫn và đưa dữ liệu vào runtime

Định nghĩa tập trung `PROJECT_ROOT`, `DATA_PATH`, `OUTPUT_DIR`, `EXPORT_DIR` và tùy chọn `PERSIST_ROOT`. Có thể dùng `/content/telco_churn_lr` làm project root trên Colab; sau đó dùng đường dẫn tương đối bên dưới. Không rải đường dẫn tuyệt đối trong các cell.

Notebook phải kiểm tra file tồn tại, in đường dẫn đã chọn và báo lỗi có hướng dẫn nếu thiếu. Không giả định `C:\...` trên máy Windows hoặc file cạnh notebook trong VS Code đã tự có trên Colab.

Các cách đưa dữ liệu vào, chọn một cách chính và ghi vào README:

- Với CSV người dùng cung cấp: dùng **Upload to Colab** từ VS Code Explorer; kiểm tra vị trí thực tế trong Colab Contents rồi đặt `DATA_PATH`.
- Với bản CSV công khai đã xác minh: tải bằng code trong runtime, ghi URL/commit, checksum và kiểm tra schema trước khi dùng.
- Với dữ liệu trên Drive: dùng lệnh **Colab: Mount Google Drive to Server...**, chạy cell được thêm và cấu hình đường dẫn trong Drive.

Nguồn thao tác truyền file/mount: [User Guide — Working with files](https://github.com/googlecolab/colab-vscode/wiki/User-Guide#working-with-files). Không yêu cầu dùng SSH tunnel, server mount thử nghiệm hoặc quyền truy cập nâng cao cho bài này.

`config.json` phải được đưa vào runtime theo cách tương tự, hoặc sinh từ một cấu hình mặc định được định nghĩa rõ trong notebook. Nếu có cả hai, quy định nguồn ưu tiên, in cấu hình thực dùng và lưu lại; không để cấu hình local và runtime lệch nhau âm thầm.

### 7.3. Cài phụ thuộc trong kernel

Có cell cài gói bằng `%pip` trong notebook, ví dụ dùng file requirements đã đưa vào runtime. Với notebook tự chứa, có thể khai báo danh sách phiên bản phụ thuộc trực tiếp ngay trong cell setup rồi xuất cùng danh sách thành `requirements-lock.txt`.

- Chỉ cài những gói cần cho bài; không nâng cấp hàng loạt mọi gói sẵn có của Colab.
- Dùng phiên bản tương thích thực tế đã kiểm tra. Lần đầu chưa xác nhận phiên bản phải ghi trạng thái; sau khi chạy thành công lưu bộ phiên bản.
- Nếu thay một gói đã import và cần restart kernel, hướng dẫn restart rồi chạy lại từ setup; không tiếp tục trong trạng thái trộn phiên bản.
- CPU đủ cho bài toán này. Không chuyển GPU chỉ để tăng tài nguyên khi thuật toán không cần.

### 7.4. Chạy và lưu kết quả

Quy trình chuẩn: **kết nối kernel → hoàn tất setup tương tác → kiểm tra data/config → chạy tuần tự toàn bộ phần phân tích → kiểm tra → xuất kết quả → lưu notebook**.

Hai nơi lưu phải được kiểm tra riêng:

| Sản phẩm | Nơi tạo | Cách giữ lại |
|---|---|---|
| Mã nguồn và output hiển thị của `.ipynb` | File notebook đang mở trong VS Code | Save notebook, kiểm tra file local đã có output mới |
| CSV/JSON/hình/báo cáo/ZIP do code ghi | Filesystem của runtime Colab | Sao chép ra nơi bền vững đã cấu hình; không coi Save notebook là đã lưu các file này |

Mặc định thực hiện phân tích trong `/content`, đóng gói `results`, `reports` và config/requirements có trong runtime thành ZIP rồi sao chép sang thư mục Drive của dự án nếu người dùng đã mount. Không đóng gói token, cache không cần thiết hoặc tự lồng ZIP vào chính nó. Lưu notebook local riêng cùng gói kết quả; việc code trong runtime tự tìm thấy file notebook local không phải giả định hợp lệ.

Nếu không dùng Drive, phải chọn và kiểm tra một cách tải ZIP về máy tương thích với môi trường hiện tại; nếu chỉ lưu được ở ổ tạm thì đánh dấu bước lưu bền vững còn thiếu. Có thể tải ZIP đã lưu trên Drive về thư mục dự án local. Không khẳng định các file đã tự đồng bộ giữa hai nơi.

Colab runtime có thể bị thu hồi và file trên ổ tạm không phải nơi lưu lâu dài. Sau reset, dùng lại notebook đã lưu, dữ liệu và cấu hình đã ghi để tạo lại môi trường. Xem [Colab FAQ](https://research.google.com/colaboratory/faq.html).

### 7.5. Hợp đồng kết quả và bằng chứng chạy

`metrics.json` tối thiểu gồm:

- Thông tin dữ liệu/split: hash, seed, số mẫu train/test và từng lớp.
- Cấu hình cuối, threshold, metric chọn và tóm tắt CV.
- Metric test của LR và baseline; TN/FP/FN/TP.
- Run ID/thời điểm, phiên bản thư viện và loại môi trường thực thi.

Các file results phải thuộc cùng run/config. Nếu đổi dữ liệu hoặc config, ghi nhận run mới; không trộn bảng/hình giữa các lần chạy.

`reports/verification.md` phân biệt ba trạng thái: **notebook đã tạo**, **đã kiểm tra/chạy cục bộ**, **đã chạy trên Colab từ VS Code**. Chỉ đánh dấu trạng thái cuối khi có bằng chứng thực thi tương ứng. Nếu Codex chưa truy cập được VS Code/runtime của người dùng, hoàn thành notebook và kiểm tra khả dụng trước, rồi nêu chính xác bước kết nối/chạy còn cần người dùng thực hiện. Không giả vờ đã điều khiển môi trường chưa truy cập được.

Giai đoạn 7 chỉ được đánh dấu hoàn thành khi đã xác minh chạy trên Colab từ VS Code và lưu được kết quả. Nếu còn chờ kết nối, bàn giao các phần đã xong nhưng giữ trạng thái nghiệm thu môi trường là chưa hoàn thành.

## 8. Checklist nghiệm thu cuối cùng

### Dữ liệu và khả năng tái lập

- [x] Có nguồn dữ liệu, phiên bản/checksum, schema và audit thực tế.
- [x] ID không trùng giữa train/test; ID và target không đi vào feature.
- [x] Có cấu hình và seed, chạy từ đầu được; notebook không phụ thuộc trạng thái ẩn.
- [x] Không có số liệu giả hoặc số tham khảo bị gắn nhãn là kết quả tự chạy.

### Phương pháp

- [x] Imputer, encoder và scaler nằm trong pipeline được fit trong từng fold.
- [x] Chọn cấu hình bằng train-CV, giữ ngưỡng 0.5 và không sửa theo test.
- [x] Có baseline; Precision/Recall/F1 tính đúng cho churn; AUC từ score.
- [x] Confusion matrix khớp số mẫu và metric.
- [x] Có giải thích giới hạn của dữ liệu công khai, một split và dữ liệu mẫu.

### Notebook, kết quả và giải thích

- [x] Biểu đồ tỷ lệ có mẫu số và n của nhóm, ghi rõ train/test/toàn dữ liệu.
- [x] Diễn giải hệ số có dấu, nhóm tham chiếu/đơn vị; odds không bị gọi thành xác suất.
- [x] Đề xuất marketing là giả thuyết cần kiểm chứng, không là kết quả nhân quả.
- [x] Notebook, báo cáo và results dùng cùng số liệu.
- [x] Notebook có đủ Markdown, code, output và nhận xét; chạy theo thứ tự sau setup.
- [x] Hình/bảng đọc rõ, font tiếng Việt và nhãn lớp/đơn vị đúng.
- [x] Đã xác minh kernel Colab trong VS Code, hoặc ghi rõ bước này còn chờ thực hiện.
- [ ] Đã lưu notebook kèm output và xuất các file kết quả khỏi ổ tạm runtime.
- [x] Chưa đưa sản phẩm thuyết trình vào điều kiện hoàn thành giai đoạn phân tích.
- [x] Có source audit và verification phân biệt rõ đã đọc, đã chạy và chưa xác minh.

Các kiểm tra cần tập trung vào nguy cơ thật: leakage, metric sai lớp, số mẫu không khớp, sai nhóm tham chiếu và tài liệu lệch số liệu. Không xây bộ test lớn chỉ để kiểm tra lại hành vi sẵn có của scikit-learn.

## 9. Quy tắc xử lý khi bị chặn

| Tình huống | Hành động |
|---|---|
| Không tải được nguồn tham khảo | Ghi trạng thái, dùng nguồn đã đọc/tài liệu chuẩn cho phần còn lại; không nhận đã kiểm chứng nguồn bị chặn |
| Không có dữ liệu và không tải được | Hoàn thành cấu trúc, config, code chưa chạy và khung nội dung có chỗ trống; yêu cầu đúng file CSV; không tạo dữ liệu giả để điền kết quả |
| Dữ liệu khác schema | Đối chiếu phiên bản và báo khác biệt trước khi đổi thiết kế; không trộn bản 21 cột với bản mở rộng |
| API thư viện không tương thích | Sửa theo tài liệu của phiên bản đang dùng, ghi thay đổi và chạy lại kiểm tra cần thiết |
| Kết quả kém mong đợi | Kiểm tra bug/leakage/nhãn; nếu đúng thì giữ kết quả và giải thích giới hạn |
| Chưa kết nối được Colab trong VS Code | Hoàn thiện notebook và kiểm tra cục bộ nếu có thể; cung cấp đúng các bước kết nối, ghi còn chờ xác minh Colab |
| Runtime ngắt hoặc reset | Tạo lại môi trường từ setup, nạp đúng dữ liệu/config và chạy lại; dùng các kết quả đã xuất để đối chiếu |
| File ở máy cá nhân nhưng Colab không thấy | Upload/tải/mount theo hướng dẫn, kiểm tra đường dẫn runtime; không đổi sang kernel local mà vẫn nhận là Colab |
| Chưa có nơi lưu bền vững | Tạo ZIP, hướng dẫn lưu/tải về và ghi trạng thái chưa hoàn tất lưu kết quả |

Không yêu cầu người dùng xác nhận lại mọi giai đoạn. Báo tiến độ theo kết quả thực tế, tiếp tục phần đã đủ điều kiện và chỉ dừng phần thực sự bị chặn.

## 10. Câu lệnh khởi động để gửi cho Codex

Sao chép đoạn dưới cùng với file này vào phiên Codex triển khai:

> Hãy đọc toàn bộ PLAN.md và triển khai đồ án theo kế hoạch này. Ưu tiên notebook `.ipynb` đầy đủ và kết quả thực nghiệm trước; chưa tạo slide hoặc lời thuyết trình, không đặt giới hạn số slide/thời lượng. Tôi sẽ mở và sửa notebook trong VS Code, dùng extension Google Colab chính thức để chạy bằng kernel Colab. Hãy kiểm tra workspace, đề bài và dữ liệu; tạo notebook với các cell setup, kiểm tra dữ liệu, EDA, pipeline, cross-validation, đánh giá Logistic Regression, diễn giải và xuất kết quả. Viết Markdown giải thích tiếng Việt, chạy kiểm tra trong môi trường truy cập được, cập nhật checklist và cung cấp hướng dẫn VS Code–Colab cụ thể. Không bịa số liệu hoặc nhận đã chạy trên Colab khi chưa có bằng chứng. Không chọn mô hình/ngưỡng bằng test. Nếu bị chặn bởi đăng nhập hoặc kết nối, hoàn thành các phần còn làm được rồi báo chính xác thao tác còn cần. Khi bàn giao, nêu notebook và file kết quả, môi trường đã chạy, kiểm tra đã thực hiện và giới hạn còn lại.

## 11. Trạng thái triển khai

- [x] Giai đoạn 0: đầu vào, notebook và thiết lập VS Code–Colab.
- [x] Giai đoạn 1: kiểm tra dữ liệu và split.
- [x] Giai đoạn 2: EDA.
- [x] Giai đoạn 3: pipeline và cross-validation.
- [x] Giai đoạn 4: đánh giá cuối và phân tích lỗi.
- [x] Giai đoạn 5: diễn giải và đề xuất.
- [x] Giai đoạn 6: hoàn thiện notebook và tổng hợp kết quả.
- [ ] Giai đoạn 7: chạy lại trên Colab, xuất kết quả và nghiệm thu.

Các ô còn để trống có chủ đích: chúng là các bước nghiệm thu Colab chưa có bằng chứng thực thi trong môi trường hiện tại.

**Cập nhật triển khai 01/10/2026:** notebook và các giai đoạn 0–6 đã được chạy/kiểm tra cục bộ bằng Python 3.11.15, run ID `20260930T190508Z`. Giai đoạn 7 vẫn mở vì môi trường hiện tại không có quyền đăng nhập/chọn kernel Colab trong VS Code. Các mục riêng của Giai đoạn 7 liên quan restart/chạy Colab, lưu output Colab và đưa kết quả khỏi `/content` tiếp tục để trống; xem `telco_churn_lr/reports/verification.md` và `telco_churn_lr/reports/colab_vscode_setup.md`.
