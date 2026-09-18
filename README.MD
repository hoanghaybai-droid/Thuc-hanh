# ⚖️ HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH (BCTC)
### Ứng dụng Học máy & Mô hình Định lượng Beneish M-Score trên nền tảng Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Giới Thiệu Dự Án

Ứng dụng web được xây dựng nhằm hỗ trợ **Kiểm toán viên**, **Chuyên viên thẩm định tín dụng**, **Nhà phân tích tài chính** và **Nhà đầu tư chứng khoán** phát hiện sớm các dấu hiệu thao túng, bóp méo hoặc gian lận trên Báo cáo tài chính (BCTC) của doanh nghiệp.

Hệ thống kết hợp giữa:
1. **Mô hình định lượng Beneish M-Score (8 biến)** kinh điển của Giáo sư Messod Beneish (1999) - công cụ từng phát hiện vụ gian lận lịch sử tại tập đoàn Enron.
2. **Thuật toán Học máy Phân loại (Logistic Regression)** được huấn luyện trên dữ liệu 500 doanh nghiệp kiểm toán để tính toán xác suất rủi ro một cách linh hoạt, khách quan và trực quan hóa toàn diện.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
├── app.py                         # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt               # Danh sách thư viện phụ thuộc để triển khai
├── README.MD                      # Tài liệu hướng dẫn sử dụng và triển khai
├── MScore_data.csv                # Bộ dữ liệu 500 BCTC doanh nghiệp dùng huấn luyện
└── mo_hinh_du_bao_gian_lan_bctc.py# Script huấn luyện và kiểm định mô hình gốc từ Colab
```

---

## 🎯 8 Chỉ Số Beneish M-Score Được Sử Dụng

| Tên biến | Tên chỉ số đầy đủ | Ý nghĩa phát hiện gian lận | Ngưỡng rủi ro |
| :--- | :--- | :--- | :---: |
| **DSRI** | Days Sales in Receivables Index | Khoản phải thu tăng bất thường so với doanh thu; nghi vấn ghi nhận doanh thu ảo. | $> 1.05$ |
| **GMI** | Gross Margin Index | Biên lợi nhuận gộp suy giảm; doanh nghiệp chịu áp lực làm đẹp số liệu. | $> 1.05$ |
| **AQI** | Asset Quality Index | Tỷ lệ tài sản phi vật chất tăng; nghi vấn vốn hóa chi phí bất hợp lý. | $> 1.05$ |
| **SGI** | Sales Growth Index | Tăng trưởng doanh thu đột biến; áp lực duy trì chỉ tiêu tăng trưởng. | $> 1.20$ |
| **DEPI** | Depreciation Index | Tỷ lệ trích khấu hao chậm lại; trì hoãn chi phí để nâng lợi nhuận. | $> 1.05$ |
| **SGAI** | Sales, General & Admin Expense Index | Chi phí bán hàng và quản lý tăng nhanh hơn doanh thu; giảm hiệu quả kiểm soát. | $> 1.05$ |
| **LVGI** | Leverage Index | Đòn bẩy nợ tài chính tăng cao; rủi ro vi phạm cam kết trả nợ. | $> 1.10$ |
| **TATA** | Total Accruals to Total Assets | Chênh lệch giữa lợi nhuận kế toán và dòng tiền thực (Accruals cao = chất lượng lãi kém). | $> 0.05$ |

---

## 🚀 Các Tính Năng Nổi Bật Của Ứng Dụng

1. **🔮 Thẩm định Đơn lẻ (Doanh nghiệp cụ thể):**
   - Nhập linh hoạt 8 chỉ số qua thanh trượt hoặc ô số.
   - Các nút nạp nhanh dữ liệu mẫu: *Doanh nghiệp Minh Bạch*, *Doanh nghiệp Nguy Cơ Cao (Red Flag)*, *Dữ liệu mẫu từ Notebook*.
   - Đồng hồ đo rủi ro trực quan (Plotly Gauge Chart).
   - Đối chiếu song song với công thức M-Score truyền thống (ngưỡng cắt $-1.78$).
   - Biểu đồ Radar (Spider Chart) so sánh đa chiều giữa doanh nghiệp với mức trung bình ngành.
   - Danh sách cảnh báo chi tiết các chỉ số vi phạm ngưỡng an toàn.

2. **📁 Dự báo Hàng loạt (Batch Prediction):**
   - Tải lên tệp danh sách nhiều doanh nghiệp dưới dạng `.csv` hoặc `.xlsx`.
   - Có nút tải tệp mẫu chuẩn định dạng để người dùng dễ chuẩn bị.
   - Bảng tổng hợp kết quả tự động tô màu các mã có rủi ro cao.
   - Biểu đồ tỷ lệ rủi ro trong toàn bộ danh mục và nút tải về kết quả hoàn chỉnh (`.csv`).

3. **📊 Đánh giá Năng lực Mô hình & Trực quan hóa Dữ liệu:**
   - Thống kê đầy đủ các chỉ số: **Accuracy**, **Precision**, **Recall**, **F1-Score**, **AUC-ROC**.
   - Ma trận nhầm lẫn (Confusion Matrix) dạng Heatmap trực quan.
   - Đường cong ROC (Receiver Operating Characteristic Curve).
   - Biểu đồ tầm quan trọng và chiều tác động của từng biến đặc trưng (Logistic Regression Coefficients).

4. **📖 Cẩm nang Kế toán & 8 Chỉ số M-Score:**
   - Tra cứu nhanh ý nghĩa kinh tế, công thức chi tiết và các thủ thuật "phù phép" BCTC thường gặp.

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)

### Bước 1: Tải mã nguồn về máy
Mở Terminal / PowerShell và di chuyển vào thư mục dự án:
```bash
cd "C:\Users\ADMIN\Desktop\Tao app thuc hanh"
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Khuyến nghị)
- **Trên Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```
- **Trên macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Bước 4: Khởi chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Trình duyệt web sẽ tự động mở trang web tại địa chỉ: `http://localhost:8501`.

---

## 🌐 Hướng Dẫn Đẩy Lên GitHub & Triển Khai Miễn Phí Trên Streamlit Cloud

Để đưa ứng dụng lên mạng internet (miễn phí, có tên miền riêng để gửi cho khách hàng, giảng viên, đồng nghiệp truy cập), hãy làm theo các bước sau:

### 1. Đẩy mã nguồn lên GitHub:
1. Đăng nhập vào [GitHub](https://github.com/) và tạo một repository mới (ví dụ đặt tên: `du-bao-gian-lan-bctc`). Đặt chế độ là **Public**.
2. Mở terminal tại thư mục dự án và chạy các lệnh:
```bash
git init
git add app.py requirements.txt README.MD MScore_data.csv
git commit -m "Khoi tao web app du bao gian lan BCTC tren Streamlit"
git branch -M main
git remote add origin https://github.com/<ten-tai-khoan-github-cua-ban>/du-bao-gian-lan-bctc.git
git push -u origin main
```

### 2. Triển khai lên Streamlit Community Cloud:
1. Truy cập vào **[share.streamlit.io](https://share.streamlit.io/)** và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn vào nút **"Create app"** (hoặc **"New app"**).
3. Điền các thông tin:
   - **Repository:** Chọn repository bạn vừa tải lên (`du-bao-gian-lan-bctc`).
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (Tùy chọn):** Đặt tên miền con tùy thích (ví dụ: `du-bao-gian-lan-bctc.streamlit.app`).
4. Nhấn **"Deploy!"**.
5. Đợi khoảng 1-2 phút để hệ thống tự động cài đặt `requirements.txt` và khởi chạy ứng dụng. Sau khi hoàn tất, ứng dụng đã chính thức online 24/7!

---

## 📋 Yêu Cầu Định Dạng Tệp Dữ Liệu Khi Dự Báo Hàng Loạt

Tệp tải lên (`.csv` hoặc `.xlsx`) cần có tối thiểu các cột sau (không phân biệt chữ hoa hay thường):

```csv
MA_CK,DSRI,GMI,AQI,SGI,DEPI,SGAI,LVGI,TATA
VNM,0.95,0.98,0.90,1.08,1.01,0.97,0.92,-0.01
HPG,1.02,1.01,0.94,1.12,0.98,1.00,1.05,0.02
NVL,1.62,1.35,1.38,1.65,0.78,1.20,1.42,0.16
```

---

## ⚖️ Giấy Phép & Tuyên Bố Miễn Trừ Trách Nhiệm (Disclaimer)
- Ứng dụng được xây dựng cho mục đích nghiên cứu học thuật, hỗ trợ phân tích định lượng và cảnh báo rủi ro ban đầu.
- Kết quả cảnh báo (Red-Flag) là dấu hiệu thống kê gợi mở cho chuyên viên cần đi sâu kiểm tra chứng từ kế toán, không cấu thành kết luận pháp lý hay phán quyết tư pháp cuối cùng về gian lận.
