# -*- coding: utf-8 -*-
"""
ỨNG DỤNG WEB DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH (BCTC)
Sử dụng mô hình Beneish M-Score 8 biến kết hợp Thuật toán Logistic Regression
Phát triển trên nền tảng Streamlit
"""

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# ------------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG & GIAO DIỆN STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Dự Báo Gian Lận BCTC | Beneish M-Score",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy biến CSS để giao diện chuyên nghiệp, trực quan chuẩn ứng dụng tài chính
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-safe {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        color: #065F46;
        padding: 14px 20px;
        border-radius: 12px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
    }
    .status-danger {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
        padding: 14px 20px;
        border-radius: 12px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
    }
    .warning-item {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 0 6px 6px 0;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. ĐỊNH NGHĨA BIẾN & THUẬT TOÁN BENEISH M-SCORE
# ------------------------------------------------------------------------------
FEATURE_COLS = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']

FEATURE_LABELS = {
    'DSRI': 'DSRI (Chỉ số Phải thu trên Doanh thu)',
    'GMI': 'GMI (Chỉ số Biên lợi nhuận gộp)',
    'AQI': 'AQI (Chỉ số Chất lượng tài sản)',
    'SGI': 'SGI (Chỉ số Tăng trưởng doanh thu)',
    'DEPI': 'DEPI (Chỉ số Tỷ lệ khấu hao)',
    'SGAI': 'SGAI (Chỉ số Chi phí bán hàng & QLDN)',
    'LVGI': 'LVGI (Chỉ số Đòn bẩy nợ tài chính)',
    'TATA': 'TATA (Tổng biến dồn tích trên Tổng tài sản)'
}

FEATURE_THRESHOLDS = {
    'DSRI': (1.05, 'lớn hơn', 'Khoản phải thu tăng bất thường so với doanh thu; rủi ro ghi nhận doanh thu ảo/doanh thu chưa thu được tiền.'),
    'GMI': (1.05, 'lớn hơn', 'Biên lợi nhuận gộp giảm sút; doanh nghiệp đối mặt áp lực tài chính dễ dẫn đến động cơ thổi phồng lợi nhuận.'),
    'AQI': (1.05, 'lớn hơn', 'Tỷ trọng tài sản phi vật chất/vô hình tăng; dấu hiệu vốn hóa chi phí bất hợp lý để trì hoãn ghi nhận lỗ.'),
    'SGI': (1.20, 'lớn hơn', 'Tăng trưởng doanh thu đột biến; áp lực duy trì chỉ tiêu tăng trưởng có thể thúc đẩy gian lận.'),
    'DEPI': (1.05, 'lớn hơn', 'Tỷ lệ khấu hao giảm; dấu hiệu kéo dài thời gian khấu hao tài sản cố định nhằm giảm chi phí, tăng lãi.'),
    'SGAI': (1.05, 'lớn hơn', 'Chi phí bán hàng và quản lý tăng nhanh hơn doanh thu; giảm hiệu quả kiểm soát chi phí hoạt động.'),
    'LVGI': (1.10, 'lớn hơn', 'Đòn bẩy nợ tăng vọt; nguy cơ vi phạm các cam kết vay nợ thúc đẩy gian lận báo cáo.'),
    'TATA': (0.05, 'lớn hơn', 'Biến dồn tích (Accruals) cao; lợi nhuận kế toán không đi kèm dòng tiền từ hoạt động kinh doanh thực tế.')
}

def tinh_beneish_mscore_chuan(dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata):
    """Tính chỉ số Beneish M-Score theo công thức gốc 8 biến của GS. Messod Beneish"""
    m_score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.037 * tata
        + 0.0327 * lvgi
    )
    return m_score

# ------------------------------------------------------------------------------
# 3. HUẤN LUYỆN & CACHING MÔ HÌNH LOGISTIC REGRESSION
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang tải dữ liệu và huấn luyện mô hình học máy...")
def load_and_train_model(dataset_source):
    """Đọc dữ liệu, chia 80% train / 20% test có stratify, huấn luyện Logistic Regression."""
    if isinstance(dataset_source, str) and os.path.exists(dataset_source):
        df = pd.read_csv(dataset_source)
    elif hasattr(dataset_source, 'read'):
        df = pd.read_csv(dataset_source)
    else:
        # Dữ liệu fallback nếu chưa có file
        raise FileNotFoundError("Không tìm thấy tệp dữ liệu huấn luyện!")

    # Đảm bảo đúng thứ tự 8 biến
    X = df[['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']]
    y = df['FRAUD_FLAG']

    # Phân chia dữ liệu theo đúng tỷ lệ trong notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Khởi tạo và huấn luyện Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Đánh giá trên tập test
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

    cm = confusion_matrix(y_test, y_pred)

    coef_df = pd.DataFrame({
        'Chỉ số': X.columns,
        'Hệ số (Weight)': model.coef_[0],
        'Chiều tác động': ['Tăng nguy cơ (Dương)' if c > 0 else 'Giảm nguy cơ (Âm)' for c in model.coef_[0]]
    }).sort_values(by='Hệ số (Weight)', ascending=False)

    # Giá trị trung bình của nhóm minh bạch và nhóm gian lận để so sánh đối chuẩn
    mean_safe = df[df['FRAUD_FLAG'] == 0][FEATURE_COLS].mean()
    mean_fraud = df[df['FRAUD_FLAG'] == 1][FEATURE_COLS].mean()

    return {
        'model': model,
        'metrics': metrics,
        'cm': cm,
        'coef_df': coef_df,
        'df': df,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_prob': y_prob,
        'mean_safe': mean_safe,
        'mean_fraud': mean_fraud
    }

# ------------------------------------------------------------------------------
# 4. THANH BÊN (SIDEBAR) ĐIỀU KHIỂN
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("Hệ Thống Phân Tích")
    st.caption("Ứng dụng Trí tuệ Nhân tạo & Tài chính Doanh nghiệp")

    st.markdown("---")
    st.subheader("⚙️ Cấu hình Mô hình")

    # Tùy chọn nguồn dữ liệu
    data_file_default = "MScore_data.csv"
    uploaded_data = st.file_uploader(
        "Tải lên bộ dữ liệu huấn luyện mới (Tùy chọn):",
        type=["csv"],
        help="Mặc định hệ thống sẽ dùng file MScore_data.csv có sẵn trong kho mã nguồn."
    )

    data_source = uploaded_data if uploaded_data is not None else data_file_default

    # Ngưỡng phân loại linh hoạt
    threshold = st.slider(
        "🎯 Ngưỡng quyết định rủi ro (Threshold):",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Nếu Xác suất gian lận >= Ngưỡng này -> Cảnh báo Red-Flag. Trong kiểm toán và đầu tư thận trọng, có thể hạ ngưỡng xuống 0.35 - 0.45 để phát hiện tối đa gian lận."
    )

    st.markdown("---")
    st.markdown("""
    **Thông tin kỹ thuật:**
    - **Thuật toán:** Logistic Regression
    - **Bộ chỉ số:** Beneish M-Score (8 biến)
    - **Dữ liệu chuẩn:** 500 BCTC kiểm toán
    - **Tỷ lệ Train/Test:** 80% / 20% Phân tầng
    """)

    st.markdown("---")
    st.caption("Phiên bản v1.0.0 • Triển khai trên Streamlit Cloud")

# Tải mô hình
try:
    trained_bundle = load_and_train_model(data_source)
    model = trained_bundle['model']
    metrics = trained_bundle['metrics']
    coef_df = trained_bundle['coef_df']
    mean_safe = trained_bundle['mean_safe']
    mean_fraud = trained_bundle['mean_fraud']
    dataset_df = trained_bundle['df']
except Exception as e:
    st.error(f"❌ Không thể tải hoặc huấn luyện mô hình: {str(e)}")
    st.stop()

# ------------------------------------------------------------------------------
# 5. TIÊU ĐỀ CHÍNH CỦA ỨNG DỤNG
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">⚖️ HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Mô hình Định lượng Beneish M-Score kết hợp Trí tuệ Nhân tạo (Logistic Regression) hỗ trợ Kiểm toán viên & Nhà đầu tư phát hiện thao túng BCTC</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 6. CÁC PHÂN HỆ CHỨC NĂNG (TABS)
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Dự Báo Đơn Lẻ (Doanh Nghiệp)",
    "📁 Dự Báo Hàng Loạt (Batch)",
    "📊 Hiệu Năng Mô Hình & Dữ Liệu",
    "📖 Cẩm Nang 8 Chỉ Số M-Score"
])

# ==============================================================================
# TAB 1: DỰ BÁO ĐƠN LẺ
# ==============================================================================
with tab1:
    st.markdown("### 📝 Nhập thông số BCTC của Doanh nghiệp cần thẩm định")
    
    # Nút bấm nạp nhanh dữ liệu mẫu
    col_preset1, col_preset2, col_preset3 = st.columns(3)
    
    # Khởi tạo giá trị mặc định trong session_state nếu chưa có
    default_vals = {
        'dsri': 1.45, 'gmi': 1.15, 'aqi': 1.30, 'sgi': 1.60,
        'depi': 0.82, 'sgai': 1.12, 'lvgi': 1.25, 'tata': 0.12
    }
    for k, v in default_vals.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with col_preset1:
        if st.button("🟢 Nạp Mẫu: Doanh Nghiệp Minh Bạch", use_container_width=True):
            st.session_state.dsri = 0.92
            st.session_state.gmi = 0.95
            st.session_state.aqi = 0.88
            st.session_state.sgi = 1.05
            st.session_state.depi = 1.02
            st.session_state.sgai = 0.96
            st.session_state.lvgi = 0.94
            st.session_state.tata = -0.02
            st.rerun()

    with col_preset2:
        if st.button("🔴 Nạp Mẫu: Doanh Nghiệp Nguy Cơ Cao (Red Flag)", use_container_width=True):
            st.session_state.dsri = 1.68
            st.session_state.gmi = 1.45
            st.session_state.aqi = 1.42
            st.session_state.sgi = 1.85
            st.session_state.depi = 0.75
            st.session_state.sgai = 1.25
            st.session_state.lvgi = 1.40
            st.session_state.tata = 0.18
            st.rerun()

    with col_preset3:
        if st.button("⚖️ Nạp Mẫu: Mặc Định từ Notebook", use_container_width=True):
            st.session_state.dsri = 1.45
            st.session_state.gmi = 1.15
            st.session_state.aqi = 1.30
            st.session_state.sgi = 1.60
            st.session_state.depi = 0.82
            st.session_state.sgai = 1.12
            st.session_state.lvgi = 1.25
            st.session_state.tata = 0.12
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Form nhập liệu 8 chỉ số M-Score chia làm 2 cột
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("##### 📌 Nhóm Chỉ Số Doanh Thu & Biên Lợi Nhuận")
        val_dsri = st.number_input(
            "1. DSRI - Days Sales in Receivables Index:",
            min_value=0.0, max_value=10.0,
            value=float(st.session_state.dsri), step=0.01,
            help="Tỷ lệ số ngày thu tiền khách hàng kỳ này so với kỳ trước. Giá trị > 1.05 cho thấy khoản phải thu tăng nhanh bất thường."
        )
        val_gmi = st.number_input(
            "2. GMI - Gross Margin Index:",
            min_value=0.0, max_value=10.0,
            value=float(st.session_state.gmi), step=0.01,
            help="Tỷ lệ biên lợi nhuận gộp kỳ trước so với kỳ này. Giá trị > 1.05 cho thấy biên lợi nhuận gộp đang xấu đi."
        )
        val_aqi = st.number_input(
            "3. AQI - Asset Quality Index:",
            min_value=0.0, max_value=10.0,
            value=float(st.session_state.aqi), step=0.01,
            help="Chỉ số chất lượng tài sản phi lưu động. Giá trị > 1.05 cho thấy doanh nghiệp đang tăng vốn hóa chi phí."
        )
        val_sgi = st.number_input(
            "4. SGI - Sales Growth Index:",
            min_value=0.0, max_value=10.0,
            value=float(st.session_state.sgi), step=0.01,
            help="Tỷ lệ tăng trưởng doanh thu so với kỳ trước."
        )

    with c_right:
        st.markdown("##### 📌 Nhóm Chỉ Số Chi Phí, Đòn Bẩy & Dồn Tích")
        val_depi = st.number_input(
            "5. DEPI - Depreciation Index:",
            min_value=0.0, max_value=10.0,
            value=float(st.session_state.depi), step=0.01,
            help="Tỷ lệ khấu hao kỳ trước so với kỳ này. Giá trị > 1.05 cho thấy tốc độ khấu hao bị hạ thấp để đẩy lợi nhuận kế toán."
        )
        val_sgai = st.number_input(
            "6. SGAI - Sales, General & Admin Expense Index:",
            min_value=0.0, max_value=10.0,
            value=float(st.session_state.sgai), step=0.01,
            help="Tỷ lệ chi phí quản lý & bán hàng so với doanh thu."
        )
        val_lvgi = st.number_input(
            "7. LVGI - Leverage Index:",
            min_value=0.0, max_value=10.0,
            value=float(st.session_state.lvgi), step=0.01,
            help="Chỉ số thay đổi đòn bẩy nợ tài chính trên tổng tài sản."
        )
        val_tata = st.number_input(
            "8. TATA - Total Accruals to Total Assets:",
            min_value=-2.0, max_value=2.0,
            value=float(st.session_state.tata), step=0.01,
            help="Tổng biến dồn tích trên tổng tài sản = (Lợi nhuận sau thuế - Dòng tiền từ HĐKD) / Tổng tài sản. TATA > 0.05 là dấu hiệu rủi ro cao."
        )

    st.markdown("---")

    # Thực hiện dự báo
    input_df = pd.DataFrame([[val_dsri, val_gmi, val_aqi, val_sgi, val_depi, val_sgai, val_lvgi, val_tata]],
                            columns=FEATURE_COLS)
    
    fraud_prob = model.predict_proba(input_df)[0][1]
    is_fraud = fraud_prob >= threshold
    beneish_m = tinh_beneish_mscore_chuan(val_dsri, val_gmi, val_aqi, val_sgi, val_depi, val_sgai, val_lvgi, val_tata)
    beneish_flag = beneish_m > -1.78

    st.markdown("### 📊 Kết Quả Thẩm Định Rủi Ro Gian Lận")

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        # Hiển thị huy hiệu kết luận
        if is_fraud:
            st.markdown(f"""
            <div class="status-danger">
                🚨 CẢNH BÁO RED-FLAG: NGUY CƠ THAO TÚNG BCTC CAO!<br>
                <span style="font-size: 0.95rem; font-weight: normal;">
                Xác suất rủi ro gian lận vượt ngưỡng kiểm soát ({fraud_prob:.1%} &ge; {threshold:.0%})
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-safe">
                ✅ ĐÁNH GIÁ: BCTC CÓ ĐỘ TIN CẬY CAO (MINH BẠCH)<br>
                <span style="font-size: 0.95rem; font-weight: normal;">
                Xác suất rủi ro gian lận trong mức cho phép ({fraud_prob:.1%} &lt; {threshold:.0%})
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Biểu đồ Gauge Chart thể hiện xác suất
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fraud_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "<b>Xác Suất Gian Lận (AI Dự Báo)</b>", 'font': {'size': 18}},
            delta={'reference': threshold * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            number={'suffix': "%", 'font': {'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': "#EF4444" if is_fraud else "#10B981"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E2E8F0",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.25)'},
                    {'range': [30, 50], 'color': 'rgba(245, 158, 11, 0.25)'},
                    {'range': [50, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # So sánh với mô hình Beneish M-Score truyền thống
        st.markdown("##### 📌 Đối chiếu Mô hình Gốc Beneish M-Score (1999):")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Điểm số M-Score", f"{beneish_m:.3f}")
        with m_col2:
            m_eval = "🔴 Gian lận (M > -1.78)" if beneish_flag else "🟢 Bình thường (M &le; -1.78)"
            st.metric("Đánh giá M-Score", m_eval)

    with res_col2:
        st.markdown("##### 🕸️ Biểu đồ Radar: Đối chiếu với Dữ liệu Kiểm toán")
        
        # Biểu đồ Radar so sánh với giá trị trung bình mẫu
        radar_categories = FEATURE_COLS
        user_values = [val_dsri, val_gmi, val_aqi, val_sgi, val_depi, val_sgai, val_lvgi, val_tata]
        safe_values = [mean_safe[col] for col in FEATURE_COLS]
        fraud_values = [mean_fraud[col] for col in FEATURE_COLS]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=user_values,
            theta=radar_categories,
            fill='toself',
            name='DN Hiện Tại',
            line_color='#2563EB'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=safe_values,
            theta=radar_categories,
            name='TB Minh Bạch',
            line=dict(color='#10B981', dash='dot')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=fraud_values,
            theta=radar_categories,
            name='TB Gian Lận',
            line=dict(color='#EF4444', dash='dash')
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[min(0, min(user_values)), max(2.5, max(user_values))])),
            showlegend=True,
            height=320,
            margin=dict(l=30, r=30, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Liệt kê các chỉ số cảnh báo vi phạm
    st.markdown("---")
    st.markdown("##### ⚠️ Phân Tích Cảnh Báo Các Chỉ Số Có Dấu Hiệu Bất Thường:")
    
    current_metrics_map = {
        'DSRI': val_dsri, 'GMI': val_gmi, 'AQI': val_aqi, 'SGI': val_sgi,
        'DEPI': val_depi, 'SGAI': val_sgai, 'LVGI': val_lvgi, 'TATA': val_tata
    }
    
    flagged_indicators = []
    for col, (thresh, relation, desc) in FEATURE_THRESHOLDS.items():
        val = current_metrics_map[col]
        if (relation == 'lớn hơn' and val > thresh):
            flagged_indicators.append((col, val, thresh, desc))

    if flagged_indicators:
        cols_flag = st.columns(2)
        for idx, (col, val, thresh, desc) in enumerate(flagged_indicators):
            with cols_flag[idx % 2]:
                st.markdown(f"""
                <div class="warning-item">
                    <strong>⚠️ {col} = {val:.2f}</strong> (Vượt mức tham chiếu an toàn {thresh}):<br>
                    <span style="color: #4B5563;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("✨ Không có chỉ số nào vượt ngưỡng cảnh báo đỏ. Báo cáo tài chính phản ánh cấu trúc hoạt động ổn định.")

# ==============================================================================
# TAB 2: DỰ BÁO HÀNG LOẠT (BATCH PREDICTION)
# ==============================================================================
with tab2:
    st.markdown("### 📁 Dự Báo Gian Lận Hàng Loạt Từ Tệp CSV / Excel")
    st.markdown("""
    Phân hệ này cho phép các chuyên viên phân tích hoặc kiểm toán viên tải lên danh mục gồm hàng chục hoặc hàng trăm mã chứng khoán 
    để rà soát rủi ro thao túng BCTC đồng loạt chỉ trong vài giây.
    """)

    col_upload, col_template = st.columns([2, 1])

    with col_upload:
        batch_file = st.file_uploader(
            "Tải lên tệp danh sách doanh nghiệp (.csv hoặc .xlsx):",
            type=["csv", "xlsx"]
        )

    with col_template:
        st.markdown("##### 📥 Tải tệp dữ liệu mẫu:")
        st.caption("Tệp mẫu chứa sẵn các cột chuẩn: DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA")
        
        # Tạo file mẫu
        template_df = pd.DataFrame({
            'MA_CK': ['VNM', 'HPG', 'VIC', 'FPT', 'NVL', 'DQC'],
            'DSRI': [0.95, 1.02, 1.45, 0.98, 1.62, 1.85],
            'GMI': [0.98, 1.01, 1.18, 0.95, 1.35, 1.40],
            'AQI': [0.90, 0.94, 1.25, 0.89, 1.38, 1.52],
            'SGI': [1.08, 1.12, 1.55, 1.15, 1.65, 1.70],
            'DEPI': [1.01, 0.98, 0.85, 1.00, 0.78, 0.72],
            'SGAI': [0.97, 1.00, 1.14, 0.99, 1.20, 1.30],
            'LVGI': [0.92, 1.05, 1.30, 0.88, 1.42, 1.55],
            'TATA': [-0.01, 0.02, 0.14, -0.03, 0.16, 0.22]
        })
        csv_template = template_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="⬇️ Tải File Mẫu Chuẩn (.CSV)",
            data=csv_template,
            file_name="mau_du_lieu_mscore_doanh_nghiep.csv",
            mime="text/csv",
            use_container_width=True
        )

    if batch_file is not None:
        try:
            if batch_file.name.endswith('.csv'):
                input_batch = pd.read_csv(batch_file)
            else:
                input_batch = pd.read_excel(batch_file)

            # Kiểm tra xem có đủ 8 cột cần thiết không (không phân biệt hoa thường)
            df_cols_upper = {c.upper().strip(): c for c in input_batch.columns}
            missing_features = [col for col in FEATURE_COLS if col not in df_cols_upper]

            if missing_features:
                st.error(f"❌ Tệp tải lên còn thiếu các cột bắt buộc: {', '.join(missing_features)}")
                st.info("Vui lòng tải tệp mẫu chuẩn ở góc trên bên phải để đối chiếu tên cột.")
            else:
                # Trích xuất dữ liệu đúng 8 cột
                eval_features_cols = [df_cols_upper[col] for col in FEATURE_COLS]
                X_batch = input_batch[eval_features_cols].copy()
                X_batch.columns = FEATURE_COLS # Chuẩn hóa tên cột
                
                # Dự báo xác suất
                batch_probs = model.predict_proba(X_batch)[:, 1]
                batch_flags = (batch_probs >= threshold).astype(int)
                
                # Tính M-score chuẩn
                batch_mscore = [
                    tinh_beneish_mscore_chuan(
                        row['DSRI'], row['GMI'], row['AQI'], row['SGI'],
                        row['DEPI'], row['SGAI'], row['LVGI'], row['TATA']
                    ) for _, row in X_batch.iterrows()
                ]

                # Gắn kết quả vào dataframe
                result_df = input_batch.copy()
                result_df['Xac_Suat_Gian_Lan'] = np.round(batch_probs, 4)
                result_df['Phan_Loai_AI'] = ['🔴 RED-FLAG (Gian lận)' if f == 1 else '🟢 MINH BẠCH (An toàn)' for f in batch_flags]
                result_df['Beneish_M_Score'] = np.round(batch_mscore, 3)
                result_df['M_Score_Eval'] = ['🔴 Cảnh báo' if m > -1.78 else '🟢 An toàn' for m in batch_mscore]

                st.markdown("---")
                st.markdown("### 📈 Báo Cáo Tổng Hợp Danh Mục Doanh Nghiệp")

                total_count = len(result_df)
                fraud_count = int(np.sum(batch_flags))
                safe_count = total_count - fraud_count

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Tổng số doanh nghiệp", f"{total_count} DN")
                kpi2.metric("Số lượng Minh bạch", f"{safe_count} DN", delta=f"{safe_count/total_count:.1%}")
                kpi3.metric("Số lượng Cảnh báo Red-Flag", f"{fraud_count} DN", delta=f"{fraud_count/total_count:.1%}", delta_color="inverse")
                kpi4.metric("Xác suất gian lận TB", f"{np.mean(batch_probs):.1%}")

                # Biểu đồ phân bổ tỷ lệ rủi ro
                fig_pie = px.pie(
                    names=['Minh bạch (An toàn)', 'Cảnh báo Red-Flag (Gian lận)'],
                    values=[safe_count, fraud_count],
                    color=['Minh bạch (An toàn)', 'Cảnh báo Red-Flag (Gian lận)'],
                    color_discrete_map={'Minh bạch (An toàn)': '#10B981', 'Cảnh báo Red-Flag (Gian lận)': '#EF4444'},
                    hole=0.45,
                    title="Phân bổ Mức độ Rủi ro trong Danh mục"
                )
                fig_pie.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("##### 📋 Bảng Dữ Liệu Chi Tiết:")
                st.dataframe(
                    result_df.style.apply(
                        lambda row: ['background-color: #FEE2E2' if row['Phan_Loai_AI'].startswith('🔴') else '' for _ in row],
                        axis=1
                    ),
                    use_container_width=True
                )

                # Nút tải kết quả phân tích
                csv_out = result_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Tải Kết Quả Rà Soát Đầy Đủ (.CSV)",
                    data=csv_out,
                    file_name="ket_qua_du_bao_gian_lan_bctc.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Xử lý tệp thất bại: {str(e)}")

# ==============================================================================
# TAB 3: HIỆU NĂNG MÔ HÌNH & DỮ LIỆU
# ==============================================================================
with tab3:
    st.markdown("### 📊 Đánh Giá Năng Lực Mô Hình & Trực Quan Hóa")

    st.markdown("##### 🎯 Các Chỉ Số Kiểm Định Hiệu Suất Trên Tập Test Độc Lập (20%):")
    m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
    m_c1.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.2%}")
    m_c2.metric("Độ chuẩn xác (Precision)", f"{metrics['precision']:.2%}")
    m_c3.metric("Độ nhạy phát hiện (Recall)", f"{metrics['recall']:.2%}")
    m_c4.metric("Chỉ số F1-Score", f"{metrics['f1']:.2%}")
    m_c5.metric("Chỉ số AUC-ROC", f"{metrics['roc_auc']:.4f}")

    st.markdown("---")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("##### 🔍 Ma Trận Nhầm Lẫn (Confusion Matrix)")
        cm_data = trained_bundle['cm']
        fig_cm = px.imshow(
            cm_data,
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="Dự Báo của Mô Hình", y="Thực Tế Kiểm Toán", color="Số lượng"),
            x=['Minh Bạch (0)', 'Gian Lận (1)'],
            y=['Minh Bạch (0)', 'Gian Lận (1)']
        )
        fig_cm.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Ma trận phản ánh độ chính xác giữa nhãn thực tế của doanh nghiệp và dự báo của mô hình trên 100 doanh nghiệp kiểm thử độc lập.")

    with col_roc:
        st.markdown("##### 📈 Đường Cong ROC (Receiver Operating Characteristic)")
        fpr, tpr, _ = roc_curve(trained_bundle['y_test'], trained_bundle['y_prob'])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})", line=dict(color='#2563EB', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline ngẫu nhiên', line=dict(color='gray', dash='dash')))
        fig_roc.update_layout(
            xaxis_title="Tỷ lệ Dương tính giả (FPR)",
            yaxis_title="Tỷ lệ Dương tính thật (TPR / Recall)",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(x=0.5, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.markdown("##### ⚖️ Trọng Số & Mức Độ Đóng Góp Của Từng Biến Trong Mô Hình (Feature Importance):")
    st.markdown("""
    Các biến có **hệ số dương lớn** (như TATA, DSRI, SGI) là những biến làm tăng mạnh nguy cơ BCTC bị thao túng. 
    Ngược lại, hệ số âm phản ánh xu hướng ngược chiều.
    """)

    fig_coef = px.bar(
        coef_df,
        x='Hệ số (Weight)',
        y='Chỉ số',
        orientation='h',
        color='Hệ số (Weight)',
        color_continuous_scale=['#10B981', '#F59E0B', '#EF4444'],
        text='Hệ số (Weight)'
    )
    fig_coef.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_coef.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📦 Khám Phá Tập Dữ Liệu Huấn Luyện (`MScore_data.csv`):")
    st.dataframe(dataset_df.head(10), use_container_width=True)

# ==============================================================================
# TAB 4: CẨM NANG 8 CHỈ SỐ M-SCORE
# ==============================================================================
with tab4:
    st.markdown("### 📖 Cẩm Nang Chi Tiết Về Mô Hình Beneish M-Score")
    
    st.markdown("""
    Mô hình **Beneish M-Score** được phát triển bởi Giáo sư **Messod Beneish** (Trường Kinh doanh Kelley thuộc Đại học Indiana) vào năm 1999. 
    Đây là một trong những công cụ định lượng kinh điển nhất trên thị trường tài chính thế giới, từng giúp phát hiện vụ gian lận lịch sử 
    tại tập đoàn năng lượng **Enron** trước khi tập đoàn này sụp đổ.
    
    Mô hình kết hợp **8 chỉ số tài chính** phản ánh các thủ thuật kế toán phổ biến:
    """)

    guide_data = [
        {
            "Chỉ số": "1. DSRI (Days Sales in Receivables Index)",
            "Công thức": "(Khoản phải thu_t / Doanh thu_t) / (Khoản phải thu_t-1 / Doanh thu_t-1)",
            "Mục đích & Dấu hiệu bất thường": "Đo lường sự biến động của tỷ trọng nợ phải thu trên doanh thu. Nếu DSRI > 1.0, tốc độ tăng khoản phải thu đang vượt quá tốc độ tăng trưởng doanh thu. Đây là dấu hiệu của việc nới lỏng chính sách tín dụng để ghi khống doanh thu, hoặc ghi nhận doanh thu trước khi đủ điều kiện."
        },
        {
            "Chỉ số": "2. GMI (Gross Margin Index)",
            "Công thức": "[(Doanh thu_t-1 - Giá vốn_t-1) / Doanh thu_t-1] / [(Doanh thu_t - Giá vốn_t) / Doanh thu_t]",
            "Mục đích & Dấu hiệu bất thường": "Đo lường sự suy giảm của biên lợi nhuận gộp. Khi GMI > 1.0, biên lợi nhuận gộp kỳ này đang sụt giảm so với kỳ trước. Sự suy giảm này tạo động lực rất lớn cho ban điều hành thực hiện các hành vi thao túng kế toán để che giấu tình hình kinh doanh đi xuống."
        },
        {
            "Chỉ số": "3. AQI (Asset Quality Index)",
            "Công thức": "[1 - (Tài sản ngắn hạn_t + TSCĐ_t + Chứng khoán_t) / Tổng tài sản_t] / [Kỳ t-1]",
            "Mục đích & Dấu hiệu bất thường": "Đo lường tỷ trọng tài sản phi vật chất, tài sản dài hạn khác. Nếu AQI > 1.0, doanh nghiệp có xu hướng chuyển chi phí phát sinh trong kỳ thành tài sản (vốn hóa chi phí bất hợp lý) nhằm trì hoãn việc ghi nhận chi phí vào báo cáo kết quả hoạt động kinh doanh."
        },
        {
            "Chỉ số": "4. SGI (Sales Growth Index)",
            "Công thức": "Doanh thu_t / Doanh thu_t-1",
            "Mục đích & Dấu hiệu bất thường": "Tốc độ tăng trưởng doanh thu thuần. Bản thân tăng trưởng không phải là hành vi xấu, nhưng các công ty có tốc độ tăng trưởng cao thường chịu áp lực rất lớn từ cổ đông và thị trường để duy trì tốc độ đó, do đó dễ thực hiện các hành vi làm đẹp số liệu."
        },
        {
            "Chỉ số": "5. DEPI (Depreciation Index)",
            "Công thức": "(Tỷ lệ khấu hao kỳ t-1) / (Tỷ lệ khấu hao kỳ t)",
            "Mục đích & Dấu hiệu bất thường": "Đo lường sự thay đổi của tỷ lệ khấu hao TSCĐ. Nếu DEPI > 1.0, tốc độ khấu hao trong kỳ hiện tại đang chậm lại. Đây là dấu hiệu doanh nghiệp đã kéo dài thời gian sử dụng hữu ích ước tính hoặc thay đổi phương pháp khấu hao để giảm chi phí khấu hao trong kỳ."
        },
        {
            "Chỉ số": "6. SGAI (Sales, General and Administrative expenses Index)",
            "Công thức": "(Chi phí bán hàng & QLDN_t / Doanh thu_t) / [Kỳ t-1]",
            "Mục đích & Dấu hiệu bất thường": "Đo lường hiệu quả quản lý chi phí hoạt động. Nếu SGAI > 1.0, chi phí bán hàng và quản lý doanh nghiệp đang gia tăng nhanh hơn tốc độ tăng trưởng doanh thu, báo hiệu sự suy giảm hiệu quả kiểm soát chi phí."
        },
        {
            "Chỉ số": "7. LVGI (Leverage Index)",
            "Công thức": "[(Nợ ngắn hạn_t + Nợ dài hạn_t) / Tổng tài sản_t] / [Kỳ t-1]",
            "Mục đích & Dấu hiệu bất thường": "Đo lường tỷ lệ gia tăng đòn bẩy nợ tài chính. LVGI > 1.0 cho thấy mức độ sử dụng nợ đang tăng lên, làm gia tăng gánh nặng trả lãi và nguy cơ vi phạm các điều khoản cam kết nợ với ngân hàng (debt covenants)."
        },
        {
            "Chỉ số": "8. TATA (Total Accruals to Total Assets)",
            "Công thức": "(Lợi nhuận thuần sau thuế - Dòng tiền thuần từ HĐKD) / Tổng tài sản",
            "Mục đích & Dấu hiệu bất thường": "Đo lường mức độ chênh lệch giữa lợi nhuận kế toán và dòng tiền thực thu từ hoạt động kinh doanh (Accruals). TATA càng cao phản ánh chất lượng lợi nhuận càng kém, doanh nghiệp ghi nhận lãi lớn trên giấy tờ nhưng không thu được tiền mặt thực tế."
        }
    ]

    for item in guide_data:
        with st.expander(f"📌 {item['Chỉ số']}", expanded=True):
            st.markdown(f"**Công thức tính:** `{item['Công thức']}`")
            st.markdown(f"**Ý nghĩa phân tích:** {item['Mục đích & Dấu hiệu bất thường']}")

    st.markdown("---")
    st.markdown("""
    ##### ⚖️ Công thức M-Score gốc của Giáo sư Messod Beneish:
    $$M = -4.84 + 0.920 \\times DSRI + 0.528 \\times GMI + 0.404 \\times AQI + 0.892 \\times SGI + 0.115 \\times DEPI - 0.172 \\times SGAI + 4.037 \\times TATA + 0.0327 \\times LVGI$$
    
    - **Nếu M-Score > -1.78:** Khả năng cao doanh nghiệp có thực hiện các thủ thuật gian lận, thao túng báo cáo tài chính (Red-Flag).
    - **Nếu M-Score &le; -1.78:** Doanh nghiệp có tính minh bạch cao, chưa ghi nhận dấu hiệu thao túng nghiêm trọng.
    """)
