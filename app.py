# -*- coding: utf-8 -*-
"""
ỨNG DỤNG WEB DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH (BCTC)
Sử dụng mô hình Beneish M-Score 8 biến kết hợp Thuật toán Logistic Regression & XGBoost
Tích hợp Dashboard Phân Tích Số Liệu Chuyên Sâu
"""

import os
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

# Kiểm tra an toàn thư viện XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ModuleNotFoundError:
    HAS_XGBOOST = False

# ------------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Dự Báo Gian Lận BCTC | Beneish M-Score",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.1rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .status-safe { background-color: #ECFDF5; border: 2px solid #10B981; color: #065F46; padding: 14px 20px; border-radius: 12px; font-size: 1.25rem; font-weight: 700; text-align: center; }
    .status-danger { background-color: #FEF2F2; border: 2px solid #EF4444; color: #991B1B; padding: 14px 20px; border-radius: 12px; font-size: 1.25rem; font-weight: 700; text-align: center; }
    .warning-item { background-color: #FFFBEB; border-left: 4px solid #F59E0B; padding: 8px 12px; margin-bottom: 6px; border-radius: 0 6px 6px 0; font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

FEATURE_COLS = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']

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
    return (
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

# ------------------------------------------------------------------------------
# 2. TẠO DỮ LIỆU MẶC ĐỊNH NẾU KHÔNG CÓ FILE CSV
# ------------------------------------------------------------------------------
def generate_sample_data():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        'DSRI': np.random.uniform(0.8, 1.8, n),
        'GMI': np.random.uniform(0.8, 1.6, n),
        'AQI': np.random.uniform(0.7, 1.7, n),
        'SGI': np.random.uniform(0.9, 1.9, n),
        'DEPI': np.random.uniform(0.7, 1.3, n),
        'SGAI': np.random.uniform(0.8, 1.5, n),
        'LVGI': np.random.uniform(0.8, 1.6, n),
        'TATA': np.random.uniform(-0.1, 0.3, n),
        'FRAUD_FLAG': np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    })
    return df

# ------------------------------------------------------------------------------
# 3. HUẤN LUYỆN VÀ CACHE MÔ HÌNH
# ------------------------------------------------------------------------------
@st.cache_resource
def train_models_cached():
    file_path = "MScore_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = generate_sample_data()

    # Tính M-Score cho toàn bộ dataset để làm Dashboard
    df['M_SCORE'] = df.apply(
        lambda r: tinh_beneish_mscore_chuan(
            r['DSRI'], r['GMI'], r['AQI'], r['SGI'], r['DEPI'], r['SGAI'], r['LVGI'], r['TATA']
        ), axis=1
    )

    X = df[FEATURE_COLS]
    y = df['FRAUD_FLAG']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    lr_pred = lr_model.predict(X_test)

    lr_metrics = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred, zero_division=0),
        'recall': recall_score(y_test, lr_pred, zero_division=0),
        'f1': f1_score(y_test, lr_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, lr_prob)
    }

    lr_coef_df = pd.DataFrame({
        'Chỉ số': X.columns,
        'Hệ số (Weight)': lr_model.coef_[0]
    }).sort_values(by='Hệ số (Weight)', ascending=False)

    # XGBoost
    xgb_model, xgb_metrics, xgb_coef_df, xgb_prob = None, None, None, None

    if HAS_XGBOOST:
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            random_state=42, eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred = xgb_model.predict(X_test)

        xgb_metrics = {
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred, zero_division=0),
            'recall': recall_score(y_test, xgb_pred, zero_division=0),
            'f1': f1_score(y_test, xgb_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, xgb_prob)
        }

        xgb_coef_df = pd.DataFrame({
            'Chỉ số': X.columns,
            'Hệ số (Weight)': xgb_model.feature_importances_
        }).sort_values(by='Hệ số (Weight)', ascending=False)

    mean_safe = df[df['FRAUD_FLAG'] == 0][FEATURE_COLS].mean()
    mean_fraud = df[df['FRAUD_FLAG'] == 1][FEATURE_COLS].mean()

    return {
        'lr_model': lr_model, 'lr_metrics': lr_metrics, 'lr_coef_df': lr_coef_df, 'lr_prob': lr_prob,
        'xgb_model': xgb_model, 'xgb_metrics': xgb_metrics, 'xgb_coef_df': xgb_coef_df, 'xgb_prob': xgb_prob,
        'df': df, 'X_test': X_test, 'y_test': y_test, 'mean_safe': mean_safe, 'mean_fraud': mean_fraud
    }

bundle = train_models_cached()

# ------------------------------------------------------------------------------
# 4. THANH BÊN (SIDEBAR)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("⚙️ Cấu Hình Mô Hình")
    
    model_options = ["Logistic Regression"]
    if HAS_XGBOOST:
        model_options.append("XGBoost Classifier")
    else:
        st.warning("⚠️ Chưa cài 'xgboost'. Thêm vào requirements.txt để kích hoạt.")

    selected_model_name = st.selectbox("🤖 Chọn Thuật Toán AI:", options=model_options)
    threshold = st.slider("🎯 Ngưỡng Quyết Định (Threshold):", 0.10, 0.90, 0.50, 0.05)

if selected_model_name == "XGBoost Classifier" and HAS_XGBOOST:
    model = bundle['xgb_model']
    metrics = bundle['xgb_metrics']
    coef_df = bundle['xgb_coef_df']
else:
    model = bundle['lr_model']
    metrics = bundle['lr_metrics']
    coef_df = bundle['lr_coef_df']

# ------------------------------------------------------------------------------
# 5. GIAO DIỆN CHÍNH & CÁC TAB
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">⚖️ HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Mô hình Định lượng Beneish M-Score kết hợp Trí tuệ Nhân tạo (<strong>{selected_model_name}</strong>)</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Dự Báo Đơn Lẻ", 
    "📈 Dashboard Phân Tích Số Liệu", 
    "📁 Dự Báo Hàng Loạt", 
    "📊 Hiệu Năng Mô Hình"
])

# ==============================================================================
# TAB 1: DỰ BÁO ĐƠN LẺ
# ==============================================================================
with tab1:
    st.markdown("### 📝 Thẩm định Rủi ro Doanh nghiệp Đơn lẻ")
    
    if 'dsri' not in st.session_state:
        st.session_state.dsri, st.session_state.gmi = 1.45, 1.15
        st.session_state.aqi, st.session_state.sgi = 1.30, 1.60
        st.session_state.depi, st.session_state.sgai = 0.82, 1.12
        st.session_state.lvgi, st.session_state.tata = 1.25, 0.12

    c_btn1, c_btn2, c_btn3 = st.columns(3)
    with c_btn1:
        if st.button("🟢 Mẫu DN Minh Bạch", use_container_width=True):
            st.session_state.dsri, st.session_state.gmi = 0.92, 0.95
            st.session_state.aqi, st.session_state.sgi = 0.88, 1.05
            st.session_state.depi, st.session_state.sgai = 1.02, 0.96
            st.session_state.lvgi, st.session_state.tata = 0.94, -0.02
            st.rerun()

    with c_btn2:
        if st.button("🔴 Mẫu DN Gian Lận (Red-Flag)", use_container_width=True):
            st.session_state.dsri, st.session_state.gmi = 1.68, 1.45
            st.session_state.aqi, st.session_state.sgi = 1.42, 1.85
            st.session_state.depi, st.session_state.sgai = 0.75, 1.25
            st.session_state.lvgi, st.session_state.tata = 1.40, 0.18
            st.rerun()

    with c_btn3:
        if st.button("⚖️ Mẫu Mặc Định", use_container_width=True):
            st.session_state.dsri, st.session_state.gmi = 1.45, 1.15
            st.session_state.aqi, st.session_state.sgi = 1.30, 1.60
            st.session_state.depi, st.session_state.sgai = 0.82, 1.12
            st.session_state.lvgi, st.session_state.tata = 1.25, 0.12
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        val_dsri = st.number_input("1. DSRI (Khoản phải thu / Doanh thu):", value=float(st.session_state.dsri), step=0.01)
        val_gmi = st.number_input("2. GMI (Biên lợi nhuận gộp):", value=float(st.session_state.gmi), step=0.01)
        val_aqi = st.number_input("3. AQI (Chất lượng tài sản):", value=float(st.session_state.aqi), step=0.01)
        val_sgi = st.number_input("4. SGI (Tăng trưởng doanh thu):", value=float(st.session_state.sgi), step=0.01)
    with col2:
        val_depi = st.number_input("5. DEPI (Tỷ lệ khấu hao):", value=float(st.session_state.depi), step=0.01)
        val_sgai = st.number_input("6. SGAI (Chi phí bán hàng & QLDN):", value=float(st.session_state.sgai), step=0.01)
        val_lvgi = st.number_input("7. LVGI (Đòn bẩy tài chính):", value=float(st.session_state.lvgi), step=0.01)
        val_tata = st.number_input("8. TATA (Biến dồn tích / Tổng tài sản):", value=float(st.session_state.tata), step=0.01)

    input_df = pd.DataFrame([[val_dsri, val_gmi, val_aqi, val_sgi, val_depi, val_sgai, val_lvgi, val_tata]], columns=FEATURE_COLS)
    
    fraud_prob = model.predict_proba(input_df)[0][1]
    is_fraud = fraud_prob >= threshold
    beneish_m = tinh_beneish_mscore_chuan(val_dsri, val_gmi, val_aqi, val_sgi, val_depi, val_sgai, val_lvgi, val_tata)

    st.markdown("---")
    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        if is_fraud:
            st.markdown(f'<div class="status-danger">🚨 CẢNH BÁO RED-FLAG: NGUY CƠ GIAN LẬN CAO<br><small>Xác suất: {fraud_prob:.1%} (Ngưỡng: {threshold:.0%})</small></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-safe">✅ ĐÁNH GIÁ: BCTC CÓ ĐỘ TIN CẬY CAO<br><small>Xác suất: {fraud_prob:.1%} (Ngưỡng: {threshold:.0%})</small></div>', unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            number={'suffix': "%"},
            title={'text': "<b>Xác Suất Gian Lận</b>"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#EF4444" if is_fraud else "#10B981"},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [30, 50], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                ],
                'threshold': {'line': {'color': "black", 'width': 3}, 'value': threshold * 100}
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown("##### 🕸️ Biểu đồ Radar So Sánh Chỉ Số")
        radar_categories = FEATURE_COLS
        user_vals = [val_dsri, val_gmi, val_aqi, val_sgi, val_depi, val_sgai, val_lvgi, val_tata]
        safe_vals = [bundle['mean_safe'][c] for c in FEATURE_COLS]
        fraud_vals = [bundle['mean_fraud'][c] for c in FEATURE_COLS]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=user_vals, theta=radar_categories, fill='toself', name='DN Hiện Tại'))
        fig_radar.add_trace(go.Scatterpolar(r=safe_vals, theta=radar_categories, name='TB Minh Bạch', line=dict(dash='dot')))
        fig_radar.add_trace(go.Scatterpolar(r=fraud_vals, theta=radar_categories, name='TB Gian Lận', line=dict(dash='dash')))
        fig_radar.update_layout(height=280, margin=dict(l=30, r=30, t=20, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)

# ==============================================================================
# TAB 2: DASHBOARD PHÂN TÍCH SỐ LIỆU TOÀN DIỆN
# ==============================================================================
with tab2:
    st.markdown("### 📈 Dashboard Phân Tích & Trực Quan Hóa Dữ Liệu BCTC")
    df_data = bundle['df']

    # 1. Thống kê tổng quan
    st.markdown("##### 1. Thống Kê Tổng Quan Mẫu Dữ Liệu")
    m1, m2, m3, m4 = st.columns(4)
    total_samples = len(df_data)
    fraud_samples = df_data['FRAUD_FLAG'].sum()
    safe_samples = total_samples - fraud_samples
    
    m1.metric("Tổng Số BCTC Thẩm Định", f"{total_samples} DN")
    m2.metric("Doanh Nghiệp Minh Bạch", f"{safe_samples} DN", f"{safe_samples/total_samples:.1%}")
    m3.metric("Doanh Nghiệp Gian Lận", f"{fraud_samples} DN", f"{fraud_samples/total_samples:.1%}", delta_color="inverse")
    m4.metric("Beneish M-Score Trung Bình", f"{df_data['M_SCORE'].mean():.2f}")

    st.markdown("---")

    # 2. Phân phối M-Score & Ma trận tương quan
    dash_col1, dash_col2 = st.columns(2)

    with dash_col1:
        st.markdown("##### 📊 Phân Phối Điểm Beneish M-Score Theo Nhóm")
        fig_hist = px.histogram(
            df_data, x="M_SCORE", color="FRAUD_FLAG",
            barmode="overlay", nbins=40,
            labels={'M_SCORE': 'Beneish M-Score', 'FRAUD_FLAG': 'Nhóm (0: Safe, 1: Fraud)'},
            color_discrete_map={0: '#10B981', 1: '#EF4444'}
        )
        fig_hist.add_vline(x=-1.78, line_dash="dash", line_color="black", annotation_text="Ngưỡng M = -1.78")
        fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with dash_col2:
        st.markdown("##### 🎯 Ma Trận Tương Quan Giữa Các Biến Tài Chính")
        corr_matrix = df_data[FEATURE_COLS + ['FRAUD_FLAG']].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # 3. Phân tích đồ thị hộp Boxplot cho 8 chỉ số
    st.markdown("##### 📦 Biến Động 8 Chỉ Số Kế Toán Giữa Nhóm Minh Bạch vs Gian Lận")
    selected_feature = st.selectbox("Chọn chỉ số tài chính cần soi chi tiết:", options=FEATURE_COLS, index=0)
    
    fig_box = px.box(
        df_data, x="FRAUD_FLAG", y=selected_feature, color="FRAUD_FLAG",
        points="all",
        color_discrete_map={0: '#10B981', 1: '#EF4444'},
        labels={'FRAUD_FLAG': 'Nhóm Doanh Nghiệp (0: Minh Bạch, 1: Gian Lận)'}
    )
    fig_box.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("##### 📋 Bảng Thống Kê Mô Tả Chi Tiết (Mean, Median, Std)")
    summary_df = df_data.groupby('FRAUD_FLAG')[FEATURE_COLS].agg(['mean', 'median', 'std']).T
    st.dataframe(summary_df, use_container_width=True)

# ==============================================================================
# TAB 3: DỰ BÁO HÀNG LOẠT
# ==============================================================================
with tab3:
    st.markdown("### 📁 Dự Báo Gian Lận Hàng Loạt Từ Tệp CSV")
    batch_file = st.file_uploader("Tải tệp CSV danh sách doanh nghiệp:", type=["csv"])
    
    if batch_file:
        try:
            batch_df = pd.read_csv(batch_file)
            missing_cols = [c for c in FEATURE_COLS if c not in batch_df.columns]
            if missing_cols:
                st.error(f"❌ File thiếu các cột bắt buộc: {missing_cols}")
            else:
                probs = model.predict_proba(batch_df[FEATURE_COLS])[:, 1]
                batch_df['Xac_Suat_Gian_Lan'] = np.round(probs, 4)
                batch_df['Ket_Luan'] = ['🔴 RED-FLAG' if p >= threshold else '🟢 MINH BẠCH' for p in probs]
                
                st.markdown("##### 📋 Kết quả dự báo:")
                st.dataframe(batch_df, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Lỗi xử lý file: {str(e)}")

# ==============================================================================
# TAB 4: HIỆU NĂNG MÔ HÌNH
# ==============================================================================
with tab4:
    st.markdown("### 📊 Đánh Giá Hiệu Năng & Ma Trận Nhầm Lẫn")
    
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown(f"##### 🎯 Chỉ số kiểm định: {selected_model_name}")
        st.json(metrics)
    
    with c_m2:
        st.markdown("##### ⚖️ Trọng Số / Trắc Nghiệm Tầm Quan Trọng Của Biến")
        fig_coef = px.bar(
            coef_df, x='Hệ số (Weight)', y='Chỉ số', orientation='h',
            color='Hệ số (Weight)', color_continuous_scale='Reds'
        )
        fig_coef.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_coef, use_container_width=True)