# -*- coding: utf-8 -*-
"""
ỨNG DỤNG WEB DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH (BCTC)
Sử dụng mô hình Beneish M-Score 8 biến kết hợp Thuật toán Logistic Regression & XGBoost
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
# 1. CẤU HÌNH TRANG
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
# 2. TẠO DỮ LIỆU MẶC ĐỊNH NẾU THIẾU FILE
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
# 3. HUẤN LUYỆN MÔ HÌNH
# ------------------------------------------------------------------------------
@st.cache_resource
def train_models_cached():
    file_path = "MScore_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = generate_sample_data()

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
    xgb_model = None
    xgb_metrics = None
    xgb_coef_df = None
    xgb_prob = None

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

# Load mô hình
bundle = train_models_cached()

# ------------------------------------------------------------------------------
# 4. THANH BÊN (SIDEBAR)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Cấu Hình")
    
    model_options = ["Logistic Regression"]
    if HAS_XGBOOST:
        model_options.append("XGBoost Classifier")
    else:
        st.warning("⚠️ Chưa cài 'xgboost'. Vui lòng thêm vào requirements.txt để dùng XGBoost.")

    selected_model_name = st.selectbox("🤖 Choose Model:", options=model_options)

    threshold = st.slider("🎯 Ngưỡng rủi ro (Threshold):", 0.10, 0.90, 0.50, 0.05)

# Lựa chọn mô hình hiện tại
if selected_model_name == "XGBoost Classifier" and HAS_XGBOOST:
    model = bundle['xgb_model']
    metrics = bundle['xgb_metrics']
    coef_df = bundle['xgb_coef_df']
else:
    model = bundle['lr_model']
    metrics = bundle['lr_metrics']
    coef_df = bundle['lr_coef_df']

# ------------------------------------------------------------------------------
# 5. MÀN HÌNH CHÍNH
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">⚖️ HỆ THỐNG DỰ BÁO GIAN LẬN BCTC</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Mô hình Beneish M-Score kết hợp AI (<strong>{selected_model_name}</strong>)</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Dự Báo Đơn Lẻ", "📁 Dự Báo Hàng Loạt", "📊 Hiệu Năng Mô Hình"])

with tab1:
    st.markdown("### 📝 Nhập chỉ số tài chính")
    
    if 'dsri' not in st.session_state:
        st.session_state.dsri, st.session_state.gmi = 1.45, 1.15
        st.session_state.aqi, st.session_state.sgi = 1.30, 1.60
        st.session_state.depi, st.session_state.sgai = 0.82, 1.12
        st.session_state.lvgi, st.session_state.tata = 1.25, 0.12

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🟢 Mẫu Minh Bạch", use_container_width=True):
            st.session_state.dsri, st.session_state.gmi = 0.92, 0.95
            st.session_state.aqi, st.session_state.sgi = 0.88, 1.05
            st.session_state.depi, st.session_state.sgai = 1.02, 0.96
            st.session_state.lvgi, st.session_state.tata = 0.94, -0.02
            st.rerun()

    with c_btn2:
        if st.button("🔴 Mẫu Gian Lận (Red Flag)", use_container_width=True):
            st.session_state.dsri, st.session_state.gmi = 1.68, 1.45
            st.session_state.aqi, st.session_state.sgi = 1.42, 1.85
            st.session_state.depi, st.session_state.sgai = 0.75, 1.25
            st.session_state.lvgi, st.session_state.tata = 1.40, 0.18
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        val_dsri = st.number_input("1. DSRI:", value=float(st.session_state.dsri), step=0.01)
        val_gmi = st.number_input("2. GMI:", value=float(st.session_state.gmi), step=0.01)
        val_aqi = st.number_input("3. AQI:", value=float(st.session_state.aqi), step=0.01)
        val_sgi = st.number_input("4. SGI:", value=float(st.session_state.sgi), step=0.01)
    with col2:
        val_depi = st.number_input("5. DEPI:", value=float(st.session_state.depi), step=0.01)
        val_sgai = st.number_input("6. SGAI:", value=float(st.session_state.sgai), step=0.01)
        val_lvgi = st.number_input("7. LVGI:", value=float(st.session_state.lvgi), step=0.01)
        val_tata = st.number_input("8. TATA:", value=float(st.session_state.tata), step=0.01)

    input_df = pd.DataFrame([[val_dsri, val_gmi, val_aqi, val_sgi, val_depi, val_sgai, val_lvgi, val_tata]], columns=FEATURE_COLS)
    
    fraud_prob = model.predict_proba(input_df)[0][1]
    is_fraud = fraud_prob >= threshold

    st.markdown("---")
    if is_fraud:
        st.markdown(f'<div class="status-danger">🚨 CẢNH BÁO RED-FLAG: NGUY CƠ GIAN LẬN CAO ({fraud_prob:.1%})</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-safe">✅ BCTC CÓ ĐỘ TIN CẬY CAO ({fraud_prob:.1%})</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### 📁 Tải file CSV kiểm tra danh mục")
    batch_file = st.file_uploader("Tải tệp CSV:", type=["csv"])
    if batch_file:
        batch_df = pd.read_csv(batch_file)
        st.dataframe(batch_df.head(), use_container_width=True)

with tab3:
    st.markdown(f"### 📊 Hiệu năng mô hình: {selected_model_name}")
    st.json(metrics)