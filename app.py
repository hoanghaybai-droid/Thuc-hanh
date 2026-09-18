import streamlit as st
import pandas as pd
import numpy as np

# Kiểm tra an toàn thư viện XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ModuleNotFoundError:
    HAS_XGBOOST = False
    st.error("⚠️ Thư viện 'xgboost' chưa được cài đặt trong môi trường. Vui lòng thêm 'xgboost' vào file requirements.txt!")