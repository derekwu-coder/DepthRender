"""
Nationality options loader (Nationality.csv).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core.i18n import tr

@st.cache_data
def load_nationality_options(csv_path: Path) -> pd.DataFrame:
    """
    從 CSV 讀取國籍清單，欄位需包含：
    - Country
    - Code（Alpha-3）
    
    並組合成選單 label，例如：
        Taiwan (TWN)
        Japan (JPN)
    """

    # --- 防呆：檔案不存在 ---
    if not csv_path.exists():
        st.error(tr("nationality_file_not_found", path=csv_path))
        return pd.DataFrame(columns=["Country", "Code", "label"])

    # --- 讀取 CSV ---
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(tr("nationality_read_error", error=e))
        return pd.DataFrame(columns=["Country", "Code", "label"])

    # --- 檢查必要欄位 ---
    required_cols = {"Country", "Code"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(tr("nationality_missing_columns", missing=list(missing)))
        return pd.DataFrame(columns=["Country", "Code", "label"])

    # --- 整理資料 ---
    df = df.dropna(subset=["Country", "Code"]).copy()
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Code"] = df["Code"].astype(str).str.upper().str.strip()

    # --- 下拉選單顯示字串 ---
    df["label"] = df["Country"] + " (" + df["Code"] + ")"

    return df

# ================================
# 共用：把潛水資料重採樣成「每秒一點」並計算速率
# ================================
