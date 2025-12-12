from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from pathlib import Path
import pandas as pd
from io import BytesIO
import numpy as np
import altair as alt
import time
import threading

from core.parser_garmin import parse_garmin_fit_to_dives
from core.parser_atmos import parse_atmos_uddf
from core.video_renderer import render_video  # 之後實作
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

st.set_page_config(page_title="Dive Overlay Generator", layout="wide")

# ==================================
# 全局 CSS：讓畫面更像 App
# ==================================
APP_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ===== Layout width ===== */
.main > div {display:flex; justify-content:center;}
.main > div > div {max-width: 1200px; width:100%;}

/* ===== Reserve space for fixed header + fixed tabs ===== */
.block-container{
  padding-top: 128px;   /* adjust if header/tabs overlap */
}

/* ===== Fixed top header (brand + language) ===== */
.app-header-row{
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 120;
  padding: 0.10rem 0.10rem 0.15rem 0.10rem;  /* tighter */
  backdrop-filter: blur(10px);
  background: rgba(248,250,252,0.96);
}
@media (prefers-color-scheme: dark){
  .app-header-row{ background: rgba(15,23,42,0.98); }
}

.app-top-bar{
  display:flex;
  align-items:center;
  justify-content: space-between;
  gap: 0.55rem;
  padding: 0.15rem 0.6rem 0.15rem;
}

.app-top-icon{ display:none; } /* remove wave icon */

.app-title-text{
  font-size: 1.50rem;
  font-weight: 700;
  line-height: 1.40rem;
}
@media (max-width: 600px){
  .app-title-text{
    font-size: 1.25rem !important;
    line-height: 1.25rem !important;
  }
}

/* ===== Tabs: fixed bar, full-width background (no notch) ===== */
div[data-testid="stTabs"]{ border-bottom:none !important; box-shadow:none !important; background:transparent !important; }
div[data-testid="stTabs"] div[role="tablist"]{
  position: fixed;
  top: 78px;           /* just under header */
  left: 0; right: 0;
  z-index: 110;
  padding: 0.10rem 0.55rem 0.20rem 0.55rem !important;
  margin: 0 !important;
  background: #f8fafc !important;
  border-bottom: none !important;
  box-shadow: none !important;
}
@media (prefers-color-scheme: dark){
  div[data-testid="stTabs"] div[role="tablist"]{ background: #0E1117 !important; }
}

/* Remove moving highlight / border bars */
div[data-baseweb="tab-highlight"]{ display:none !important; height:0 !important; opacity:0 !important; }
div[data-baseweb="tab-border"]{ background: transparent !important; border:none !important; height: 0 !important; }

/* Center pills + keep them closer (without shifting left) */
div[data-baseweb="tab-list"]{
  justify-content: center !important;
  gap: 10px !important;
  margin: 0 auto !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
}

/* Pill button style (keeps selected color) */
div[data-testid="stTabs"] button[role="tab"]{
  border-radius: 999px !important;
  padding: 0.18rem 0.90rem !important;
  margin: 0 !important;
  border: 1px solid rgba(148,163,184,0.7) !important;
  background-color: #f3f4f6 !important;
  color: #111827 !important;
  font-size: 0.90rem !important;
  font-weight: 500 !important;
  box-shadow: none !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
  background-color: #dbeafe !important;
  border-color: #38bdf8 !important;
  color: #0f172a !important;
}
@media (prefers-color-scheme: dark){
  div[data-testid="stTabs"] button[role="tab"]{
    background-color: #111827 !important;
    border-color: rgba(55,65,81,0.9) !important;
    color: #e5e7eb !important;
  }
  div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
    background-color: #1f2937 !important;
    border-color: #38bdf8 !important;
    color: #e5f2ff !important;
  }
}

/* ===== Card ===== */
.app-card{
  background-color: rgba(255,255,255,0.90);
  border-radius: 18px;
  padding: 0.85rem 1.2rem 1.1rem 1.2rem;
  box-shadow: 0 8px 20px rgba(15,23,42,0.10);
}
@media (prefers-color-scheme: dark){
  .app-card{
    background-color: rgba(15,23,42,0.90);
    box-shadow: 0 8px 20px rgba(0,0,0,0.60);
  }
}

/* ===== Subheaders ===== */
h3{
  font-size: 1.05rem !important;
  margin-top: 0.55rem !important;
  margin-bottom: 0.20rem !important;
}

/* ===== Upload labels (keep same color as other labels) ===== */
.upload-label{
  font-size: 0.95rem;
  font-weight: 600;
  color: rgba(17,24,39,0.92);
  margin-bottom: 0.25rem;
}
@media (prefers-color-scheme: dark){
  .upload-label{ color: rgba(229,231,235,0.92); }
}

/* ===== Align time block: desktop ~50%, mobile 100%, left aligned ===== */
.align-wrap{
  max-width: 560px;   /* approx half-column on desktop */
  width: 100%;
  margin: 0.15rem 0 0.35rem 0;
}
@media (max-width: 768px){
  .align-wrap{ max-width: 100% !important; width: 100% !important; }
}

/* Tighten spacing inside align block */
.align-wrap div[data-testid="stMarkdown"]{ margin-bottom: 0.20rem !important; }
.align-wrap div[data-testid="stRadio"]{ margin-top: -0.20rem !important; margin-bottom: 0.05rem !important; }
.align-wrap div[data-testid="stTextInput"]{ margin-top: -0.10rem !important; margin-bottom: 0.10rem !important; }

/* +/- buttons: near 1:1 and not full-row width */
.align-wrap div[data-testid="stButton"] button{
  width: 52px !important;
  height: 52px !important;
  padding: 0 !important;
  font-size: 28px !important; /* for full-width symbols */
  font-weight: 800 !important;
  line-height: 1 !important;
  text-align: center !important;
}
@media (max-width: 768px){
  .align-wrap div[data-testid="stButton"] button{ width: 46px !important; height: 46px !important; }
}

/* Center the time input and keep it compact */
.align-wrap div[data-testid="stTextInput"] input{
  text-align: center !important;
  max-width: 220px !important;
  margin: 0 auto !important;
  font-variant-numeric: tabular-nums;
}

/* ===== Mobile layout helpers ===== */

/* Force the FIRST st.columns in Overlay tab to stay 50/50 on mobile */
@media (max-width: 768px){
  div[data-testid="stTabs"] div[role="tabpanel"]:first-of-type div[data-testid="stHorizontalBlock"]:first-of-type{
    flex-direction: row !important;
    flex-wrap: nowrap !important;
  }
  div[data-testid="stTabs"] div[role="tabpanel"]:first-of-type div[data-testid="stHorizontalBlock"]:first-of-type > div{
    flex: 0 0 50% !important;
    max-width: 50% !important;
    min-width: 0 !important;
  }
}
@media (max-width: 768px){
  .app-card{
    padding: 0.75rem 0.9rem 1.0rem 0.9rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(15,23,42,0.15);
  }

  /* Other two-column blocks can stack */
  .overlay-stack-mobile div[data-testid="stHorizontalBlock"]{
    flex-direction: column !important;
    flex-wrap: nowrap !important;
  }
  .overlay-stack-mobile div[data-testid="stHorizontalBlock"] > div{
    max-width: 100% !important;
    width: 100% !important;
  }

  /* Keep upload section in two columns (50/50) */
  .upload-cols div[data-testid="stHorizontalBlock"]{
    flex-direction: row !important;
    flex-wrap: nowrap !important;
  }
  .upload-cols div[data-testid="stHorizontalBlock"] > div{
    max-width: 50% !important;
    flex: 0 0 50% !important;
    min-width: 0 !important;
  }

  /* Header columns should not be forced to 50/50 by generic rules */
  .header-cols div[data-testid="stHorizontalBlock"] > div{ max-width: unset !important; }
  .header-cols div[data-testid="stHorizontalBlock"] > div:first-child{ flex: 0 0 70% !important; max-width: 70% !important; }
  .header-cols div[data-testid="stHorizontalBlock"] > div:last-child{ flex: 0 0 30% !important; max-width: 30% !important; }
}

</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)

# ==================================
# 🗣️ 多語系字典 & 文字取得函式
# ==================================

# 預設語言
if "lang" not in st.session_state:
    st.session_state["lang"] = "zh"  # 先預設中文

LANG_OPTIONS = {
    "zh": "中文",
    "en": "English",
}

TRANSLATIONS = {
    "zh": {
        # ======================
        # App / 共用
        # ======================
        "app_title": "Dive Overlay Generator",
        "top_brand": "DepthRender",
        "language_label": "🌐 語言",

        # ======================
        # Tabs 標題
        # ======================
        "tab_overlay_title": "疊加影片產生器",
        "tab_compare_title": "潛水數據比較",
        "compare_coming_soon": (
            "這裡未來會加入不同潛水之間的曲線比較功能，例如：\n\n"
            "- 深度曲線對比\n"
            "- 速率 / FF 比例比較\n"
            "- 不同比賽 / 不同天的表現差異"
        ),

        # ======================
        # Overlay tab：上傳 / 預覽
        # ======================
        "upload_watch_subheader": "1️⃣ 上傳手錶數據",
        "upload_watch_label": "手錶數據 (.fit/.uddf)",
        "upload_video_subheader": "2️⃣ 上傳潛水影片",
        "upload_video_label": "影片檔（任意解析度）",

        "fit_detected": "偵測到 Garmin .fit 檔，開始解析多潛資料...",
        "fit_no_dives": "這個 .fit 裡面沒有偵測到有效的潛水紀錄。",
        "select_dive_label": "選擇要使用的那一潛：",
        "uddf_detected": "偵測到 ATMOS UDDF 檔，開始解析單一潛水紀錄...",
        "no_depth_samples": "成功讀取手錶檔，但沒有找到任何深度樣本點。",

        "dive_time_detected": "偵測到的 Dive Time：約 {mm:02d}:{ss:02d} （從深度 ≥ 0.7 m 開始，到回到 0 m）",
        "preview_subheader": "3️⃣ 潛水曲線預覽（時間 vs 深度 / 速率）",

        "axis_time_seconds": "時間（秒）",
        "axis_depth_m": "深度（m）",
        "axis_rate_mps": "速率（m/s）",
        "tooltip_time": "時間 (s)",
        "tooltip_depth": "深度 (m)",
        "tooltip_rate": "速率 (m/s)",
        "depth_chart_title": "深度 vs 時間",
        "rate_chart_title": "速率 vs 時間",

        # 目前你雖然拿掉了 caption 的顯示，但保留 key 不影響使用
        "preview_caption": "原始資料點數：{n_points}，重採樣時間範圍：{t_min:.0f}～{t_max:.0f} 秒，最大深度：約 {max_depth:.1f} m",

        # ======================
        # Overlay tab：對齊與版型
        # ======================
        "align_layout_subheader": "4️⃣ 影片對齊與版型",
        "time_offset_label": "潛水開始時間調整",
        "time_offset_help": "如果影片比實際下潛早開始，請用負值調整。",
        "layout_select_label": "選擇影片版型",
        "layout_preview_title": "版型示意圖（目前選擇會加黃色外框）",

        "layout_a_label": "A: 深度＋心率＋速率",
        "layout_a_desc": "",
        "layout_b_label": "B: 賽事風格",
        "layout_b_desc": "",
        "layout_c_label": "C: 單純深度",
        "layout_c_desc": "Simple_A",
        "layout_d_label": "D: 單純深度",
        "layout_d_desc": "Simple_B",

        # ======================
        # Overlay tab：潛水員資訊
        # ======================
        "diver_info_subheader": "5️⃣ 潛水員資訊（選填，主要給 Layout B 使用）",
        "diver_name_label": "潛水員姓名 / Nickname",
        "nationality_label": "國籍",
        "discipline_label": "潛水項目（Discipline）",
        "not_specified": "（不指定）",

        # ======================
        # Overlay tab：產生影片 + 錯誤訊息
        # ======================
        "render_button": "🚀 產生疊加數據影片",
        "error_need_both_files": "請先上傳手錶數據與影片檔。",
        "progress_init": "初始化中...",
        "progress_rendering": "產生影片中...",
        "progress_done": "影片產生完成！",
        "render_success": "影片產生完成！",
        "download_button": "下載 1080p 影片",
        "render_error": "產生影片時發生錯誤：{error}",

        "nationality_file_not_found": "找不到 Nationality 檔案：{path}",
        "nationality_read_error": "讀取 Nationality.csv 時發生錯誤：{error}",
        "nationality_missing_columns": "Nationality.csv 缺少必要欄位：{missing}",

        # ======================
        # Compare tab：標題 / 上傳
        # ======================
        "compare_title": "📊 潛水數據比較",
        "compare_upload_a": "上傳數據 A（.fit / .uddf）",
        "compare_upload_b": "上傳數據 B（.fit / .uddf）",
        "compare_select_dive_a": "數據A 要比較的那一潛：",
        "compare_select_dive_b": "數據B 要比較的那一潛：",

        "compare_smooth_label": "速率平滑度",
        "compare_align_label": "調整數據 B 的時間偏移（秒，用來對齊兩組曲線）",
        "compare_no_data": "請先上傳並選擇兩組有效的潛水數據。",

        "compare_depth_chart_title": "深度 vs 時間",
        "compare_rate_chart_title": "速率 vs 時間",
        "compare_series_legend": "數據來源",
        "compare_align_current": "偏移：{offset:.1f} 秒",

        "compare_desc_rate_label": "下潛速率 (m/s)",
        "compare_asc_rate_label": "上升速率 (m/s)",
        "compare_ff_depth_label_a": "數據A：FF 開始深度 (m)",
        "compare_ff_depth_label_b": "數據B：FF 開始深度 (m)",
        "compare_ff_rate_label": "Free Fall 速率 (m/s)",
        "compare_metric_unit_mps": "{value:.2f} m/s",
        "compare_metric_not_available": "—",

        # ======================
        # Overlay：速率分析 + 潛水時間顯示（單一潛水）
        # ======================
        "overlay_speed_analysis_title": "潛水速率分析",
        "overlay_ff_depth_label": "FF 開始深度 (m)",
        "metric_dive_time_label": "潛水時間",
        "metric_dive_time_value": "{mm:02d}:{ss:02d}",

        "overlay_rate_section_title": "潛水速率分析",
        "overlay_desc_rate_label": "下潛速率 (m/s)",
        "overlay_asc_rate_label": "上升速率 (m/s)",
        "overlay_ff_rate_label": "Free Fall 速率 (m/s)",
        "overlay_metric_unit_mps": "{value:.2f} m/s",
        "overlay_metric_not_available": "—",

        # ======================
        # Overlay：影片對齊 UI
        # ======================
        "align_mode_label": "對齊方式",
        "align_mode_start": "對齊下潛時間 (開始躬身)",
        "align_mode_bottom": "對齊最深時間 (轉身/摘到 tag)",
        "align_mode_end": "對齊出水時間 (手錶出水)",

        "align_video_time_label": "影片時間（mm:ss.ss，例如 01:10.05）",
        "align_video_time_help": "請輸入分鐘:秒.小數，秒與小數最多 2 位，例如 00:03.18",
        "align_video_time_invalid": "影片時間格式不正確，請使用 mm:ss 或 mm:ss.ss，例如 00:03.18",

        # ======================
        # 渲染剩餘時間提示
        # ======================
        "render_estimate_pending": "剩餘時間預估中⋯⋯",
        "render_do_not_leave": "請勿離開此畫面或關閉螢幕",
        "render_estimate_eta": "預估剩餘時間：約 {eta}",
        
        "align_video_time_title": "影片時間",
        "align_step_label": "調整級距",
        "align_step_min": "分 (1 min)",
        "align_step_sec": "秒 (1 s)",
        "align_step_csec": "0.01 秒 (10 ms)",
        "align_minus": "-",
        "align_plus": "＋",
        "align_time_invalid": "影片時間格式不正確，請使用 mm:ss 或 mm:ss.ss，例如 00:03.18",
        "align_step_label": "調整級距",
        "align_step_min": "分 (1 min)",
        "align_step_sec": "秒 (1 s)",
        "align_step_csec": "0.1 秒 (100 ms)",
        "align_video_time_seconds_label": "影片時間（秒）",
        "align_video_time_seconds_help": "用右側 +/- 依級距微調；上方可切換分 / 秒 / 0.02s。",
        "align_video_time_display": "顯示格式",
        "upload_file_short": "上傳檔案",

    
    },

    "en": {
        # ======================
        # App / Common
        # ======================
        "app_title": "Dive Overlay Generator",
        "top_brand": "DepthRender",
        "language_label": "🌐 Language",

        # ======================
        # Tabs titles
        # ======================
        "tab_overlay_title": "Overlay Generator",
        "tab_compare_title": "Dive Comparison",
        "compare_coming_soon": (
            "This tab will later provide dive-to-dive comparison, such as:\n\n"
            "- Depth curve comparison\n"
            "- Speed / free-fall ratio\n"
            "- Performance across different sessions / competitions"
        ),

        # ======================
        # Overlay tab: upload / preview
        # ======================
        "upload_watch_subheader": "1️⃣ Upload dive log",
        "upload_watch_label": "Dive log (.fit/.uddf)",
        "upload_video_subheader": "2️⃣ Upload video",
        "upload_video_label": "Video file",

        "fit_detected": "Detected Garmin .fit file. Parsing multi-dive data...",
        "fit_no_dives": "No valid dives found in this .fit file.",
        "select_dive_label": "Select which dive to use:",
        "uddf_detected": "Detected ATMOS UDDF file. Parsing single dive...",
        "no_depth_samples": "Log file loaded, but no depth samples were found.",

        "dive_time_detected": "Detected dive time: approx {mm:02d}:{ss:02d} (from depth ≥ 0.7 m until back to 0 m)",
        "preview_subheader": "3️⃣ Dive curve preview (time vs depth / speed)",

        "axis_time_seconds": "Time (s)",
        "axis_depth_m": "Depth (m)",
        "axis_rate_mps": "Speed (m/s)",
        "tooltip_time": "Time (s)",
        "tooltip_depth": "Depth (m)",
        "tooltip_rate": "Speed (m/s)",
        "depth_chart_title": "Depth vs Time",
        "rate_chart_title": "Speed vs Time",

        "preview_caption": "Raw samples: {n_points}, resampled time range: {t_min:.0f}–{t_max:.0f} s, max depth: ~{max_depth:.1f} m",

        # ======================
        # Overlay tab: alignment & layout
        # ======================
        "align_layout_subheader": "4️⃣ Video alignment & layout",
        "time_offset_label": "Align video start",
        "time_offset_help": "If the video starts before the actual dive, use a negative offset.",
        "layout_select_label": "Choose overlay layout",
        "layout_preview_title": "Layout preview (selected layout highlighted in yellow)",

        "layout_a_label": "A: Depth + HR + Speed",
        "layout_a_desc": "",
        "layout_b_label": "B: Competition-style",
        "layout_b_desc": "",
        "layout_c_label": "C: Depth only",
        "layout_c_desc": "Simple_A",
        "layout_d_label": "D: Depth only",
        "layout_d_desc": "Simple_B",

        # ======================
        # Overlay tab: diver info
        # ======================
        "diver_info_subheader": "5️⃣ Diver info (optional, mainly for Layout B)",
        "diver_name_label": "Diver name / Nickname",
        "nationality_label": "Nationality",
        "discipline_label": "Discipline",
        "not_specified": "(Not specified)",

        # ======================
        # Overlay tab: render + errors
        # ======================
        "render_button": "🚀 Generate overlay video",
        "error_need_both_files": "Please upload both dive log and video file.",
        "progress_init": "Initializing...",
        "progress_rendering": "Rendering video...",
        "progress_done": "Rendering finished!",
        "render_success": "Video rendered successfully!",
        "download_button": "Download 1080p video",
        "render_error": "Error while rendering video: {error}",

        "nationality_file_not_found": "Nationality file not found: {path}",
        "nationality_read_error": "Error reading Nationality.csv: {error}",
        "nationality_missing_columns": "Nationality.csv is missing required columns: {missing}",

        # ======================
        # Compare tab
        # ======================
        "compare_title": "📊 Dual-dive comparison",
        "compare_upload_a": "Upload log A (.fit / .uddf)",
        "compare_upload_b": "Upload log B (.fit / .uddf)",
        "compare_select_dive_a": "Dive A:",
        "compare_select_dive_b": "Dive B:",

        "compare_smooth_label": "Speed smoothing",
        "compare_align_label": "Time offset for log B (seconds, to align two curves)",
        "compare_no_data": "Please upload and select two valid dive logs first.",

        "compare_depth_chart_title": "Depth vs Time (comparison)",
        "compare_rate_chart_title": "Speed vs Time (comparison)",
        "compare_series_legend": "Series",
        "compare_align_current": "Offset: {offset:.1f}s",

        "compare_desc_rate_label": "Descent Rate (m/s)",
        "compare_asc_rate_label": "Ascent Rate (m/s)",
        "compare_ff_depth_label_a": "A: FF start depth (m)",
        "compare_ff_depth_label_b": "B: FF start depth (m)",
        "compare_ff_rate_label": "Free-fall Descent Rate (m/s)",
        "compare_metric_unit_mps": "{value:.2f} m/s",
        "compare_metric_not_available": "—",

        # ======================
        # Overlay: single-dive metrics
        # ======================
        "overlay_speed_analysis_title": "Dive speed analysis",
        "overlay_ff_depth_label": "FF start depth (m)",
        "metric_dive_time_label": "Dive time",
        "metric_dive_time_value": "{mm:02d}:{ss:02d}",

        "overlay_rate_section_title": "Dive speed metrics",
        "overlay_desc_rate_label": "Descent speed (m/s)",
        "overlay_asc_rate_label": "Ascent speed (m/s)",
        "overlay_ff_rate_label": "Free-fall speed (m/s)",
        "overlay_metric_unit_mps": "{value:.2f} m/s",
        "overlay_metric_not_available": "—",

        # ======================
        # Overlay: alignment UI
        # ======================
        "align_mode_label": "Alignment mode",
        "align_mode_start": "Align descent time (start of duck dive)",
        "align_mode_bottom": "Align bottom time (turn / tag grab)",
        "align_mode_end": "Align surfacing time (watch exits water)",

        "align_video_time_label": "Video time (mm:ss.ss, e.g. 01:10.05)",
        "align_video_time_help": "Use mm:ss or mm:ss.ss, with up to 2 decimal places, e.g. 00:03.18",
        "align_video_time_invalid": "Invalid video time format. Please use mm:ss or mm:ss.ss, e.g. 00:03.18",

        # ======================
        # Rendering ETA messages
        # ======================
        "render_estimate_pending": "Estimating remaining time…",
        "render_do_not_leave": "Do not leave this page or turn off the screen",
        "render_estimate_eta": "Estimated remaining time: approx. {eta}",
        
        "align_video_time_title": "Video time",
        "align_step_label": "Step size",
        "align_step_min": "Minute (1 min)",
        "align_step_sec": "Second (1 s)",
        "align_step_csec": "0.01 s (10 ms)",
        "align_minus": "−",
        "align_plus": "+",
        "align_time_invalid": "Invalid time format. Use mm:ss or mm:ss.ss, e.g. 00:03.18",
        "align_step_label": "Step",
        "align_step_min": "Min (1 min)",
        "align_step_sec": "Sec (1 s)",
        "align_step_csec": "0.1 s (100 ms)",
        "align_video_time_seconds_label": "Video time (seconds)",
        "align_video_time_seconds_help": "Use +/- to adjust by the selected step; switch step above (min / sec / 0.02s).",
        "align_video_time_display": "Display",
        "upload_file_short": "Upload file",

    },
}


def tr(key: str, **kwargs) -> str:
    """依據目前語言取得對應字串，可帶入 format 參數。"""
    lang = st.session_state.get("lang", "zh")
    text = TRANSLATIONS.get(lang, TRANSLATIONS["zh"]).get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text

def set_language():
    """讓 selectbox 改變時更新 session_state['lang']"""
    label_to_code = {v: k for k, v in LANG_OPTIONS.items()}
    # 🔁 這裡改成新的 key 名稱
    selected_label = st.session_state.get("_lang_select_top", LANG_OPTIONS["zh"])
    st.session_state["lang"] = label_to_code.get(selected_label, "zh")

# -------------------------------
# 頂部：左邊品牌、右邊語言選單（整排 sticky）
# -------------------------------
st.markdown("<div class='app-header-row'>", unsafe_allow_html=True)

st.markdown("<div class='header-cols'>", unsafe_allow_html=True)  # ✅ NEW
top_left, top_right = st.columns([8, 1])
st.markdown("</div>", unsafe_allow_html=True)                     # ✅ NEW

with top_left:
    st.markdown(
        f"""
        <div class="app-top-bar">
            <div>
                <div class="app-title-text">{tr('top_brand')}</div>
                <div class="app-title-sub">Dive Overlay Generator</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.selectbox(
        tr("language_label"),
        options=list(LANG_OPTIONS.values()),
        key="_lang_select_top",
        index=list(LANG_OPTIONS.keys()).index(st.session_state["lang"]),
        on_change=set_language,
    )

st.markdown("</div>", unsafe_allow_html=True)  # >🌊<

# 讀取國籍 / 國碼清單
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
def prepare_dive_curve(
    dive_df: pd.DataFrame,
    smooth_window: int
) -> Optional[pd.DataFrame]:

    """
    輸入原始 dive_df（需含 time_s, depth_m）
    回傳：
        time_s: 每秒一個點（整數秒）
        depth_m: 線性插值後深度
        rate_abs_mps: 每秒深度變化量的絕對值
        rate_abs_mps_smooth: 平滑後速率
    """
    if dive_df is None or len(dive_df) == 0:
        return None

    df = dive_df.sort_values("time_s").reset_index(drop=True).copy()
    if "time_s" not in df.columns or "depth_m" not in df.columns:
        return None

    t_min = float(df["time_s"].min())
    t_max = float(df["time_s"].max())
    t0 = int(np.floor(t_min))
    t1 = int(np.ceil(t_max))

    if t1 <= t0:
        return None

    uniform_time = np.arange(t0, t1 + 1, 1.0)

    depth_interp = np.interp(
        uniform_time,
        df["time_s"].to_numpy(),
        df["depth_m"].to_numpy()
    )

    rate_uniform = np.diff(depth_interp, prepend=depth_interp[0])
    rate_abs = np.abs(rate_uniform)
    rate_abs_clipped = np.clip(rate_abs, 0.0, 3.0)

    out = pd.DataFrame({
        "time_s": uniform_time.astype(float),
        "depth_m": depth_interp,
        "rate_abs_mps": rate_abs_clipped,
    })

    if smooth_window <= 1:
        out["rate_abs_mps_smooth"] = out["rate_abs_mps"]
    else:
        out["rate_abs_mps_smooth"] = (
            out["rate_abs_mps"]
            .rolling(window=smooth_window, center=True, min_periods=1)
            .mean()
        )

    return out

def compute_dive_metrics(
    df_rate: pd.DataFrame,
    dive_df_raw: Optional[pd.DataFrame],
    ff_start_depth_m: float,
) -> dict:
    """
    根據重採樣後的 df_rate（time_s, depth_m, rate_abs_mps_smooth）
    與原始 dive_df_raw（time_s, depth_m），計算：
      - descent_avg: 下潛平均速率（扣除開頭與底部各 1 秒）
      - ascent_avg: 上升平均速率（扣除一開始 1 秒）
      - ff_avg: Free Fall 開始後到最低點前 1 秒的平均速率
    回傳 dict，若無法計算則為 None。
    """
    result = {
        "descent_avg": None,
        "ascent_avg": None,
        "ff_avg": None,
    }

    if df_rate is None or dive_df_raw is None:
        return result
    if "time_s" not in dive_df_raw.columns or "depth_m" not in dive_df_raw.columns:
        return result
    if "time_s" not in df_rate.columns or "depth_m" not in df_rate.columns:
        return result
    if "rate_abs_mps_smooth" not in df_rate.columns:
        return result

    raw = dive_df_raw.sort_values("time_s").reset_index(drop=True).copy()

    # 1) 找 Dive start / end（跟 Overlay tab 一樣邏輯）
    start_rows = raw[raw["depth_m"] >= 0.7]
    if start_rows.empty:
        return result

    t_start = float(start_rows["time_s"].iloc[0])
    after = raw[raw["time_s"] >= t_start]
    end_candidates = after[after["depth_m"] <= 0.05]
    if not end_candidates.empty:
        t_end = float(end_candidates["time_s"].iloc[-1])
    else:
        t_end = float(after["time_s"].iloc[-1])

    if t_end <= t_start:
        return result

    # 2) 找到底點時間 t_bottom（只看 t_start ~ t_end 區間）
    within_dive = after[after["time_s"] <= t_end]
    if within_dive.empty:
        return result

    idx_bottom = within_dive["depth_m"].idxmax()
    t_bottom = float(within_dive.loc[idx_bottom, "time_s"])

    # 3) 在 df_rate 上切出對應區段
    df = df_rate.sort_values("time_s").reset_index(drop=True)

    # ---- 下潛平均速率：從 t_start + 1 到 t_bottom - 1 ----
    desc_start = t_start + 1.0
    desc_end = t_bottom - 1.0
    if desc_end > desc_start:
        mask_desc = (df["time_s"] >= desc_start) & (df["time_s"] <= desc_end)
        seg_desc = df.loc[mask_desc]
        if not seg_desc.empty:
            result["descent_avg"] = float(seg_desc["rate_abs_mps_smooth"].mean())

    # ---- 上升平均速率：從 t_bottom + 1 到 t_end ----
    asc_start = t_bottom + 1.0
    asc_end = t_end
    if asc_end > asc_start:
        mask_asc = (df["time_s"] >= asc_start) & (df["time_s"] <= asc_end)
        seg_asc = df.loc[mask_asc]
        if not seg_asc.empty:
            result["ascent_avg"] = float(seg_asc["rate_abs_mps_smooth"].mean())

    # ---- Free Fall 段平均速率 ----
    # 從指定 FF 深度開始，到 t_bottom - 1
    max_depth = float(within_dive["depth_m"].max())
    if ff_start_depth_m > 0.0 and ff_start_depth_m < max_depth:
        # 找 raw 裡面第一次達到 FF 深度的時間（只看下潛區間）
        ff_zone = within_dive[
            (within_dive["time_s"] >= t_start) &
            (within_dive["time_s"] <= t_bottom)
        ]
        ff_candidates = ff_zone[ff_zone["depth_m"] >= ff_start_depth_m]
        if not ff_candidates.empty:
            t_ff_start = float(ff_candidates["time_s"].iloc[0])
            ff_end = t_bottom - 1.0
            if ff_end > t_ff_start:
                mask_ff = (df["time_s"] >= t_ff_start) & (df["time_s"] <= ff_end)
                seg_ff = df.loc[mask_ff]
                if not seg_ff.empty:
                    result["ff_avg"] = float(seg_ff["rate_abs_mps_smooth"].mean())

    return result

# ================================
# 主畫面內容開始（卡片 + Tabs）
# ================================
with st.container():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    
    # Tabs：目前功能 + 比較分頁
    tab_overlay, tab_compare = st.tabs([
        tr("tab_overlay_title"),
        tr("tab_compare_title"),
    ])

    # ============================
    # Tab 1：疊加影片產生器
    # ============================
    with tab_overlay:

        # --- 1. 上傳區 ---
        st.markdown("<div class='upload-cols'>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader(tr("upload_watch_subheader"))
            st.markdown(f"<div class='upload-label'>{tr('upload_watch_label')}</div>", unsafe_allow_html=True)

            # ✅ 手錶：type=None（手機端允許任何檔案選），再用檔名判斷 .fit/.uddf
            watch_file = st.file_uploader(
                label="",
                type=None,
                key="overlay_watch_uploader",
                label_visibility="collapsed",
            )

        with col_right:
            st.subheader(tr("upload_video_subheader"))
            st.markdown(f"<div class='upload-label'>{tr('upload_video_label')}</div>", unsafe_allow_html=True)

            # ✅ 影片：限制只顯示影片類型（避免照片也出現）
            video_file = st.file_uploader(
                label="",
                type=["mp4", "mov", "m4v", "avi", "mkv", "webm"],
                key="overlay_video_uploader",
                label_visibility="collapsed",
            )

        st.markdown("</div>", unsafe_allow_html=True)


        # --- 2. 選手錶類型 & 解析 ---
        dive_df = None
        df_rate = None          # 速率重採樣後的 df
        dive_time_s = None      # Dive time（秒）
        dive_start_s = None     # 計時起點（深度 ≥ 0.7 m 的時間）
        dive_end_s = None       # 計時終點（回到 0 m 的時間）
        selected_dive_index = None

        if watch_file is not None:
            suffix = Path(watch_file.name).suffix.lower()

            if suffix == ".fit":
                st.info(tr("fit_detected"))
                dives = parse_garmin_fit_to_dives(BytesIO(watch_file.read()))
                # dives: List[pd.DataFrame]

                if len(dives) == 0:
                    st.error(tr("fit_no_dives"))
                else:
                    # 用最大深度當顯示文字
                    options = [
                        f"Dive #{i+1}（{df['depth_m'].max():.1f} m）"
                        for i, df in enumerate(dives)
                    ]

                    selected_dive_index = st.selectbox(
                        tr("select_dive_label"),
                        options=list(range(len(dives))),
                        format_func=lambda i: options[i],
                        key="overlay_dive_index",
                    )

                    dive_df = dives[selected_dive_index]

            elif suffix == ".uddf":
                st.info(tr("uddf_detected"))
                dive_df = parse_atmos_uddf(BytesIO(watch_file.read()))

        # --- 3. 顯示時間–深度曲線供確認 ---
        if dive_df is not None:
            if len(dive_df) == 0:
                st.warning(tr("no_depth_samples"))
            else:
                # 先確保按時間排序
                dive_df = dive_df.sort_values("time_s").reset_index(drop=True)

                # --- 強制加入起始/結束的 0 m 點 ---
                if len(dive_df) > 0 and "time_s" in dive_df.columns and "depth_m" in dive_df.columns:
                    dive_df["time_s"] = dive_df["time_s"] + 1.0

                    first_row = dive_df.iloc[0].copy()
                    first_row["time_s"] = 0.0
                    first_row["depth_m"] = 0.0

                    last_row = dive_df.iloc[-1].copy()
                    last_row["time_s"] = float(dive_df["time_s"].max()) + 1.0
                    last_row["depth_m"] = 0.0

                    dive_df = pd.concat(
                        [first_row.to_frame().T, dive_df, last_row.to_frame().T],
                        ignore_index=True,
                    )
                    dive_df = dive_df.sort_values("time_s").reset_index(drop=True)

                # 重採樣 + 速率（讓使用者選擇平滑度 1 / 2 / 3 秒）
                if "overlay_smooth_level" not in st.session_state:
                    st.session_state["overlay_smooth_level"] = 1  # 預設 1 秒

                smooth_level_overlay = int(st.session_state["overlay_smooth_level"])

                df_rate = prepare_dive_curve(
                    dive_df,
                    smooth_window=smooth_level_overlay,
                )

                # ====== 偵測 Dive Time（不再用 st.info 顯示，而是放到數據區） ======
                dive_time_s = None
                dive_start_s = None
                dive_end_s = None

                if df_rate is not None:
                    df_sorted = dive_df.sort_values("time_s").reset_index(drop=True)
                    start_rows = df_sorted[df_sorted["depth_m"] >= 0.7]
                    if not start_rows.empty:
                        t_start = float(start_rows["time_s"].iloc[0])
                        after = df_sorted[df_sorted["time_s"] >= t_start]
                        end_candidates = after[after["depth_m"] <= 0.05]

                        if not end_candidates.empty:
                            t_end = float(end_candidates["time_s"].iloc[-1])
                        else:
                            t_end = float(after["time_s"].iloc[-1])

                        dive_start_s = t_start
                        dive_end_s = t_end
                        dive_time_s = max(0.0, dive_end_s - dive_start_s)

                # ====== 3-1. 圖表（上下排列，不再用 columns） ======
                if df_rate is not None:
                    t_min = df_rate["time_s"].min()
                    t_resample_max = df_rate["time_s"].max()
                    max_display_time = int(np.ceil(t_resample_max / 5)) * 5

                    st.subheader(tr("preview_subheader"))

                    # 深度 vs 時間
                    depth_chart = (
                        alt.Chart(df_rate)
                        .mark_line()
                        .encode(
                            x=alt.X(
                                "time_s:Q",
                                title=tr("axis_time_seconds"),
                                scale=alt.Scale(domain=[t_min, max_display_time]),
                            ),
                            y=alt.Y(
                                "depth_m:Q",
                                title=tr("axis_depth_m"),
                                scale=alt.Scale(reverse=True),
                            ),
                            tooltip=[
                                alt.Tooltip("time_s:Q", title=tr("tooltip_time"), format=".1f"),
                                alt.Tooltip("depth_m:Q", title=tr("tooltip_depth"), format=".1f"),
                            ],
                        )
                        .properties(
                            title=tr("depth_chart_title"),
                            height=300,
                        )
                    )
                    st.altair_chart(depth_chart, use_container_width=True)

                    # 速率 vs 時間（平滑線）
                    # 給速率圖用的動態 Y 軸上限（最小 0.5，每 0.5 一級）
                    max_rate_plot = float(df_rate["rate_abs_mps_smooth"].max())
                    max_rate_domain = max(0.5, np.ceil(max_rate_plot * 2.0) / 2.0)

                    rate_chart = (
                        alt.Chart(df_rate)
                        .mark_line(interpolate="basis")
                        .encode(
                            x=alt.X(
                                "time_s:Q",
                                title=tr("axis_time_seconds"),
                                scale=alt.Scale(domain=[t_min, max_display_time]),
                            ),
                            y=alt.Y(
                                "rate_abs_mps_smooth:Q",
                                title=tr("axis_rate_mps"),
                                scale=alt.Scale(domain=[0, max_rate_domain]),
                            ),
                            tooltip=[
                                alt.Tooltip("time_s:Q", title=tr("tooltip_time"), format=".1f"),
                                alt.Tooltip("rate_abs_mps_smooth:Q", title=tr("tooltip_rate"), format=".2f"),
                            ],
                        )
                        .properties(
                            title=tr("rate_chart_title"),
                            height=300,
                        )
                    )

                    st.altair_chart(rate_chart, use_container_width=True)

                    # 在速率圖下方放「速率平滑度」選單（靠右，小一點）
                    spacer_l, spacer_mid, smooth_col_overlay = st.columns([10, 1, 1])
                    with smooth_col_overlay:
                        st.markdown(
                            f"<div style='text-align:right; font-size:0.85rem; margin-bottom:2px;'>"
                            f"{tr('compare_smooth_label')}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.selectbox(
                            "",
                            options=[1, 2, 3],
                            key="overlay_smooth_level",   # 用一樣的 key，控制 df_rate 平滑度
                            label_visibility="collapsed",
                        )

                    # 原始資料說明
                    #st.caption(
                    #    tr(
                    #        "preview_caption",
                    #        n_points=len(dive_df),
                    #        t_min=df_rate["time_s"].min(),
                    #        t_max=df_rate["time_s"].max(),
                    #        max_depth=df_rate["depth_m"].max(),
                    #    )
                    #)

                    # ====== 3-2. 潛水速率分析（含 Dive Time） ======
                    st.markdown(f"### {tr('overlay_speed_analysis_title')}")

                    # --- FF 開始深度 ---
                    max_depth = float(df_rate["depth_m"].max())
                    default_ff = min(15.0, max_depth)

                    ff_start_overlay = st.number_input(
                        tr("overlay_ff_depth_label"),
                        min_value=0.0,
                        max_value=max_depth,
                        step=1.0,
                        value=default_ff,
                        key="overlay_ff_depth_main",   # ✅ 全新的、不會和其他地方重複的 key
                    )


                    # 使用與「潛水數據比較」相同的公式
                    metrics_overlay = compute_dive_metrics(df_rate, dive_df, ff_start_overlay)

                    def fmt_mps_local(value: Optional[float]) -> str:
                        if value is None or np.isnan(value):
                            return tr("compare_metric_not_available")
                        return tr("compare_metric_unit_mps", value=round(value, 2))

                    def fmt_dive_time_local(t: Optional[float]) -> str:
                        if t is None or (isinstance(t, float) and np.isnan(t)) or t <= 0:
                            return tr("compare_metric_not_available")
                        mm = int(t // 60)
                        ss = int(round(t % 60))
                        return tr("metric_dive_time_value", mm=mm, ss=ss)

                    def render_metric_block_local(title: str, value_str: str):
                        st.markdown(
                            f"""
                            <div style="margin-bottom:6px;">
                                <div style="font-weight:700; font-size:1.05rem; margin-top:0; margin-bottom:0;">
                                    {title}
                                </div>
                                <div style="font-size:0.95rem; margin-top:0; margin-bottom:0.1rem;">
                                    {value_str}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # 顯示順序：潛水時間 → 下潛速率 → 上升速率 → FF 速率
                    render_metric_block_local(
                        tr("metric_dive_time_label"),
                        fmt_dive_time_local(dive_time_s),
                    )
                    render_metric_block_local(
                        tr("compare_desc_rate_label"),
                        fmt_mps_local(metrics_overlay["descent_avg"]),
                    )
                    render_metric_block_local(
                        tr("compare_asc_rate_label"),
                        fmt_mps_local(metrics_overlay["ascent_avg"]),
                    )
                    render_metric_block_local(
                        tr("compare_ff_rate_label"),
                        fmt_mps_local(metrics_overlay["ff_avg"]),
                    )

        # --- 4. 設定時間偏移 & 版型選擇 ---
        st.subheader(tr("align_layout_subheader"))
        
        # ==========================================================
        # 4-1) 對齊模式
        # ==========================================================
        align_mode = st.radio(
            tr("align_mode_label"),
            options=["start", "bottom", "end"],
            format_func=lambda m: {
                "start": tr("align_mode_start"),
                "bottom": tr("align_mode_bottom"),
                "end": tr("align_mode_end"),
            }[m],
            horizontal=False,
            key="overlay_align_mode",
        )

        # ==========================================================
        # 4-2) 影片時間輸入（FF 同款：［－］［input］［＋］）
        #     - widget key: overlay_align_video_time_str
        #     - calc key:   overlay_align_video_time_s
        # ==========================================================
        def parse_time_str_to_seconds_safe(s: str):
            s = (s or "").strip()
            if not s:
                return 0.0
            try:
                parts = s.split(":")
                if len(parts) != 2:
                    return None
                mm = int(parts[0].strip())
                ss = float(parts[1].strip())
                if mm < 0 or ss < 0:
                    return None
                return mm * 60.0 + ss
            except Exception:
                return None

        def seconds_to_mmss_cc(sec: float) -> str:
            sec = max(0.0, float(sec))
            mm = int(sec // 60)
            ss = sec - mm * 60
            return f"{mm:02d}:{ss:05.2f}"  # mm:ss.cc

        def clamp_time(sec: float, max_sec: float = 3600.0) -> float:
            return max(0.0, min(float(sec), float(max_sec)))

        # --- 初始化 state ---
        if "overlay_align_video_time_s" not in st.session_state:
            st.session_state["overlay_align_video_time_s"] = 0.0
        if "overlay_align_video_time_str" not in st.session_state:
            st.session_state["overlay_align_video_time_str"] = "00:00.00"
        if "overlay_align_step_unit" not in st.session_state:
            st.session_state["overlay_align_step_unit"] = "sec"

        # --- 級距設定 ---
        step_map = {
            "min": 60.0,
            "sec": 1.0,
            "csec": 0.1,   # 0.1 秒
        }

        def sync_time_str_from_seconds():
            st.session_state["overlay_align_video_time_str"] = seconds_to_mmss_cc(
                st.session_state["overlay_align_video_time_s"]
            )

        def on_minus():
            step = step_map.get(st.session_state["overlay_align_step_unit"], 1.0)
            st.session_state["overlay_align_video_time_s"] = round(
                clamp_time(st.session_state["overlay_align_video_time_s"] - step), 2
            )
            sync_time_str_from_seconds()

        def on_plus():
            step = step_map.get(st.session_state["overlay_align_step_unit"], 1.0)
            st.session_state["overlay_align_video_time_s"] = round(
                clamp_time(st.session_state["overlay_align_video_time_s"] + step), 2
            )
            sync_time_str_from_seconds()

        # --- 4-2 UI 容器：桌機約 50% 寬、靠左；手機強制全寬 ---
        st.markdown('<div class="align-block align-left">', unsafe_allow_html=True)

        # ① Label
        st.markdown(f"**{tr('align_video_time_label')}**")

        # ② 級距選擇（同區塊，不要多餘空白）
        st.radio(
            label="",
            options=["min", "sec", "csec"],
            horizontal=True,
            format_func=lambda k: {
                "min": tr("align_step_min"),
                "sec": tr("align_step_sec"),
                "csec": tr("align_step_csec"),
            }[k],
            key="overlay_align_step_unit",
            label_visibility="collapsed",
        )

        # ③ － / input / ＋（全形，避免「+」消失）
        b1, sp1, mid, sp2, b2 = st.columns([1, 0.35, 2.3, 0.35, 1], vertical_alignment="center")

        with b1:
            st.button("－", key="overlay_align_minus", on_click=on_minus)

        with sp1:
            st.write("")

        with mid:
            video_time_str = st.text_input(
                label="",
                key="overlay_align_video_time_str",
                label_visibility="collapsed",
                help=tr("align_video_time_help"),
            )

            v_ref_from_text = parse_time_str_to_seconds_safe(video_time_str)
            if v_ref_from_text is None:
                st.warning(tr("align_video_time_invalid"))
            else:
                st.session_state["overlay_align_video_time_s"] = float(v_ref_from_text)

        with sp2:
            st.write("")

        with b2:
            st.button("＋", key="overlay_align_plus", on_click=on_plus)

        st.markdown("</div>", unsafe_allow_html=True)

        # 最終 v_ref（秒）
        v_ref = float(st.session_state["overlay_align_video_time_s"])


        # ==========================================================
        # 4-3) 準備事件時間 + time_offset（確保第 6 段不會 undefined）
        # ==========================================================
        t_ref_raw = None
        if df_rate is not None and dive_df is not None:
            if align_mode == "start":
                t_ref_raw = dive_start_s
            elif align_mode == "end":
                t_ref_raw = dive_end_s
            elif align_mode == "bottom":
                raw = dive_df.sort_values("time_s").reset_index(drop=True)
                after = raw[raw["time_s"] >= dive_start_s]
                within = after[after["time_s"] <= dive_end_s]
                if not within.empty:
                    idx_bottom = within["depth_m"].idxmax()
                    t_ref_raw = float(within.loc[idx_bottom, "time_s"])
        
        # 🔧 只在「對齊」移除那 1 秒 offset（其他邏輯不動）
        t_ref_for_align = (t_ref_raw - 1.0) if (t_ref_raw is not None) else None
        
        if t_ref_for_align is not None:
            time_offset = t_ref_for_align - v_ref
            st.caption(f"目前計算出的偏移：{time_offset:+.2f} 秒（會套用到渲染）")
        else:
            time_offset = 0.0
            st.caption("尚未偵測到潛水事件，暫時使用 0 秒偏移。")
        
        # ==========================================================
        # 4-4) Layout 選擇（確保 selected_id 一定存在）
        # ==========================================================
        LAYOUTS_DIR = ASSETS_DIR / "layouts"
        layouts_config = [
            {"id": "A", "label_key": "layout_a_label", "filename": "layout_a.png", "desc_key": "layout_a_desc", "uses_diver_info": False},
            {"id": "B", "label_key": "layout_b_label", "filename": "layout_b.png", "desc_key": "layout_b_desc", "uses_diver_info": False},
            {"id": "C", "label_key": "layout_c_label", "filename": "layout_c.png", "desc_key": "layout_c_desc", "uses_diver_info": False},
            {"id": "D", "label_key": "layout_d_label", "filename": "layout_d.png", "desc_key": "layout_d_desc", "uses_diver_info": True},
        ]
        layout_ids = [cfg["id"] for cfg in layouts_config]
        
        selected_id = st.selectbox(
            tr("layout_select_label"),
            options=layout_ids,
            format_func=lambda i: tr(f"layout_{i.lower()}_label"),
            key="overlay_layout_id",
        )
        
        def load_layout_image(cfg, is_selected: bool):
            img_path = LAYOUTS_DIR / cfg["filename"]
            img = Image.open(img_path).convert("RGBA")
            if not is_selected:
                return img
        
            border_color = "#FFD700"
            border_width = 12
            corner_radius = 15
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            w, h = img.size
            pad = border_width // 2
            draw.rounded_rectangle(
                [(-pad, -pad), (w + pad - 1, h + pad - 1)],
                radius=corner_radius,
                outline=border_color,
                width=border_width,
            )
            return Image.alpha_composite(img, overlay)
        
        st.markdown("### " + tr("layout_preview_title"))
        cols = st.columns(len(layouts_config))
        for col, cfg in zip(cols, layouts_config):
            with col:
                img = load_layout_image(cfg, cfg["id"] == selected_id)
                st.image(img, caption=tr(cfg["label_key"]), use_container_width=True)
                if cfg.get("desc_key"):
                    st.caption(tr(cfg["desc_key"]))
        
        # --- 5. 輸入潛水員資訊---
        st.subheader(tr("diver_info_subheader"))
        nationality_file = ASSETS_DIR / "Nationality.csv"
        nat_df = load_nationality_options(nationality_file)
        
        not_spec_label = tr("not_specified")
        if nat_df.empty:
            nationality_options = [not_spec_label]
        else:
            nationality_labels = nat_df["label"].tolist()
            nationality_options = [not_spec_label] + nationality_labels
        
        default_label = "Taiwan (TWN)"
        default_index = nationality_options.index(default_label) if default_label in nationality_options else 0
        
        col_info_1, col_info_2 = st.columns(2)
        with col_info_1:
            diver_name = st.text_input(tr("diver_name_label"), value="", key="overlay_diver_name")
            nationality_label = st.selectbox(
                tr("nationality_label"),
                options=nationality_options,
                index=default_index,
                key="overlay_nationality",
            )
            nationality = "" if nationality_label == not_spec_label else nationality_label
        
        with col_info_2:
            discipline = st.selectbox(
                tr("discipline_label"),
                options=[not_spec_label, "CWT", "CWTB", "CNF", "FIM"],
                key="overlay_discipline",
            )
        
        # --- 6. 產生影片 ---
        if st.button(tr("render_button"), type="primary", key="overlay_render_btn"):
            if (dive_df is None) or (video_file is None):
                st.error(tr("error_need_both_files"))
            else:
                progress_bar = st.progress(0, text=tr("progress_init"))
                status_placeholder = st.empty()
                start_time = time.time()
        
                def format_seconds(sec: float) -> str:
                    sec = max(0, int(round(sec)))
                    mm = sec // 60
                    ss = sec % 60
                    return f"{mm:02d}:{ss:02d}"
        
                def progress_callback(p: float, message: str = ""):
                    p = max(0.0, min(1.0, float(p)))
                    percent = int(round(p * 100))
        
                    bar_text = f"{message} {percent}%" if message else f"{tr('progress_rendering')} {percent}%"
        
                    if p >= 1.0:
                        progress_bar.progress(100, text=tr("progress_done"))
                        status_placeholder.empty()
                        return
        
                    progress_bar.progress(percent, text=bar_text)
        
                    elapsed = time.time() - start_time
                    eta_seconds = None
                    if p >= 0.40 and p > 0:
                        total_est = elapsed / p
                        eta_seconds = max(0.0, total_est - elapsed)
        
                    if eta_seconds is None:
                        status_placeholder.info(f"{tr('render_estimate_pending')}\n{tr('render_do_not_leave')}")
                    else:
                        eta_str = format_seconds(eta_seconds)
                        status_placeholder.info(f"{tr('render_estimate_eta', eta=eta_str)}\n{tr('render_do_not_leave')}")
        
                tmp_video_path = Path("/tmp") / video_file.name
                with open(tmp_video_path, "wb") as f:
                    f.write(video_file.read())
        
                try:
                    output_path = render_video(
                        video_path=tmp_video_path,
                        dive_df=dive_df,
                        df_rate=df_rate,
                        time_offset=time_offset,
                        layout=selected_id,
                        assets_dir=ASSETS_DIR,
                        output_resolution=(1080, 1920),
                        diver_name=diver_name,
                        nationality=nationality,
                        discipline=discipline if discipline != not_spec_label else "",
                        dive_time_s=dive_time_s,
                        dive_start_s=dive_start_s,
                        dive_end_s=dive_end_s,
                        progress_callback=progress_callback,
                    )
        
                    progress_callback(1.0, tr("progress_done"))
                    st.success(tr("render_success"))
        
                    with open(output_path, "rb") as f:
                        st.download_button(
                            tr("download_button"),
                            data=f,
                            file_name="dive_overlay_1080p.mp4",
                            mime="video/mp4",
                        )
        
                    col_preview, col_empty = st.columns([1, 1])
                    with col_preview:
                        st.video(str(output_path))
        
                except Exception as e:
                    status_placeholder.empty()
                    st.error(tr("render_error", error=e))


    # ============================
    # Tab 2：潛水數據比較功能
    # ============================
    with tab_compare:
    
        # -------------------------
        # 1. 上傳 A / B（手機上並排）
        # -------------------------
        with st.container():
            st.markdown('<div class="cmp-two-col">', unsafe_allow_html=True)
            cmp_col1, cmp_col2 = st.columns(2)
    
            with cmp_col1:
                cmp_file_a = st.file_uploader(
                    tr("compare_upload_a"),
                    type=["mp4","mov","m4v","avi","mkv","webm"],
                    key="cmp_file_a",
                )
    
            with cmp_col2:
                cmp_file_b = st.file_uploader(
                    tr("compare_upload_b"),
                    type=None,
                    key="cmp_file_b",
                )
            st.markdown('</div>', unsafe_allow_html=True)
    
        # 解析結果容器
        dives_a = []
        dives_b = []
        dive_a = None
        dive_b = None
        label_a = None
        label_b = None
    
        # -------------------------
        # 2. 處理數據 A
        # -------------------------
        if cmp_file_a is not None:
            suffix_a = Path(cmp_file_a.name).suffix.lower()
    
            if suffix_a == ".fit":
                file_bytes_a = cmp_file_a.read()
                dives_a = parse_garmin_fit_to_dives(BytesIO(file_bytes_a))
    
            elif suffix_a == ".uddf":
                dive_a = parse_atmos_uddf(BytesIO(cmp_file_a.read()))
                if dive_a is not None and len(dive_a) > 0:
                    max_depth_a = dive_a["depth_m"].max()
                    label_a = f"ATMOS A ({max_depth_a:.1f} m)"
    
        # -------------------------
        # 3. 處理數據 B
        # -------------------------
        if cmp_file_b is not None:
            suffix_b = Path(cmp_file_b.name).suffix.lower()
    
            if suffix_b == ".fit":
                file_bytes_b = cmp_file_b.read()
                dives_b = parse_garmin_fit_to_dives(BytesIO(file_bytes_b))
    
            elif suffix_b == ".uddf":
                dive_b = parse_atmos_uddf(BytesIO(cmp_file_b.read()))
                if dive_b is not None and len(dive_b) > 0:
                    max_depth_b = dive_b["depth_m"].max()
                    label_b = f"ATMOS B ({max_depth_b:.1f} m)"
    
        # -------------------------
        # 4. Garmin 多潛選擇：A / B 並排顯示
        # -------------------------
        if dives_a or dives_b:
            with st.container():
                st.markdown('<div class="cmp-two-col">', unsafe_allow_html=True)
                sel_col_a, sel_col_b = st.columns(2)
    
                with sel_col_a:
                    if dives_a:
                        options_a = [
                            f"Dive #{i+1}（{df['depth_m'].max():.1f} m）"
                            for i, df in enumerate(dives_a)
                        ]
                        idx_a = st.selectbox(
                            tr("compare_select_dive_a"),
                            options=list(range(len(dives_a))),
                            format_func=lambda i: options_a[i],
                            key="cmp_select_a",
                        )
                        dive_a = dives_a[idx_a]
                        label_a = options_a[idx_a]
    
                with sel_col_b:
                    if dives_b:
                        options_b = [
                            f"Dive #{i+1}（{df['depth_m'].max():.1f} m）"
                            for i, df in enumerate(dives_b)
                        ]
                        idx_b = st.selectbox(
                            tr("compare_select_dive_b"),
                            options=list(range(len(dives_b))),
                            format_func=lambda i: options_b[i],
                            key="cmp_select_b",
                        )
                        dive_b = dives_b[idx_b]
                        label_b = options_b[idx_b]
    
                st.markdown('</div>', unsafe_allow_html=True)
    
        # -------------------------
        # 5. 初始化平滑視窗 / 時間偏移狀態
        # -------------------------
        if "cmp_smooth_level" not in st.session_state:
            st.session_state["cmp_smooth_level"] = 2  # 預設 2 秒
    
        smooth_level = int(st.session_state["cmp_smooth_level"])
    
        if "cmp_align_offset_b" not in st.session_state:
            st.session_state["cmp_align_offset_b"] = 0.0
    
        # -------------------------
        # 6. 準備重採樣後的 df
        # -------------------------
        df_a = prepare_dive_curve(dive_a, smooth_window=smooth_level) if dive_a is not None else None
        df_b = prepare_dive_curve(dive_b, smooth_window=smooth_level) if dive_b is not None else None
    
        if (df_a is None) or (df_b is None):
            st.info(tr("compare_no_data"))
        else:
            # 預設 label
            if label_a is None:
                max_depth_a = df_a["depth_m"].max()
                label_a = f"Dive A ({max_depth_a:.1f} m)"
            if label_b is None:
                max_depth_b = df_b["depth_m"].max()
                label_b = f"Dive B ({max_depth_b:.1f} m)"
    
            # -------------------------
            # 7. Free Fall 開始深度控制（左右各一組）
            # -------------------------
            max_depth_a = float(df_a["depth_m"].max())
            max_depth_b = float(df_b["depth_m"].max())
    
            with st.container():
                st.markdown('<div class="cmp-two-col">', unsafe_allow_html=True)
                ff_col_a, ff_col_b = st.columns(2)
    
                with ff_col_a:
                    ff_start_a = st.number_input(
                        tr("compare_ff_depth_label_a"),
                        min_value=0.0,
                        max_value=max_depth_a,
                        step=1.0,
                        value=min(15.0, max_depth_a),
                        key="cmp_ff_depth_a",
                    )
                with ff_col_b:
                    ff_start_b = st.number_input(
                        tr("compare_ff_depth_label_b"),
                        min_value=0.0,
                        max_value=max_depth_b,
                        step=1.0,
                        value=min(15.0, max_depth_b),
                        key="cmp_ff_depth_b",
                    )
    
                st.markdown('</div>', unsafe_allow_html=True)
    
            # -------------------------
            # 8. 計算各種平均速率 + 潛水時間
            # -------------------------

            def detect_dive_time(dive_df_raw: Optional[pd.DataFrame]) -> Optional[float]:
                """依照 Overlay 頁面相同邏輯，從 depth >= 0.7 m 到回到 0 m 的時間差。"""
                if dive_df_raw is None or len(dive_df_raw) == 0:
                    return None
                if "time_s" not in dive_df_raw.columns or "depth_m" not in dive_df_raw.columns:
                    return None

                raw = dive_df_raw.sort_values("time_s").reset_index(drop=True)

                start_rows = raw[raw["depth_m"] >= 0.7]
                if start_rows.empty:
                    return None

                t_start = float(start_rows["time_s"].iloc[0])
                after = raw[raw["time_s"] >= t_start]
                end_candidates = after[after["depth_m"] <= 0.05]

                if not end_candidates.empty:
                    t_end = float(end_candidates["time_s"].iloc[-1])
                else:
                    t_end = float(after["time_s"].iloc[-1])

                if t_end <= t_start:
                    return None
                return max(0.0, t_end - t_start)

            # Dive Time（秒）
            dive_time_a = detect_dive_time(dive_a)
            dive_time_b = detect_dive_time(dive_b)

            # 平均速率指標
            metrics_a = compute_dive_metrics(df_a, dive_a, ff_start_a)
            metrics_b = compute_dive_metrics(df_b, dive_b, ff_start_b)

            def fmt_mps(value: Optional[float]) -> str:
                if value is None or np.isnan(value):
                    return tr("compare_metric_not_available")
                return tr("compare_metric_unit_mps", value=round(value, 2))

            def fmt_dive_time(t: Optional[float]) -> str:
                if t is None or (isinstance(t, float) and np.isnan(t)) or t <= 0:
                    return tr("compare_metric_not_available")
                mm = int(t // 60)
                ss = int(round(t % 60))
                return tr("metric_dive_time_value", mm=mm, ss=ss)

            def render_metric_block(title: str, value_str: str):
                """子標題和數值之間不留空白行，且子標題字體略大。"""
                st.markdown(
                    f"""
                    <div style="margin-bottom:6px;">
                        <div style="font-weight:700; font-size:1.05rem; margin-top:0; margin-bottom:0;">
                            {title}
                        </div>
                        <div style="font-size:0.95rem; margin-top:0; margin-bottom:0.1rem;">
                            {value_str}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
            # -------------------------
            # 9. A / B 指標顯示（含潛水時間）
            # -------------------------
            with st.container():
                st.markdown('<div class="cmp-two-col">', unsafe_allow_html=True)
                m_col_a, m_col_b = st.columns(2)

                with m_col_a:
                    st.markdown(f"### {label_a}")
                    render_metric_block(tr("metric_dive_time_label"), fmt_dive_time(dive_time_a))
                    render_metric_block(tr("compare_desc_rate_label"), fmt_mps(metrics_a["descent_avg"]))
                    render_metric_block(tr("compare_asc_rate_label"),  fmt_mps(metrics_a["ascent_avg"]))
                    render_metric_block(tr("compare_ff_rate_label"),   fmt_mps(metrics_a["ff_avg"]))

                with m_col_b:
                    st.markdown(f"### {label_b}")
                    render_metric_block(tr("metric_dive_time_label"), fmt_dive_time(dive_time_b))
                    render_metric_block(tr("compare_desc_rate_label"), fmt_mps(metrics_b["descent_avg"]))
                    render_metric_block(tr("compare_asc_rate_label"),  fmt_mps(metrics_b["ascent_avg"]))
                    render_metric_block(tr("compare_ff_rate_label"),   fmt_mps(metrics_b["ff_avg"]))

                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
    
            # -------------------------
            # 10. 調整 B 的時間偏移（移到圖表上方，3 欄排版）
            # -------------------------
            st.markdown(f"**{tr('compare_align_label')}**")
    
            with st.container():
                st.markdown('<div class="cmp-three-col">', unsafe_allow_html=True)
                align_col1, align_col2, align_col3 = st.columns([1, 2, 1])
    
                with align_col1:
                    if st.button("◀ -0.2 s", key="cmp_align_minus"):
                        st.session_state["cmp_align_offset_b"] = max(
                            -20.0, st.session_state["cmp_align_offset_b"] - 0.2
                        )
    
                with align_col2:
                    st.markdown(
                        f"<div style='text-align:center; font-size:1.05rem; "
                        f"margin-top:0.25rem; margin-bottom:0.25rem;'>"
                        f"{tr('compare_align_current', offset=st.session_state['cmp_align_offset_b'])}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
    
                with align_col3:
                    if st.button("+0.2 s ▶", key="cmp_align_plus"):
                        st.session_state["cmp_align_offset_b"] = min(
                            20.0, st.session_state["cmp_align_offset_b"] + 0.2
                        )
    
                st.markdown('</div>', unsafe_allow_html=True)
    
            align_offset_b = float(st.session_state["cmp_align_offset_b"])
    
            # -------------------------
            # 11. 準備繪圖用資料（含時間偏移）
            # -------------------------
            plot_a_depth = df_a[["time_s", "depth_m"]].copy()
            plot_a_depth["series"] = label_a
            plot_a_depth["time_plot"] = plot_a_depth["time_s"]
    
            plot_b_depth = df_b[["time_s", "depth_m"]].copy()
            plot_b_depth["series"] = label_b
            plot_b_depth["time_plot"] = plot_b_depth["time_s"] + align_offset_b
    
            depth_plot_df = pd.concat([plot_a_depth, plot_b_depth], ignore_index=True)
    
            # 深度負值直接剪掉，避免 Y 軸往下多拉一截
            depth_plot_df["depth_plot"] = depth_plot_df["depth_m"].clip(lower=0.0)
    
            plot_a_rate = df_a[["time_s", "rate_abs_mps_smooth"]].copy()
            plot_a_rate["series"] = label_a
            plot_a_rate["time_plot"] = plot_a_rate["time_s"]
    
            plot_b_rate = df_b[["time_s", "rate_abs_mps_smooth"]].copy()
            plot_b_rate["series"] = label_b
            plot_b_rate["time_plot"] = plot_b_rate["time_s"] + align_offset_b
    
            rate_plot_df = pd.concat([plot_a_rate, plot_b_rate], ignore_index=True)
    
            # -------------------------
            # 11-1. X / Y 軸 domain 設定（X 軸鎖定 0～max，Y 不顯示負數）
            # -------------------------
            max_time_plot = float(
                max(depth_plot_df["time_plot"].max(), rate_plot_df["time_plot"].max())
            )
            if max_time_plot < 0:
                max_time_plot = 0.0
    
            max_depth_plot = float(depth_plot_df["depth_plot"].max())
            max_depth_plot = max(max_depth_plot, 0.0)
    
            max_rate_plot = float(rate_plot_df["rate_abs_mps_smooth"].max())
            max_rate_domain = max(0.5, np.ceil(max_rate_plot * 2.0) / 2.0)
    
            # -------------------------
            # 12. 深度 vs 時間（比較）— 不顯示圖例
            # -------------------------
            depth_chart_cmp = (
                alt.Chart(depth_plot_df)
                .mark_line(interpolate="monotone")
                .encode(
                    x=alt.X(
                        "time_plot:Q",
                        title=tr("axis_time_seconds"),
                        scale=alt.Scale(domain=[0, max_time_plot], nice=False),
                    ),
                    y=alt.Y(
                        "depth_plot:Q",
                        title=tr("axis_depth_m"),
                        scale=alt.Scale(
                            domain=[max_depth_plot, 0],  # 反轉，且不顯示 < 0
                            nice=False,
                        ),
                    ),
                    color=alt.Color(
                        "series:N",
                        title=tr("compare_series_legend"),
                        legend=None,  # 這裡不要圖例
                    ),
                    tooltip=[
                        alt.Tooltip("series:N", title=tr("compare_series_legend")),
                        alt.Tooltip("time_plot:Q", title=tr("tooltip_time"), format=".1f"),
                        alt.Tooltip("depth_plot:Q", title=tr("tooltip_depth"), format=".1f"),
                    ],
                )
                .properties(
                    title=tr("compare_depth_chart_title"),
                    height=320,
                )
            )
    
            # -------------------------
            # 13. 速率 vs 時間（比較）— 圖例放在下方
            # -------------------------
            rate_chart_cmp = (
                alt.Chart(rate_plot_df)
                .mark_line(interpolate="monotone")
                .encode(
                    x=alt.X(
                        "time_plot:Q",
                        title=tr("axis_time_seconds"),
                        scale=alt.Scale(domain=[0, max_time_plot], nice=False),
                    ),
                    y=alt.Y(
                        "rate_abs_mps_smooth:Q",
                        title=tr("axis_rate_mps"),
                        scale=alt.Scale(domain=[0, max_rate_domain], nice=False),
                    ),
                    color=alt.Color(
                        "series:N",
                        title=tr("compare_series_legend"),
                        legend=alt.Legend(orient="bottom"),
                    ),
                    tooltip=[
                        alt.Tooltip("series:N", title=tr("compare_series_legend")),
                        alt.Tooltip("time_plot:Q", title=tr("tooltip_time"), format=".1f"),
                        alt.Tooltip("rate_abs_mps_smooth:Q", title=tr("tooltip_rate"), format=".2f"),
                    ],
                )
                .properties(
                    title=tr("compare_rate_chart_title"),
                    height=320,
                )
            )
    
            st.altair_chart(depth_chart_cmp, use_container_width=True)
            st.altair_chart(rate_chart_cmp, use_container_width=True)
    
            # -------------------------
            # 14. 速率平滑視窗（圖表下面、縮小並貼最右邊）
            # -------------------------
            spacer_l, spacer_mid, smooth_col = st.columns([10, 1, 1])
            with smooth_col:
                st.markdown(
                    f"<div style='text-align:right; font-size:0.85rem; margin-bottom:2px;'>"
                    f"{tr('compare_smooth_label')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.selectbox(
                    "",
                    options=[1, 2, 3],
                    key="cmp_smooth_level",  # 保持同一個 key，改變會觸發重新計算
                    label_visibility="collapsed",
                )

    st.markdown('</div>', unsafe_allow_html=True)
