from PIL import Image, ImageDraw
import streamlit as st
from pathlib import Path
import pandas as pd
from io import BytesIO
import numpy as np
import altair as alt
from typing import Optional

from core.parser_garmin import parse_garmin_fit_to_dives
from core.parser_atmos import parse_atmos_uddf
from core.video_renderer import render_video

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

st.set_page_config(page_title="Dive Overlay Generator", layout="wide")

# ==================================
# 全局 CSS：版面 + 手機優化
# ==================================
APP_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* 讓整個主內容置中、限制最大寬度 */
.main > div {
    display: flex;
    justify-content: center;
}
.main > div > div {
    max-width: 1200px;
    width: 100%;
}

/* 白底卡片容器（主內容） */
.app-card {
    background-color: rgba(255,255,255,0.90);
    border-radius: 18px;
    padding: 1rem 1.2rem 1.4rem 1.2rem;
    box-shadow: 0 8px 20px rgba(15,23,42,0.10);
}

/* 深色模式下卡片變暗一點 */
@media (prefers-color-scheme: dark) {
    .app-card {
        background-color: rgba(15,23,42,0.90);
        box-shadow: 0 8px 20px rgba(0,0,0,0.60);
    }
}

/* Sticky 頂部列 */
.app-top-bar {
    padding: 0.2rem 0.6rem;
    backdrop-filter: blur(6px);
}

/* Subheader 標題（st.subheader）- 縮小一點 */
h3 {
    font-size: 1.05rem !important;
    margin-top: 0.6rem;
    margin-bottom: 0.2rem;
}

/* 預設整體左右留一點空間（桌機） */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* 所有 columns 的間距稍微縮小一點 */
.stHorizontalBlock {
    gap: 0.6rem !important;
}

/* 手機優化：兩欄並排、不產生橫向捲軸 */
@media (max-width: 768px) {
    /* 手機左右邊界縮小 */
    .block-container {
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !entire !important;
    }

    .app-card {
        padding: 0.8rem 0.9rem 1.1rem 0.9rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(15,23,42,0.15);
    }

    /* 讓所有 st.columns(2) 的容器用 grid，一行 2 欄 */
    .stHorizontalBlock {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        column-gap: 0.6rem !important;
    }

    /* 每一個 column 占 1 格，避免再被撐滿一整行 */
    .stHorizontalBlock > div[data-testid="column"] {
        max-width: 100% !important;
    }

    /* 手機上標題稍微縮小，避免換行太多 */
    .app-top-bar span {
        font-size: 1.4rem !important;
    }
    h1, .block-container h1 {
        font-size: 1.4rem !important;
    }
    h2, .block-container h2 {
        font-size: 1.2rem !important;
    }
    h3, .block-container h3 {
        font-size: 1.0rem !important;
    }

    /* 按鈕在手機維持整行寬度 */
    .stButton > button,
    .stDownloadButton > button {
        width: 100%;
    }
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)

# ==================================
# 多語系
# ==================================
if "lang" not in st.session_state:
    st.session_state["lang"] = "zh"  # 預設中文

LANG_OPTIONS = {
    "zh": "中文",
    "en": "English",
}

TRANSLATIONS = {
    "zh": {
        "app_title": "Dive Overlay Generator",
        "top_brand": "DepthRender",
        "language_label": "🌐 語言",

        "tab_overlay_title": "🎬 疊加影片產生器",
        "tab_compare_title": "📊 潛水數據比較",
        "compare_coming_soon": "這裡未來會加入不同潛水之間的曲線比較功能，例如：\n\n- 深度曲線對比\n- 速率 / FF 比例比較\n- 不同比賽 / 不同天的表現差異",

        # Overlay tab
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
        "preview_caption": "原始資料點數：{n_points}，重採樣時間範圍：{t_min:.0f}～{t_max:.0f} 秒，最大深度：約 {max_depth:.1f} m",

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

        "diver_info_subheader": "5️⃣ 潛水員資訊（選填，主要給 Layout B 使用）",
        "diver_name_label": "潛水員姓名 / Nickname",
        "nationality_label": "國籍",
        "discipline_label": "潛水項目（Discipline）",
        "not_specified": "（不指定）",

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

        # Compare tab
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
    },
    "en": {
        "app_title": "Dive Overlay Generator",
        "top_brand": "DepthRender",
        "language_label": "🌐 Language",

        "tab_overlay_title": "🎬 Overlay Generator",
        "tab_compare_title": "📊 Dive Comparison",
        "compare_coming_soon": "This tab will later provide dive-to-dive comparison, such as:\n\n- Depth curve comparison\n- Speed / free-fall ratio\n- Performance across different sessions / competitions",

        # Overlay tab
        "upload_watch_subheader": "1️⃣ Upload dive log",
        "upload_watch_label": "Dive log (.fit/.uddf)",
        "upload_video_subheader": "2️⃣ Upload dive video",
        "upload_video_label": "Video file (any resolution)",
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

        "diver_info_subheader": "5️⃣ Diver info (optional, mainly for Layout B)",
        "diver_name_label": "Diver name / Nickname",
        "nationality_label": "Nationality",
        "discipline_label": "Discipline",
        "not_specified": "(Not specified)",

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

        # Compare tab
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
    },
}

def tr(key: str, **kwargs) -> str:
    lang = st.session_state.get("lang", "zh")
    text = TRANSLATIONS.get(lang, TRANSLATIONS["zh"]).get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text

def set_language():
    label_to_code = {v: k for k, v in LANG_OPTIONS.items()}
    selected_label = st.session_state.get("_lang_select", LANG_OPTIONS["zh"])
    st.session_state["lang"] = label_to_code.get(selected_label, "zh")

# -------------------------------
# 頂部：品牌 + 語言切換
# -------------------------------
top_left, top_right = st.columns([6, 2])

with top_left:
    st.markdown(
        f"""
        <div class="app-top-bar">
            <div class="app-title-text">
                🌊 {tr('top_brand')}
            </div>
            <div style="font-size:0.95rem; opacity:0.75; margin-top:0.2rem;">
                {tr('app_title')}
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
        key="_lang_select",
        index=list(LANG_OPTIONS.keys()).index(st.session_state["lang"]),
        on_change=set_language,
    )

# 讀取國籍 / 國碼清單
@st.cache_data
def load_nationality_options(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        st.error(tr("nationality_file_not_found", path=csv_path))
        return pd.DataFrame(columns=["Country", "Code", "label"])
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(tr("nationality_read_error", error=e))
        return pd.DataFrame(columns=["Country", "Code", "label"])

    required_cols = {"Country", "Code"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(tr("nationality_missing_columns", missing=list(missing)))
        return pd.DataFrame(columns=["Country", "Code", "label"])

    df = df.dropna(subset=["Country", "Code"]).copy()
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Code"] = df["Code"].astype(str).str.upper().str.strip()
    df["label"] = df["Country"] + " (" + df["Code"] + ")"
    return df

# ================================
# 共用：重採樣 + 速率
# ================================
def prepare_dive_curve(dive_df: pd.DataFrame, smooth_window: int) -> Optional[pd.DataFrame]:
    if dive_df is None or len(dive_df) == 0:
        return None
    if "time_s" not in dive_df.columns or "depth_m" not in dive_df.columns:
        return None

    df = dive_df.sort_values("time_s").reset_index(drop=True).copy()
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

    # 1) Dive start/end
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

    # 2) Bottom time
    within_dive = after[after["time_s"] <= t_end]
    if within_dive.empty:
        return result
    idx_bottom = within_dive["depth_m"].idxmax()
    t_bottom = float(within_dive.loc[idx_bottom, "time_s"])

    df = df_rate.sort_values("time_s").reset_index(drop=True)

    # Descent: t_start+1 ~ t_bottom-1
    desc_start = t_start + 1.0
    desc_end = t_bottom - 1.0
    if desc_end > desc_start:
        mask_desc = (df["time_s"] >= desc_start) & (df["time_s"] <= desc_end)
        seg_desc = df.loc[mask_desc]
        if not seg_desc.empty:
            result["descent_avg"] = float(seg_desc["rate_abs_mps_smooth"].mean())

    # Ascent: t_bottom+1 ~ t_end
    asc_start = t_bottom + 1.0
    asc_end = t_end
    if asc_end > asc_start:
        mask_asc = (df["time_s"] >= asc_start) & (df["time_s"] <= asc_end)
        seg_asc = df.loc[mask_asc]
        if not seg_asc.empty:
            result["ascent_avg"] = float(seg_asc["rate_abs_mps_smooth"].mean())

    # Free-fall segment
    max_depth = float(within_dive["depth_m"].max())
    if ff_start_depth_m > 0.0 and ff_start_depth_m < max_depth:
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
# 主畫面：卡片 + Tabs
# ================================
with st.container():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)

    tab_overlay, tab_compare = st.tabs([
        tr("tab_overlay_title"),
        tr("tab_compare_title"),
    ])

    # ============================
    # Tab 1：疊加影片產生器
    # ============================
    with tab_overlay:
        st.markdown(
            f"<h1 style='font-size:1.5rem; margin-top:0.5rem; margin-bottom:0.6rem; font-weight:700;'>"
            f"{tr('tab_overlay_title')}"
            f"</h1>",
            unsafe_allow_html=True,
        )

        # --- 1. 上傳區 ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(tr("upload_watch_subheader"))
            watch_file = st.file_uploader(
                tr("upload_watch_label"),
                type=None,
                key="overlay_watch_file",
            )
        with col2:
            st.subheader(tr("upload_video_subheader"))
            video_file = st.file_uploader(
                tr("upload_video_label"),
                type=["mp4", "mov", "m4v"],
                key="overlay_video_file",
            )

        # --- 2. 解析手錶資料 ---
        dive_df = None
        df_rate = None
        dive_time_s = None
        dive_start_s = None
        dive_end_s = None

        if watch_file is not None:
            suffix = Path(watch_file.name).suffix.lower()
            if suffix == ".fit":
                st.info(tr("fit_detected"))
                dives = parse_garmin_fit_to_dives(BytesIO(watch_file.read()))
                if len(dives) == 0:
                    st.error(tr("fit_no_dives"))
                else:
                    options = [
                        f"Dive #{i+1}（{df['depth_m'].max():.1f} m）"
                        for i, df in enumerate(dives)
                    ]
                    idx = st.selectbox(
                        tr("select_dive_label"),
                        options=list(range(len(dives))),
                        format_func=lambda i: options[i],
                        key="overlay_dive_index",
                    )
                    dive_df = dives[idx]
            elif suffix == ".uddf":
                st.info(tr("uddf_detected"))
                dive_df = parse_atmos_uddf(BytesIO(watch_file.read()))

        # --- 3. 曲線預覽 ---
        if dive_df is not None:
            if len(dive_df) == 0:
                st.warning(tr("no_depth_samples"))
            else:
                dive_df = dive_df.sort_values("time_s").reset_index(drop=True)

                # 強制加上起始/結束 0 m 點
                if len(dive_df) > 0:
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
                    ).sort_values("time_s").reset_index(drop=True)

                df_rate = prepare_dive_curve(dive_df, smooth_window=3)
                if df_rate is not None:
                    t_min = df_rate["time_s"].min()
                    t_resample_max = df_rate["time_s"].max()
                    max_display_time = int(np.ceil(t_resample_max / 5)) * 5

                    # 計算 Dive Time
                    df_sorted = dive_df.sort_values("time_s").reset_index(drop=True)
                    start_rows = df_sorted[df_sorted["depth_m"] >= 0.7]
                    if not start_rows.empty:
                        t_start = start_rows["time_s"].iloc[0]
                        after = df_sorted[df_sorted["time_s"] >= t_start]
                        end_candidates = after[after["depth_m"] <= 0.05]
                        if not end_candidates.empty:
                            t_end = end_candidates["time_s"].iloc[-1]
                        else:
                            t_end = after["time_s"].iloc[-1]

                        dive_start_s = float(t_start)
                        dive_end_s = float(t_end)
                        dive_time_s = max(0.0, dive_end_s - dive_start_s)

                    if dive_time_s is not None:
                        mm = int(dive_time_s // 60)
                        ss = int(round(dive_time_s % 60))
                        st.info(tr("dive_time_detected", mm=mm, ss=ss))

                    st.subheader(tr("preview_subheader"))
                    col_depth, col_rate = st.columns(2)

                    with col_depth:
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
                            .interactive()
                        )
                        st.altair_chart(depth_chart, use_container_width=True)

                    with col_rate:
                        rate_chart = (
                            alt.Chart(df_rate)
                            .mark_line()
                            .encode(
                                x=alt.X(
                                    "time_s:Q",
                                    title=tr("axis_time_seconds"),
                                    scale=alt.Scale(domain=[t_min, max_display_time]),
                                ),
                                y=alt.Y(
                                    "rate_abs_mps_smooth:Q",
                                    title=tr("axis_rate_mps"),
                                    scale=alt.Scale(domain=[0, 3]),
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
                            .interactive()
                        )
                        st.altair_chart(rate_chart, use_container_width=True)

                    st.caption(
                        tr(
                            "preview_caption",
                            n_points=len(dive_df),
                            t_min=df_rate["time_s"].min(),
                            t_max=df_rate["time_s"].max(),
                            max_depth=df_rate["depth_m"].max(),
                        )
                    )

        # --- 4. 時間偏移 + 版型 ---
        st.subheader(tr("align_layout_subheader"))
        col3, col4 = st.columns(2)

        with col3:
            time_offset = st.slider(
                tr("time_offset_label"),
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.1,
                help=tr("time_offset_help"),
                key="overlay_time_offset",
            )

        LAYOUTS_DIR = ASSETS_DIR / "layouts"
        layouts_config = [
            {"id": "A", "label_key": "layout_a_label", "filename": "layout_a.png", "desc_key": "layout_a_desc", "uses_diver_info": False},
            {"id": "B", "label_key": "layout_b_label", "filename": "layout_b.png", "desc_key": "layout_b_desc", "uses_diver_info": False},
            {"id": "C", "label_key": "layout_c_label", "filename": "layout_c.png", "desc_key": "layout_c_desc", "uses_diver_info": False},
            {"id": "D", "label_key": "layout_d_label", "filename": "layout_d.png", "desc_key": "layout_d_desc", "uses_diver_info": True},
        ]
        layout_ids = [cfg["id"] for cfg in layouts_config]

        with col4:
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
            img = Image.alpha_composite(img, overlay)
            return img

        st.markdown("### " + tr("layout_preview_title"))
        cols = st.columns(len(layouts_config))
        for col, cfg in zip(cols, layouts_config):
            with col:
                img = load_layout_image(cfg, cfg["id"] == selected_id)
                st.image(img, caption=tr(cfg["label_key"]), use_container_width=True)
                if cfg.get("desc_key"):
                    st.caption(tr(cfg["desc_key"]))

        # --- 5. 潛水員資訊 ---
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
        if default_label in nationality_options:
            default_index = nationality_options.index(default_label)
        else:
            default_index = 0

        col_info_1, col_info_2 = st.columns(2)
        with col_info_1:
            diver_name = st.text_input(tr("diver_name_label"), value="", key="overlay_diver_name")
            nationality_label = st.selectbox(
                tr("nationality_label"),
                options=nationality_options,
                index=default_index,
                key="overlay_nationality",
            )
            if nationality_label == not_spec_label:
                nationality = ""
            else:
                nationality = nationality_label
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

                def progress_callback(p: float, message: str = ""):
                    p = max(0.0, min(1.0, float(p)))
                    percent = int(p * 100)
                    if message:
                        text = f"{message} {percent}%"
                    else:
                        text = f"{tr('progress_rendering')} {percent}%"
                    progress_bar.progress(percent, text=text)

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
                    col_preview, _ = st.columns([1, 1])
                    with col_preview:
                        st.video(str(output_path))
                except Exception as e:
                    st.error(tr("render_error", error=e))

    # ============================
    # Tab 2：潛水數據比較
    # ============================
    with tab_compare:
        st.markdown(
            f"<h2 style='font-size:1.4rem; margin-top:0.5rem; margin-bottom:0.8rem; font-weight:600;'>"
            f"{tr('compare_title')}"
            f"</h2>",
            unsafe_allow_html=True,
        )

        # 1. 上傳 A / B
        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            cmp_file_a = st.file_uploader(
                tr("compare_upload_a"),
                type=None,
                key="cmp_file_a",
            )
        with cmp_col2:
            cmp_file_b = st.file_uploader(
                tr("compare_upload_b"),
                type=None,
                key="cmp_file_b",
            )

        dives_a, dives_b = [], []
        dive_a = dive_b = None
        label_a = label_b = None

        # 2. 處理數據 A
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

        # 3. 處理數據 B
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

        # 4. Garmin 多潛選擇（A/B 並排）
        if dives_a or dives_b:
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

        # 5. 狀態：平滑視窗 / 時間偏移
        if "cmp_smooth_level" not in st.session_state:
            st.session_state["cmp_smooth_level"] = 2
        smooth_level = int(st.session_state["cmp_smooth_level"])

        if "cmp_align_offset_b" not in st.session_state:
            st.session_state["cmp_align_offset_b"] = 0.0

        # 6. 重採樣
        df_a = prepare_dive_curve(dive_a, smooth_window=smooth_level) if dive_a is not None else None
        df_b = prepare_dive_curve(dive_b, smooth_window=smooth_level) if dive_b is not None else None

        if (df_a is None) or (df_b is None):
            st.info(tr("compare_no_data"))
        else:
            if label_a is None:
                max_depth_a = df_a["depth_m"].max()
                label_a = f"Dive A ({max_depth_a:.1f} m)"
            if label_b is None:
                max_depth_b = df_b["depth_m"].max()
                label_b = f"Dive B ({max_depth_b:.1f} m)"

            # 7. FF 開始深度控制
            max_depth_a = float(df_a["depth_m"].max())
            max_depth_b = float(df_b["depth_m"].max())

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

            # 8. 平均速率
            metrics_a = compute_dive_metrics(df_a, dive_a, ff_start_a)
            metrics_b = compute_dive_metrics(df_b, dive_b, ff_start_b)

            def fmt_mps(value: Optional[float]) -> str:
                if value is None or np.isnan(value):
                    return tr("compare_metric_not_available")
                return tr("compare_metric_unit_mps", value=value)

            def render_metric_block(title: str, value: Optional[float]):
                value_str = fmt_mps(value)
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

            m_col_a, m_col_b = st.columns(2)
            with m_col_a:
                st.markdown(f"### {label_a}")
                render_metric_block(tr("compare_desc_rate_label"), metrics_a["descent_avg"])
                render_metric_block(tr("compare_asc_rate_label"), metrics_a["ascent_avg"])
                render_metric_block(tr("compare_ff_rate_label"), metrics_a["ff_avg"])
            with m_col_b:
                st.markdown(f"### {label_b}")
                render_metric_block(tr("compare_desc_rate_label"), metrics_b["descent_avg"])
                render_metric_block(tr("compare_asc_rate_label"), metrics_b["ascent_avg"])
                render_metric_block(tr("compare_ff_rate_label"), metrics_b["ff_avg"])

            st.markdown("---")

            # 9. 時間偏移控制
            st.markdown(f"**{tr('compare_align_label')}**")
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

            align_offset_b = float(st.session_state["cmp_align_offset_b"])

            # 10. 繪圖資料（含時間偏移）
            plot_a_depth = df_a[["time_s", "depth_m"]].copy()
            plot_a_depth["series"] = label_a
            plot_a_depth["time_plot"] = plot_a_depth["time_s"]

            plot_b_depth = df_b[["time_s", "depth_m"]].copy()
            plot_b_depth["series"] = label_b
            plot_b_depth["time_plot"] = plot_b_depth["time_s"] + align_offset_b

            depth_plot_df = pd.concat([plot_a_depth, plot_b_depth], ignore_index=True)
            depth_plot_df["depth_plot"] = depth_plot_df["depth_m"].clip(lower=0.0)

            plot_a_rate = df_a[["time_s", "rate_abs_mps_smooth"]].copy()
            plot_a_rate["series"] = label_a
            plot_a_rate["time_plot"] = plot_a_rate["time_s"]

            plot_b_rate = df_b[["time_s", "rate_abs_mps_smooth"]].copy()
            plot_b_rate["series"] = label_b
            plot_b_rate["time_plot"] = plot_b_rate["time_s"] + align_offset_b

            rate_plot_df = pd.concat([plot_a_rate, plot_b_rate], ignore_index=True)

            max_time_plot = float(
                max(depth_plot_df["time_plot"].max(), rate_plot_df["time_plot"].max())
            )
            if max_time_plot < 0:
                max_time_plot = 0.0

            max_depth_plot = float(depth_plot_df["depth_plot"].max())
            max_depth_plot = max(max_depth_plot, 0.0)

            max_rate_plot = float(rate_plot_df["rate_abs_mps_smooth"].max())
            max_rate_domain = max(0.5, np.ceil(max_rate_plot * 2.0) / 2.0)

            # 11. 深度 vs 時間（不顯示 <0，Legend 拿掉）
            depth_zoom = alt.selection_interval(bind="scales", encodings=["x"])
            depth_chart_cmp = (
                alt.Chart(depth_plot_df)
                .mark_line()
                .encode(
                    x=alt.X(
                        "time_plot:Q",
                        title=tr("axis_time_seconds"),
                        scale=alt.Scale(domain=[0, max_time_plot], nice=False),
                    ),
                    y=alt.Y(
                        "depth_plot:Q",
                        title=tr("axis_depth_m"),
                        scale=alt.Scale(domain=[0, max_depth_plot], reverse=True, nice=False),
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
                .add_selection(depth_zoom)
            )

            # 12. 速率 vs 時間（保留 Legend）
            rate_zoom = alt.selection_interval(bind="scales", encodings=["x"])
            rate_chart_cmp = (
                alt.Chart(rate_plot_df)
                .mark_line()
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
                .add_selection(rate_zoom)
            )

            st.altair_chart(depth_chart_cmp, use_container_width=True)
            st.altair_chart(rate_chart_cmp, use_container_width=True)

            # 13. 速率平滑視窗（右下角縮小）
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
                    key="cmp_smooth_level",
                    label_visibility="collapsed",
                )

    st.markdown('</div>', unsafe_allow_html=True)
