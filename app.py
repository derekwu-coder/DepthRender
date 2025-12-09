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

/* 讓內容置中並限制最大寬度 */
.main > div {
    display: flex;
    justify-content: center;
}

.main > div > div {
    max-width: 1200px;
}

/* Sticky 頂部列 */
.app-top-bar {
    padding: 0.2rem 0.6rem;
    backdrop-filter: blur(6px);
}

/* 白底卡片容器 */
.app-card {
    background-color: rgba(255,255,255,0.90);
    border-radius: 18px;
    padding: 1rem 1.2rem 1.4rem 1.2rem;
    box-shadow: 0 8px 20px rgba(15,23,42,0.10);
}

/* Subheader 標題（st.subheader）- 縮小一點 */
h3 {
    font-size: 1.05rem !important;
    margin-top: 0.6rem;
    margin-bottom: 0.2rem;
}

/* 手機優化 */
@media (max-width: 768px) {
    .app-card {
        padding: 0.8rem 0.9rem 1.1rem 0.9rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(15,23,42,0.15);
    }

    h3 {
        font-size: 0.95rem !important;
    }

    .stButton>button {
        width: 100%;
    }

    .stDownloadButton>button {
        width: 100%;
    }
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
        "app_title": "🌊 Dive Data Overlay Generator (Beta)",
        "top_brand": "DepthRender",
        "language_label": "🌐 語言",

        "tab_overlay_title": "🎬 疊加影片產生器",
        "tab_compare_title": "📊 潛水數據比較（即將推出）",
        "compare_coming_soon": "這裡未來會加入不同潛水之間的曲線比較功能，例如：\n\n- 深度曲線對比\n- 速率 / FF 比例比較\n- 不同比賽 / 不同天的表現差異",

        "upload_watch_subheader": "1️⃣ 上傳手錶數據",
        "upload_watch_label": "上傳潛水錶原始紀錄檔（.fit / .uddf）",
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
        "time_offset_label": "影片起點相對於潛水開始時間的偏移（秒）",
        "time_offset_help": "如果影片比實際下潛早開始，請用負值調整。",
        "layout_select_label": "選擇影片版型",
        "layout_preview_title": "版型示意圖（目前選擇會加黃色外框）",

        "layout_a_label": "Layout A：深度 + 心率 + 速率",
        "layout_a_desc": "（ATMOS 無法顯示心率）",
        "layout_b_label": "Layout B：包含姓名 / 國籍 / 潛水項目",
        "layout_b_desc": "賽事風格版型。",
        "layout_c_label": "Layout C：單純深度",
        "layout_c_desc": "Simple_A",
        "layout_d_label": "Layout D：單純深度",
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
    },
    "en": {
        "app_title": "🌊 Dive Data Overlay Generator (Beta)",
        "top_brand": "DepthRender",
        "language_label": "🌐 Language",

        "tab_overlay_title": "🎬 Overlay Generator",
        "tab_compare_title": "📊 Dive Comparison (Coming Soon)",
        "compare_coming_soon": "This tab will later provide dive-to-dive comparison, such as:\n\n- Depth curve comparison\n- Speed / free-fall ratio\n- Performance across different sessions / competitions",

        "upload_watch_subheader": "1️⃣ Upload dive log",
        "upload_watch_label": "Upload dive computer log (.fit / .uddf)",
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
        "tooltip_rate": "speed (m/s)",
        "depth_chart_title": "Depth vs Time",
        "rate_chart_title": "Speed vs Time",
        "preview_caption": "Raw samples: {n_points}, resampled time range: {t_min:.0f}–{t_max:.0f} s, max depth: ~{max_depth:.1f} m",

        "align_layout_subheader": "4️⃣ Video alignment & layout",
        "time_offset_label": "Video start offset relative to dive start (seconds)",
        "time_offset_help": "If the video starts before the actual dive, use a negative offset.",
        "layout_select_label": "Choose overlay layout",
        "layout_preview_title": "Layout preview (selected layout highlighted in yellow)",

        "layout_a_label": "Layout A: Depth + Heart rate + Speed",
        "layout_a_desc": "(Heart rate not available for ATMOS logs)",
        "layout_b_label": "Layout B: Name / Nationality / Discipline",
        "layout_b_desc": "Competition-style layout.",
        "layout_c_label": "Layout C: Depth only",
        "layout_c_desc": "Simple_A",
        "layout_d_label": "Layout D: Depth only",
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
    selected_label = st.session_state.get("_lang_select", LANG_OPTIONS["zh"])
    st.session_state["lang"] = label_to_code.get(selected_label, "zh")

# -------------------------------
# 頂部：左邊品牌、右邊語言選單
# -------------------------------
top_left, top_right = st.columns([8, 1])

with top_left:
    # 使用自訂頂部列，搭配 CSS
    st.markdown(
        f"""
        <div class="app-top-bar">
            <span style="font-size: 2.5rem; font-weight: 700;">  
                🌊 {tr('top_brand')}
            </span>
            <span style="font-size: 0.9rem; opacity: 0.7; margin-left: 0.4rem;">
                Dive Overlay Generator
            </span>
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
# 主畫面內容開始（卡片 + Tabs）
# ================================
with st.container():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)

    # Tabs：目前功能 + 預留比較分頁
    tab_overlay, tab_compare = st.tabs([
        tr("tab_overlay_title"),
        tr("tab_compare_title"),
    ])

    # ============================
    # Tab 1：疊加影片產生器
    # ============================
    with tab_overlay:
        # 🔹 主功能標題：比 subheader 大一階
        st.markdown(
            f"<h1 style='font-size:2.1rem; margin-top:0.5rem; margin-bottom:0.6rem; font-weight:700;'>"
            f"{tr('app_title')}"
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
            )

        with col2:
            st.subheader(tr("upload_video_subheader"))
            video_file = st.file_uploader(
                tr("upload_video_label"),
                type=["mp4", "mov", "m4v"],
            )

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
                    t_first = float(dive_df["time_s"].iloc[0])
                    t_last  = float(dive_df["time_s"].iloc[-1])

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

                # ================================
                # 方案 A：重採樣成「每秒一筆」再計算速率
                # ================================
                t_min = int(np.floor(dive_df["time_s"].min()))
                t_raw_max = dive_df["time_s"].max()
                t_resample_max = int(np.ceil(t_raw_max))

                uniform_time = np.arange(t_min, t_resample_max + 1, 1)

                depth_interp = np.interp(
                    uniform_time,
                    dive_df["time_s"].to_numpy(),
                    dive_df["depth_m"].to_numpy()
                )

                rate_uniform = np.diff(depth_interp, prepend=depth_interp[0])

                rate_abs = np.abs(rate_uniform)
                rate_abs_clipped = np.clip(rate_abs, 0.0, 3.0)

                df_rate = pd.DataFrame({
                    "time_s": uniform_time,
                    "depth_m": depth_interp,
                    "rate_abs_mps": rate_abs_clipped,
                })

                window_sec = 3
                df_rate["rate_abs_mps_smooth"] = (
                    df_rate["rate_abs_mps"]
                    .rolling(window=window_sec, center=True, min_periods=1)
                    .mean()
                )

                max_display_time = int(np.ceil(t_resample_max / 5)) * 5

                # ================================
                # 計算 Dive Time
                # ================================
                dive_time_s = None
                dive_start_s = None
                dive_end_s = None

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
                    dive_end_s   = float(t_end)
                    dive_time_s  = max(0.0, dive_end_s - dive_start_s)

                if dive_time_s is not None:
                    mm = int(dive_time_s // 60)
                    ss = int(round(dive_time_s % 60))
                    st.info(tr("dive_time_detected", mm=mm, ss=ss))

                # ================================
                # 3️⃣ 左右並排圖表
                # ================================
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

        # --- 4. 設定時間偏移 & 版型選擇 ---
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
            )

        # ------------ 動態 Layout 設定區 ------------
        LAYOUTS_DIR = ASSETS_DIR / "layouts"

        layouts_config = [
            {
                "id": "A",
                "label_key": "layout_a_label",
                "filename": "layout_a.png",
                "desc_key": "layout_a_desc",
                "uses_diver_info": False,
            },
            {
                "id": "B",
                "label_key": "layout_b_label",
                "filename": "layout_b.png",
                "desc_key": "layout_b_desc",
                "uses_diver_info": False,
            },
            {
                "id": "C",
                "label_key": "layout_c_label",
                "filename": "layout_c.png",
                "desc_key": "layout_c_desc",
                "uses_diver_info": False,
            },
            {
                "id": "D",
                "label_key": "layout_d_label",
                "filename": "layout_d.png",
                "desc_key": "layout_d_desc",
                "uses_diver_info": True,
            },
        ]

        layout_ids = [cfg["id"] for cfg in layouts_config]

        with col4:
            selected_id = st.selectbox(
                tr("layout_select_label"),
                options=layout_ids,
                format_func=lambda i: tr(f"layout_{i.lower()}_label"),
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
                [
                    (-pad, -pad),
                    (w + pad - 1, h + pad - 1),
                ],
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
                st.image(
                    img,
                    caption=tr(cfg["label_key"]),
                    use_container_width=True,
                )
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
        if default_label in nationality_options:
            default_index = nationality_options.index(default_label)
        else:
            default_index = 0

        col_info_1, col_info_2 = st.columns(2)

        with col_info_1:
            diver_name = st.text_input(tr("diver_name_label"), value="")

            nationality_label = st.selectbox(
                tr("nationality_label"),
                options=nationality_options,
                index=default_index,
            )

            if nationality_label == not_spec_label:
                nationality = ""
            else:
                nationality = nationality_label

        with col_info_2:
            discipline = st.selectbox(
                tr("discipline_label"),
                options=[not_spec_label, "CWT", "CWTB", "CNF", "FIM"],
            )

        # --- 6. 產生影片 ---
        if st.button(tr("render_button"), type="primary"):
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
                        output_resolution=(1080, 1920),  # 直式 9:16
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
                    st.error(tr("render_error", error=e))

    # ============================
    # Tab 2：未來的比較功能
    # ============================
    with tab_compare:
        st.info(tr("compare_coming_soon"))

    st.markdown('</div>', unsafe_allow_html=True)
