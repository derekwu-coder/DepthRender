"""
Language / translation utilities for the Streamlit app.

This module is intentionally UI-agnostic (no page layout here); it only manages
language state and translation lookup.
"""
import streamlit as st

def init_lang(default: str = "zh") -> None:
    if "lang" not in st.session_state:
        st.session_state["lang"] = default

LANG_OPTIONS = {
    "zh": "中文",
    "en": "English",
}

TRANSLATIONS = {
    "zh": {
        "app_title": "Dive Overlay Generator",
        "top_brand": "DepthRender",
        "language_label": "🌐 語言",

        "tab_overlay_title": "疊加影片產生器",
        "tab_compare_title": "潛水數據比較",
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
        "align_mode_label": "對齊方式",
        "align_mode_start": "對齊下潛時間 (開始躬身)",
        "align_mode_bottom": "對齊最深時間 (轉身/摘到tag)",
        "align_mode_end": "對齊出水時間 (手錶出水)",

        "align_video_time_label": "影片時間（mm:ss.ss，例如 01:10.05）",
        "align_video_time_help": "請輸入分鐘:秒.小數，秒與小數最多 2 位，例如 00:03.18",
        "align_video_time_invalid": "影片時間格式不正確，請使用 mm:ss 或 mm:ss.ss，例如 00:03.18",

        "align_step_min": "分 (1 min)",
        "align_step_sec": "秒 (1 s)",
        "align_step_csec": "0.1 秒 (100 ms)",
        "layout_select_label": "選擇影片版型",
        "layout_preview_title": "版型示意圖",

        "layout_a_label": "賽事風格 1",
        "layout_a_desc": "",
        "layout_b_label": "賽事風格 2",
        "layout_b_desc": "",
        "layout_c_label": "單純數據",
        "layout_c_desc": "Simple_A",
        "layout_d_label": "開發中 請勿使用",
        "layout_d_desc": "Simple_B",

        "diver_info_subheader": "5️⃣ 潛水員資訊（選填，主要給賽事風格使用）",
        "diver_name_label": "姓名（暫不支援中文）",
        "nationality_label": "國籍",
        "discipline_label": "潛水項目（Discipline）",
        "not_specified": "（不指定）",

        "render_button": "🚀 產生疊加數據影片",
        "error_need_both_files": "請先上傳手錶數據與影片檔。",
        "error_watch_file_missing": "找不到手錶檔案，請重新上傳。",
        "error_video_missing": "找不到影片檔案，請重新上傳。",
        "error_video_ext_not_supported": "不支援的影片格式，請上傳 mp4 / mov / m4v。",
        "progress_init": "初始化中...",
        "progress_rendering": "產生影片中...",
        "progress_done": "影片產生完成！",
        "progress_eta_estimating": "剩餘時間預估中⋯⋯請勿離開此畫面或關閉螢幕",
        "progress_eta": "預估剩餘時間：約 {mm:02d}:{ss:02d} ⋯⋯請勿離開此畫面或關閉螢幕",
        "render_success": "影片產生完成！",
        "download_button": "下載 1080p 影片",
        "render_error": "產生影片時發生錯誤：{error}",

        "nationality_file_not_found": "找不到 Nationality 檔案：{path}",
        "nationality_read_error": "讀取 Nationality.csv 時發生錯誤：{error}",
        "nationality_missing_columns": "Nationality.csv 缺少必要欄位：{missing}",
        
        "preview_skipped_large_file": "檔案較大，為降低雲端記憶體峰值，已略過預覽（請直接下載）。",
        "post_render_tip": "請先下載影片；如要開始下一支，請點「開始新任務」。",
        "start_new_job_btn": "開始新任務",

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
        "compare_ff_rate_label": "Free Fall 速率 (m/s)",
        "compare_metric_unit_mps": "{value:.2f} m/s",
        "compare_metric_not_available": "—",

        # Overlay 速率分析 + 潛水時間顯示
        "overlay_speed_analysis_title": "潛水速率分析",
        "overlay_ff_depth_label": "FF 開始深度 (m)",
        "metric_dive_time_label": "潛水時間",
        "metric_dive_time_value": "{mm:02d}:{ss:02d}",

        
        # Overlay rate analysis (單一潛水速率分析)
        "overlay_rate_section_title": "潛水速率分析",
        "overlay_ff_depth_label": "FF 開始深度 (m)",
        "overlay_desc_rate_label": "下潛速率 (m/s)",
        "overlay_asc_rate_label": "上升速率 (m/s)",
        "overlay_ff_rate_label": "Free Fall 速率 (m/s)",
        "overlay_metric_unit_mps": "{value:.2f} m/s",
        "overlay_metric_not_available": "—",
        "layout_a_tuning_title": "Layout A 版面參數（底部平行四邊形）",
        "layout_a_show_tuning": "顯示進階調整（賽事風格 1）",
        "layout_a_alpha": "背景板透明度",
        "layout_a_tuning_hint": "此區只影響 Layout A。若你不調整，會使用預設值。",
        "layout_a_x_start": "起始 X（左邊界）",
        "layout_a_y_from_bottom": "距離底部 Y",
        "layout_a_height": "高度 H",
        "layout_a_skew": "斜度（skew）",
        "layout_a_gap": "板塊間距（gap）",
        "layout_a_w1": "板 1 寬度（國籍/國旗）",
        "layout_a_w2": "板 2 寬度（姓名）",
        "layout_a_w3": "板 3 寬度（項目）",
        "layout_a_w4": "板 4 寬度（時間）",
        "layout_a_w5": "板 5 寬度（深度）",
        "layout_a_text_title": "文字大小",
        "layout_a_fs_code": "國籍三碼 字體大小",
        "layout_a_fs_name": "姓名 字體大小",
        "layout_a_fs_disc": "項目 字體大小",
        "layout_a_fs_time": "時間 字體大小",
        "layout_a_fs_depth": "深度 字體大小",
        "layout_a_inner_pad": "內距 padding",
        "layout_a_offsets_title": "微調偏移（X/Y）",
        "layout_a_off_code_x": "國籍 X",
        "layout_a_off_code_y": "國籍 Y",
        "layout_a_off_flag_x": "國旗 X",
        "layout_a_off_flag_y": "國旗 Y",
        "layout_a_off_name_x": "姓名 X",
        "layout_a_off_name_y": "姓名 Y",
        "layout_a_off_disc_x": "項目 X",
        "layout_a_off_disc_y": "項目 Y",
        "layout_a_off_time_x": "時間 X",
        "layout_a_off_time_y": "時間 Y",
        "layout_a_off_depth_x": "深度 X",
        "layout_a_off_depth_y": "深度 Y",
        "select_layout_btn": "select"
        
},
    "en": {
        "app_title": "Dive Overlay Generator",
        "top_brand": "DepthRender",
        "language_label": "🌐 Language",

        "tab_overlay_title": "Overlay Generator",
        "tab_compare_title": "Dive Comparison",
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
        "align_mode_label": "Alignment mode",
        "align_mode_start": "Align descent time (start of duck dive)",
        "align_mode_bottom": "Align bottom time (turn / tag grab)",
        "align_mode_end": "Align surfacing time (watch exits water)",

        "align_video_time_label": "Video time (mm:ss.ss, e.g. 01:10.05)",
        "align_video_time_help": "Use mm:ss or mm:ss.ss, up to 2 decimals (e.g. 00:03.18)",
        "align_video_time_invalid": "Invalid video time format. Use mm:ss or mm:ss.ss (e.g. 00:03.18)",

        "align_step_min": "Min (1 min)",
        "align_step_sec": "Sec (1 s)",
        "align_step_csec": "0.1 s (100 ms)",
        "layout_select_label": "Choose overlay layout",
        "layout_preview_title": "Layout preview",

        "layout_a_label": "Competition Style 1",
        "layout_a_desc": "",
        "layout_b_label": "Competition Style 2",
        "layout_b_desc": "",
        "layout_c_label": "Dive indicator",
        "layout_c_desc": "Simple_A",
        "layout_d_label": "Under Development",
        "layout_d_desc": "Simple_B",

        "diver_info_subheader": "5️⃣ Diver info (optional, mainly for Competition Style)",
        "diver_name_label": "Diver name / Nickname",
        "nationality_label": "Nationality",
        "discipline_label": "Discipline",
        "not_specified": "(Not specified)",

        "render_button": "🚀 Generate overlay video",
        "error_need_both_files": "Please upload both dive log and video file.",
        "error_watch_file_missing": "Watch file is missing. Please re-upload.",
        "error_video_missing": "Video file is missing. Please re-upload.",
        "error_video_ext_not_supported": "Unsupported video format. Please upload mp4 / mov / m4v.",
        "progress_init": "Initializing...",
        "progress_rendering": "Rendering video...",
        "progress_done": "Rendering finished!",
        "progress_eta_estimating": "Estimating remaining time... Please stay on this page and keep the screen on.",
        "progress_eta": "Estimated remaining time: ~{mm:02d}:{ss:02d} ... Please stay on this page and keep the screen on.",

        "render_success": "Video rendered successfully!",
        "download_button": "Download 1080p video",
        "render_error": "Error while rendering video: {error}",

        "nationality_file_not_found": "Nationality file not found: {path}",
        "nationality_read_error": "Error reading Nationality.csv: {error}",
        "nationality_missing_columns": "Nationality.csv is missing required columns: {missing}",
        
        "preview_skipped_large_file": "Large output file detected. Preview is skipped to reduce memory spikes. Please download the video.",
        "post_render_tip": "Please download the video first. To start the next job, click “Start new job”.",
        "start_new_job_btn": "Start new job",

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
        "compare_ff_rate_label": "Free-fall Descent Rate (m/s)",
        "compare_metric_unit_mps": "{value:.2f} m/s",
        "compare_metric_not_available": "—",
        
        # Overlay speed analysis + dive time display
        "overlay_speed_analysis_title": "Dive speed analysis",
        "overlay_ff_depth_label": "FF start depth (m)",
        "metric_dive_time_label": "Dive time",
        "metric_dive_time_value": "{mm:02d}:{ss:02d}",

        # Overlay rate analysis (single-dive metrics)
        "overlay_rate_section_title": "Dive speed metrics",
        "overlay_ff_depth_label": "FF start depth (m)",
        "overlay_desc_rate_label": "Descent speed (m/s)",
        "overlay_asc_rate_label": "Ascent speed (m/s)",
        "overlay_ff_rate_label": "Free-fall speed (m/s)",
        "overlay_metric_unit_mps": "{value:.2f} m/s",
        "overlay_metric_not_available": "—",
        "layout_a_tuning_title": "Layout A tuning (Bottom parallelograms)",
        "layout_a_show_tuning": "Show advanced tuning (Race style 1)",
        "layout_a_alpha": "Background opacity",
        "layout_a_tuning_hint": "These controls only affect Layout A. Leave as-is to use defaults.",
        "layout_a_x_start": "X start (left)",
        "layout_a_y_from_bottom": "Y from bottom",
        "layout_a_height": "Height H",
        "layout_a_skew": "Skew",
        "layout_a_gap": "Gap between plates",
        "layout_a_w1": "Plate 1 width (Code/Flag)",
        "layout_a_w2": "Plate 2 width (Name)",
        "layout_a_w3": "Plate 3 width (Discipline)",
        "layout_a_w4": "Plate 4 width (Time)",
        "layout_a_w5": "Plate 5 width (Depth)",
        "layout_a_text_title": "Font sizes",
        "layout_a_fs_code": "Code font size",
        "layout_a_fs_name": "Name font size",
        "layout_a_fs_disc": "Discipline font size",
        "layout_a_fs_time": "Time font size",
        "layout_a_fs_depth": "Depth font size",
        "layout_a_inner_pad": "Inner padding",
        "layout_a_offsets_title": "Fine offsets (X/Y)",
        "layout_a_off_code_x": "Code X",
        "layout_a_off_code_y": "Code Y",
        "layout_a_off_flag_x": "Flag X",
        "layout_a_off_flag_y": "Flag Y",
        "layout_a_off_name_x": "Name X",
        "layout_a_off_name_y": "Name Y",
        "layout_a_off_disc_x": "Discipline X",
        "layout_a_off_disc_y": "Discipline Y",
        "layout_a_off_time_x": "Time X",
        "layout_a_off_time_y": "Time Y",
        "layout_a_off_depth_x": "Depth X",
        "layout_a_off_depth_y": "Depth Y",
        "select_layout_btn": "select"

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
