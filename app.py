from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from pathlib import Path
from io import BytesIO
from typing import Optional, Set

import pandas as pd
import numpy as np
import altair as alt
import time
import os
import base64

from core.parser_garmin import parse_garmin_fit_to_dives, parse_garmin_fit_to_dives_with_hr
from core.parser_atmos import parse_atmos_uddf
from core.video_renderer import render_video

from core.i18n import init_lang, tr, LANG_OPTIONS, set_language
from core.tmp_store import persist_upload_to_tmp, cleanup_tmp_dir, reset_overlay_job
from core.nationality import load_nationality_options
from core.dive_curve import prepare_dive_curve, compute_dive_metrics
from ui.styles import inject_app_css
from ui.layout_selector import render_layout_selector
from typing import Dict, List, Optional, Union


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"


st.set_page_config(page_title="Dive Overlay Generator", layout="wide")

# Overlay job state (idle/rendering/done/error)
if "ov_job_state" not in st.session_state:
    st.session_state["ov_job_state"] = "idle"

# Best-effort cleanup of old /tmp files (keep current session files)
_keep = set()
for _k in ("ov_video_meta", "ov_output_path"):
    _v = st.session_state.get(_k)
    if isinstance(_v, dict) and "path" in _v:
        _keep.add(_v["path"])
    elif isinstance(_v, (str, Path)):
        _keep.add(str(_v))
cleanup_tmp_dir(max_age_sec=60*60*2, keep_paths=_keep)

# -------------------------------
# UI bootstrap: CSS + language + header
# -------------------------------
init_lang(default="zh")
inject_app_css()

# 頂部：左邊品牌、右邊語言選單（整排 sticky）
# -------------------------------
st.markdown("<div class='app-header-row'>", unsafe_allow_html=True)

top_left, top_right = st.columns([8, 1])

with top_left:
    st.markdown(
        f"""
        <div class="app-top-bar">
            <div class="app-top-icon"></div>
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

st.markdown("</div>", unsafe_allow_html=True)



# 頂部：左邊品牌、右邊語言選單（整排 sticky）
# -------------------------------
st.markdown("<div class='app-header-row'>", unsafe_allow_html=True)

top_left, top_right = st.columns([8, 1])

with top_right:
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
# 讀取國籍 / 國碼清單
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
    
        # --- 0. 若上一輪已完成/失敗，而使用者開始新上傳：先做「下一輪前清理」的兩段式流程 ---
        # 第一步：本次偵測到新檔案 → 設 flag → rerun
        # 第二步：rerun 一開始看到 flag → 執行 reset_overlay_job()（避免動到已建立的 widget）
        if st.session_state.get("ov_reset_on_next_run") is True:
            st.session_state["ov_reset_on_next_run"] = False
            reset_overlay_job()
            cleanup_tmp_dir(max_age_sec=60 * 60 * 2)  # best-effort sweep
            # 重要：reset 完立刻 rerun，確保狀態乾淨
            st.rerun()

        # --- 1. 上傳區 ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(tr("upload_watch_subheader"))
            watch_file = st.file_uploader(
                tr("upload_watch_label"),
                type=None,
                key="overlay_watch_file",
                disabled=(st.session_state.get("ov_job_state") == "rendering"),
            )
        
        with col2:
            st.subheader(tr("upload_video_subheader"))
            video_file = st.file_uploader(
                tr("upload_video_label"),
                type=["mp4", "mov", "m4v"],
                key="overlay_video_file",
                disabled=(st.session_state.get("ov_job_state") == "rendering"),
            )
        
        # --- 2. 新工作開始前的清理（安全觸發）：只在上一輪 done/error 時啟動 ---
        # 注意：這裡不要直接 reset（因為 widget 已建立），改用 flag + rerun
        if (watch_file is not None or video_file is not None) and st.session_state.get("ov_job_state") in ("done", "error"):
            st.session_state["ov_reset_on_next_run"] = True
            st.rerun()
        
        # --- 3. 寫入 /tmp（各自獨立處理）---
        # watch（通常很小，但仍用串流落地，並避免 rerun 重複寫）
        if watch_file is not None:
            wmeta = st.session_state.get("ov_watch_meta")
            if (wmeta is None) or (wmeta.get("name") != watch_file.name) or (not wmeta.get("path")) or (not os.path.exists(wmeta.get("path", ""))):
                st.session_state["ov_watch_meta"] = persist_upload_to_tmp(watch_file)
        
        # video（大檔必須用串流落地）
        if video_file is not None:
            vmeta = st.session_state.get("ov_video_meta")
            if (vmeta is None) or (vmeta.get("name") != video_file.name) or (not vmeta.get("path")) or (not os.path.exists(vmeta.get("path", ""))):
                st.session_state["ov_video_meta"] = persist_upload_to_tmp(video_file)
        
        # --- 4. 標記 job 進入 idle（可開始下一步） ---
        if (
            st.session_state.get("ov_job_state") != "rendering"
            and st.session_state.get("ov_watch_meta") is not None
            and st.session_state.get("ov_video_meta") is not None
        ):
            st.session_state["ov_job_state"] = "idle"

        # --- 5. 選手錶類型 & 解析 ---
        dive_df = None
        hr_df = None
        df_rate = None          # 速率重採樣後的 df
        dive_time_s = None      # Dive time（秒）
        dive_start_s = None     # 計時起點（深度 ≥ 0.7 m 的時間）
        dive_end_s = None       # 計時終點（回到 0 m 的時間）
        selected_dive_index = None

        watch_meta = st.session_state.get("ov_watch_meta")
        watch_meta_ok = bool(watch_meta and watch_meta.get("path") and os.path.exists(watch_meta.get("path", "")))

        if watch_file is not None or watch_meta_ok:
            watch_name = watch_file.name if watch_file is not None else (watch_meta.get("name", "") if watch_meta else "")
            suffix = Path(watch_name).suffix.lower()

            # ✅ 一律從已落地的 /tmp 檔讀，避免 UploadedFile 被讀到 EOF
            watch_path = None
            if watch_meta and watch_meta.get("path") and os.path.exists(watch_meta["path"]):
                watch_path = watch_meta["path"]
        
            if watch_path is None:
                st.error(tr("error_watch_file_missing"))
            else:
                if suffix == ".fit":
                    st.info(tr("fit_detected"))
                    with open(watch_path, "rb") as f:
                        file_like = BytesIO(f.read())
                        result = parse_garmin_fit_to_dives_with_hr(file_like)
                        dives = result["dives"]
                        hr_list = result.get("heart_rate", [])

                    if len(dives) == 0:
                        st.error(tr("fit_no_dives"))
                    else:
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
                        dive_df = dives[int(selected_dive_index)]
                        hr_df = None
                        if isinstance(hr_list, list) and 0 <= int(selected_dive_index) < len(hr_list):
                            hr_df = hr_list[int(selected_dive_index)]
        
                elif suffix == ".uddf":
                    st.info(tr("uddf_detected"))
                    with open(watch_path, "rb") as f:
                        dive_df = parse_atmos_uddf(BytesIO(f.read()))

        # --- 6. 顯示時間–深度曲線供確認 ---
        if dive_df is not None:
            if len(dive_df) == 0:
                st.warning(tr("no_depth_samples"))
            else:
                # 先確保按時間排序
                dive_df = dive_df.sort_values("time_s").reset_index(drop=True)

                # --- 強制加入起始/結束的 0 m 點 ---
                if len(dive_df) > 0 and "time_s" in dive_df.columns and "depth_m" in dive_df.columns:
                    dive_df["time_s"] = dive_df["time_s"] + 1.0
                    if hr_df is not None and len(hr_df) > 0 and "time_s" in hr_df.columns:
                         hr_df = hr_df.sort_values("time_s").reset_index(drop=True)
                         hr_df["time_s"] = hr_df["time_s"] + 1.0

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

                # 重採樣 + 速率（可調平滑度；預設 2 秒）
                if "ov_smooth_level" not in st.session_state:
                    st.session_state["ov_smooth_level"] = 1
                smooth_level = int(st.session_state["ov_smooth_level"])
                df_rate = prepare_dive_curve(dive_df, smooth_window=smooth_level)

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

                # ====== 6-1. 圖表（上下排列，不再用 columns） ======
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
                    # 速率座標軸上限（自動浮動，0.5 m/s 間距進位）
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
                                scale=alt.Scale(domain=[0, max_rate_domain], nice=False),
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

                    # 速率平滑度（放在速率圖下方，可選 1~3 秒）
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
                            key="ov_smooth_level",
                            label_visibility="collapsed",
                        )

                    # 原始資料說明
                    st.caption(
                        tr(
                            "preview_caption",
                            n_points=len(dive_df),
                            t_min=df_rate["time_s"].min(),
                            t_max=df_rate["time_s"].max(),
                            max_depth=df_rate["depth_m"].max(),
                        )
                    )

                    # ====== 6-2. 潛水速率分析（含 Dive Time） ======
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


                # --- 7. 影片對齊與版型 ---
                st.subheader(tr("align_layout_subheader"))

                # ==========================================================
                # 7-1) 對齊方式（下潛 / 最深 / 出水）
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
                # 7-2) 影片時間輸入（[-] [time] [+] + 級距選擇）
                #     - 注意：避免直接修改 text_input 的 session_state key
                # ==========================================================
                def _parse_time_str_to_seconds_safe(s: str):
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

                def _seconds_to_mmss_cc(sec: float) -> str:
                    sec = max(0.0, float(sec))
                    mm = int(sec // 60)
                    ss = sec - mm * 60
                    return f"{mm:02d}:{ss:05.2f}"  # => 00:00.00

                def _clamp_time(sec: float, max_sec: float = 3600.0) -> float:
                    return max(0.0, min(float(sec), float(max_sec)))

                # --- 初始化 state（只寫入「非 widget key」或在 widget 建立前寫入）---
                if "overlay_align_video_time_s" not in st.session_state:
                    st.session_state["overlay_align_video_time_s"] = 0.0
                if "overlay_align_step_unit" not in st.session_state:
                    st.session_state["overlay_align_step_unit"] = "sec"
                if "overlay_align_video_time_str" not in st.session_state:
                    st.session_state["overlay_align_video_time_str"] = "00:00.00"

                step_map = {"min": 60.0, "sec": 1.0, "csec": 0.1}

                def _sync_str_from_seconds():
                    # 這個函式會被按鈕 callback 呼叫；按鈕在 text_input 之前建立，避免 StreamlitAPIException
                    st.session_state["overlay_align_video_time_str"] = _seconds_to_mmss_cc(
                        st.session_state["overlay_align_video_time_s"]
                    )

                def _on_minus():
                    step = step_map.get(st.session_state.get("overlay_align_step_unit", "sec"), 1.0)
                    st.session_state["overlay_align_video_time_s"] = round(
                        _clamp_time(st.session_state["overlay_align_video_time_s"] - step), 2
                    )
                    _sync_str_from_seconds()

                def _on_plus():
                    step = step_map.get(st.session_state.get("overlay_align_step_unit", "sec"), 1.0)
                    st.session_state["overlay_align_video_time_s"] = round(
                        _clamp_time(st.session_state["overlay_align_video_time_s"] + step), 2
                    )
                    _sync_str_from_seconds()

                # label
                st.markdown(f"**{tr('align_video_time_label')}**")

                # ✅ 手機端不要被壓成 50%：這裡用容器 class 強制全寬
                st.markdown("<div class='align-time-block'>", unsafe_allow_html=True)

                # 級距選擇（放在上方，避免手機被擠到最右側）
                st.radio(
                    label="",
                    options=["min", "sec", "csec"],
                    horizontal=True,
                    format_func=lambda k: {"min": tr("align_step_min"), "sec": tr("align_step_sec"), "csec": tr("align_step_csec")}[k],
                    key="overlay_align_step_unit",
                    label_visibility="collapsed",
                )

                # [-] [time] [+]
                bcol1, bcol2, bcol3 = st.columns([1.0, 2.4, 1.0], vertical_alignment="center")

                with bcol1:
                    st.button("－", key="overlay_align_minus", on_click=_on_minus, use_container_width=True)

                with bcol3:
                    st.button("＋", key="overlay_align_plus", on_click=_on_plus, use_container_width=True)

                with bcol2:
                    video_time_str = st.text_input(
                        label="",
                        key="overlay_align_video_time_str",
                        label_visibility="collapsed",
                        help=tr("align_video_time_help"),
                    )
                    v_ref_from_text = _parse_time_str_to_seconds_safe(video_time_str)
                    if v_ref_from_text is None:
                        st.warning(tr("align_video_time_invalid"))
                    else:
                        st.session_state["overlay_align_video_time_s"] = float(v_ref_from_text)

                st.markdown("</div>", unsafe_allow_html=True)

                # 最終 v_ref（秒）
                v_ref = float(st.session_state["overlay_align_video_time_s"])

                # ==========================================================
                # 7-3) 對齊參考時間（手錶端：下潛 / 最深 / 出水）
                # ==========================================================
                t_ref_raw = None
                if df_rate is not None and dive_df is not None:
                    if align_mode == "start":
                        t_ref_raw = dive_start_s
                    elif align_mode == "end":
                        t_ref_raw = dive_end_s
                    elif align_mode == "bottom":
                        raw = dive_df.sort_values("time_s").reset_index(drop=True)
                        within = raw[(raw["time_s"] >= dive_start_s) & (raw["time_s"] <= dive_end_s)]
                        if not within.empty:
                            idx_bottom = within["depth_m"].idxmax()
                            t_ref_raw = float(within.loc[idx_bottom, "time_s"])

                # ==========================================================
                # 7-4) time_offset：自動計算並直接套用到 render（不用再按「套用」）
                #     time_offset = t_ref_for_align - v_ref
                # ==========================================================
                if t_ref_raw is not None:
                    t_ref_for_align = float(t_ref_raw)
                    time_offset = t_ref_for_align - v_ref - 1.0
                    st.caption(f"目前計算出的偏移：{time_offset:+.2f} 秒（會套用到渲染）")
                else:
                    time_offset = 0.0
                    st.caption("尚未偵測到潛水事件，暫時使用 0 秒偏移。")

                # ==========================================================
                # 7-5) 動態 Layout 設定區（2 欄 Grid）
                # ==========================================================
                st.subheader(tr("layout_select_label"))
                
                LAYOUTS_DIR = ASSETS_DIR / "layouts"
                
                layouts_config = [
                    {"id": "A", "label_key": "layout_a_label", "filename": "layout_a.png"},
                    {"id": "B", "label_key": "layout_b_label", "filename": "layout_b.png"},
                    {"id": "C", "label_key": "layout_c_label", "filename": "layout_c.png"},
                    {"id": "D", "label_key": "layout_d_label", "filename": "layout_d.png"},
                ]
                
                # 初始化選擇狀態
                if "overlay_layout_id" not in st.session_state:
                    st.session_state["overlay_layout_id"] = layouts_config[0]["id"]
                
                selected_id = st.session_state["overlay_layout_id"]
                
                def _img_to_base64_png(path: Path) -> str:
                    if not path.exists():
                        return ""
                    return base64.b64encode(path.read_bytes()).decode("utf-8")
                
                cols = st.columns(2, gap="small")
                
                for idx, cfg in enumerate(layouts_config):
                    col = cols[idx % 2]
                    layout_id = cfg["id"]
                    label = tr(cfg["label_key"])
                
                    img_path = LAYOUTS_DIR / cfg["filename"]
                    img_b64 = _img_to_base64_png(img_path)
                
                    is_selected = (layout_id == selected_id)
                    card_class = "layout-card selected" if is_selected else "layout-card dimmed"
                
                    # ✅ 這裡用「字串拼接」避免三引號縮排變 code block
                    check_html = '<span class="layout-check">✓</span>' if is_selected else ""
                
                    card_html = (
                        f'<div class="{card_class}">'
                        f'  <img src="data:image/png;base64,{img_b64}" />'
                        f'  <div class="layout-footer">'
                        f'    {check_html}'
                        f'    <span class="layout-title">{label}</span>'
                        f'  </div>'
                        f'</div>'
                    )
                
                    with col:
                        st.markdown(card_html, unsafe_allow_html=True)
                
                        if st.button(
                            tr("select_layout_btn"),              # 你已經修好 select_label 的翻譯就用這個
                            key=f"select_layout_btn_{layout_id}",
                            use_container_width=True,
                        ):
                            st.session_state["overlay_layout_id"] = layout_id
                            st.rerun()

        # Defaults for diver info (only shown for Layout A/B, but must exist for all layouts)
        diver_name = st.session_state.get("overlay_diver_name", "")
        nationality = ""
        not_spec_label = tr("not_specified")
        discipline = st.session_state.get("overlay_discipline", not_spec_label)
        
        # Current selected layout
        selected_id = st.session_state.get("overlay_layout_id", "C")

# --- 8. 輸入潛水員資訊---
        if selected_id in ("A", "B"):
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
# --- 9. 產生影片 ---
        if st.button(tr("render_button"), type="primary", key="overlay_render_btn"):
            video_meta = st.session_state.get("ov_video_meta")
            video_path_ok = bool(video_meta and video_meta.get("path") and os.path.exists(video_meta.get("path", "")))
            watch_meta = st.session_state.get("ov_watch_meta")
            watch_path_ok = bool(watch_meta and watch_meta.get("path") and os.path.exists(watch_meta.get("path", "")))
            if (dive_df is None) or (not video_path_ok) or (not watch_path_ok):
                st.error(tr("error_need_both_files"))
            else:
                progress_bar = st.progress(0, text=tr("progress_init"))
                eta_box = st.empty()

                render_t0 = time.time()

                def progress_callback(p: float, message: str = ""):
                    p = max(0.0, min(1.0, float(p)))
                    percent = int(p * 100)

                    # Progress bar text (keep existing behavior)
                    if message:
                        text = f"{message} {percent}%"
                    else:
                        text = f"{tr('progress_rendering')} {percent}%"
                    progress_bar.progress(percent, text=text)

                    # ETA info box under the progress bar
                    if percent <= 26:                                  # original = 40
                        eta_box.info(tr("progress_eta_estimating"))
                        return

                    if 27 <= percent <= 99:                            # original = 41 - 99
                        elapsed = max(0.1, time.time() - render_t0)
                        if p > 0.0001:
                            total_est = elapsed / p
                            remaining = max(0.0, total_est - elapsed)
                        else:
                            remaining = 0.0

                        mm = int(remaining // 60)
                        ss = int(remaining % 60)
                        eta_box.info(tr("progress_eta", mm=mm, ss=ss))
                        return

                    # 100%: close the info box
                    if percent >= 100:
                        eta_box.empty()


                
                # Persist upload to /tmp without reading into RAM (important for 300-400MB files)
                # Reuse persisted path across reruns for this session
                video_meta = st.session_state.get("ov_video_meta")
                if (video_meta is None) or (not video_meta.get("path")) or (not os.path.exists(video_meta.get("path", ""))):
                    st.error(tr("error_video_missing"))
                    st.stop()

                tmp_video_path = Path(video_meta["path"])
                try:
                    st.session_state["ov_job_state"] = "rendering"
                    st.session_state.pop("ov_render_error", None)
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
                        hr_df=hr_df,
)

                    progress_callback(1.0, tr("progress_done"))
                    st.success(tr("render_success"))

                    st.session_state["ov_output_path"] = str(output_path)
                    st.session_state["ov_job_state"] = "done"

                    with open(output_path, "rb") as f:
                        st.download_button(
                            tr("download_button"),
                            data=f,
                            file_name="dive_overlay_1080p.mp4",
                            mime="video/mp4",
                        )

                    # Preview policy: skip inline preview for large outputs to reduce memory spikes on cloud
                    try:
                        out_size_mb = os.path.getsize(output_path) / 1024 / 1024
                    except Exception:
                        out_size_mb = 9999.0

                    if out_size_mb <= 120:
                        col_preview, col_empty = st.columns([1, 1])
                        with col_preview:
                            st.video(str(output_path))
                    else:
                        st.info(tr("preview_skipped_large_file"))

                    # ---- Post-render UX: lock UI and offer clean restart ----
                    st.info(tr("post_render_tip"))

                    if st.button(tr("start_new_job_btn"), type="primary", key="ov_start_new_job"):
                        reset_overlay_job()
                        cleanup_tmp_dir(max_age_sec=60*60*2)  # best-effort sweep
                        st.rerun()

                except Exception as e:
                    st.session_state["ov_job_state"] = "error"
                    st.session_state["ov_render_error"] = str(e)
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
                    type=None,
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
            st.session_state["cmp_smooth_level"] = 1  # 預設 2 秒
    
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
                    key="cmp_smooth_level",
                    label_visibility="collapsed",
                )

    st.markdown('</div>', unsafe_allow_html=True)
