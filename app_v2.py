# app.py
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

st.title("🌊 Dive Data Overlay Generator (Beta)")

# --- 1. 上傳區 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ 上傳手錶數據")
    watch_file = st.file_uploader("Garmin .fit 或 ATMOS .uddf", type=["fit", "uddf"])

with col2:
    st.subheader("2️⃣ 上傳潛水影片")
    video_file = st.file_uploader("上傳原始潛水影片（mp4 格式）", type=["mp4"])

# --- 側邊欄：選項設定 ---
st.sidebar.header("⚙️ 參數設定")

# 補償時間（秒）
time_offset = st.sidebar.number_input(
    "手錶時間與影片時間的差值（秒，正數代表手錶時間比影片慢）",
    value=0.0,
    step=0.1,
    format="%.1f",
)

# 選擇解析來源
parser_type = st.sidebar.radio(
    "手錶資料類型",
    options=["Garmin", "ATMOS"],
    index=0,
)

# --- Layout 設定 ---
st.sidebar.header("🧩 Layout 選擇")

# ------------ 動態 Layout 設定區（未來要加 layout，改這裡就好） ------------
LAYOUTS_DIR = ASSETS_DIR / "layouts"

layouts_config = [
    {
        "id": "layout_a",
        "label": "Layout A：深度 + 心率 + 速率",
        "filename": "layout_a.png",
        "description": "直式 9:16，左上角深度，右上角心率，下方速率條。",
    },
    {
        "id": "layout_b",
        "label": "Layout B：包含姓名 / 國籍 / 潛水項目",
        "filename": "layout_b.png",
        "description": "直式 9:16，包含選手姓名、國籍小國旗、項目名稱。",
    },
    {
        "id": "layout_c",
        "label": "Layout C：單純深度",
        "filename": "layout_c.png",
        "description": "直式 9:16，只顯示深度資訊，適合乾淨畫面。",
    },
    {
        "id": "layout_d",
        "label": "Layout D：單純深度（變體）",
        "filename": "layout_d.png",
        "description": "另一種深度排版配置。",
    },
]

layout_labels = [cfg["label"] for cfg in layouts_config]

selected_label = st.sidebar.radio(
    "選擇影片版型",
    options=layout_labels,
    index=0,
)

# 找出目前被選取的 layout 設定
selected_layout = next(cfg for cfg in layouts_config if cfg["label"] == selected_label)
selected_id = selected_layout["id"]

# ------------ 輔助函式：載入圖片 & 為選到的那張加白框 ------------

def load_layout_image(cfg, is_selected: bool):
    img_path = LAYOUTS_DIR / cfg["filename"]
    img = Image.open(img_path).convert("RGBA")

    if not is_selected:
        return img

    border_color = "#FFD700"   # 黃色框線
    border_width = 12
    corner_radius = 15         # 可以自己調整圓角程度

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size

    # 讓外框和示意圖完全同尺寸：線條中心貼齊邊緣
    # 座標略超出畫布，超出部分會被裁掉，看起來就不會「縮進去」一圈
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

# ------------ 版型示意圖區：一次顯示全部，卡片式排版 ------------
st.markdown("### 版型示意圖（目前選擇會加白色外框）")

# 依 layout 數量動態建立欄位（目前是 3 個，就會是 3 欄）
cols = st.columns(len(layouts_config))

for col, cfg in zip(cols, layouts_config):
    with col:
        img = load_layout_image(cfg, cfg["id"] == selected_id)

        # 每張示意圖下面放文字說明
        st.image(img, use_column_width=True)
        st.caption(cfg["label"])
        st.write(
            f"<span style='font-size: 0.85em; color: #888;'>{cfg['description']}</span>",
            unsafe_allow_html=True,
        )

# --- 基本資訊輸入 ---
st.markdown("### 基本潛水資訊")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    diver_name = st.text_input("潛水員姓名", value="Derek")

with info_col2:
    nationality = st.text_input("國籍（國旗代碼，例如：TPE、JPN）", value="TPE")

with info_col3:
    discipline = st.text_input("比賽項目（例如：CWT、CNF、FIM）", value="CNF")

# --- 2. 解析手錶數據 ---
st.markdown("### 解析手錶潛水紀錄")

if watch_file is not None:
    suffix = Path(watch_file.name).suffix.lower()

    if parser_type == "Garmin" and suffix == ".fit":
        st.info("偵測到 Garmin .fit 檔，開始解析多潛資料")
        dives = parse_garmin_fit_to_dives(BytesIO(watch_file.read()))

        dive_options = [f"潛水 #{i+1}（開始時間：{d['start_time']}）" for i, d in enumerate(dives)]
        selected_index = st.selectbox("選擇要使用的潛水紀錄", range(len(dives)), format_func=lambda i: dive_options[i])
        selected_dive = dives[selected_index]
        dive_df = selected_dive["df"]
        df_rate = selected_dive.get("df_rate", None)

    elif parser_type == "ATMOS" and suffix == ".uddf":
        st.info("偵測到 ATMOS UDDF 檔，開始解析單一潛水紀錄")
        dive_df = parse_atmos_uddf(BytesIO(watch_file.read()))
        df_rate = None
    else:
        st.error("檔案格式與所選手錶類型不符，請重新確認。")
        dive_df = None
        df_rate = None
else:
    dive_df = None
    df_rate = None

# --- 如果有成功解析出 dive_df，顯示預覽與圖表 ---
if dive_df is not None and not dive_df.empty:
    st.success("手錶數據解析成功！以下是資料預覽：")
    st.dataframe(dive_df.head(50))

    # 假設 dive_df 至少有 time_s, depth_m 欄位
    if "time_s" in dive_df.columns and "depth_m" in dive_df.columns:
        st.markdown("### 深度曲線預覽")

        # 1. 把 time_s 當作 X 軸, depth_m 當作 Y 軸
        chart_data = dive_df[["time_s", "depth_m"]].copy()
        chart_data = chart_data.rename(columns={"time_s": "時間（秒）", "depth_m": "深度（m）"})

        # 2. 使用 Altair 畫出折線圖
        depth_chart = (
            alt.Chart(chart_data)
            .mark_line()
            .encode(
                x=alt.X("時間（秒）:Q", title="時間（秒）"),
                y=alt.Y("深度（m）:Q", title="深度（m）", scale=alt.Scale(zero=False)),
            )
            .properties(width=800, height=300)
        )

        st.altair_chart(depth_chart, use_container_width=True)

    # ================================
    # 計算 Dive Time（從 depth >= 0.7 m 開始，到回到 0 附近）
    # ================================
    dive_time_s = None
    dive_start_s = None
    dive_end_s = None

    df_sorted = dive_df.sort_values("time_s").reset_index(drop=True)

    # 1. 開始時間：第一個 depth >= 0.7 m
    start_candidates = df_sorted[df_sorted["depth_m"] >= 0.7]
    if not start_candidates.empty:
        dive_start_s = start_candidates["time_s"].iloc[0]

    # 2. 結束時間：在 dive_start_s 之後，最後一段「接近水面」的時間點
    if dive_start_s is not None:
        near_surface = df_sorted[
            (df_sorted["time_s"] >= dive_start_s) & (df_sorted["depth_m"] <= 0.7)
        ]
        if not near_surface.empty:
            dive_end_s = near_surface["time_s"].iloc[-1]

    if (dive_start_s is not None) and (dive_end_s is not None):
        dive_time_s = float(dive_end_s - dive_start_s)
    else:
        dive_time_s = None

    st.markdown("### Dive Time 計算結果")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.write(f"開始時間（depth ≥ 0.7 m）: {dive_start_s:.2f} s" if dive_start_s is not None else "開始時間無法判定")
    with col_d2:
        st.write(f"結束時間（接近水面）: {dive_end_s:.2f} s" if dive_end_s is not None else "結束時間無法判定")
    with col_d3:
        st.write(f"Dive Time: {dive_time_s:.2f} 秒" if dive_time_s is not None else "Dive Time 無法計算")

else:
    dive_time_s = None
    dive_start_s = None
    dive_end_s = None

# --- 6. 產生影片 ---
if st.button("🚀 產生疊加數據影片", type="primary"):
    if (dive_df is None) or (video_file is None):
        st.error("請先上傳手錶數據與影片檔。")
    else:
        # 建立進度條（取代單純的 spinner）
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0, text="產生影片中，請稍候...")

        # 1. 先把上傳的影片暫存到 /tmp，這段通常很快
        tmp_video_path = Path("/tmp") / video_file.name
        with open(tmp_video_path, "wb") as f:
            f.write(video_file.read())

        # 2. 在背景執行 render_video，前景負責更新進度條
        result = {"output_path": None, "error": None}

        def worker():
            try:
                result["output_path"] = render_video(
                    video_path=tmp_video_path,
                    dive_df=dive_df,
                    df_rate=df_rate,
                    time_offset=time_offset,
                    layout=selected_id,
                    assets_dir=ASSETS_DIR,
                    output_resolution=(1080, 1920),  # 直式 9:16
                    diver_name=diver_name,
                    nationality=nationality,
                    discipline=discipline,
                    dive_time_s=dive_time_s,      # 總 Dive time
                    dive_start_s=dive_start_s,    # ⭐ 計時起點
                    dive_end_s=dive_end_s,        # ⭐ 計時終點
                )
            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=worker)
        thread.start()

        percent = 0
        # 簡單的「假進度」：直到 render_video 完成前都慢慢前進，不會到 100%
        while thread.is_alive():
            percent = min(percent + 1, 99)
            progress_bar.progress(percent, text=f"產生影片中... {percent}%")
            time.sleep(0.2)

        thread.join()

        if result["error"] is not None:
            progress_placeholder.empty()
            st.error(f"產生影片時發生錯誤：{result['error']}")
        else:
            progress_bar.progress(100, text="影片產生完成！ 100%")
            st.success("影片產生完成！")
            output_path = result["output_path"]

            # 下載按鈕
            with open(output_path, "rb") as f:
                st.download_button(
                    "下載 1080p 影片",
                    data=f,
                    file_name="dive_overlay_1080p.mp4",
                    mime="video/mp4",
                )

            # 把影片放在較窄的欄位裡，視覺上就不會佔滿整個畫面
            col_preview, col_empty = st.columns([1, 1])  # 左右各 50%，你也可以改成 [1,2]

            with col_preview:
                st.video(str(output_path))
