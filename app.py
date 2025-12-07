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
        st.error(f"找不到 Nationality 檔案：{csv_path}")
        return pd.DataFrame(columns=["Country", "Code", "label"])

    # --- 讀取 CSV ---
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"讀取 Nationality.csv 時發生錯誤：{e}")
        return pd.DataFrame(columns=["Country", "Code", "label"])

    # --- 檢查必要欄位 ---
    required_cols = {"Country", "Code"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Nationality.csv 缺少必要欄位：{missing}")
        return pd.DataFrame(columns=["Country", "Code", "label"])

    # --- 整理資料 ---
    df = df.dropna(subset=["Country", "Code"]).copy()
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Code"] = df["Code"].astype(str).str.upper().str.strip()

    # --- 下拉選單顯示字串 ---
    df["label"] = df["Country"] + " (" + df["Code"] + ")"

    return df

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
    video_file = st.file_uploader("影片檔 (任意解析度)", type=["mp4", "mov", "m4v"])

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
        st.info("偵測到 Garmin .fit 檔，開始解析多潛資料...")
        dives = parse_garmin_fit_to_dives(BytesIO(watch_file.read()))
        # dives: List[pd.DataFrame]

        if len(dives) == 0:
            st.error("這個 .fit 裡面沒有偵測到有效的潛水紀錄。")
        else:
            # 用最大深度當顯示文字
            options = [
                f"Dive #{i+1}（{df['depth_m'].max():.1f} m）"
                for i, df in enumerate(dives)
            ]

            selected_dive_index = st.selectbox(
                "選擇要使用的那一潛：",
                options=list(range(len(dives))),
                format_func=lambda i: options[i],
            )

            # ✅ 一定要放在 else 裡面（確認 dives 不為空，且有選項）
            dive_df = dives[selected_dive_index]

    elif suffix == ".uddf":
        st.info("偵測到 ATMOS UDDF 檔，開始解析單一潛水紀錄...")
        dive_df = parse_atmos_uddf(BytesIO(watch_file.read()))

# --- 3. 顯示時間–深度曲線供確認 ---
if dive_df is not None:
    if len(dive_df) == 0:
        st.warning("成功讀取手錶檔，但沒有找到任何深度樣本點。")
    else:
        # 先確保按時間排序
        dive_df = dive_df.sort_values("time_s").reset_index(drop=True)
        
        # --- 強制加入起始/結束的 0 m 點 ---
        if len(dive_df) > 0 and "time_s" in dive_df.columns and "depth_m" in dive_df.columns:
            # 先記錄原始最小 / 最大時間
            t_first = float(dive_df["time_s"].iloc[0])
            t_last  = float(dive_df["time_s"].iloc[-1])

            # 1) 先把原本所有 time_s 整體平移 +1 秒
            dive_df["time_s"] = dive_df["time_s"] + 1.0

            # 2) 建立新的開頭點：time=0, depth=0
            first_row = dive_df.iloc[0].copy()
            first_row["time_s"] = 0.0
            first_row["depth_m"] = 0.0

            # 3) 建立新的結尾點：最後時間 + 1 秒, depth=0
            last_row = dive_df.iloc[-1].copy()
            last_row["time_s"] = float(dive_df["time_s"].max()) + 1.0
            last_row["depth_m"] = 0.0

            # 4) 把首尾點加回去，重新排序一次
            dive_df = pd.concat(
                [first_row.to_frame().T, dive_df, last_row.to_frame().T],
                ignore_index=True,
            )
            dive_df = dive_df.sort_values("time_s").reset_index(drop=True)


        # ================================
        # 方案 A：重採樣成「每秒一筆」再計算速率
        # ================================

        # 1. 建立每秒一個時間點（取整數秒）
        t_min = int(np.floor(dive_df["time_s"].min()))
        t_raw_max = dive_df["time_s"].max()          # 真實最大時間（可能 82.x）
        t_resample_max = int(np.ceil(t_raw_max))     # 重採樣做到的最後一秒（例如 83）

        # 用實際重採樣範圍做 uniform_time（這是「真的資料範圍」）
        uniform_time = np.arange(t_min, t_resample_max + 1, 1)  # 1 秒一個點

        # 2. 用線性插值補出每秒深度
        depth_interp = np.interp(
            uniform_time,
            dive_df["time_s"].to_numpy(),
            dive_df["depth_m"].to_numpy()
        )

        # 3. 計算速率：下一秒深度 - 當前秒深度（m/s）
        rate_uniform = np.diff(depth_interp, prepend=depth_interp[0])

        # 4. 取絕對值 + 限制在 0～3 m/s
        rate_abs = np.abs(rate_uniform)
        rate_abs_clipped = np.clip(rate_abs, 0.0, 3.0)

        # 5. 建立畫圖用 DataFrame（重採樣後，真實範圍）
        df_rate = pd.DataFrame({
            "time_s": uniform_time,
            "depth_m": depth_interp,
            "rate_abs_mps": rate_abs_clipped,
        })

        # 6. 平滑處理：滑動平均（例如 3 秒窗）
        window_sec = 3
        df_rate["rate_abs_mps_smooth"] = (
            df_rate["rate_abs_mps"]
            .rolling(window=window_sec, center=True, min_periods=1)
            .mean()
        )

        # 7. 圖表用 X 軸顯示上限：以 5 秒為單位進位（只影響顯示）
        max_display_time = int(np.ceil(t_resample_max / 5)) * 5

        # ================================
        # 計算 Dive Time（從 depth >= 0.7 m 開始，到回到 0 附近）
        # ================================
        dive_time_s = None
        dive_start_s = None
        dive_end_s = None

        df_sorted = dive_df.sort_values("time_s").reset_index(drop=True)

        # 1. 開始時間：第一個 depth >= 0.7 m
        start_rows = df_sorted[df_sorted["depth_m"] >= 0.7]
        if not start_rows.empty:
            t_start = start_rows["time_s"].iloc[0]

            # 2. 結束時間：之後 depth 回到 0 附近（例如 <= 0.05 當作 0）
            after = df_sorted[df_sorted["time_s"] >= t_start]
            end_candidates = after[after["depth_m"] <= 0.05]

            if not end_candidates.empty:
                t_end = end_candidates["time_s"].iloc[-1]
            else:
                # 如果沒有回到 0，就用最後一個點當結束（保底）
                t_end = after["time_s"].iloc[-1]

            dive_start_s = float(t_start)
            dive_end_s   = float(t_end)
            dive_time_s  = max(0.0, dive_end_s - dive_start_s)

        # 也在畫面上讓你確認一下計算結果
        if dive_time_s is not None:
            mm = int(dive_time_s // 60)
            ss = int(round(dive_time_s % 60))
            st.info(f"偵測到的 Dive Time：約 {mm:02d}:{ss:02d} （從深度 ≥ 0.7 m 開始，到回到 0 m）")

        # ================================
        # 3️⃣ 左右並排圖表
        # ================================
        st.subheader("3️⃣ 潛水曲線預覽（時間 vs 深度 / 速率）")

        col_depth, col_rate = st.columns(2)

        # 左邊：深度 vs 時間
        with col_depth:
            depth_chart = (
                alt.Chart(df_rate)
                .mark_line()
                .encode(
                    x=alt.X(
                        "time_s:Q",
                        title="時間（秒）",
                        scale=alt.Scale(domain=[t_min, max_display_time]),
                    ),
                    y=alt.Y(
                        "depth_m:Q",
                        title="深度（m）",
                        scale=alt.Scale(reverse=True),  # 上淺下深
                    ),
                    tooltip=[
                        alt.Tooltip("time_s:Q", title="時間 (s)", format=".1f"),
                        alt.Tooltip("depth_m:Q", title="深度 (m)", format=".1f"),
                    ],
                )
                .properties(
                    title="深度 vs 時間（重採樣 1 秒）",
                    height=300,
                )
            )
            st.altair_chart(depth_chart, use_container_width=True)

        # 右邊：速率 vs 時間
        with col_rate:
            rate_chart = (
                alt.Chart(df_rate)
                .mark_line()
                .encode(
                    x=alt.X(
                        "time_s:Q",
                        title="時間（秒）",
                        scale=alt.Scale(domain=[t_min, max_display_time]),
                    ),
                    y=alt.Y(
                        "rate_abs_mps_smooth:Q",
                        title="速率（m/s）",
                        scale=alt.Scale(domain=[0, 3]),
                    ),
                    tooltip=[
                        alt.Tooltip("time_s:Q", title="時間 (s)", format=".1f"),
                        alt.Tooltip("rate_abs_mps_smooth:Q", title="平滑速率 (|m/s|)", format=".2f"),
                    ],
                )
                .properties(
                    title="速率 vs 時間（每秒深度差，經平滑處理）",
                    height=300,
                )
            )
            st.altair_chart(rate_chart, use_container_width=True)

        st.caption(
            f"原始資料點數：{len(dive_df)}，"
            f"重採樣時間範圍：{df_rate['time_s'].min():.0f}～{df_rate['time_s'].max():.0f} 秒，"
            f"最大深度：約 {df_rate['depth_m'].max():.1f} m"
        )


# --- 4. 設定時間偏移 & 版型選擇 ---
st.subheader("4️⃣ 影片對齊與版型")

col3, col4 = st.columns(2)

with col3:
    time_offset = st.slider(
        "影片起點相對於潛水開始時間的偏移（秒）",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
        help="如果影片比實際下潛早開始，請用正值調整。"
    )

# ------------ 動態 Layout 設定區（未來要加 layout，改這裡就好） ------------
LAYOUTS_DIR = ASSETS_DIR / "layouts"

layouts_config = [
    {
        "id": "A",
        "label": "Layout A：深度 + 心率 + 速率",
        "filename": "layout_a.png",
        "description": "（ATMOS無法顯示心律）",
        "uses_diver_info": False,
    },
    {
        "id": "B",
        "label": "Layout B：包含姓名 / 國籍 / 潛水項目",
        "filename": "layout_b.png",
        "description": "賽事風格版型。",
        "uses_diver_info": False,
    },
    {
        "id": "C",
        "label": "Layout C：單純深度",
        "filename": "layout_c.png",
        "description": "Simple_A",
        "uses_diver_info": False,
    },
    {
        "id": "D",
        "label": "Layout D：單純深度",
        "filename": "layout_d.png",
        "description": "Simple_B",
        "uses_diver_info": True,
    },
]


# 用 label 當顯示文字，id 當內部 key
layout_labels = [cfg["label"] for cfg in layouts_config]

with col4:
    selected_label = st.selectbox(
        "選擇影片版型",
        options=layout_labels,
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
    corner_radius = 15         # 👈 可以自己調整圓角程度

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
st.markdown("### 版型示意圖（目前選擇會加黃色外框）")

# 依 layout 數量動態建立欄位（目前是 3 個，就會是 3 欄）
cols = st.columns(len(layouts_config))

for col, cfg in zip(cols, layouts_config):
    with col:
        img = load_layout_image(cfg, cfg["id"] == selected_id)
        st.image(
            img,
            caption=cfg["label"],
            use_container_width=True,   # ✅ 新版參數，取代 use_column_width
        )
        if cfg.get("description"):
            st.caption(cfg["description"])

# --- 5. 輸入潛水員資訊---
st.subheader("5️⃣ 潛水員資訊（選填 只有Layout B 需要填寫）")

# --- 國籍選單資料 ---
nationality_file = ASSETS_DIR / "Nationality.csv"

nat_df = load_nationality_options(nationality_file)

if nat_df.empty:
    nationality_options = ["（不指定）"]
else:
    nationality_labels = nat_df["label"].tolist()
    nationality_options = ["（不指定）"] + nationality_labels

# 設定預設值：如果有 Taiwan (TWN) 就優先選它，否則選「不指定」
default_label = "Taiwan (TWN)"
if default_label in nationality_options:
    default_index = nationality_options.index(default_label)
else:
    default_index = 0


col_info_1, col_info_2 = st.columns(2)

with col_info_1:
    diver_name = st.text_input("潛水員姓名 / Nickname", value="")

    nationality_label = st.selectbox(
        "國籍",
        options=nationality_options,
        index=default_index,
    )

    # 之後 render_video 使用的字串
    if nationality_label == "（不指定）":
        nationality = ""
    else:
        nationality = nationality_label

with col_info_2:
    discipline = st.selectbox(
        "潛水項目（Discipline）",
        options=["（不指定）", "CWT", "CWTB", "CNF", "FIM"]
    )
    # 之後你也可以改成只顯示適合「深度」的幾個項目

# --- 6. 產生影片 ---
if st.button("🚀 產生疊加數據影片", type="primary"):
    if (dive_df is None) or (video_file is None):
        st.error("請先上傳手錶數據與影片檔。")
    else:
        # 建立真正會被 render_video 更新的進度條
        progress_bar = st.progress(0, text="初始化中...")

        def progress_callback(p: float, message: str = ""):
            """
            p: 0.0 ~ 1.0
            message: 顯示在進度條上的文字
            """
            p = max(0.0, min(1.0, float(p)))  # clamp
            percent = int(p * 100)
            if message:
                text = f"{message} {percent}%"
            else:
                text = f"產生影片中... {percent}%"
            progress_bar.progress(percent, text=text)

        # 先把上傳的影片暫存到 /tmp
        tmp_video_path = Path("/tmp") / video_file.name
        with open(tmp_video_path, "wb") as f:
            f.write(video_file.read())

        try:
            # 同步呼叫 render_video，並把 progress_callback 傳進去
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
                discipline=discipline,
                dive_time_s=dive_time_s,
                dive_start_s=dive_start_s,
                dive_end_s=dive_end_s,
                progress_callback=progress_callback,  # ⭐ 新增這個
            )

            # 確保最後是 100%
            progress_callback(1.0, "影片產生完成！")
            st.success("影片產生完成！")

            # 下載按鈕
            with open(output_path, "rb") as f:
                st.download_button(
                    "下載 1080p 影片",
                    data=f,
                    file_name="dive_overlay_1080p.mp4",
                    mime="video/mp4",
                )

            # 預覽影片
            col_preview, col_empty = st.columns([1, 1])
            with col_preview:
                st.video(str(output_path))

        except Exception as e:
            st.error(f"產生影片時發生錯誤：{e}")
