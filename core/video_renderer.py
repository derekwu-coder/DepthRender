# core/video_renderer.py

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import VideoClip
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageFont, Image
import time

# --- FONT PATH 設定 ---
BASE_DIR = Path(__file__).resolve().parent.parent
FONT_PATH = BASE_DIR / "assets" / "fonts" / "RobotoCondensedBold.ttf"

print(f"[FONT] FONT_PATH = {FONT_PATH}")

if not FONT_PATH.exists():
    print(f"[WARN] Font file NOT found: {FONT_PATH}")
else:
    print(f"[INFO] Font file FOUND: {FONT_PATH}")



# --- Pillow ANTIALIAS patch（修補新版 Pillow 沒有 ANTIALIAS 的問題） ---
if not hasattr(PILImage, "ANTIALIAS"):
    try:
        from PIL import Image as _ImgMod
        if hasattr(_ImgMod, "Resampling"):
            PILImage.ANTIALIAS = _ImgMod.Resampling.LANCZOS
        else:
            PILImage.ANTIALIAS = _ImgMod.LANCZOS
    except Exception:
        PILImage.ANTIALIAS = PILImage.BICUBIC


# --- Layout anchor 設定：右上 info 卡要放哪一角 ---
LAYOUT_CONFIG = {
    "A": {"anchor": "top_left"},
    "B": {"anchor": "top_right"},
    "C": {"anchor": "bottom_right"},
    "D": {"anchor": "top_right"},
}

# ============================================================
# 🎛 Layout B：右下角選手資訊模組（黃/黑背板 + 國旗 + 姓名 + 項目）
# ============================================================

# 黃色背板（Board 2，上面那塊）
BOARD2_ENABLE = True
BOARD2_WIDTH  = 550
BOARD2_HEIGHT = 70
BOARD2_RADIUS = 13
BOARD2_COLOR  = (254, 168, 23, 255)   # 你原本調過的鵝黃偏橘
BOARD2_LEFT   = 480                  # 跟 Mac 版一樣，從左邊算起
BOARD2_BOTTOM = 175                  # 距離畫面底部的距離（px）

# 黑色背板（Board 3，下面那塊）
BOARD3_ENABLE = True
BOARD3_WIDTH  = 550
BOARD3_HEIGHT = 80
BOARD3_RADIUS = 13
BOARD3_COLOR  = (0, 0, 0, 255)
BOARD3_LEFT   = 480
BOARD3_BOTTOM = 115

# Board 3 內文字（速率 + Dive Time）
BOARD3_RATE_FONT_SIZE   = 34
BOARD3_TIME_FONT_SIZE   = 34
BOARD3_TEXT_COLOR       = (255, 255, 255, 255)

BOARD3_RATE_OFFSET_X    = 20   # 從板子左側算起的微調（正數往右）
BOARD3_RATE_OFFSET_Y    = 0   # 垂直微調（正數往下）

BOARD3_TIME_OFFSET_X    = 20    # 以「置中」為基準的 X 微調
BOARD3_TIME_OFFSET_Y    = 0   # 垂直微調（正數往下）


# 國旗 + 三碼國碼（放在黃色背板左側）
FLAG_ENABLE            = True
FLAG_LEFT_OFFSET       = 15   # 黃色背板左邊到國旗的水平距離
FLAG_TOP_BOTTOM_MARGIN = 15   # 黃板上下留白，決定國旗高度
FLAG_ALPHA3_TEXT_GAP   = 6    # 國旗與三碼文字間距
FLAG_ALPHA3_FONT_SIZE  = 34
FLAG_ALPHA3_FONT_COLOR = (0, 0, 0, 255)
FLAG_ALPHA3_OFFSET_Y   = -8   # 三碼文字略微往上

# 位置微調（你後面可以慢慢調）
COMP_DISC_OFFSET_RIGHT = 15  # 項目靠右對齊時，距離黃板右邊的距離
COMP_DISC_OFFSET_Y      = -8 # 項目在黃板中的 Y 偏移

COMP_ALPHA3_OFFSET_X = 0     # 三碼國碼額外 X 調整（在國旗下）
# （Y 用上面的 FLAG_ALPHA3_OFFSET_Y）


# --- 右上 info 卡 ---
INFO_CARD_FONT_SIZE = 48              # 字體大小   original = 48
INFO_TEXT_OFFSET_X = 0                # 👉 info 卡文字整體 X 位移
INFO_TEXT_OFFSET_Y = -7               # 👉 info 卡文字整體 Y 位移（負值 = 往上）

# --- 深度刻度文字 ---
DEPTH_TICK_LABEL_FONT_SIZE = 32
DEPTH_TICK_LABEL_OFFSET_X = 0         # 👉 刻度數字 X 位移
DEPTH_TICK_LABEL_OFFSET_Y = -8         # 👉 刻度數字 Y 位移

# --- 泡泡內文字 ---
BUBBLE_FONT_SIZE = 36
BUBBLE_TEXT_OFFSET_X = 0              # 👉 泡泡內文字 X 位移
BUBBLE_TEXT_OFFSET_Y = -10              # 👉 泡泡內文字 Y 位移

# --- 賽事資訊文字（右下模組）---
COMP_NAME_FONT_SIZE = 34              # 姓名         original = 34
COMP_SUB_FONT_SIZE  = 34              # 國籍 / 項目  original = 34
COMP_CODE_FONT_SIZE = 34              # 三碼國碼     original = 34

COMP_NAME_OFFSET_X = 30                # 👉 姓名文字 X 位移
COMP_NAME_OFFSET_Y = -8                # 👉 姓名文字 Y 位移
COMP_SUB_OFFSET_X  = 0                # 👉 國籍 / 項目 X 位移
COMP_SUB_OFFSET_Y  = -8                # 👉 國籍 / 項目 Y 位移
COMP_CODE_OFFSET_X = 0                # 👉 國碼 X 位移
COMP_CODE_OFFSET_Y = -2                # 👉 國碼 Y 位移

# ----- Layout B：黑背板 + 深度條 + 泡泡 -----

# 黑色背板（寬 x 高）
DEPTH_PANEL_WIDTH = 100          # px
DEPTH_PANEL_HEIGHT = 980         # px
DEPTH_PANEL_LEFT_MARGIN = 40     # 距離畫面左邊的距離（px）
DEPTH_PANEL_RADIUS = 20          # 🔸背板導角半徑

# 深度條
DEPTH_BAR_TOTAL_HEIGHT = 850     # 深度條總高度（px）
DEPTH_TICK_WIDTH = 4             # 刻度線寬度（px）

# 刻度長度（px）
DEPTH_TICK_LEN_10M = 36          # 整十刻度長度
DEPTH_TICK_LEN_5M = 27           # 整五刻度長度（但非十的倍數）
DEPTH_TICK_LEN_1M = 22           # 其他刻度長度

# 刻度文字
DEPTH_TICK_LABEL_FONT_SIZE = 30  # 🔸刻度數字字體大小

# 泡泡標籤
BUBBLE_WIDTH = 80               # 泡泡主體寬度（px）
BUBBLE_HEIGHT = 45               # 泡泡主體高度（px）
BUBBLE_RADIUS = 10               # 泡泡圓角半徑（px）
BUBBLE_TAIL_WIDTH = 22           # 泡泡指向左邊的小三角形寬度（px）
BUBBLE_TAIL_HEIGHT_RATIO = 0.5  # 泡泡小三角形高度 = BUBBLE_HEIGHT * 這個比例
BUBBLE_FONT_SIZE = 32            # 泡泡內深度字體大小

# 泡泡顏色
BUBBLE_CURRENT_COLOR = (254, 168, 23, 255)   # 當前深度泡泡：橘黃色
BUBBLE_BEST_COLOR = (255, 255, 255, 255)     # 最大深度泡泡：白色
BUBBLE_TEXT_COLOR_DARK = (0, 0, 0, 255)      # 文字：黑色
BUBBLE_TEXT_COLOR_LIGHT = (0, 0, 0, 255)     # 目前兩顆都用深色字

# ----- 預留：未來 Layout A / C / D 專用參數可加在這裡 -----
# 例如：
# LAYOUT_A_PARAMS = {...}
# LAYOUT_C_PARAMS = {...}
# LAYOUT_D_PARAMS = {...}

# ============================================================
# 小工具函式
# ============================================================

def text_size(draw_obj, text: str, font_obj):
    """
    安全取得文字寬高，兼容 Pillow 舊版 / 新版。
    """
    try:
        bbox = draw_obj.textbbox((0, 0), text, font=font_obj)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        return font_obj.getsize(text)


def format_dive_time(seconds: float) -> str:
    """
    秒數 -> MM:SS 字串。
    """
    if seconds is None:
        return ""
    total_sec = int(round(float(seconds)))
    minutes = total_sec // 60
    sec = total_sec % 60
    return f"{minutes:02d}:{sec:02d}"


# ============================================================
# 右下角賽事資訊卡（Layout B）
# ============================================================

def _infer_country_code_3(nationality: str) -> Tuple[Optional[str], str]:

    """
    嘗試從使用者輸入的 nationality 推出三碼國碼：
    - 若有括號，如 'Chinese Taipei (TPE)' -> 回傳 ('TPE', 'Chinese Taipei')
    - 若整串是 2~3 碼英文字母，如 'TPE' / 'JPN' -> 當成國碼
    - 其他情況 -> 不顯示國旗，只顯示原文字
    回傳：(code3 或 None, 顯示用文字 label)
    """
    if not nationality:
        return None, ""

    txt = nationality.strip()
    if "(" in txt and ")" in txt:
        # 例如：Chinese Taipei (TPE)
        before, _, after = txt.partition("(")
        code = after.split(")")[0].strip().upper()
        label = before.strip()
        if len(code) in (2, 3) and code.isalpha():
            return code, label or code

    if txt.isalpha() and len(txt) in (2, 3):
        code = txt.upper()
        return code, code

    return None, txt


def _load_flag_png(flags_dir: Path, code3: Optional[str]) -> Optional[Image.Image]:

    """
    從 assets/flags 底下載入三碼國旗 PNG（檔名假設為 tpe.png / jpn.png ...）
    """
    if not code3:
        return None
    path = flags_dir / f"{code3.lower()}.png"
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGBA")
    except Exception:
        return None

def draw_competition_panel_bottom_right(
    base_img: PILImage.Image,
    diver_name: Optional[str],
    nationality: Optional[str],
    discipline: Optional[str],
    flags_dir: Path,
    rate_text: str,
    time_text: str,
) -> PILImage.Image:
    """
    在畫面右下方畫出：
    - 黃色 Backplate 2：國旗 + 三碼 + 姓名 + 項目
    - 黑色 Backplate 3：速率 + Dive Time
    """
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # ---------- 先畫黑色背板（Board 3，底下那塊） ----------
    if BOARD3_ENABLE:
        b3_w = int(BOARD3_WIDTH)
        b3_h = int(BOARD3_HEIGHT)
        b3_x = int(BOARD3_LEFT)
        b3_y = int(H - BOARD3_BOTTOM - b3_h)
        b3_rect = [b3_x, b3_y, b3_x + b3_w, b3_y + b3_h]

        draw.rounded_rectangle(
            b3_rect,
            radius=int(BOARD3_RADIUS),
            fill=BOARD3_COLOR,
        )

        # --- Board 3 裡的文字：左邊是速率，正中間是 Dive Time ---
        # 速率
        if rate_text:
            try:
                font_rate = load_font(BOARD3_RATE_FONT_SIZE)
            except Exception:
                font_rate = ImageFont.load_default()

            rw, rh = text_size(draw, rate_text, font_rate)
            rate_x = b3_x + BOARD3_RATE_OFFSET_X
            rate_y = b3_y + (b3_h - rh) // 2 + BOARD3_RATE_OFFSET_Y

            draw.text(
                (rate_x, rate_y),
                rate_text,
                font=font_rate,
                fill=BOARD3_TEXT_COLOR,
            )

        # Dive Time（置中）
        if time_text:
            try:
                font_time = load_font(BOARD3_TIME_FONT_SIZE)
            except Exception:
                font_time = ImageFont.load_default()

            tw, th = text_size(draw, time_text, font_time)
            time_x = b3_x + (b3_w - tw) // 2 + BOARD3_TIME_OFFSET_X
            time_y = b3_y + (b3_h - th) // 2 + BOARD3_TIME_OFFSET_Y

            draw.text(
                (time_x, time_y),
                time_text,
                font=font_time,
                fill=BOARD3_TEXT_COLOR,
            )

    # ---------- 黃色背板（Board 2，上面那塊） ----------
    if BOARD2_ENABLE:
        b2_w = int(BOARD2_WIDTH)
        b2_h = int(BOARD2_HEIGHT)
        b2_x = int(BOARD2_LEFT)
        b2_y = int(H - BOARD2_BOTTOM - b2_h)
        b2_rect = [b2_x, b2_y, b2_x + b2_w, b2_y + b2_h]

        draw.rounded_rectangle(
            b2_rect,
            radius=int(BOARD2_RADIUS),
            fill=BOARD2_COLOR,
        )

        # ---------- 國旗 + 三碼國碼在黃板左側 ----------
        code3, country_label = _infer_country_code_3(nationality or "")
        flag_img = _load_flag_png(flags_dir, code3) if FLAG_ENABLE else None

        flag_right_x = b2_x  # 先給個預設值，下面如果有旗幟會覆蓋它

        if FLAG_ENABLE and flag_img is not None:
            margin_tb = int(FLAG_TOP_BOTTOM_MARGIN)
            left_off  = int(FLAG_LEFT_OFFSET)

            # 依照黃板高度縮放國旗
            target_h = max(1, b2_h - margin_tb * 2)
            scale = target_h / flag_img.height
            target_w = int(flag_img.width * scale)

            if target_w > 0:
                flag_resized = flag_img.resize((target_w, target_h), PILImage.ANTIALIAS)
                fx = b2_x + left_off
                fy = b2_y + margin_tb
                img.paste(flag_resized, (fx, fy), flag_resized)

                flag_right_x = fx + target_w

                # 三碼國碼文字（例如 TPE）在旗右側
                if code3:
                    try:
                        font_code = load_font(int(FLAG_ALPHA3_FONT_SIZE))
                    except Exception:
                        font_code = ImageFont.load_default()

                    code_text = code3.upper()
                    tw, th = text_size(draw, code_text, font_code)
                    gap = int(FLAG_ALPHA3_TEXT_GAP)
                    tx = flag_right_x + gap + COMP_ALPHA3_OFFSET_X
                    ty = (
                        b2_y
                        + (b2_h - th) // 2
                        + int(FLAG_ALPHA3_OFFSET_Y)
                    )
                    draw.text(
                        (tx, ty),
                        code_text,
                        font=font_code,
                        fill=FLAG_ALPHA3_FONT_COLOR,
                    )

        # ---------- 姓名：置中在黃板 ----------
        if diver_name:
            try:
                font_name = load_font(COMP_NAME_FONT_SIZE)
            except Exception:
                font_name = ImageFont.load_default()

            dn_text = str(diver_name)
            nw, nh = text_size(draw, dn_text, font_name)
            name_x = b2_x + (b2_w - nw) // 2 + COMP_NAME_OFFSET_X
            name_y = b2_y + (b2_h - nh) // 2 + COMP_NAME_OFFSET_Y

            draw.text(
                (name_x, name_y),
                dn_text,
                font=font_name,
                fill=(0, 0, 0, 255),
            )

        # ---------- 項目（discipline）：靠右顯示在黃板裡 ----------
        if discipline and discipline != "（不指定）":
            try:
                font_disc = load_font(COMP_SUB_FONT_SIZE)
            except Exception:
                font_disc = ImageFont.load_default()

            dt_text = str(discipline)
            dw, dh = text_size(draw, dt_text, font_disc)
            right_off = int(COMP_DISC_OFFSET_RIGHT)
            disc_x = b2_x + b2_w - right_off - dw
            disc_y = b2_y + (b2_h - dh) // 2 + COMP_DISC_OFFSET_Y

            draw.text(
                (disc_x, disc_y),
                dt_text,
                font=font_disc,
                fill=(0, 0, 0, 255),
            )

    return img


# ============================================================
# 左側黑背板 + 深度條 + 泡泡（Layout B）
# ============================================================

def draw_speech_bubble(
    draw: ImageDraw.ImageDraw,
    left_x: int,
    center_y: int,
    text: str,
    fill_color: tuple,
    text_color: tuple,
    font: ImageFont.FreeTypeFont,
):
    """
    畫一個「左邊有小三角形」的泡泡：
    - left_x：小三角形尖端的位置（會貼在黑色背板右邊）
    - 泡泡主體在右側，寬 BUBBLE_WIDTH、高 BUBBLE_HEIGHT
    """
    w = BUBBLE_WIDTH
    h = BUBBLE_HEIGHT
    r = BUBBLE_RADIUS
    tail_w = BUBBLE_TAIL_WIDTH
    tail_h = int(h * BUBBLE_TAIL_HEIGHT_RATIO)


    tip_x = left_x                     # 小三角形尖端（貼在背板右邊）
    base_x = left_x + tail_w          # 三角形與矩形交界
    rect_x0 = base_x
    rect_x1 = rect_x0 + w
    y0 = center_y - h // 2
    y1 = y0 + h

    # 主體圓角矩形（在右邊）
    draw.rounded_rectangle(
        [rect_x0, y0, rect_x1, y1],
        radius=r,
        fill=fill_color,
    )

    # 左邊小三角形
    tri_y_top = center_y - tail_h // 2
    tri_y_bot = center_y + tail_h // 2
    draw.polygon(
        [
            (tip_x, center_y),          # 尖端（貼背板）
            (base_x, tri_y_top),        # 右上
            (base_x, tri_y_bot),        # 右下
        ],
        fill=fill_color,
    )

    # 文字置中在「矩形」裡
    tw, th = text_size(draw, text, font)
    text_x = rect_x0 + (w - tw) // 2 + BUBBLE_TEXT_OFFSET_X
    text_y = center_y - th // 2 + BUBBLE_TEXT_OFFSET_Y
    draw.text(
        (text_x, text_y),
        text,
        font=font,
        fill=text_color,
    )


def draw_depth_bar_and_bubbles(
    base_overlay: PILImage.Image,
    depth_val: float,
    max_depth_for_scale: float,
    best_depth: float,
    show_best_bubble: bool,
    base_font: ImageFont.FreeTypeFont,
):
    """
    Layout B 專用：
    - 左邊黑色背板（100 x 980 px），左右位置、大小固定，上下置中，導角 DEPTH_PANEL_RADIUS
    - 深度條總高度固定為 DEPTH_BAR_TOTAL_HEIGHT（例如 850px）
    - 每 1 m 一個刻度：
        - 整十刻度長度 = DEPTH_TICK_LEN_10M
        - 整五刻度長度 = DEPTH_TICK_LEN_5M
        - 其他刻度長度 = DEPTH_TICK_LEN_1M
      刻度線寬度 = DEPTH_TICK_WIDTH
      ✦ 不畫中間直線，只畫橫線刻度
    - 刻度數字在刻度「左側」，字體大小 DEPTH_TICK_LABEL_FONT_SIZE
    - 泡泡 1（橘色）：顯示當前深度，隨深度上下移動
    - 泡泡 2（白色）：當 show_best_bubble=True 時顯示在最大深度位置
      （也就是：到達最大深度那一刻出現，之後一路顯示）
    - 兩顆泡泡的小三角形尖端都貼齊黑背板的右側邊界
    """
    overlay = base_overlay.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size

    if max_depth_for_scale <= 0:
        return overlay

    # --- 黑色背板 ---
    panel_x0 = DEPTH_PANEL_LEFT_MARGIN
    panel_x1 = panel_x0 + DEPTH_PANEL_WIDTH
    panel_y0 = (h - DEPTH_PANEL_HEIGHT) // 2
    panel_y1 = panel_y0 + DEPTH_PANEL_HEIGHT

    draw.rounded_rectangle(
        [panel_x0, panel_y0, panel_x1, panel_y1],
        radius=DEPTH_PANEL_RADIUS,
        fill=(0, 0, 0, 200),
    )

    # --- 深度條幾何位置（只決定 Y 範圍，不畫直線） ---
    bar_h = DEPTH_BAR_TOTAL_HEIGHT
    bar_y0 = (h - bar_h) // 2
    bar_y1 = bar_y0 + bar_h

    # 刻度最右邊對齊：離背板右側 10px
    tick_x_end = panel_x1 - 10

    max_d = max_depth_for_scale

    # 刻度文字字型
    try:
        tick_font = load_font(DEPTH_TICK_LABEL_FONT_SIZE)
    except:
        tick_font = base_font


    # --- 刻度：每 1 m 一格 ---
    for d in range(0, int(max_d) + 1):
        ratio = d / max_d
        y = int(bar_y0 + ratio * bar_h)

        # 決定刻度長度
        if d % 10 == 0:
            tick_len = DEPTH_TICK_LEN_10M
        elif d % 5 == 0:
            tick_len = DEPTH_TICK_LEN_5M
        else:
            tick_len = DEPTH_TICK_LEN_1M

        tick_x_start = tick_x_end - tick_len

        # 刻度線（往左畫，右邊對齊 tick_x_end）
        draw.line(
            [(tick_x_start, y), (tick_x_end, y)],
            fill=(255, 255, 255, 220),
            width=DEPTH_TICK_WIDTH,
        )

        # 每 10 m 顯示數字（在刻度左側）
        if d % 10 == 0:
            label = f"{d}"
            lw, lh = text_size(draw, label, tick_font)
            lx = tick_x_start - 6 - lw + DEPTH_TICK_LABEL_OFFSET_X
            ly = y - lh // 2 + DEPTH_TICK_LABEL_OFFSET_Y
            draw.text(
                (lx, ly),
                label,
                font=tick_font,
                fill=(255, 255, 255, 255),
            )

    # --- 深度數值 -> Y 座標 ---
    def depth_to_y(dv: float) -> int:
        d_clamped = max(0.0, min(max_d, float(dv)))
        ratio = d_clamped / max_d
        return int(bar_y0 + ratio * bar_h)

    # 泡泡字型
    try:
        bubble_font = load_font(BUBBLE_FONT_SIZE)
    except:
        bubble_font = base_font


    bubble_attach_x = panel_x1  # 小三角形尖端貼齊背板右側

    # --- 泡泡 1：當前深度（橘色） ---
    current_y = depth_to_y(depth_val)
    current_text = f"{depth_val:.1f}"

    draw_speech_bubble(
        draw=draw,
        left_x=bubble_attach_x,
        center_y=current_y,
        text=current_text,
        fill_color=BUBBLE_CURRENT_COLOR,
        text_color=BUBBLE_TEXT_COLOR_DARK,
        font=bubble_font,
    )

    # --- 泡泡 2：最大深度（白色） ---
    # 交給呼叫端決定什麼時候開始顯示（例如：到達最大深度那一刻後就一直顯示）
    if best_depth > 0 and show_best_bubble:
        best_y = depth_to_y(best_depth)
        best_text = f"{best_depth:.1f}"

        draw_speech_bubble(
            draw=draw,
            left_x=bubble_attach_x,
            center_y=best_y,
            text=best_text,
            fill_color=BUBBLE_BEST_COLOR,
            text_color=BUBBLE_TEXT_COLOR_DARK,
            font=bubble_font,
        )

    return overlay

# ============================================================
# 主渲染函式（所有 Layout 共用）
# ============================================================

def render_video(
    video_path: Path,
    dive_df: pd.DataFrame,
    df_rate: pd.DataFrame,
    time_offset: float,
    layout: str,
    assets_dir: Path,
    output_resolution=(1080, 1920),
    diver_name: str = "",
    nationality: str = "",
    discipline: str = "",
    dive_time_s: Optional[float] = None,   # 目前沒直接用，先保留
    dive_start_s: Optional[float] = None,  # 起始 time_s（深度 >= 0.7m）
    dive_end_s: Optional[float] = None,    # 結束 time_s（回到 0）
    progress_callback=None,                # ⭐ 新增：由 app.py 傳進來
):
    """
    progress_callback(p: float, message: str) 會被用來更新 Streamlit 進度條：
    - p: 0.0 ~ 1.0
    """

    flags_dir = assets_dir / "flags"

    # 小工具：安全呼叫 progress_callback
    def update_progress(p: float, msg: str = ""):
        if progress_callback is None:
            return
        try:
            p = max(0.0, min(1.0, float(p)))
            progress_callback(p, msg)
        except Exception:
            # 不讓 UI 的錯影響主流程
            pass

    t0 = time.perf_counter()
    update_progress(0.02, "初始化中...")

    # =========================
    # 1. 讀影片 + Resize
    # =========================
    t_load_start = time.perf_counter()

    clip = VideoFileClip(str(video_path))
    W, H = output_resolution
    clip = clip.resize((W, H))

    t_load_end = time.perf_counter()
    print(f"[render_video] 載入 + resize 影片耗時 {t_load_end - t_load_start:.2f} 秒")
    update_progress(0.08, "載入影片完成")

    # =========================
    # 2. 深度 / 速率 插值用 & 前處理
    # =========================
    t_pre_start = time.perf_counter()

    times_d = dive_df["time_s"].to_numpy()
    depths_d = dive_df["depth_m"].to_numpy()

    times_r = df_rate["time_s"].to_numpy()
    rates_r = df_rate["rate_abs_mps_smooth"].to_numpy()

    # 最大深度 & 最大深度發生時間
    if len(depths_d) > 0:
        max_depth_raw = float(np.nanmax(depths_d))
        best_idx = int(np.nanargmax(depths_d))
        best_time_global = float(times_d[best_idx])  # 🔸最大深度發生的 time_s（log 的時間）
    else:
        max_depth_raw = 0.0
        best_time_global = None

    # 深度刻度顯示邏輯
    if max_depth_raw <= 0:
        max_depth_for_scale = 30.0
    elif max_depth_raw < 30.0:
        max_depth_for_scale = 30.0
    elif max_depth_raw <= 40.0:
        max_depth_for_scale = 40.0
    else:
        max_depth_for_scale = float(int(np.ceil(max_depth_raw / 10.0)) * 10.0)

    best_depth = max_depth_raw

    def depth_at(t_video: float) -> float:
        t = t_video + time_offset
        if t <= times_d[0]:
            return float(depths_d[0])
        if t >= times_d[-1]:
            return float(depths_d[-1])
        return float(np.interp(t, times_d, depths_d))

    def rate_at(t_video: float) -> float:
        t = t_video + time_offset
        if t <= times_r[0]:
            return float(rates_r[0])
        if t >= times_r[-1]:
            return float(rates_r[-1])
        return float(np.interp(t, times_r, rates_r))

    # --- Dive time 起訖 ---
    if dive_start_s is None:
        mask_start = dive_df["depth_m"] >= 0.7
        if mask_start.any():
            dive_start_s = float(dive_df.loc[mask_start, "time_s"].iloc[0])

    if dive_end_s is None:
        mask_end = dive_df["depth_m"] <= 0.05
        if mask_end.any():
            dive_end_s = float(dive_df.loc[mask_end, "time_s"].iloc[-1])

    def elapsed_dive_time(t_video: float) -> Optional[float]:
        """
        依目前影片時間 t_video 推出「潛水已經過多久」：
        - 前段：固定 0:00
        - 中段：從 start 開始累加
        - 結束後：鎖在總 Dive Time
        """
        if dive_start_s is None:
            return None

        t_global = t_video + time_offset

        if t_global <= dive_start_s:
            return 0.0

        if dive_end_s is not None and t_global >= dive_end_s:
            return max(0.0, float(dive_end_s - dive_start_s))

        return max(0.0, float(t_global - dive_start_s))

    # --- 字型 ---
    try:
        base_font = load_font(INFO_CARD_FONT_SIZE)
    except:
        base_font = ImageFont.load_default()

    t_pre_end = time.perf_counter()
    print(f"[render_video] 前處理耗時 {t_pre_end - t_pre_start:.2f} 秒")
    update_progress(0.12, "資料前處理完成")

    # =========================
    # 3. 每幀繪製（主迴圈進度）
    # =========================

    duration = float(clip.duration) if clip.duration else 0.0
    # 用 dict 包起來讓內層 make_frame 可以修改
    last_p = {"value": 0.12}

    def make_frame(t):
        # --- 進度條：用影片時間推估 ---
        if duration > 0:
            frac = max(0.0, min(1.0, t / duration))
            # 這一段佔整體 0.12 ~ 0.98
            p = 0.12 + 0.86 * frac
            # 只在進度有明顯差距時才更新，避免太頻繁呼叫
            if p - last_p["value"] >= 0.01:
                last_p["value"] = p
                update_progress(p, "產生疊加畫面中...")

        frame = clip.get_frame(t)
        img = PILImage.fromarray(frame).convert("RGBA")
        img_w, img_h = img.size

        overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 影片時間對應到 log 的時間
        t_global = t + time_offset

        depth_val = depth_at(t)
        rate_val = rate_at(t)
        text_depth = f"{depth_val:.1f} m"
        text_rate = f"{rate_val:.1f} m/s"
        elapsed = elapsed_dive_time(t)
        time_text = format_dive_time(elapsed) if elapsed is not None else ""

        # ===== 右上角 info 卡（深度 + 速率 + 動態時間）=====
        # 👉 只在「不是 Layout B」時才畫；Layout B 用右下 Board 2/3
        if layout != "B":
            lines = [text_depth, text_rate]
            if time_text:
                lines.append(time_text)

            line_heights = []
            max_w = 0
            line_spacing = 8
            for txt in lines:
                w_txt, h_txt = text_size(draw, txt, base_font)
                max_w = max(max_w, w_txt)
                line_heights.append(h_txt)
            total_text_h = sum(line_heights) + (len(lines) - 1) * line_spacing

            padding = 16
            box_w = max_w + 2 * padding
            box_h = total_text_h + 2 * padding
            margin_edge = 40

            cfg_layout = LAYOUT_CONFIG.get(layout, LAYOUT_CONFIG["A"])
            anchor = cfg_layout.get("anchor", "top_left")

            if anchor == "top_left":
                x0 = margin_edge
                y0 = margin_edge
            elif anchor == "top_right":
                x0 = img_w - margin_edge - box_w
                y0 = margin_edge
            elif anchor == "bottom_left":
                x0 = margin_edge
                y0 = img_h - margin_edge - box_h
            elif anchor == "bottom_right":
                x0 = img_w - margin_edge - box_w
                y0 = img_h - margin_edge - box_h
            else:
                x0 = margin_edge
                y0 = margin_edge
            x1 = x0 + box_w
            y1 = y0 + box_h

            draw.rounded_rectangle(
                [x0, y0, x1, y1],
                radius=22,
                fill=(0, 0, 0, 170),
            )

            text_x = x0 + padding + INFO_TEXT_OFFSET_X
            cur_y = y0 + padding + INFO_TEXT_OFFSET_Y

            for txt, h_txt in zip(lines, line_heights):
                draw.text(
                    (text_x, cur_y),
                    txt,
                    font=base_font,
                    fill=(255, 255, 255, 255),
                )
                cur_y += h_txt + line_spacing

        # ===== Layout B 專屬元件 =====
        if layout == "B":
            # 是否顯示「最大深度泡泡」：
            #  - 規則：當 t_global >= best_time_global 時，一路顯示到影片結束
            if best_time_global is not None and t_global >= best_time_global:
                show_best_bubble = True
            else:
                show_best_bubble = False

            # 左側深度條 + 動態泡泡
            overlay = draw_depth_bar_and_bubbles(
                overlay,
                depth_val=depth_val,
                max_depth_for_scale=max_depth_for_scale,
                best_depth=best_depth,
                show_best_bubble=show_best_bubble,
                base_font=base_font,
            )

            # 右下角賽事資訊（姓名 / 國籍 / 項目 / 國旗 + Board3 中的速率 & 時間）
            overlay = draw_competition_panel_bottom_right(
                overlay,
                diver_name=diver_name or "",
                nationality=nationality or "",
                discipline=discipline or "",
                flags_dir=flags_dir,
                rate_text=text_rate,
                time_text=time_text,
            )

        composed = PILImage.alpha_composite(img, overlay).convert("RGB")
        return np.array(composed)

    # =========================
    # 4. 寫出影片（MoviePy / ffmpeg encode）
    # =========================
    t_encode_start = time.perf_counter()
    update_progress(last_p["value"], "編碼影片中...")

    new_clip = VideoClip(make_frame, duration=clip.duration)
    new_clip = new_clip.set_fps(clip.fps).set_audio(clip.audio)

    output_path = Path("/tmp/dive_overlay_output.mp4")
    new_clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio=True,
        audio_codec="aac",
        fps=clip.fps,
        # logger=None  # 如果不想在終端機看到 MoviePy 自帶進度條，可以關掉
    )

    t_encode_end = time.perf_counter()
    print(f"[render_video] 編碼 / 寫檔耗時 {t_encode_end - t_encode_start:.2f} 秒")

    t1 = time.perf_counter()
    print(f"[render_video] 總耗時 {t1 - t0:.2f} 秒")

    update_progress(1.0, "影片產生完成！")

    return output_path
