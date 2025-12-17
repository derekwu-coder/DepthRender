# core/video_renderer.py

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import VideoClip
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageFont, Image
from dataclasses import dataclass
import time

# ============================================================
# 字型設定：使用專案內的 RobotoCondensedBold.ttf
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_PATH = BASE_DIR / "assets" / "fonts" / "RobotoCondensedBold.ttf"

# ============================================================
# Layout C 專用字型（深度值 / 單位）
# ============================================================

LAYOUT_C_VALUE_FONT_PATH = BASE_DIR / "assets" / "fonts" / "nereus-bold.ttf"

if not LAYOUT_C_VALUE_FONT_PATH.exists():
    print(f"[WARN] Layout C value font NOT found: {LAYOUT_C_VALUE_FONT_PATH}")
else:
    print(f"[INFO] Layout C value font FOUND: {LAYOUT_C_VALUE_FONT_PATH}")


print(f"[FONT] FONT_PATH = {FONT_PATH}")

if not FONT_PATH.exists():
    print(f"[WARN] Font file NOT found: {FONT_PATH}")
else:
    print(f"[INFO] Font file FOUND: {FONT_PATH}")


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """
    統一載入字型：
    - 成功：回傳對應大小的 RobotoCondensedBold
    - 失敗：印出警告，改用預設字型
    """
    try:
        if FONT_PATH.exists():
            print(f"[FONT LOAD] Using {FONT_PATH} size={size}")
            return ImageFont.truetype(str(FONT_PATH), size)
        else:
            print(f"[FONT LOAD] NOT FOUND, fallback to default. PATH={FONT_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load font {FONT_PATH} (size={size}): {e}")

    print(f"[FONT LOAD] Fallback to default (size={size})")
    return ImageFont.load_default()

def resolve_flags_dir(assets_dir: Path) -> Path:
    """
    嘗試在 assets/flags、assets/Flags 裡找國旗資料夾，
    避免 mac / Linux 檔名大小寫不一致的問題。
    """
    candidates = [assets_dir / "flags", assets_dir / "Flags"]
    for p in candidates:
        if p.exists():
            print(f"[FLAGS_DIR] Using {p}")
            return p
    print(f"[FLAGS_DIR] No flags dir found under {assets_dir}, fallback to {assets_dir}")
    return assets_dir

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
BOARD2_WIDTH = 550
BOARD2_HEIGHT = 70
BOARD2_RADIUS = 13
BOARD2_COLOR = (254, 168, 23, 255)  # 鵝黃偏橘
BOARD2_LEFT = 480                   # 距離畫面左側
BOARD2_BOTTOM = 175                 # 距離畫面底部

# 黑色背板（Board 3，下面那塊）
BOARD3_ENABLE = True
BOARD3_WIDTH = 550
BOARD3_HEIGHT = 80
BOARD3_RADIUS = 13
BOARD3_COLOR = (0, 0, 0, 255)
BOARD3_LEFT = 480
BOARD3_BOTTOM = 115

# Board 3 內文字（速率 + Dive Time）
BOARD3_RATE_FONT_SIZE = 34
BOARD3_TIME_FONT_SIZE = 34
BOARD3_TEXT_COLOR = (255, 255, 255, 255)

BOARD3_RATE_OFFSET_X = 20   # 從板子左側算起的 X 微調（正數往右）
BOARD3_RATE_OFFSET_Y = 0    # 垂直微調（正數往下）

BOARD3_TIME_OFFSET_X = 20   # 以「置中」為基準的 X 微調
BOARD3_TIME_OFFSET_Y = 0    # 垂直微調（正數往下）

# 國旗 + 三碼國碼（放在黃色背板左側）
FLAG_ENABLE = True
FLAG_LEFT_OFFSET = 15        # 黃色背板左邊到國旗的水平距離
FLAG_TOP_BOTTOM_MARGIN = 15  # 黃板上下留白，決定國旗高度
FLAG_ALPHA3_TEXT_GAP = 6     # 國旗與三碼文字間距
FLAG_ALPHA3_FONT_SIZE = 34
FLAG_ALPHA3_FONT_COLOR = (0, 0, 0, 255)
FLAG_ALPHA3_OFFSET_Y = -8    # 三碼文字略微往上

# 位置微調
COMP_DISC_OFFSET_RIGHT = 15  # 項目靠右對齊時，距離黃板右邊的距離
COMP_DISC_OFFSET_Y = -8      # 項目在黃板中的 Y 偏移

COMP_ALPHA3_OFFSET_X = 0     # 三碼國碼額外 X 調整（在國旗右側）
# （Y 用上面的 FLAG_ALPHA3_OFFSET_Y）

# --- 右上 info 卡 ---
INFO_CARD_FONT_SIZE = 48
INFO_TEXT_OFFSET_X = 0
INFO_TEXT_OFFSET_Y = -7

# --- 深度刻度文字 ---
DEPTH_TICK_LABEL_FONT_SIZE = 30
DEPTH_TICK_LABEL_OFFSET_X = 0
DEPTH_TICK_LABEL_OFFSET_Y = -8

# --- 泡泡內文字 ---
BUBBLE_FONT_SIZE = 32
BUBBLE_TEXT_OFFSET_X = 0
BUBBLE_TEXT_OFFSET_Y = -10

# --- 賽事資訊文字（右下模組）---
COMP_NAME_FONT_SIZE = 34   # 姓名
COMP_SUB_FONT_SIZE = 34    # 國籍 / 項目
COMP_CODE_FONT_SIZE = 34   # 三碼國碼

COMP_NAME_OFFSET_X = 30
COMP_NAME_OFFSET_Y = -8
COMP_SUB_OFFSET_X = 0
COMP_SUB_OFFSET_Y = -8
COMP_CODE_OFFSET_X = 0
COMP_CODE_OFFSET_Y = -2

# ----- Layout B：黑背板 + 深度條 + 泡泡 -----

# 黑色背板（寬 x 高）
DEPTH_PANEL_WIDTH = 100     # px
DEPTH_PANEL_HEIGHT = 980    # px
DEPTH_PANEL_LEFT_MARGIN = 40
DEPTH_PANEL_RADIUS = 20

# 深度條
DEPTH_BAR_TOTAL_HEIGHT = 850
DEPTH_TICK_WIDTH = 4

# 刻度長度（px）
DEPTH_TICK_LEN_10M = 36
DEPTH_TICK_LEN_5M = 27
DEPTH_TICK_LEN_1M = 22

# 泡泡標籤
BUBBLE_WIDTH = 80
BUBBLE_HEIGHT = 45
BUBBLE_RADIUS = 10
BUBBLE_TAIL_WIDTH = 22
BUBBLE_TAIL_HEIGHT_RATIO = 0.5  # 泡泡小三角形高度 = BUBBLE_HEIGHT * 這個比例

# 泡泡顏色
BUBBLE_CURRENT_COLOR = (254, 168, 23, 255)  # 當前深度泡泡：橘黃色
BUBBLE_BEST_COLOR = (255, 255, 255, 255)    # 最大深度泡泡：白色
BUBBLE_TEXT_COLOR_DARK = (0, 0, 0, 255)
BUBBLE_TEXT_COLOR_LIGHT = (0, 0, 0, 255)

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
# Layout A helpers (Bottom parallelogram bar)
# ============================================================
def _draw_parallelogram(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, skew: int,
                        fill_rgba: tuple, outline_rgba: Optional[tuple] = None):
    """Draw a right-leaning parallelogram. skew > 0 means bottom edge shifts left."""
    p = [(x, y), (x + w, y), (x + w - skew, y + h), (x - skew, y + h)]
    draw.polygon(p, fill=fill_rgba)
    if outline_rgba is not None:
        draw.line(p + [p[0]], fill=outline_rgba, width=1)

def _safe_lower(s: str) -> str:
    try:
        return str(s).lower()
    except Exception:
        return ""

def _layout_a_defaults():
    return {
        "x_start": 80,
        "y_from_bottom": 110,
        "h": 50,
        "skew": 25,
        "gap": 20,
        "w": [165, 250, 135, 130, 125],
        "alpha": 0.65,
        "inner_pad": 14,
        "font_sizes": {"code": 32, "name": 32, "disc": 32, "time": 32, "depth": 32},
        "offsets": {
            "code": (-10, -10), 
            "flag": (-20, 0), 
            "name": (-10, -10), 
            "disc": (-10, -10), 
            "time": (-13, -10), 
            "depth": (-10, -10)
        },
    }


# ============================================================
# Layout C - Depth Module (v1)
# Moving scale + clipping window + fixed indicator
# ============================================================

@dataclass
class LayoutCDepthConfig:
    enabled: bool = True

    # Window (clipping)
    window_top: int = 1600
    window_bottom_margin: int = 80   # bottom = H - margin
    px_per_m: int = 20
    
    # Window fade (alpha gradient at top/bottom)
    fade_enable: bool = True
    fade_margin_px: int = 80          # 1600~1680, 1760~1840
    fade_edge_transparency: float = 0.95  # 95% transparent at the very edge

    # Scale
    depth_min_m: int = 0
    depth_max_m: int = 140
    scale_x: int = 80
    
    # Scale offset (global)
    scale_x_offset: int = 38   # +right / -left original = 40
    scale_y_offset: int = 0   # +down / -up
    scale_pad_top: int = 80
    scale_pad_bottom: int = 80

    tick_len_10m: int = 73
    tick_len_5m: int = 55
    tick_len_1m: int = 55
    tick_len_max: int = 73
    tick_w_10m: int = 5
    tick_w_5m: int = 3
    tick_w_1m: int = 3
    tick_w_max: int = 6
    max_label_font_size: int = 28

    tick_color: tuple = (220, 220, 220, 255)

    # Numbers
    num_left_margin: int = 45 # original = 40
    num_offset_x: int = 0     # +right / -left
    num_offset_y: int = -2     # +down / -up
    num_font_size: int = 28
    num_color: tuple = (220, 220, 220, 255)
    num_clip_padding_top: int = 6
    zero_top_pad_px: int = 6   # 0 若被切，上緣至少留 6px
    zero_num_offset_y: int = 0   # 只影響 m==0 的數字，負數往上
    
    # --- Special number offsets ---
    zero_num_offset_x: int = 10   # only for m == 0
    zero_num_offset_y: int = 0
    
    max_num_offset_x: int = 0    # only for MaxDepth label
    max_num_offset_y: int = 4

    # Indicator (fixed)
    depth_value_font_size: int = 110
    depth_unit_font_size: int = 65
    depth_value_color: tuple = (255, 215, 0, 255)
    unit_gap_px: int = 8
    value_y_offset: int = -36
    unit_offset_x: int = 0    # +right / -left
    unit_offset_y: int = 20    # +down / -up
    unit_follow_value: bool = True   # True: 跟著數字寬度動（目前行為）
    unit_x_fixed: int = 360            # 當 unit_follow_value=False 時使用的固定 X（0 表示未啟用/用預設）

    arrow_w: int = 24
    arrow_h: int = 20
    arrow_color: tuple = (220, 220, 220, 255)

    arrow_to_value_gap: int = 17  #original = 14
    value_x: int = 205
    arrow_y_offset: int = 35

from typing import Optional

def render_layout_c_depth_module(
    base_img: Image.Image,
    current_depth_m: float,
    cfg: LayoutCDepthConfig,
    font_path: Optional[str] = None,
    max_depth_m: Optional[float] = None,   # NEW
) -> Image.Image:

    """Render Layout C depth module onto base_img (RGBA PIL Image)."""
    if not cfg.enabled:
        return base_img

    W, H = base_img.size

    y0 = int(cfg.window_top)
    y1 = int(H - cfg.window_bottom_margin)
    indicator_y = (y0 + y1) // 2

    # ---------- Moving layer ----------
    pad_top = int(getattr(cfg, "scale_pad_top", 0))
    pad_bot = int(getattr(cfg, "scale_pad_bottom", 0))
    
    moving_h = H + pad_top + pad_bot
    moving = PILImage.new("RGBA", (W, moving_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(moving)

    # --- Fonts ---
    if font_path:
        # 刻度數字：沿用原本（RobotoCondensedBold）
        num_font = ImageFont.truetype(font_path, cfg.num_font_size)
    
        # 深度值 / 單位：Layout C 專用字體（nereus-bold）
        if LAYOUT_C_VALUE_FONT_PATH.exists():
            value_font = ImageFont.truetype(
                str(LAYOUT_C_VALUE_FONT_PATH),
                cfg.depth_value_font_size,
            )
            unit_font = ImageFont.truetype(
                str(LAYOUT_C_VALUE_FONT_PATH),
                cfg.depth_unit_font_size,
            )
        else:
            # fallback
            value_font = ImageFont.truetype(font_path, cfg.depth_value_font_size)
            unit_font = ImageFont.truetype(font_path, cfg.depth_unit_font_size)
    else:
        num_font = ImageFont.load_default()
        value_font = ImageFont.load_default()
        unit_font = ImageFont.load_default()
        
    # Decide scale max depth to display
    if max_depth_m is not None and np.isfinite(max_depth_m):
        depth_max_display = int(np.ceil(float(max_depth_m)))  # 無條件進位到個位數
        depth_max_display = max(cfg.depth_min_m, depth_max_display)
    else:
        depth_max_display = int(cfg.depth_max_m)

    for m in range(cfg.depth_min_m, depth_max_display + 1):
        y = pad_top + m * cfg.px_per_m + cfg.scale_y_offset

        # --- 判斷是否為 MaxDepth ---
        is_max = (m == depth_max_display)
    
        # --- 刻度線樣式 ---
        if is_max:
            w, L = cfg.tick_w_max, cfg.tick_len_max
        elif m % 10 == 0:
            w, L = cfg.tick_w_10m, cfg.tick_len_10m
        elif m % 5 == 0:
            w, L = cfg.tick_w_5m, cfg.tick_len_5m
        else:
            w, L = cfg.tick_w_1m, cfg.tick_len_1m
    
        # --- 刻度線位置 ---
        center_x = cfg.scale_x + cfg.scale_x_offset
        x1 = center_x - L // 2
        x2 = center_x + L // 2
    
        d.line(
            [(x1, y), (x2, y)],
            fill=cfg.tick_color,
            width=w,
        )
    
        # --- 是否顯示左側數字 ---
        if (m % 10 == 0) or is_max:
            txt = str(m)
    
            # MaxDepth 的字體（可選）
            if is_max and hasattr(cfg, "max_label_font_size"):
                try:
                    max_font = ImageFont.truetype(
                        font_path,
                        int(cfg.max_label_font_size),
                    ) if font_path else ImageFont.load_default()
                    use_font = max_font
                except Exception:
                    use_font = num_font
            else:
                use_font = num_font
    
            # --- base position ---
            num_x = cfg.num_left_margin + cfg.num_offset_x
            num_y_center = y + cfg.num_offset_y
            
            # --- special offsets ---
            if m == 0:
                num_x += int(getattr(cfg, "zero_num_offset_x", 0))
                num_y_center += int(getattr(cfg, "zero_num_offset_y", 0))
            
            if is_max:
                num_x += int(getattr(cfg, "max_num_offset_x", 0))
                num_y_center += int(getattr(cfg, "max_num_offset_y", 0))
            
            d.text(
                (int(num_x), int(num_y_center)),
                txt,
                font=use_font,
                fill=cfg.num_color,
                anchor="lm",   # left + middle
            )

    # Align current depth to indicator
    offset_y = int(round(indicator_y - (pad_top + current_depth_m * cfg.px_per_m)))

    moved = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    moved.alpha_composite(moving, (0, offset_y))

    # Clip window
    clipped = moved.crop((0, y0, W, y1))
    
    # Apply top/bottom fade (alpha gradient)
    if cfg.fade_enable and cfg.fade_margin_px > 0 and 0.0 <= cfg.fade_edge_transparency < 1.0:
        win_h = clipped.size[1]
        fade = int(cfg.fade_margin_px)
        fade = max(0, min(fade, win_h // 2))  # avoid overlap
    
        # Build opacity factors per row: center=1.0, edges=(1 - transparency)
        edge_opacity = 1.0 - float(cfg.fade_edge_transparency)  # e.g. 0.10
        factors = np.ones((win_h,), dtype=np.float32)
    
        if fade > 0:
            # top fade: from edge_opacity (at y=0) -> 1.0 (at y=fade)
            top = np.linspace(edge_opacity, 1.0, fade, dtype=np.float32)
            factors[:fade] = top
    
            # bottom fade: from 1.0 (at y=win_h-fade) -> edge_opacity (at y=win_h-1)
            bot = np.linspace(1.0, edge_opacity, fade, dtype=np.float32)
            factors[win_h - fade:] = bot
    
        # Multiply clipped alpha by factors row-wise
        a = np.array(clipped.getchannel("A"), dtype=np.float32)  # (H, W)
        a *= factors[:, None]
        a = np.clip(a, 0, 255).astype(np.uint8)
    
        clipped.putalpha(Image.fromarray(a, mode="L"))
    
    # Composite back into base image
    base_img.alpha_composite(clipped, (0, y0))

    # ---------- Fixed layer ----------
    draw = ImageDraw.Draw(base_img)

    depth_txt = f"{current_depth_m:.1f}"
    vb = draw.textbbox((0, 0), depth_txt, font=value_font)
    v_w = vb[2] - vb[0]
    v_h = vb[3] - vb[1]

    # Arrow (left pointing)
    value_center_y = indicator_y + cfg.value_y_offset
    arrow_cy = value_center_y + cfg.arrow_y_offset

    arrow_right_x = cfg.value_x - cfg.arrow_to_value_gap
    
    tri = [
        (arrow_right_x, arrow_cy - cfg.arrow_h // 2),
        (arrow_right_x, arrow_cy + cfg.arrow_h // 2),
        (arrow_right_x - cfg.arrow_w, arrow_cy),
    ]

    draw.polygon(tri, fill=cfg.arrow_color)

    # Depth value
    value_center_y = indicator_y + cfg.value_y_offset
    
    draw.text(
        (cfg.value_x, value_center_y - v_h // 2),
        depth_txt,
        font=value_font,
        fill=cfg.depth_value_color,
    )

    # Unit
    unit_txt = "m"
    ub = draw.textbbox((0, 0), unit_txt, font=unit_font)
    u_h = ub[3] - ub[1]
    
    if getattr(cfg, "unit_follow_value", True):
        unit_x = cfg.value_x + v_w + cfg.unit_gap_px
    else:
        # 固定 X：如果 unit_x_fixed 沒設（<=0），就用「數字最大可能寬度」當基準
        if getattr(cfg, "unit_x_fixed", 0) > 0:
            unit_x = cfg.unit_x_fixed
        else:
            # fallback：以 "88.8" 當最大寬度估計，避免 unit 太靠近數字
            v_max_bbox = draw.textbbox((0, 0), "88.8", font=value_font)
            v_max_w = v_max_bbox[2] - v_max_bbox[0]
            unit_x = cfg.value_x + v_max_w + cfg.unit_gap_px
    
    unit_x = int(unit_x + cfg.unit_offset_x)
    unit_y = int((value_center_y - u_h // 2) + cfg.unit_offset_y)
    
    draw.text(
        (unit_x, unit_y),
        "m",
        font=unit_font,
        fill=cfg.depth_value_color,
    )

    return base_img


def draw_layout_a_bottom_bar(
    overlay: PILImage.Image,
    assets_dir: Path,
    base_font_path: Path,
    nationality: str,
    diver_name: str,
    discipline: str,
    dive_time_s: Optional[float],
    depth_val: float,
    params: Optional[dict] = None,
):
    """Layout A: five parallelogram plates at bottom (code+flag / name / discipline / time / depth)."""
    cfg = _layout_a_defaults()
    if isinstance(params, dict):
        # shallow merge
        cfg.update({k: v for k, v in params.items() if k in cfg})
        if "font_sizes" in params and isinstance(params["font_sizes"], dict):
            cfg["font_sizes"].update(params["font_sizes"])
        if "offsets" in params and isinstance(params["offsets"], dict):
            cfg["offsets"].update(params["offsets"])

    W, H = overlay.size
    x = int(cfg["x_start"])
    y = int(H - cfg["y_from_bottom"] - cfg["h"])
    h = int(cfg["h"])
    skew = int(cfg["skew"])
    gap = int(cfg["gap"])
    w_list = list(cfg["w"])
    pad = int(cfg["inner_pad"])
    alpha = float(cfg.get("alpha", 0.20))
    panel_alpha = max(0, min(255, int(round(alpha * 255))))

    # panel fill: black with alpha
    fill = (0, 0, 0, panel_alpha)

    # draw plates
    draw = ImageDraw.Draw(overlay, "RGBA")
    xs = []
    cur_x = x
    for w in w_list:
        xs.append(cur_x)
        _draw_parallelogram(draw, cur_x, y, int(w), h, skew, fill)
        cur_x += int(w) + gap

    # prepare fonts
    def _load_font(size: int):
        try:
            return ImageFont.truetype(str(base_font_path), size=size)
        except Exception:
            return ImageFont.load_default()

    fs = cfg["font_sizes"]
    f_code = _load_font(int(fs.get("code", 34)))
    f_name = _load_font(int(fs.get("name", 34)))
    f_disc = _load_font(int(fs.get("disc", 34)))
    f_time = _load_font(int(fs.get("time", 34)))
    f_depth = _load_font(int(fs.get("depth", 34)))

    # nationality code + flag
    code3, _label = _infer_country_code_3(nationality)  # returns (code3, label)
    code3 = (code3 or "").upper()

    # flag image
    flag_img = None
    if code3:
        flags_dir = resolve_flags_dir(assets_dir)
        flag_img = _load_flag_png(flags_dir, code3)  # load from assets/flags
    # plate1: code left, flag right
    x1 = xs[0]
    w1 = int(w_list[0])
    off_code = cfg["offsets"].get("code", (0, 0))
    off_flag = cfg["offsets"].get("flag", (0, 0))

    code_text = code3 if code3 else (nationality.strip()[:3].upper() if nationality else "")
    # code position: left padding
    tx = x1 + pad + int(off_code[0])
    ty = y + (h - text_size(draw, code_text, f_code)[1]) // 2 + int(off_code[1])
    if code_text:
        draw.text((tx, ty), code_text, font=f_code, fill=(255, 255, 255, 255))

    if flag_img is not None:
        # scale flag to ~60% of height
        target_h = int(h * 0.62)
        if target_h <= 0:
            target_h = 1
        scale = target_h / float(flag_img.size[1])
        target_w = max(1, int(round(flag_img.size[0] * scale)))
        flag_resized = flag_img.resize((target_w, target_h), PILImage.LANCZOS)
        fx = x1 + w1 - pad - target_w + int(off_flag[0])
        fy = y + (h - target_h) // 2 + int(off_flag[1])
        overlay.alpha_composite(flag_resized.convert("RGBA"), (fx, fy))

    # plate2: name centered
    x2 = xs[1]; w2 = int(w_list[1])
    name = diver_name or ""
    off_name = cfg["offsets"].get("name", (0, 0))
    tw, th = text_size(draw, name, f_name)
    nx = x2 + (w2 - tw) // 2 + int(off_name[0])
    ny = y + (h - th) // 2 + int(off_name[1])
    if name:
        draw.text((nx, ny), name, font=f_name, fill=(255, 255, 255, 255))

    # plate3: discipline centered
    x3 = xs[2]; w3 = int(w_list[2])
    disc = discipline or ""
    off_disc = cfg["offsets"].get("disc", (0, 0))
    tw, th = text_size(draw, disc, f_disc)
    dx = x3 + (w3 - tw) // 2 + int(off_disc[0])
    dy = y + (h - th) // 2 + int(off_disc[1])
    if disc:
        draw.text((dx, dy), disc, font=f_disc, fill=(255, 255, 255, 255))

    # plate4: time mm:ss centered
    x4 = xs[3]; w4 = int(w_list[3])
    off_time = cfg["offsets"].get("time", (0, 0))
    ttxt = format_dive_time(dive_time_s) if dive_time_s is not None else ""
    tw, th = text_size(draw, ttxt, f_time)
    tx4 = x4 + (w4 - tw) // 2 + int(off_time[0])
    ty4 = y + (h - th) // 2 + int(off_time[1])
    if ttxt:
        draw.text((tx4, ty4), ttxt, font=f_time, fill=(255, 255, 255, 255))

    # plate5: depth with 1 decimal (e.g. 12.3 m)
    x5 = xs[4]; w5 = int(w_list[4])
    off_depth = cfg["offsets"].get("depth", (0, 0))
    
    d_val = float(depth_val) if depth_val is not None else 0.0
    d_val = max(0.0, d_val)
    
    dtxt = f"{d_val:.1f} m"
    
    tw, th = text_size(draw, dtxt, f_depth)
    dx5 = x5 + (w5 - tw) // 2 + int(off_depth[0])
    dy5 = y + (h - th) // 2 + int(off_depth[1])
    draw.text((dx5, dy5), dtxt, font=f_depth, fill=(255, 255, 255, 255))

    return overlay

# ============================================================
# 國碼 / 國旗工具
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
    print(f"[FLAG] Try load flag: {path}")
    if not path.exists():
        print(f"[FLAG] NOT FOUND: {path}")
        return None
    try:
        img = Image.open(path).convert("RGBA")
        print(f"[FLAG] Loaded OK: {path}")
        return img
    except Exception as e:
        print(f"[FLAG] ERROR loading {path}: {e}")
        return None


# ============================================================
# 右下角賽事資訊卡（Layout B）
# ============================================================

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
    - 黃色 Backplate 2：國旗 + 三碼 / 國籍文字 + 姓名 + 項目
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

        # ---------- 國旗 + 三碼國碼 / 國籍文字在黃板左側 ----------
        code3, country_label = _infer_country_code_3(nationality or "")
        flag_img = _load_flag_png(flags_dir, code3) if FLAG_ENABLE else None

        flag_right_x = b2_x  # 先給個預設值，下面如果有旗幟會覆蓋它

        if FLAG_ENABLE and flag_img is not None:
            margin_tb = int(FLAG_TOP_BOTTOM_MARGIN)
            left_off = int(FLAG_LEFT_OFFSET)

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
        else:
            # ❗ 沒有 flag 檔案時，改用「三碼國碼」優先，其次才是國籍文字
            label_text = None
            if code3:
                label_text = code3.upper()         # 例如 "TWN"
            elif country_label:
                label_text = str(country_label)    # 才退回 "Taiwan"

            if label_text:
                try:
                    font_nat = load_font(int(FLAG_ALPHA3_FONT_SIZE))
                except Exception:
                    font_nat = ImageFont.load_default()

                tw, th = text_size(draw, label_text, font_nat)
                tx = b2_x + int(FLAG_LEFT_OFFSET)
                ty = b2_y + (b2_h - th) // 2 + int(FLAG_ALPHA3_OFFSET_Y)
                draw.text(
                    (tx, ty),
                    label_text,
                    font=font_nat,
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

    tip_x = left_x              # 小三角形尖端（貼在背板右邊）
    base_x = left_x + tail_w    # 三角形與矩形交界
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
            (tip_x, center_y),       # 尖端（貼背板）
            (base_x, tri_y_top),     # 右上
            (base_x, tri_y_bot),     # 右下
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
    - 左邊黑色背板（100 x 980 px），上下置中
    - 深度條總高度 DEPTH_BAR_TOTAL_HEIGHT
    - 每 1 m 一個刻度，10m / 5m / 1m 不同長度
    - 每 10 m 顯示數字（在刻度左側）
    - 泡泡 1：當前深度
    - 泡泡 2：最大深度（show_best_bubble=True 時顯示）
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

    # --- 深度條 Y 範圍 ---
    bar_h = DEPTH_BAR_TOTAL_HEIGHT
    bar_y0 = (h - bar_h) // 2
    bar_y1 = bar_y0 + bar_h

    # 刻度最右邊對齊：離背板右側 10px
    tick_x_end = panel_x1 - 10

    max_d = max_depth_for_scale

    # 刻度文字字型
    try:
        tick_font = load_font(DEPTH_TICK_LABEL_FONT_SIZE)
    except Exception:
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

        # 刻度線
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
    except Exception:
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
    progress_callback=None,
    layout_params: Optional[dict] = None,                # 由 app.py 傳進來
):
    """
    progress_callback(p: float, message: str) 會被用來更新 Streamlit 進度條：
    - p: 0.0 ~ 1.0
    """

    flags_dir = resolve_flags_dir(assets_dir)

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
        best_time_global = float(times_d[best_idx])  # 最大深度發生的 time_s（log 的時間）
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
    except Exception:
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

    base_font_path = FONT_PATH  # Layout A uses this font path
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

        # ===== Layout A: bottom parallelogram bar =====
        if layout == "A":
            overlay = draw_layout_a_bottom_bar(
                overlay=overlay,
                assets_dir=assets_dir,
                base_font_path=base_font_path,
                nationality=nationality,
                diver_name=diver_name,
                discipline=discipline,
                dive_time_s=elapsed,
                depth_val=depth_val,
                params=layout_params,
            )

        # ===== 右上角 info 卡（深度 + 速率 + 動態時間）=====
        # 👉 只在「不是 Layout B」時才畫；Layout B 用右下 Board 2/3
        if layout not in ("A", "B", "C"):
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

        
        # ===== Layout C 專屬元件 (v1: Depth module only) =====
        if layout == "C":
            layout_c_cfg = LayoutCDepthConfig()
            overlay = render_layout_c_depth_module(
                base_img=overlay,
                current_depth_m=depth_val,
                cfg=layout_c_cfg,
                font_path=base_font_path,
                max_depth_m=best_depth,   # NEW：用你已算好的最大深度
            )

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
    import os
    import uuid
    import tempfile

    t_encode_start = time.perf_counter()
    update_progress(last_p["value"], "編碼影片中...")

    new_clip = None
    output_path = None
    tmp_audio_path = None

    try:
        new_clip = VideoClip(make_frame, duration=clip.duration)
        new_clip = new_clip.set_fps(clip.fps).set_audio(clip.audio)

        # 用唯一檔名，避免多人/多次渲染互相覆蓋
        output_path = Path(tempfile.gettempdir()) / f"dive_overlay_output_{uuid.uuid4().hex}.mp4"
        tmp_audio_path = str(Path(tempfile.gettempdir()) / f"dive_overlay_audio_{uuid.uuid4().hex}.m4a")

        new_clip.write_videofile(
            str(output_path),
            codec="libx264",
            fps=clip.fps,
        
            audio=True,
            audio_codec="aac",
            temp_audiofile=tmp_audio_path,
            remove_temp=True,
        
            # 降低雲端尖峰
            threads=1,
        
            # 更保守的 encoder 參數：ultrafast 先求穩
            ffmpeg_params=[
                "-movflags", "+faststart",
                "-preset", "ultrafast",
            ],
        )

        t_encode_end = time.perf_counter()
        print(f"[render_video] 編碼 / 寫檔耗時 {t_encode_end - t_encode_start:.2f} 秒")

        t1 = time.perf_counter()
        print(f"[render_video] 總耗時 {t1 - t0:.2f} 秒")

        update_progress(1.0, "影片產生完成！")
        return output_path

    finally:
        # 1) 一定要 close，否則 ffmpeg reader/proc 容易殘留
        try:
            if new_clip is not None:
                new_clip.close()
        except Exception:
            pass

        try:
            if clip is not None:
                clip.close()
        except Exception:
            pass

        # 2) 雙保險：如果 MoviePy 沒刪掉 temp audio，這裡補刪
        try:
            if tmp_audio_path and os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
        except Exception:
            pass
