# core/video_renderer.py
from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import VideoClip
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageFont, Image, ImageFilter
from dataclasses import dataclass
import time

# ============================================================
# Font paths
# ============================================================


# ==========================
# Layout C config dataclasses (minimal, required for this standalone script)
# ==========================
from dataclasses import dataclass
from typing import Tuple

@dataclass
class LayoutCShadowConfig:
    enabled: bool = True
    offset: Tuple[int, int] = (4, 4)
    blur_radius: int = 6
    color: Tuple[int, int, int, int] = (0, 0, 0, 120)

@dataclass
class LayoutCRateConfig:
    enabled: bool = True
    global_x: int = 720
    global_y: int = 60
    label_font_size: int = 48
    value_font_size: int = 120
    unit_font_size: int = 50
    decimals: int = 1
    label_ox: int = 50
    label_oy: int = 82
    arrow_ox: int = 12
    arrow_oy: int = 87
    value_ox: int = 50
    value_oy: int = 100
    unit_ox: int = -4
    unit_oy: int = 141
    unit_follow_value: bool = True
    unit_gap_px: int = 10

@dataclass
class LayoutCHeartRateConfig:
    enabled: bool = True
    global_x: int = 63
    global_y: int = 150
    icon_size: int = 100
    value_font_size: int = 120
    value_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    icon_ox: int = 0
    icon_oy: int = 0
    value_ox: int = 110
    value_oy: int = 10
    pulse_amp: float = 0.08

@dataclass
@dataclass
class LayoutCTimeConfig:
    enabled: bool = True
    global_x: int = 204
    global_y: int = 1743
    font_size: int = 75
    colon_font_size: int = 75
    color: Tuple[int, int, int, int] = (255, 255, 255, 255)

    # Offsets relative to (global_x, global_y)
    ox: int = 0
    oy: int = 0

    # Gap between "mm" ":" "ss" parts
    part_gap_px: int = 6

@dataclass
class LayoutCAllConfig:
    depth: 'LayoutCDepthConfig'
    rate: LayoutCRateConfig
    hr: LayoutCHeartRateConfig
    time: LayoutCTimeConfig
    shadow: LayoutCShadowConfig

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_PATH = BASE_DIR / "assets" / "fonts" / "RobotoCondensedBold.ttf"

# Layout C - Rate module font (Klavika)
LAYOUT_C_RATE_FONT_PATH = BASE_DIR / "assets" / "fonts" / "klavika-medium.otf"
if not LAYOUT_C_RATE_FONT_PATH.exists():
    print(f"[WARN] Layout C rate font NOT found: {LAYOUT_C_RATE_FONT_PATH}")
else:
    print(f"[INFO] Layout C rate font FOUND: {LAYOUT_C_RATE_FONT_PATH}")

# Layout C - Value font (Nereus Bold) for depth/rate numeric value
LAYOUT_C_VALUE_FONT_PATH = BASE_DIR / "assets" / "fonts" / "nereus-bold.ttf"

# Layout C Heart Rate icon (place the PNG at: assets/icons/heart_rate_icon.png)
LAYOUT_C_HR_ICON_REL = Path("icons") / "heart_rate_icon.png"
if not LAYOUT_C_VALUE_FONT_PATH.exists():
    print(f"[WARN] Layout C value font NOT found: {LAYOUT_C_VALUE_FONT_PATH}")
else:
    print(f"[INFO] Layout C value font FOUND: {LAYOUT_C_VALUE_FONT_PATH}")

print(f"[FONT] FONT_PATH = {FONT_PATH}")
if not FONT_PATH.exists():
    print(f"[WARN] Font file NOT found: {FONT_PATH}")
else:
    print(f"[INFO] Font file FOUND: {FONT_PATH}")

# ============================================================
# Timing alignment fine-tune
# ============================================================
# If you observe a systematic constant delay between the chosen alignment time
# (e.g., "align to surfacing time") and the displayed depth/time/rate reaching 0,
# adjust this value (seconds). Negative means the overlay uses data slightly earlier.
ALIGN_DISPLAY_CORRECTION_S = 1.0  # default: -1.0 to compensate ~0.9~1.0 s lag


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Unified font loader for the project default font (RobotoCondensedBold.ttf).
    """
    try:
        if FONT_PATH.exists():
            return ImageFont.truetype(str(FONT_PATH), int(size))
        else:
            print(f"[FONT LOAD] NOT FOUND, fallback to default. PATH={FONT_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load font {FONT_PATH} (size={size}): {e}")
    return ImageFont.load_default()


def resolve_flags_dir(assets_dir: Path) -> Path:
    """
    Try assets/flags and assets/Flags to avoid mac/linux case mismatch.
    """
    candidates = [assets_dir / "flags", assets_dir / "Flags"]
    for p in candidates:
        if p.exists():
            print(f"[FLAGS_DIR] Using {p}")
            return p
    print(f"[FLAGS_DIR] No flags dir found under {assets_dir}, fallback to {assets_dir}")
    return assets_dir


# --- Pillow ANTIALIAS patch (for newer Pillow versions) ---
if not hasattr(PILImage, "ANTIALIAS"):
    try:
        from PIL import Image as _ImgMod
        if hasattr(_ImgMod, "Resampling"):
            PILImage.ANTIALIAS = _ImgMod.Resampling.LANCZOS
        else:
            PILImage.ANTIALIAS = _ImgMod.LANCZOS
    except Exception:
        PILImage.ANTIALIAS = PILImage.BICUBIC


# --- Layout anchor config for generic info card ---
LAYOUT_CONFIG = {
    "A": {"anchor": "top_left"},
    "B": {"anchor": "top_right"},
    "C": {"anchor": "bottom_right"},
    "D": {"anchor": "top_right"},
}

# ============================================================
# Layout B: bottom-right athlete panel (yellow/black boards)
# ============================================================

BOARD2_ENABLE = True
BOARD2_WIDTH = 550
BOARD2_HEIGHT = 70
BOARD2_RADIUS = 13
BOARD2_COLOR = (254, 168, 23, 255)
BOARD2_LEFT = 480
BOARD2_BOTTOM = 175

BOARD3_ENABLE = True
BOARD3_WIDTH = 550
BOARD3_HEIGHT = 80
BOARD3_RADIUS = 13
BOARD3_COLOR = (0, 0, 0, 255)
BOARD3_LEFT = 480
BOARD3_BOTTOM = 115

BOARD3_RATE_FONT_SIZE = 34
BOARD3_TIME_FONT_SIZE = 34
BOARD3_TEXT_COLOR = (255, 255, 255, 255)

BOARD3_RATE_OFFSET_X = 20
BOARD3_RATE_OFFSET_Y = 0
BOARD3_TIME_OFFSET_X = 20
BOARD3_TIME_OFFSET_Y = 0

FLAG_ENABLE = True
FLAG_LEFT_OFFSET = 15
FLAG_TOP_BOTTOM_MARGIN = 15
FLAG_ALPHA3_TEXT_GAP = 6
FLAG_ALPHA3_FONT_SIZE = 34
FLAG_ALPHA3_FONT_COLOR = (0, 0, 0, 255)
FLAG_ALPHA3_OFFSET_Y = -8

COMP_DISC_OFFSET_RIGHT = 15
COMP_DISC_OFFSET_Y = -8
COMP_ALPHA3_OFFSET_X = 0

INFO_CARD_FONT_SIZE = 48
INFO_TEXT_OFFSET_X = 0
INFO_TEXT_OFFSET_Y = -7

DEPTH_TICK_LABEL_FONT_SIZE = 30
DEPTH_TICK_LABEL_OFFSET_X = 0
DEPTH_TICK_LABEL_OFFSET_Y = -8

BUBBLE_FONT_SIZE = 32
BUBBLE_TEXT_OFFSET_X = 0
BUBBLE_TEXT_OFFSET_Y = -10

COMP_NAME_FONT_SIZE = 34
COMP_SUB_FONT_SIZE = 34
COMP_CODE_FONT_SIZE = 34

COMP_NAME_OFFSET_X = 30
COMP_NAME_OFFSET_Y = -8
COMP_SUB_OFFSET_X = 0
COMP_SUB_OFFSET_Y = -8
COMP_CODE_OFFSET_X = 0
COMP_CODE_OFFSET_Y = -2

DEPTH_PANEL_WIDTH = 100
DEPTH_PANEL_HEIGHT = 980
DEPTH_PANEL_LEFT_MARGIN = 40
DEPTH_PANEL_RADIUS = 20

DEPTH_BAR_TOTAL_HEIGHT = 850
DEPTH_TICK_WIDTH = 4

DEPTH_TICK_LEN_10M = 36
DEPTH_TICK_LEN_5M = 27
DEPTH_TICK_LEN_1M = 22

BUBBLE_WIDTH = 80
BUBBLE_HEIGHT = 45
BUBBLE_RADIUS = 10
BUBBLE_TAIL_WIDTH = 22
BUBBLE_TAIL_HEIGHT_RATIO = 0.5

BUBBLE_CURRENT_COLOR = (254, 168, 23, 255)
BUBBLE_BEST_COLOR = (255, 255, 255, 255)
BUBBLE_TEXT_COLOR_DARK = (0, 0, 0, 255)
BUBBLE_TEXT_COLOR_LIGHT = (0, 0, 0, 255)

# ============================================================
# Utilities
# ============================================================

def text_size(draw_obj, text: str, font_obj):
    """Safe text size helper compatible with different Pillow versions."""
    try:
        bbox = draw_obj.textbbox((0, 0), text, font=font_obj)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        return font_obj.getsize(text)


def format_dive_time(seconds: float) -> str:
    """Seconds -> MM:SS"""
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
        "alpha": 0.75,

        "shadow_enable": True,
        "shadow_dx": 5,
        "shadow_dy": 8,
        "shadow_blur": 10,
        "shadow_alpha": 90,

        "inner_pad": 14,
        "font_sizes": {"code": 32, "name": 32, "disc": 32, "time": 32, "depth": 32},
        "offsets": {
            "code": (-10, -10),
            "flag": (-20, 0),
            "name": (-10, -10),
            "disc": (-10, -10),
            "time": (-13, -10),
            "depth": (-10, -10),
        },
    }


# ============================================================
# Layout C - Depth Module
# ============================================================

@dataclass
class LayoutCDepthConfig:
    enabled: bool = True

    window_top: int = 1600
    window_bottom_margin: int = 80
    px_per_m: int = 20

    fade_enable: bool = True
    fade_margin_px: int = 70
    fade_edge_transparency: float = 0.95

    depth_min_m: int = 0
    depth_max_m: int = 140
    scale_x: int = 80

    scale_x_offset: int = 38
    scale_y_offset: int = 0
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

    num_left_margin: int = 45
    num_offset_x: int = 0
    num_offset_y: int = -2
    num_font_size: int = 28
    num_color: tuple = (220, 220, 220, 255)

    zero_num_offset_x: int = 10
    zero_num_offset_y: int = 0
    max_num_offset_x: int = 0
    max_num_offset_y: int = 4

    depth_value_font_size: int = 110
    depth_unit_font_size: int = 65
    depth_value_color: tuple = (255, 215, 0, 255)
    unit_gap_px: int = 8
    value_y_offset: int = -36
    unit_offset_x: int = 0
    unit_offset_y: int = 20
    unit_follow_value: bool = True
    unit_x_fixed: int = 360

    arrow_w: int = 24
    arrow_h: int = 20
    arrow_color: tuple = (220, 220, 220, 255)

    arrow_to_value_gap: int = 17
    value_x: int = 205
    arrow_y_offset: int = 35


def render_layout_c_depth_module(
    base_img: Image.Image,
    current_depth_m: float,
    cfg: LayoutCDepthConfig,
    font_path: Optional[str] = None,
    max_depth_m: Optional[float] = None,
) -> Image.Image:
    """
    Render Layout C depth module with LOCAL shadow rendering.
    Key optimizations:
      - Render only within a limited module width (instead of full frame width).
      - Avoid full-frame intermediate layers; operate on the depth-window region only.
      - Apply shadow only on the module canvas, not on a full-frame layer.
    """
    if not cfg.enabled:
        return base_img

    out = base_img.copy().convert("RGBA")
    W_full, H_full = out.size

    y0 = int(cfg.window_top)
    y1 = int(H_full - cfg.window_bottom_margin)
    win_h = max(1, y1 - y0)
    indicator_y = win_h // 2

    pad_top = int(getattr(cfg, "scale_pad_top", 0))
    pad_bot = int(getattr(cfg, "scale_pad_bottom", 0))

    # Fonts
    if font_path:
        num_font = ImageFont.truetype(str(font_path), cfg.num_font_size)
        if LAYOUT_C_VALUE_FONT_PATH.exists():
            value_font = ImageFont.truetype(str(LAYOUT_C_VALUE_FONT_PATH), cfg.depth_value_font_size)
            unit_font = ImageFont.truetype(str(LAYOUT_C_VALUE_FONT_PATH), cfg.depth_unit_font_size)
        else:
            value_font = ImageFont.truetype(str(font_path), cfg.depth_value_font_size)
            unit_font = ImageFont.truetype(str(font_path), cfg.depth_unit_font_size)
    else:
        num_font = ImageFont.load_default()
        value_font = ImageFont.load_default()
        unit_font = ImageFont.load_default()

    # Determine display depth max
    if max_depth_m is not None and np.isfinite(max_depth_m):
        depth_max_display = int(np.ceil(float(max_depth_m)))
        depth_max_display = max(cfg.depth_min_m, depth_max_display)
    else:
        depth_max_display = int(cfg.depth_max_m)

    # Estimate module width (limit heavy pixel ops)
    meas = PILImage.new("RGBA", (10, 10), (0, 0, 0, 0))
    md = ImageDraw.Draw(meas)
    depth_txt = f"{current_depth_m:.{int(getattr(cfg, 'depth_decimals', 0))}f}"
    vb = _try_render_textbbox(md, depth_txt, value_font)
    value_w = vb[2] - vb[0]
    ub = _try_render_textbbox(md, "m", unit_font)
    unit_w = ub[2] - ub[0]

    rightmost = max(
        int(getattr(cfg, "unit_x_fixed", int(cfg.unit_x_fixed))),
        int(cfg.value_x) + int(getattr(cfg, "value_extra_right_pad", 0)) + value_w + int(getattr(cfg, "unit_gap_px", 8)) + unit_w + 40,
        int(cfg.scale_x) + int(getattr(cfg, "scale_x_offset", 0)) + int(getattr(cfg, "tick_w_max", 40)) + 120,
    )
    module_w = int(min(W_full, max(200, rightmost)))

    # --- 1) Build moving scale canvas (module_w only) ---
    # Height needs to cover all tick positions. We draw from depth_min..depth_max_display
    # so the max y is pad_top + depth_max_display*px_per_m + offsets.
    max_tick_y = pad_top + depth_max_display * int(cfg.px_per_m) + int(getattr(cfg, "scale_y_offset", 0)) + 10
    moving_h = int(max(win_h + pad_top + pad_bot + 10, max_tick_y + pad_bot + 10))
    moving = PILImage.new("RGBA", (module_w, moving_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(moving)

    # Draw ticks + numbers
    for m in range(int(cfg.depth_min_m), int(depth_max_display) + 1):
        y = pad_top + m * int(cfg.px_per_m) + int(getattr(cfg, "scale_y_offset", 0))
        is_max = (m == int(depth_max_display))

        if is_max:
            tick_w = int(getattr(cfg, "tick_w_max", getattr(cfg, "tick_w", 26)))
            tick_l = int(getattr(cfg, "tick_l_max", getattr(cfg, "tick_l", 4)))
        else:
            tick_w = int(getattr(cfg, "tick_w", 26))
            tick_l = int(getattr(cfg, "tick_l", 4))

        x0_tick = int(cfg.scale_x) + int(getattr(cfg, "scale_x_offset", 0))
        x1_tick = x0_tick + tick_w

        # Line
        d.rectangle([x0_tick, y - tick_l, x1_tick, y + tick_l], fill=cfg.tick_color)

        # Numbers (every num_step)
        num_step = int(getattr(cfg, "num_step", 5))
        show_num = (m % num_step == 0)

        if show_num:
            txt = str(int(m))
            num_x = x1_tick + int(getattr(cfg, "num_gap_x", 10))
            num_y_center = y

            if m == 0:
                num_x += int(getattr(cfg, "zero_num_offset_x", 0))
                num_y_center += int(getattr(cfg, "zero_num_offset_y", 0))
            if is_max:
                num_x += int(getattr(cfg, "max_num_offset_x", 0))
                num_y_center += int(getattr(cfg, "max_num_offset_y", 0))

            d.text(
                (int(num_x), int(num_y_center)),
                txt,
                font=num_font,
                fill=cfg.num_color,
                anchor="lm",
            )

    # --- 2) Crop window aligned so current depth is at indicator ---
    crop_start = int(round(pad_top + float(current_depth_m) * float(cfg.px_per_m) + int(getattr(cfg, "scale_y_offset", 0)) - indicator_y))
    crop_end = crop_start + win_h

    clipped = PILImage.new("RGBA", (module_w, win_h), (0, 0, 0, 0))
    src_top = max(0, crop_start)
    src_bot = min(moving_h, crop_end)
    if src_bot > src_top:
        part = moving.crop((0, src_top, module_w, src_bot))
        dst_y = src_top - crop_start
        clipped.paste(part, (0, int(dst_y)), part)

    # Fade edges (on clipped)
    if cfg.fade_enable and int(getattr(cfg, "fade_margin_px", 0)) > 0 and 0.0 <= float(cfg.fade_edge_transparency) < 1.0:
        win_h2 = clipped.size[1]
        fade = int(cfg.fade_margin_px)
        edge_alpha = int(round(255 * float(cfg.fade_edge_transparency)))
        mask = PILImage.new("L", (module_w, win_h2), 255)
        md2 = ImageDraw.Draw(mask)

        # Top fade
        for i in range(min(fade, win_h2)):
            a = int(edge_alpha + (255 - edge_alpha) * (i / max(1, fade - 1)))
            md2.rectangle([0, i, module_w, i], fill=a)

        # Bottom fade
        for i in range(min(fade, win_h2)):
            y = win_h2 - 1 - i
            a = int(edge_alpha + (255 - edge_alpha) * (i / max(1, fade - 1)))
            md2.rectangle([0, y, module_w, y], fill=a)

        clipped.putalpha(mask)

    # --- 3) Compose module canvas (scale + value + arrow + unit) ---
    layer = PILImage.new("RGBA", (module_w, win_h), (0, 0, 0, 0))
    layer.alpha_composite(clipped, (0, 0))
    ld = ImageDraw.Draw(layer)

    # Value text + arrow (centered around indicator)
    depth_dec = int(getattr(cfg, "depth_decimals", 0))
    depth_txt = f"{float(current_depth_m):.{depth_dec}f}"
    value_x = int(getattr(cfg, "value_x", 205))
    value_y = int(indicator_y + int(getattr(cfg, "value_y_offset", -36)))

    # Arrow beside value (optional, uses cfg.arrow_* settings)
    arrow_w = int(getattr(cfg, "arrow_w", 24))
    arrow_h = int(getattr(cfg, "arrow_h", 20))
    arrow_gap = int(getattr(cfg, "arrow_to_value_gap", 17))
    arrow_y = int(indicator_y + int(getattr(cfg, "arrow_y_offset", 35)))

    # Draw arrow (down by default; direction can be overridden by cfg if you later choose)
    ax = value_x - arrow_gap - arrow_w
    ay = arrow_y
    tri = [(ax, ay), (ax + arrow_w, ay), (ax + arrow_w // 2, ay + arrow_h)]
    ld.polygon(tri, fill=getattr(cfg, "arrow_color", (220, 220, 220, 255)))

    # Draw value
    ld.text((value_x, value_y), depth_txt, font=value_font, fill=cfg.depth_value_color, anchor="lm")

    # Unit
    if bool(getattr(cfg, "unit_follow_value", True)):
        vb2 = _try_render_textbbox(ld, depth_txt, value_font)
        value_w2 = vb2[2] - vb2[0]
        unit_x = value_x + value_w2 + int(getattr(cfg, "unit_gap_px", 8)) + int(getattr(cfg, "unit_offset_x", 0))
    else:
        unit_x = int(getattr(cfg, "unit_x_fixed", 360)) + int(getattr(cfg, "unit_offset_x", 0))
    unit_y = int(indicator_y + int(getattr(cfg, "unit_offset_y", 20)))
    ld.text((unit_x, unit_y), "m", font=unit_font, fill=cfg.depth_value_color, anchor="lm")

    # --- 4) Shadow only on module canvas and composite back ---
    if LAYOUT_C_SHADOW_ENABLE:
        shadowed, pad_x, pad_y = _apply_shadow_layer_local(
            layer,
            offset=(LAYOUT_C_SHADOW_OX, LAYOUT_C_SHADOW_OY),
            blur_radius=LAYOUT_C_SHADOW_BLUR,
            shadow_color=LAYOUT_C_SHADOW_COLOR,
        )
        out = _alpha_composite_at(out, shadowed, 0 - pad_x, y0 - pad_y)
    else:
        out = _alpha_composite_at(out, layer, 0, y0)

    return out
def get_layout_c_config() -> LayoutCAllConfig:
    """Single source of truth for Layout C defaults."""
    shadow_cfg = LayoutCShadowConfig(
        enabled=True, # original = True
        offset=(4, 4),
        blur_radius=6,
        color=(0, 0, 0, 120),
    )

    # Depth config is mostly driven by LayoutCDepthConfig defaults (tuned in dataclass)
    depth_cfg = LayoutCDepthConfig()

    # Rate config default (your tuned values)
    rate_cfg = LayoutCRateConfig(
        enabled=True, # original = True
        global_x=720,
        global_y=60,
        label_font_size=48,
        value_font_size=120,
        unit_font_size=50,
        decimals=1,
        label_ox=50, label_oy=82,
        arrow_ox=12, arrow_oy=87,
        value_ox=50, value_oy=100,
        unit_ox=-4, unit_oy=141,
        unit_follow_value=True,
        unit_gap_px=10,
    )

    # Heart rate config default (your tuned values)
    hr_cfg = LayoutCHeartRateConfig(
        enabled=True, # original = True
        global_x=63,
        global_y=150,
        icon_size=100,
        value_font_size=120,
        value_color=(255, 255, 255, 255),
        icon_ox=0,
        icon_oy=0,
        value_ox=110,
        value_oy=10,
        pulse_amp=0.08,
    )


    # Time config (Layout C): mm:ss under depth value (":" uses base font)
    time_cfg = LayoutCTimeConfig(
        enabled=True,
    
        global_x=204,
        global_y=1743,
    
        font_size=75,
        color=(255, 255, 255, 255),
    
        # ":" base font size (tunable)
        colon_font_size=50,
    
        ox=0,
        oy=-4,
        part_gap_px=0,
    )

    return LayoutCAllConfig(depth=depth_cfg, rate=rate_cfg, hr=hr_cfg, time=time_cfg, shadow=shadow_cfg)

# Default Layout C shadow parameters (used by all Layout C modules)
_layout_c_shadow_defaults = get_layout_c_config().shadow
LAYOUT_C_SHADOW_ENABLED = bool(_layout_c_shadow_defaults.enabled)
LAYOUT_C_SHADOW_OFFSET = tuple(_layout_c_shadow_defaults.offset)
LAYOUT_C_SHADOW_BLUR = int(_layout_c_shadow_defaults.blur_radius)
LAYOUT_C_SHADOW_COLOR = tuple(_layout_c_shadow_defaults.color)
# Backwards-compatible aliases (some editors/linters expect these names)
LAYOUT_C_SHADOW_ENABLE = LAYOUT_C_SHADOW_ENABLED
LAYOUT_C_SHADOW_OX = int(LAYOUT_C_SHADOW_OFFSET[0])
LAYOUT_C_SHADOW_OY = int(LAYOUT_C_SHADOW_OFFSET[1])

# Font path aliases / fallbacks
DEFAULT_FONT_PATH = str(FONT_PATH)
LAYOUT_C_HR_VALUE_FONT_PATH = str(LAYOUT_C_VALUE_FONT_PATH if 'LAYOUT_C_VALUE_FONT_PATH' in globals() and Path(LAYOUT_C_VALUE_FONT_PATH).exists() else FONT_PATH)




def _load_font_path(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        if path and Path(path).exists():
            return ImageFont.truetype(str(path), int(size))
    except Exception:
        pass
    return ImageFont.load_default()

# ==========================================================
# Shadow helper for Layout C modules
# ==========================================================
from PIL import ImageFilter


def _apply_shadow_layer_local(
    img: PILImage.Image,
    offset: tuple = (4, 4),
    blur_radius: int = 6,
    shadow_color=(0, 0, 0, 120),
) -> tuple[PILImage.Image, int, int]:
    """
    Apply a drop shadow to the non-transparent area of `img` on a LOCAL (small) canvas.

    Returns:
      (shadowed_img, pad_x, pad_y)
    Where:
      - shadowed_img is larger than `img` due to padding for blur and offset
      - pad_x/pad_y indicate where the original `img`'s top-left is located inside shadowed_img
        so that callers can paste shadowed_img at (global_x - pad_x, global_y - pad_y) to align.
    """
    img = img.convert("RGBA")
    w, h = img.size
    ox, oy = int(offset[0]), int(offset[1])
    br = max(0, int(blur_radius))

    # Padding to avoid blur clipping.
    pad = max(2, br * 3 + max(abs(ox), abs(oy)) + 2)
    W = w + pad * 2
    H = h + pad * 2

    # Shadow mask from alpha
    alpha = img.split()[-1]
    shadow_mask = PILImage.new("L", (W, H), 0)
    shadow_mask.paste(alpha, (pad, pad))

    # Colored shadow layer
    shadow_rgba = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow_rgba_draw = ImageDraw.Draw(shadow_rgba)
    shadow_rgba_draw.bitmap((0, 0), shadow_mask, fill=shadow_color)

    if br > 0:
        shadow_rgba = shadow_rgba.filter(ImageFilter.GaussianBlur(br))

    # Composite shadow (with offset) + original
    out = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(shadow_rgba, (ox, oy))

    tmp = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    tmp.paste(img, (pad, pad), img)
    out.alpha_composite(tmp, (0, 0))

    return out, pad, pad


def _alpha_composite_at(dst: PILImage.Image, src: PILImage.Image, x: int, y: int) -> PILImage.Image:
    """
    Alpha composite `src` onto `dst` at (x, y). Safe for out-of-bounds; clips automatically.
    """
    out = dst
    try:
        out.alpha_composite(src, (int(x), int(y)))
    except TypeError:
        # Older Pillow fallback: manual paste with mask
        out.paste(src, (int(x), int(y)), src)
    return out
def _try_render_textbbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    try:
        return draw.textbbox((0, 0), text, font=font)
    except Exception:
        w, h = font.getsize(text)
        return (0, 0, w, h)


def render_layout_c_rate_module(
    base_img: PILImage.Image,
    speed_mps_signed: float,
    cfg: LayoutCRateConfig,
    *,
    is_descent_override: Optional[bool] = None,
) -> PILImage.Image:
    """
    Layout C Rate module (LOCAL shadow rendering):
      - Render onto a small local canvas (module bbox) instead of a full-frame layer.
      - Apply shadow on the small canvas only (major speed-up).
    """
    if not cfg.enabled:
        return base_img

    out = base_img.copy().convert("RGBA")

    # Determine direction
    if is_descent_override is not None:
        is_descent = bool(is_descent_override)
    else:
        is_descent = float(speed_mps_signed) >= 0.0

    label_txt = "Descent Rate" if is_descent else "Ascent Rate"
    label_color = cfg.label_color if is_descent else (255, 184, 166, 255)  # #FFB8A6

    v = abs(float(speed_mps_signed))
    value_txt = f"{v:.{int(cfg.decimals)}f}"
    unit_txt = "m/s"

    label_font = _load_font_path(LAYOUT_C_RATE_FONT_PATH, cfg.label_font_size)
    value_font = _load_font_path(LAYOUT_C_VALUE_FONT_PATH, cfg.value_font_size)
    unit_font = _load_font_path(DEFAULT_FONT_PATH, cfg.unit_font_size)

    # --- Measure ---
    meas = PILImage.new("RGBA", (10, 10), (0, 0, 0, 0))
    md = ImageDraw.Draw(meas)

    lb = _try_render_textbbox(md, label_txt, label_font)
    label_w, label_h = lb[2] - lb[0], lb[3] - lb[1]

    vb = _try_render_textbbox(md, value_txt, value_font)
    value_w, value_h = vb[2] - vb[0], vb[3] - vb[1]

    ub = _try_render_textbbox(md, unit_txt, unit_font)
    unit_w, unit_h = ub[2] - ub[0], ub[3] - ub[1]

    aw, ah = int(cfg.arrow_w), int(cfg.arrow_h)

    # Compute unit X if it follows value
    if bool(getattr(cfg, "unit_follow_value", True)):
        unit_x = int(cfg.value_ox) + value_w + int(getattr(cfg, "unit_gap_px", 10))
    else:
        unit_x = int(getattr(cfg, "unit_x_fixed", int(cfg.unit_ox)))

    # Local canvas extents
    min_x = min(int(cfg.arrow_ox), int(cfg.label_ox), int(cfg.value_ox), int(cfg.unit_ox), unit_x, 0)
    min_y = min(int(cfg.arrow_oy), int(cfg.label_oy), int(cfg.value_oy), int(cfg.unit_oy), 0)

    max_x = max(
        int(cfg.arrow_ox) + aw,
        int(cfg.label_ox) + label_w,
        int(cfg.value_ox) + value_w,
        unit_x + unit_w,
    )
    max_y = max(
        int(cfg.arrow_oy) + ah,
        int(cfg.label_oy) + label_h,
        int(cfg.value_oy) + value_h,
        int(cfg.unit_oy) + unit_h,
    )

    pad_inner = 6
    w = int(max_x - min_x + pad_inner * 2)
    h = int(max_y - min_y + pad_inner * 2)

    # Local origin shift so all coordinates are positive
    sx = -int(min_x) + pad_inner
    sy = -int(min_y) + pad_inner

    layer = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Arrow triangle
    ax = sx + int(cfg.arrow_ox)
    ay = sy + int(cfg.arrow_oy)
    if is_descent:
        tri = [(ax, ay), (ax + aw, ay), (ax + aw // 2, ay + ah)]          # down
    else:
        tri = [(ax + aw // 2, ay), (ax, ay + ah), (ax + aw, ay + ah)]     # up
    draw.polygon(tri, fill=cfg.arrow_color)

    # Label
    draw.text(
        (sx + int(cfg.label_ox), sy + int(cfg.label_oy)),
        label_txt,
        font=label_font,
        fill=label_color,
        anchor="lm",
    )

    # Value
    draw.text(
        (sx + int(cfg.value_ox), sy + int(cfg.value_oy)),
        value_txt,
        font=value_font,
        fill=cfg.value_color,
        anchor="lm",
    )

    # Unit
    draw.text(
        (sx + int(unit_x), sy + int(cfg.unit_oy)),
        unit_txt,
        font=unit_font,
        fill=cfg.unit_color,
        anchor="lm",
    )

    # Shadow (local only)
    if LAYOUT_C_SHADOW_ENABLE and bool(getattr(cfg, "shadow_enable", True)):
        shadowed, pad_x, pad_y = _apply_shadow_layer_local(
            layer,
            offset=(LAYOUT_C_SHADOW_OX, LAYOUT_C_SHADOW_OY),
            blur_radius=LAYOUT_C_SHADOW_BLUR,
            shadow_color=LAYOUT_C_SHADOW_COLOR,
        )
        gx = int(cfg.global_x) - pad_x
        gy = int(cfg.global_y) - pad_y
        out = _alpha_composite_at(out, shadowed, gx, gy)
    else:
        out = _alpha_composite_at(out, layer, int(cfg.global_x), int(cfg.global_y))

    return out
def _resize_icon(base_icon: PILImage.Image, scale: float, size: int) -> PILImage.Image:
    s = max(1, int(round(float(size) * float(scale))))
    try:
        return base_icon.resize((s, s), resample=PILImage.BICUBIC)
    except Exception:
        return base_icon.resize((s, s))



def render_layout_c_heart_rate_module(
    base_img: PILImage.Image,
    *,
    hr_text: str,
    cfg: LayoutCHeartRateConfig,
    assets_dir: Path,
    pulse_scale: float = 1.0,
    show_value: bool = True,
    show_icon: bool = True,
) -> PILImage.Image:
    """
    Layout C Heart Rate module (LOCAL shadow rendering).
    Renders onto a small canvas and composites at cfg.global_x/cfg.global_y.
    """
    if not cfg.enabled:
        return base_img
    if not show_icon:
        return base_img

    out = base_img.copy().convert("RGBA")

    icon_path = (Path(assets_dir) / LAYOUT_C_HR_ICON_REL)
    if not icon_path.exists():
        return out

    icon_base = PILImage.open(str(icon_path)).convert("RGBA")

    gx = int(cfg.global_x)
    gy = int(cfg.global_y)
    icon_size = int(cfg.icon_size)

    # Fonts
    value_font = _load_font_path(LAYOUT_C_HR_VALUE_FONT_PATH, int(cfg.value_font_size))

    # Measure value text
    meas = PILImage.new("RGBA", (10, 10), (0, 0, 0, 0))
    md = ImageDraw.Draw(meas)
    vb = _try_render_textbbox(md, str(hr_text), value_font)
    value_w, value_h = vb[2] - vb[0], vb[3] - vb[1]

    # Local extents
    min_x = min(int(cfg.icon_ox), int(cfg.value_ox), 0)
    min_y = min(int(cfg.icon_oy), int(cfg.value_oy), 0)

    max_x = max(int(cfg.icon_ox) + icon_size, int(cfg.value_ox) + value_w)
    max_y = max(int(cfg.icon_oy) + icon_size, int(cfg.value_oy) + value_h)

    pad_inner = 6
    w = int(max_x - min_x + pad_inner * 2)
    h = int(max_y - min_y + pad_inner * 2)
    sx = -int(min_x) + pad_inner
    sy = -int(min_y) + pad_inner

    layer = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Icon (animated scale)
    icon_scaled = _resize_icon(icon_base, float(pulse_scale), icon_size)
    ox = (icon_size - icon_scaled.size[0]) // 2
    oy = (icon_size - icon_scaled.size[1]) // 2
    ix = sx + int(cfg.icon_ox) + ox
    iy = sy + int(cfg.icon_oy) + oy
    layer.paste(icon_scaled, (int(ix), int(iy)), icon_scaled)

    # Value text
    if show_value:
        draw.text(
            (sx + int(cfg.value_ox), sy + int(cfg.value_oy)),
            str(hr_text),
            font=value_font,
            fill=cfg.value_color,
            anchor="lm",
        )

    # Shadow (local only)
    if LAYOUT_C_SHADOW_ENABLE:
        shadowed, pad_x, pad_y = _apply_shadow_layer_local(
            layer,
            offset=(LAYOUT_C_SHADOW_OX, LAYOUT_C_SHADOW_OY),
            blur_radius=LAYOUT_C_SHADOW_BLUR,
            shadow_color=LAYOUT_C_SHADOW_COLOR,
        )
        out = _alpha_composite_at(out, shadowed, gx - pad_x, gy - pad_y)
    else:
        out = _alpha_composite_at(out, layer, gx, gy)

    return out
def _format_mmss(seconds: float) -> str:
    s = max(0, int(math.floor(float(seconds) + 1e-6)))
    mm = s // 60
    ss = s % 60
    return f"{mm:02d}:{ss:02d}"



def render_layout_c_time_module(
    base_img: PILImage.Image,
    *,
    time_s: float,
    cfg: LayoutCTimeConfig,
) -> PILImage.Image:
    """
    Render Layout C dive time (mm:ss) with LOCAL shadow rendering.
    """
    if not cfg.enabled:
        return base_img

    out = base_img.copy().convert("RGBA")

    text_mmss = _format_mmss(time_s)
    mm_txt, ss_txt = text_mmss.split(":")
    colon_txt = ":"

    nereus_font = _load_font_path(LAYOUT_C_VALUE_FONT_PATH, int(cfg.font_size))
    colon_size = int(cfg.colon_font_size) if cfg.colon_font_size else int(cfg.font_size)
    colon_font = _load_font_path(FONT_PATH, colon_size)  # base font so ":" renders

    # Measure
    meas = PILImage.new("RGBA", (10, 10), (0, 0, 0, 0))
    md = ImageDraw.Draw(meas)
    b_mm = _try_render_textbbox(md, mm_txt, nereus_font)
    b_col = _try_render_textbbox(md, colon_txt, colon_font)
    b_ss = _try_render_textbbox(md, ss_txt, nereus_font)

    w_mm, h_mm = b_mm[2]-b_mm[0], b_mm[3]-b_mm[1]
    w_col, h_col = b_col[2]-b_col[0], b_col[3]-b_col[1]
    w_ss, h_ss = b_ss[2]-b_ss[0], b_ss[3]-b_ss[1]

    gap = int(getattr(cfg, "part_gap_px", 6))
    total_w = w_mm + gap + w_col + gap + w_ss
    total_h = max(h_mm, h_col, h_ss)

    pad_inner = 6
    layer = PILImage.new("RGBA", (total_w + pad_inner*2, total_h + pad_inner*2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    x = pad_inner
    y = pad_inner

    # Draw mm
    draw.text((x, y + (total_h - h_mm)//2), mm_txt, font=nereus_font, fill=cfg.color, anchor="lt")
    x += w_mm + gap
    # Draw colon
    draw.text((x, y + (total_h - h_col)//2), colon_txt, font=colon_font, fill=cfg.color, anchor="lt")
    x += w_col + gap
    # Draw ss
    draw.text((x, y + (total_h - h_ss)//2), ss_txt, font=nereus_font, fill=cfg.color, anchor="lt")

    gx = int(cfg.global_x) + int(cfg.ox)
    gy = int(cfg.global_y) + int(cfg.oy)

    if LAYOUT_C_SHADOW_ENABLE:
        shadowed, pad_x, pad_y = _apply_shadow_layer_local(
            layer,
            offset=(LAYOUT_C_SHADOW_OX, LAYOUT_C_SHADOW_OY),
            blur_radius=LAYOUT_C_SHADOW_BLUR,
            shadow_color=LAYOUT_C_SHADOW_COLOR,
        )
        out = _alpha_composite_at(out, shadowed, gx - pad_x, gy - pad_y)
    else:
        out = _alpha_composite_at(out, layer, gx, gy)

    return out
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

    fill = (12, 12, 12, panel_alpha)
    draw = ImageDraw.Draw(overlay, "RGBA")

    polys = []
    xs = []
    cur_x = x
    for w in w_list:
        xs.append(cur_x)
        ww = int(w)
        p = [(cur_x, y), (cur_x + ww, y), (cur_x + ww - skew, y + h), (cur_x - skew, y + h)]
        polys.append(p)
        cur_x += ww + gap

    if bool(cfg.get("shadow_enable", False)):
        shadow_dx = int(cfg.get("shadow_dx", 0))
        shadow_dy = int(cfg.get("shadow_dy", 6))
        shadow_blur = int(cfg.get("shadow_blur", 8))
        shadow_alpha = int(cfg.get("shadow_alpha", 110))
        shadow_alpha = max(0, min(255, shadow_alpha))

        if shadow_alpha > 0 and (shadow_dx != 0 or shadow_dy != 0 or shadow_blur > 0):
            shadow_layer = PILImage.new("RGBA", overlay.size, (0, 0, 0, 0))
            sd = ImageDraw.Draw(shadow_layer, "RGBA")
            shadow_fill = (0, 0, 0, shadow_alpha)
            for p in polys:
                p2 = [(px + shadow_dx, py + shadow_dy) for (px, py) in p]
                sd.polygon(p2, fill=shadow_fill)
            if shadow_blur > 0:
                shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
            overlay.alpha_composite(shadow_layer)

    for p in polys:
        draw.polygon(p, fill=fill)

    def _load_font(size: int):
        try:
            return ImageFont.truetype(str(base_font_path), size=int(size))
        except Exception:
            return ImageFont.load_default()

    fs = cfg["font_sizes"]
    f_code = _load_font(int(fs.get("code", 34)))
    f_name = _load_font(int(fs.get("name", 34)))
    f_disc = _load_font(int(fs.get("disc", 34)))
    f_time = _load_font(int(fs.get("time", 34)))
    f_depth = _load_font(int(fs.get("depth", 34)))

    code3, _label = _infer_country_code_3(nationality)
    code3 = (code3 or "").upper()

    flag_img = None
    if code3:
        flags_dir = resolve_flags_dir(assets_dir)
    flag_img = _load_flag_png(flags_dir, code3)

    x1 = xs[0]
    w1 = int(w_list[0])
    off_code = cfg["offsets"].get("code", (0, 0))
    off_flag = cfg["offsets"].get("flag", (0, 0))

    code_text = code3 if code3 else (nationality.strip()[:3].upper() if nationality else "")
    tx = x1 + pad + int(off_code[0])
    ty = y + (h - text_size(draw, code_text, f_code)[1]) // 2 + int(off_code[1])
    if code_text:
        draw.text((tx, ty), code_text, font=f_code, fill=(255, 255, 255, 255))

    if flag_img is not None:
        target_h = int(h * 0.62)
        if target_h <= 0:
            target_h = 1
        scale = target_h / float(flag_img.size[1])
        target_w = max(1, int(round(flag_img.size[0] * scale)))
        flag_resized = flag_img.resize((target_w, target_h), PILImage.LANCZOS)
        fx = x1 + w1 - pad - target_w + int(off_flag[0])
        fy = y + (h - target_h) // 2 + int(off_flag[1])
        overlay.alpha_composite(flag_resized.convert("RGBA"), (fx, fy))

    x2 = xs[1]; w2 = int(w_list[1])
    name = diver_name or ""
    off_name = cfg["offsets"].get("name", (0, 0))
    tw, th = text_size(draw, name, f_name)
    nx = x2 + (w2 - tw) // 2 + int(off_name[0])
    ny = y + (h - th) // 2 + int(off_name[1])
    if name:
        draw.text((nx, ny), name, font=f_name, fill=(255, 255, 255, 255))

    x3 = xs[2]; w3 = int(w_list[2])
    disc = discipline or ""
    off_disc = cfg["offsets"].get("disc", (0, 0))
    tw, th = text_size(draw, disc, f_disc)
    dx = x3 + (w3 - tw) // 2 + int(off_disc[0])
    dy = y + (h - th) // 2 + int(off_disc[1])
    if disc:
        draw.text((dx, dy), disc, font=f_disc, fill=(255, 255, 255, 255))

    x4 = xs[3]; w4 = int(w_list[3])
    off_time = cfg["offsets"].get("time", (0, 0))
    ttxt = format_dive_time(dive_time_s) if dive_time_s is not None else ""
    tw, th = text_size(draw, ttxt, f_time)
    tx4 = x4 + (w4 - tw) // 2 + int(off_time[0])
    ty4 = y + (h - th) // 2 + int(off_time[1])
    if ttxt:
        draw.text((tx4, ty4), ttxt, font=f_time, fill=(255, 255, 255, 255))

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
# Country code / flag helpers
# ============================================================

def _infer_country_code_3(nationality: str) -> Tuple[Optional[str], str]:
    if not nationality:
        return None, ""

    txt = nationality.strip()
    if "(" in txt and ")" in txt:
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
    if not code3:
        return None
    path = flags_dir / f"{code3.lower()}.png"
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGBA")
        return img
    except Exception as e:
        print(f"[FLAG] ERROR loading {path}: {e}")
        return None


# ============================================================
# Layout B: bottom-right competition panel
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
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    if BOARD3_ENABLE:
        b3_w = int(BOARD3_WIDTH)
        b3_h = int(BOARD3_HEIGHT)
        b3_x = int(BOARD3_LEFT)
        b3_y = int(H - BOARD3_BOTTOM - b3_h)
        b3_rect = [b3_x, b3_y, b3_x + b3_w, b3_y + b3_h]

        draw.rounded_rectangle(b3_rect, radius=int(BOARD3_RADIUS), fill=BOARD3_COLOR)

        if rate_text:
            font_rate = load_font(BOARD3_RATE_FONT_SIZE)
            rw, rh = text_size(draw, rate_text, font_rate)
            rate_x = b3_x + BOARD3_RATE_OFFSET_X
            rate_y = b3_y + (b3_h - rh) // 2 + BOARD3_RATE_OFFSET_Y
            draw.text((rate_x, rate_y), rate_text, font=font_rate, fill=BOARD3_TEXT_COLOR)

        if time_text:
            font_time = load_font(BOARD3_TIME_FONT_SIZE)
            tw, th = text_size(draw, time_text, font_time)
            time_x = b3_x + (b3_w - tw) // 2 + BOARD3_TIME_OFFSET_X
            time_y = b3_y + (b3_h - th) // 2 + BOARD3_TIME_OFFSET_Y
            draw.text((time_x, time_y), time_text, font=font_time, fill=BOARD3_TEXT_COLOR)

    if BOARD2_ENABLE:
        b2_w = int(BOARD2_WIDTH)
        b2_h = int(BOARD2_HEIGHT)
        b2_x = int(BOARD2_LEFT)
        b2_y = int(H - BOARD2_BOTTOM - b2_h)
        b2_rect = [b2_x, b2_y, b2_x + b2_w, b2_y + b2_h]

        draw.rounded_rectangle(b2_rect, radius=int(BOARD2_RADIUS), fill=BOARD2_COLOR)

        code3, country_label = _infer_country_code_3(nationality or "")
        flag_img = _load_flag_png(flags_dir, code3) if FLAG_ENABLE else None

        if FLAG_ENABLE and flag_img is not None:
            margin_tb = int(FLAG_TOP_BOTTOM_MARGIN)
            left_off = int(FLAG_LEFT_OFFSET)

            target_h = max(1, b2_h - margin_tb * 2)
            scale = target_h / flag_img.height
            target_w = int(flag_img.width * scale)

            if target_w > 0:
                flag_resized = flag_img.resize((target_w, target_h), PILImage.ANTIALIAS)
                fx = b2_x + left_off
                fy = b2_y + margin_tb
                img.paste(flag_resized, (fx, fy), flag_resized)

                flag_right_x = fx + target_w

                if code3:
                    font_code = load_font(int(FLAG_ALPHA3_FONT_SIZE))
                    code_text = code3.upper()
                    tw, th = text_size(draw, code_text, font_code)
                    gap = int(FLAG_ALPHA3_TEXT_GAP)
                    tx = flag_right_x + gap + COMP_ALPHA3_OFFSET_X
                    ty = b2_y + (b2_h - th) // 2 + int(FLAG_ALPHA3_OFFSET_Y)
                    draw.text((tx, ty), code_text, font=font_code, fill=FLAG_ALPHA3_FONT_COLOR)
        else:
            label_text = None
            if code3:
                label_text = code3.upper()
            elif country_label:
                label_text = str(country_label)

            if label_text:
                font_nat = load_font(int(FLAG_ALPHA3_FONT_SIZE))
                tw, th = text_size(draw, label_text, font_nat)
                tx = b2_x + int(FLAG_LEFT_OFFSET)
                ty = b2_y + (b2_h - th) // 2 + int(FLAG_ALPHA3_OFFSET_Y)
                draw.text((tx, ty), label_text, font=font_nat, fill=FLAG_ALPHA3_FONT_COLOR)

        if diver_name:
            font_name = load_font(COMP_NAME_FONT_SIZE)
            dn_text = str(diver_name)
            nw, nh = text_size(draw, dn_text, font_name)
            name_x = b2_x + (b2_w - nw) // 2 + COMP_NAME_OFFSET_X
            name_y = b2_y + (b2_h - nh) // 2 + COMP_NAME_OFFSET_Y
            draw.text((name_x, name_y), dn_text, font=font_name, fill=(0, 0, 0, 255))

        if discipline and discipline != "（不指定）":
            font_disc = load_font(COMP_SUB_FONT_SIZE)
            dt_text = str(discipline)
            dw, dh = text_size(draw, dt_text, font_disc)
            right_off = int(COMP_DISC_OFFSET_RIGHT)
            disc_x = b2_x + b2_w - right_off - dw
            disc_y = b2_y + (b2_h - dh) // 2 + COMP_DISC_OFFSET_Y
            draw.text((disc_x, disc_y), dt_text, font=font_disc, fill=(0, 0, 0, 255))

    return img


# ============================================================
# Layout B: left depth bar + bubbles
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
    w = BUBBLE_WIDTH
    h = BUBBLE_HEIGHT
    r = BUBBLE_RADIUS
    tail_w = BUBBLE_TAIL_WIDTH
    tail_h = int(h * BUBBLE_TAIL_HEIGHT_RATIO)

    tip_x = left_x
    base_x = left_x + tail_w
    rect_x0 = base_x
    rect_x1 = rect_x0 + w
    y0 = center_y - h // 2
    y1 = y0 + h

    draw.rounded_rectangle([rect_x0, y0, rect_x1, y1], radius=r, fill=fill_color)

    tri_y_top = center_y - tail_h // 2
    tri_y_bot = center_y + tail_h // 2
    draw.polygon([(tip_x, center_y), (base_x, tri_y_top), (base_x, tri_y_bot)], fill=fill_color)

    tw, th = text_size(draw, text, font)
    text_x = rect_x0 + (w - tw) // 2 + BUBBLE_TEXT_OFFSET_X
    text_y = center_y - th // 2 + BUBBLE_TEXT_OFFSET_Y
    draw.text((text_x, text_y), text, font=font, fill=text_color)


def draw_depth_bar_and_bubbles(
    base_overlay: PILImage.Image,
    depth_val: float,
    max_depth_for_scale: float,
    best_depth: float,
    show_best_bubble: bool,
    base_font: ImageFont.FreeTypeFont,
):
    overlay = base_overlay.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size

    if max_depth_for_scale <= 0:
        return overlay

    panel_x0 = DEPTH_PANEL_LEFT_MARGIN
    panel_x1 = panel_x0 + DEPTH_PANEL_WIDTH
    panel_y0 = (h - DEPTH_PANEL_HEIGHT) // 2
    panel_y1 = panel_y0 + DEPTH_PANEL_HEIGHT

    draw.rounded_rectangle([panel_x0, panel_y0, panel_x1, panel_y1], radius=DEPTH_PANEL_RADIUS, fill=(0, 0, 0, 200))

    bar_h = DEPTH_BAR_TOTAL_HEIGHT
    bar_y0 = (h - bar_h) // 2
    tick_x_end = panel_x1 - 10
    max_d = max_depth_for_scale

    try:
        tick_font = load_font(DEPTH_TICK_LABEL_FONT_SIZE)
    except Exception:
        tick_font = base_font

    for d in range(0, int(max_d) + 1):
        ratio = d / max_d
        y = int(bar_y0 + ratio * bar_h)

        if d % 10 == 0:
            tick_len = DEPTH_TICK_LEN_10M
        elif d % 5 == 0:
            tick_len = DEPTH_TICK_LEN_5M
        else:
            tick_len = DEPTH_TICK_LEN_1M

        tick_x_start = tick_x_end - tick_len
        draw.line([(tick_x_start, y), (tick_x_end, y)], fill=(255, 255, 255, 220), width=DEPTH_TICK_WIDTH)

        if d % 10 == 0:
            label = f"{d}"
            lw, lh = text_size(draw, label, tick_font)
            lx = tick_x_start - 6 - lw + DEPTH_TICK_LABEL_OFFSET_X
            ly = y - lh // 2 + DEPTH_TICK_LABEL_OFFSET_Y
            draw.text((lx, ly), label, font=tick_font, fill=(255, 255, 255, 255))

    def depth_to_y(dv: float) -> int:
        d_clamped = max(0.0, min(max_d, float(dv)))
        ratio = d_clamped / max_d
        return int(bar_y0 + ratio * bar_h)

    try:
        bubble_font = load_font(BUBBLE_FONT_SIZE)
    except Exception:
        bubble_font = base_font

    bubble_attach_x = panel_x1

    current_y = depth_to_y(depth_val)
    current_text = f"{depth_val:.1f}"
    draw_speech_bubble(draw, bubble_attach_x, current_y, current_text, BUBBLE_CURRENT_COLOR, BUBBLE_TEXT_COLOR_DARK, bubble_font)

    if best_depth > 0 and show_best_bubble:
        best_y = depth_to_y(best_depth)
        best_text = f"{best_depth:.1f}"
        draw_speech_bubble(draw, bubble_attach_x, best_y, best_text, BUBBLE_BEST_COLOR, BUBBLE_TEXT_COLOR_DARK, bubble_font)

    return overlay


# ============================================================
# Main render function
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
    dive_time_s: Optional[float] = None,
    dive_start_s: Optional[float] = None,
    dive_end_s: Optional[float] = None,
    progress_callback=None,
    layout_params: Optional[dict] = None,
    hr_df: Optional[pd.DataFrame] = None,
):
    """
    progress_callback(p: float, message: str) updates Streamlit progress:
      - p: 0.0 ~ 1.0
    """
    flags_dir = resolve_flags_dir(assets_dir)

    # Apply global fine-tune correction (seconds)
    effective_offset = float(time_offset) + float(ALIGN_DISPLAY_CORRECTION_S)


    def update_progress(p: float, msg: str = ""):
        if progress_callback is None:
            return
        try:
            p = max(0.0, min(1.0, float(p)))
            progress_callback(p, msg)
        except Exception:
            pass

    t0 = time.perf_counter()
    update_progress(0.02, "初始化中...")

    t_load_start = time.perf_counter()
    clip = VideoFileClip(str(video_path))
    W, H = output_resolution
    clip = clip.resize((W, H))
    t_load_end = time.perf_counter()
    print(f"[render_video] 載入 + resize 影片耗時 {t_load_end - t_load_start:.2f} 秒")
    update_progress(0.08, "載入影片完成")
    
    # ===============================
    # HR animation timing base
    # ===============================
    _fps = float(getattr(clip, "fps", 10.0)) # original = 30.0
    dt = 1.0 / _fps

    # =========================
    # Depth / rate prep
    # =========================
    t_pre_start = time.perf_counter()

    times_d = dive_df["time_s"].to_numpy()
    depths_d = dive_df["depth_m"].to_numpy()

    # Layout B existing (abs rate)
    times_r = df_rate["time_s"].to_numpy()
    rates_r = df_rate["rate_abs_mps_smooth"].to_numpy()

    # Max depth / time
    if len(depths_d) > 0:
        max_depth_raw = float(np.nanmax(depths_d))
        best_idx = int(np.nanargmax(depths_d))
        best_time_global = float(times_d[best_idx])
    else:
        max_depth_raw = 0.0
        best_time_global = None

    # Layout B scale logic
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
        t = t_video + effective_offset
        if len(times_d) == 0:
            return 0.0
        if t <= times_d[0]:
            return float(depths_d[0])
        if t >= times_d[-1]:
            return float(depths_d[-1])
        return float(np.interp(t, times_d, depths_d))

    def is_descent_at(t_video: float, half_window_s: float = 0.30) -> bool:
        """
        Determine direction by depth trend around t.
        True => descent, False => ascent.
        """
        t0w = max(0.0, t_video - half_window_s)
        t1w = t_video + half_window_s
        d0 = depth_at(t0w)
        d1 = depth_at(t1w)
        return (d1 - d0) >= 0.0

    def rate_at(t_video: float) -> float:
        t = t_video + effective_offset
        if len(times_r_ext) == 0:
            return 0.0
        if t <= float(times_r_ext[0]):
            # If we padded a (dive_start_s, 0.0) point, this returns 0.0.
            return float(rates_r_ext[0])
        if t >= float(times_r_ext[-1]):
            return float(rates_r_ext[-1])
        return float(np.interp(t, times_r_ext, rates_r_ext))


    # Dive start/end inference (unified logic for A/B/C)
    # We infer start/end by the FIRST/LAST crossing of a shallow depth threshold, using linear interpolation
    # to avoid the typical ~0.5~1.5s "late stop" that happens when using the last sampled point only.
    START_DEPTH_EPS = 0.30   # meters, start when depth crosses above this
    END_DEPTH_EPS = 0.30     # meters, end when depth crosses below this (during ascent)
    SURFACE_DEPTH_EPS = 0.05  # meters, force rate=0 and depth=0 when at/near surface

    def _interp_crossing_time(times: np.ndarray, depths: np.ndarray, threshold: float, *, rising: bool) -> Optional[float]:
        """Return interpolated crossing time for depth==threshold.
        rising=True: find first crossing from <thr to >=thr (start of descent)
        rising=False: find last crossing from >thr to <=thr (end of ascent)
        """
        if times is None or depths is None:
            return None
        n = int(min(len(times), len(depths)))
        if n < 2:
            return None

        t_arr = np.asarray(times[:n], dtype=float)
        d_arr = np.asarray(depths[:n], dtype=float)

        thr = float(threshold)

        if rising:
            for i in range(1, n):
                d0 = d_arr[i - 1]; d1 = d_arr[i]
                if (d0 < thr) and (d1 >= thr):
                    t0 = t_arr[i - 1]; t1 = t_arr[i]
                    if d1 == d0:
                        return float(t1)
                    frac = (thr - d0) / (d1 - d0)
                    return float(t0 + frac * (t1 - t0))
            return None
        else:
            for i in range(n - 1, 0, -1):
                d0 = d_arr[i - 1]; d1 = d_arr[i]
                if (d0 > thr) and (d1 <= thr):
                    t0 = t_arr[i - 1]; t1 = t_arr[i]
                    if d1 == d0:
                        return float(t1)
                    frac = (thr - d0) / (d1 - d0)
                    return float(t0 + frac * (t1 - t0))
            return None

    if dive_start_s is None:
        dive_start_s = _interp_crossing_time(times_d, depths_d, START_DEPTH_EPS, rising=True)
        if dive_start_s is None:
            mask_start = dive_df["depth_m"] >= 0.1
            if mask_start.any():
                dive_start_s = float(dive_df.loc[mask_start, "time_s"].iloc[0])

    if dive_end_s is None:
        dive_end_s = _interp_crossing_time(times_d, depths_d, END_DEPTH_EPS, rising=False)
        if dive_end_s is None:
            mask_end = dive_df["depth_m"] <= 0.05
            if mask_end.any():
                dive_end_s = float(dive_df.loc[mask_end, "time_s"].iloc[-1])


    # ---------------------------------------------------------
    # Rate timeline padding (fix: rate starts late due to df_rate)
    # We build an extended time series so rate starts from 0 at dive_start_s
    # and returns to 0 at dive_end_s, then interpolate on it.
    # ---------------------------------------------------------
    times_r_ext = times_r
    rates_r_ext = rates_r

    try:
        if len(times_r) > 0 and len(rates_r) > 0:
            ext_t = list(times_r.astype(float))
            ext_v = list(rates_r.astype(float))

            # Prepend a (dive_start_s, 0.0) point if df_rate starts later than dive start
            if dive_start_s is not None:
                ds = float(dive_start_s)
                if ds < ext_t[0] - 1e-6:
                    ext_t.insert(0, ds)
                    ext_v.insert(0, 0.0)

            # Append a (dive_end_s, 0.0) point to ensure rate goes to 0 at end
            if dive_end_s is not None:
                de = float(dive_end_s)
                if de > ext_t[-1] + 1e-6:
                    ext_t.append(de)
                    ext_v.append(0.0)
                else:
                    # If de is within range, insert/overwrite a point at de = 0 to clamp
                    # (helps prevent lingering non-zero rate after surfacing)
                    # Find insert position
                    import bisect
                    i = bisect.bisect_left(ext_t, de)
                    if i < len(ext_t) and abs(ext_t[i] - de) < 1e-3:
                        ext_v[i] = 0.0
                    else:
                        ext_t.insert(i, de)
                        ext_v.insert(i, 0.0)

            times_r_ext = np.array(ext_t, dtype=float)
            rates_r_ext = np.array(ext_v, dtype=float)
    except Exception as _e:
        times_r_ext = times_r
        rates_r_ext = rates_r
    def rate_c_signed_like_layout_b(t_video: float) -> float:

        """
        Layout C rate aligned with Layout B:
        - magnitude from df_rate['rate_abs_mps_smooth'] (>=0)
        - sign from direction (is_descent_at)
        """
        mag = float(rate_at(t_video))
        return mag if is_descent_at(t_video) else -mag

    def elapsed_dive_time(t_video: float) -> Optional[float]:
        if dive_start_s is None:
            return None

        t_global = t_video + effective_offset
        if t_global <= dive_start_s:
            return 0.0

        if dive_end_s is not None and t_global >= dive_end_s:
            return max(0.0, float(dive_end_s - dive_start_s))

        return max(0.0, float(t_global - dive_start_s))

    try:
        base_font = load_font(INFO_CARD_FONT_SIZE)
    except Exception:
        base_font = ImageFont.load_default()

    t_pre_end = time.perf_counter()
    print(f"[render_video] 前處理耗時 {t_pre_end - t_pre_start:.2f} 秒")
    update_progress(0.12, "資料前處理完成")

    # =========================
    # Frame rendering loop
    # =========================
    duration = float(clip.duration) if clip.duration else 0.0
    last_p = {"value": 0.12}

    base_font_path = FONT_PATH

    # =========================
    # Layout C configs (centralized)
    # =========================
    layout_c_all = get_layout_c_config()
    layout_c_depth_cfg = layout_c_all.depth
    layout_c_rate_cfg = layout_c_all.rate
    hr_cfg = layout_c_all.hr
    layout_c_time_cfg = layout_c_all.time
    shadow_cfg = layout_c_all.shadow

    # Optional overrides from layout_params (dict)
    try:
        if isinstance(layout_params, dict):
            if isinstance(layout_params.get("layout_c_depth_cfg"), dict):
                layout_c_depth_cfg = LayoutCDepthConfig(**layout_params["layout_c_depth_cfg"])
            if isinstance(layout_params.get("layout_c_rate_cfg"), dict):
                layout_c_rate_cfg = LayoutCRateConfig(**layout_params["layout_c_rate_cfg"])
            # Backward-compatible key
            if isinstance(layout_params.get("layout_c_hr_cfg"), dict):
                hr_cfg = LayoutCHeartRateConfig(**layout_params["layout_c_hr_cfg"])
            if isinstance(layout_params.get("layout_c_time_cfg"), dict):
                layout_c_time_cfg = LayoutCTimeConfig(**layout_params["layout_c_time_cfg"])
            if isinstance(layout_params.get("layout_c_shadow_cfg"), dict):
                shadow_cfg = LayoutCShadowConfig(**layout_params["layout_c_shadow_cfg"])
    except Exception:
        pass

    # Apply Layout C shadow (global for all Layout C modules)
    global LAYOUT_C_SHADOW_ENABLED, LAYOUT_C_SHADOW_OFFSET, LAYOUT_C_SHADOW_BLUR, LAYOUT_C_SHADOW_COLOR
    LAYOUT_C_SHADOW_ENABLED = bool(getattr(shadow_cfg, "enabled", True)) # original = True
    LAYOUT_C_SHADOW_OFFSET = tuple(getattr(shadow_cfg, "offset", (4, 4)))
    LAYOUT_C_SHADOW_BLUR = int(getattr(shadow_cfg, "blur_radius", 6))
    LAYOUT_C_SHADOW_COLOR = tuple(getattr(shadow_cfg, "color", (0, 0, 0, 120)))

# =========================
    # Heart rate prep (Layout C)
    # =========================
    hr_available = False
    hr_times = None
    hr_values = None
    hr_last_value = {"v": None}
    hr_anim = {"phase": 0.0, "bpm_active": None, "bpm_pending": None, "switch_pending": False}
    # hr_cfg is prepared above (Layout C configs)

    if hr_df is not None and isinstance(hr_df, pd.DataFrame) and not hr_df.empty:
        if "time_s" in hr_df.columns:
            hr_col = "heart_rate" if "heart_rate" in hr_df.columns else ("hr" if "hr" in hr_df.columns else None)
            if hr_col is not None:
                _tmp = hr_df[["time_s", hr_col]].copy()
                _tmp = _tmp.dropna()
                _tmp["time_s"] = pd.to_numeric(_tmp["time_s"], errors="coerce")
                _tmp[hr_col] = pd.to_numeric(_tmp[hr_col], errors="coerce")
                _tmp = _tmp.dropna()
                if not _tmp.empty:
                    _tmp = _tmp.sort_values("time_s")
                    hr_times = _tmp["time_s"].to_numpy(dtype=float)
                    hr_values = _tmp[hr_col].to_numpy(dtype=float)
                    hr_available = len(hr_times) > 1
                    # --- Normalize HR time base if it looks like absolute / non-relative time_s ---
                    # If HR time_s starts far beyond the video duration, treat it as "absolute" and shift to start at 0.
                    try:
                        if duration is not None and duration > 0 and len(hr_times) > 0:
                            if float(hr_times[0]) > float(duration) + 2.0:
                                hr_times = hr_times - float(hr_times[0])
                    except Exception:
                        pass


    def hr_at(t_local: float) -> Optional[float]:
        if not hr_available or hr_times is None or hr_values is None:
            return None
        try:
            if t_local <= float(hr_times[0]):
                return float(hr_values[0])
            if t_local >= float(hr_times[-1]):
                return float(hr_values[-1])
            return float(np.interp(float(t_local), hr_times, hr_values))
        except Exception:
            return None

    # ==========================================================
    # Layout C overlay render decoupling
    #   - Video output FPS stays unchanged.
    #   - Overlay modules for Layout C are rendered at a fixed FPS and reused between frames.
    # ==========================================================
    LAYOUT_C_OVERLAY_FPS = 10  # overlay update rate for Layout C (Hz)
    _layout_c_cache = {"tq": None, "overlay": None, "size": None}


    def make_frame(t):
        if duration > 0:
            frac = max(0.0, min(1.0, t / duration))
            p = 0.12 + 0.86 * frac
            if p - last_p["value"] >= 0.01:
                last_p["value"] = p
                update_progress(p, "產生疊加畫面中...")

        frame = clip.get_frame(t)
        img = PILImage.fromarray(frame).convert("RGBA")
        img_w, img_h = img.size

        # Layout C overlay caching (render overlay at a fixed FPS, reuse between video frames)
        if layout == "C":
            step = 1.0 / float(LAYOUT_C_OVERLAY_FPS)
            tq = math.floor(float(t) / step) * step
            # Avoid floating-point drift in cache keys
            tq = round(tq, 3)
            if _layout_c_cache["tq"] == tq and _layout_c_cache["overlay"] is not None and _layout_c_cache["size"] == img.size:
                composed = PILImage.alpha_composite(img, _layout_c_cache["overlay"]).convert("RGB")
                return np.array(composed)

            t_for_calc = tq
        else:
            t_for_calc = t

        overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Global time (used for module texts / gating)
        t_global = t_for_calc + effective_offset

        depth_val = depth_at(t_for_calc)

        # Layout B (abs, from df_rate)
        rate_val_abs_raw = rate_at(t_for_calc)

        # Layout C (signed, Layout B-aligned: magnitude from df_rate + sign from depth trend)
        rate_val_signed_raw = rate_c_signed_like_layout_b(t_for_calc)

        # Heart rate (Layout C only)
        hr_text = "--"
        show_hr_module = False
        show_hr_value = False
        pulse_scale = 1.0

        if layout.upper() == "C" and hr_cfg.enabled and hr_available:
            t_data = float(t_global)
            data_start = float(hr_times[0]) if (hr_times is not None and len(hr_times) > 0) else 0.0
            data_end = float(hr_times[-1]) if (hr_times is not None and len(hr_times) > 0) else (float(duration) if duration else 0.0)


            if t_data < data_start:
                hr_text = "--"
                show_hr_module = True
                show_hr_value = True
                pulse_scale = 1.0
            else:
                if t_data <= data_end:
                    hr_now = hr_at(t_data)
                    if hr_now is not None and not np.isnan(hr_now):
                        hr_last_value["v"] = float(hr_now)

                if hr_last_value["v"] is not None:
                    hr_text = str(int(round(hr_last_value["v"])))
                    show_hr_module = True
                    show_hr_value = True

                    target_bpm = float(hr_last_value["v"])
                    if hr_anim["bpm_active"] is None:
                        hr_anim["bpm_active"] = target_bpm
                    if hr_anim["bpm_active"] is not None:
                        if abs(target_bpm - float(hr_anim["bpm_active"])) > 0.1 and not hr_anim["switch_pending"]:
                            hr_anim["bpm_pending"] = target_bpm
                            hr_anim["switch_pending"] = True

                    bpm_for_phase = float(hr_anim["bpm_active"] or 0.0)
                    hr_anim["phase"] += 2.0 * math.pi * (bpm_for_phase / 60.0) * dt
                    if hr_anim["phase"] >= 2.0 * math.pi:
                        hr_anim["phase"] -= 2.0 * math.pi
                        if hr_anim["switch_pending"]:
                            hr_anim["bpm_active"] = hr_anim["bpm_pending"]
                            hr_anim["switch_pending"] = False

                    pulse_scale = 1.0 + float(hr_cfg.pulse_amp) * math.sin(float(hr_anim["phase"]))
                else:
                    hr_text = "--"
                    show_hr_module = True
                    show_hr_value = True
                    pulse_scale = 1.0


        # Unified in-dive gating (A/B/C should share the same timing behavior)
        in_dive = (dive_start_s is not None and t_global >= dive_start_s) and (dive_end_s is None or t_global <= dive_end_s)
        near_surface = (float(depth_val) <= float(SURFACE_DEPTH_EPS))

        # Depth display: snap to 0 at/near surface when not in-dive (prevents lingering 0.1~0.3m jitter)
        depth_disp = float(depth_val)
        if (not in_dive) and near_surface:
            depth_disp = 0.0

        # Rate display: force 0 before start / after end / near surface
        rate_val_abs = 0.0 if (not in_dive or near_surface) else float(rate_val_abs_raw)
        rate_val_signed_c = 0.0 if (not in_dive or near_surface) else float(rate_val_signed_raw)

        # Direction (for Layout C arrow/label). If not in dive, default to descent label.
        direction_is_descent = bool(is_descent_at(t)) if in_dive else True

        elapsed = elapsed_dive_time(t)
        time_text = format_dive_time(elapsed) if elapsed is not None else ""

        text_depth = f"{depth_disp:.1f} m"
        text_rate = f"{rate_val_abs:.1f} m/s"

        # ===== Layout A =====
        if layout == "A":
            overlay = draw_layout_a_bottom_bar(
                overlay=overlay,
                assets_dir=assets_dir,
                base_font_path=base_font_path,
                nationality=nationality,
                diver_name=diver_name,
                discipline=discipline,
                dive_time_s=elapsed,
                depth_val=depth_disp,
                params=layout_params,
            )

        # ===== Generic info card (NOT A/B/C) =====
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

            draw.rounded_rectangle([x0, y0, x1, y1], radius=22, fill=(0, 0, 0, 170))

            text_x = x0 + padding + INFO_TEXT_OFFSET_X
            cur_y = y0 + padding + INFO_TEXT_OFFSET_Y

            for txt, h_txt in zip(lines, line_heights):
                draw.text((text_x, cur_y), txt, font=base_font, fill=(255, 255, 255, 255))
                cur_y += h_txt + line_spacing

        # ===== Layout C =====
        if layout == "C":
            overlay = render_layout_c_depth_module(
                base_img=overlay,
                current_depth_m=depth_disp,
                cfg=layout_c_depth_cfg,
                font_path=base_font_path,
                max_depth_m=best_depth,
            )

            overlay = render_layout_c_time_module(
                overlay,
                time_s=float(max(0.0, t_global)),
                cfg=layout_c_time_cfg,
            )

            overlay = render_layout_c_rate_module(
                base_img=overlay,
                speed_mps_signed=rate_val_signed_c,
                cfg=layout_c_rate_cfg,
                is_descent_override=direction_is_descent,
            )

            # Heart rate module (icon + value) - only when HR data exists
            if show_hr_module:
                overlay = render_layout_c_heart_rate_module(
                    overlay,
                    hr_text=hr_text,
                    cfg=hr_cfg,
                    assets_dir=assets_dir,
                    pulse_scale=pulse_scale,
                    show_value=show_hr_value,
                    show_icon=True,
                )

        # ===== Layout B =====
        if layout == "B":
            show_best_bubble = bool(best_time_global is not None and t_global >= best_time_global)

            overlay = draw_depth_bar_and_bubbles(
                overlay,
                depth_val=depth_disp,
                max_depth_for_scale=max_depth_for_scale,
                best_depth=best_depth,
                show_best_bubble=show_best_bubble,
                base_font=base_font,
            )

            overlay = draw_competition_panel_bottom_right(
                overlay,
                diver_name=diver_name or "",
                nationality=nationality or "",
                discipline=discipline or "",
                flags_dir=flags_dir,
                rate_text=text_rate,
                time_text=time_text,
            )

        # Update Layout C overlay cache on cache-miss renders
        if layout == "C":
            _layout_c_cache["tq"] = t_for_calc
            _layout_c_cache["overlay"] = overlay
            _layout_c_cache["size"] = img.size

        composed = PILImage.alpha_composite(img, overlay).convert("RGB")
        return np.array(composed)

    # =========================
    # Encode video
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
            threads=1,
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

        try:
            if tmp_audio_path and os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
        except Exception:
            pass
