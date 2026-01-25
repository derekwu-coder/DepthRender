# core/video_renderer.py

# Global default colors
label_color = (255, 255, 255, 255)
value_color = (0, 255, 255, 255)

from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd
import subprocess
import json
import hashlib
from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import VideoClip
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageFont, Image, ImageFilter, ImageChops
from dataclasses import dataclass
from functools import lru_cache
import time


# ---------------------------------------------------------------------------
# Caches (performance): avoid re-loading fonts / decoding PNG icons per frame
# ---------------------------------------------------------------------------
_TEMP_FONT_CACHE = {}  # key: (str(font_path), int(size)) -> ImageFont.FreeTypeFont
_TEMP_ICON_CACHE = {}  # key: (str(icon_path), int(icon_size)) -> PILImage.Image (RGBA, resized)
_HR_ICON_CACHE = {}  # key: (str(icon_path), int(icon_h)) -> (outline_rgba, inside_mask_L)

# Additional caches
_ICON_BASE_CACHE = {}  # key: str(icon_path) -> PILImage.Image (RGBA, bg-removed if applied)
_FLAG_BASE_CACHE = {}  # key: str(flag_path) -> PILImage.Image (RGBA)
_FLAG_RESIZE_CACHE = {}  # key: (str(flag_path), int(w), int(h)) -> PILImage.Image (RGBA resized)


# Shadow caches (performance): avoid per-frame alpha-mask generation
_TEMP_ICON_SHADOW_CACHE = {}  # key: (str(icon_path), int(icon_size), int(dx), int(dy), int(alpha), tuple(color_rgb)) -> PILImage.Image


def _rgba_shadow_color(color_rgb=(0, 0, 0), alpha: int = 140) -> tuple:
    try:
        r, g, b = color_rgb
    except Exception:
        r, g, b = 0, 0, 0
    return (int(r), int(g), int(b), int(max(0, min(255, int(alpha)))))


def _draw_text_hard_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple,
    text: str,
    *,
    font,
    fill,
    shadow_enable: bool,
    shadow_dx: int,
    shadow_dy: int,
    shadow_color_rgb=(0, 0, 0),
    shadow_alpha: int = 140,
):
    """Fast hard shadow: draw shadow text at offset, then normal text."""
    if text is None:
        return
    s = str(text)
    if shadow_enable and s:
        sc = _rgba_shadow_color(shadow_color_rgb, shadow_alpha)
        draw.text((int(xy[0]) + int(shadow_dx), int(xy[1]) + int(shadow_dy)), s, font=font, fill=sc)
    draw.text((int(xy[0]), int(xy[1])), s, font=font, fill=fill)


def _draw_line_hard_shadow(
    draw: ImageDraw.ImageDraw,
    p0: tuple,
    p1: tuple,
    *,
    fill,
    width: int,
    shadow_enable: bool,
    shadow_dx: int,
    shadow_dy: int,
    shadow_color_rgb=(0, 0, 0),
    shadow_alpha: int = 140,
):
    """Fast hard shadow for a line."""
    if shadow_enable:
        sc = _rgba_shadow_color(shadow_color_rgb, shadow_alpha)
        draw.line([(int(p0[0]) + int(shadow_dx), int(p0[1]) + int(shadow_dy)), (int(p1[0]) + int(shadow_dx), int(p1[1]) + int(shadow_dy))], fill=sc, width=max(1, int(width)))
    draw.line([(int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))], fill=fill, width=max(1, int(width)))


def _draw_ellipse_outline_hard_shadow(
    draw: ImageDraw.ImageDraw,
    box: tuple,
    *,
    outline,
    width: int,
    shadow_enable: bool,
    shadow_dx: int,
    shadow_dy: int,
    shadow_color_rgb=(0, 0, 0),
    shadow_alpha: int = 140,
):
    if shadow_enable:
        sc = _rgba_shadow_color(shadow_color_rgb, shadow_alpha)
        x0, y0, x1, y1 = box
        draw.ellipse((x0 + int(shadow_dx), y0 + int(shadow_dy), x1 + int(shadow_dx), y1 + int(shadow_dy)), outline=sc, width=max(1, int(width)))
    draw.ellipse(box, outline=outline, width=max(1, int(width)))


def _get_icon_shadow_cached(icon_rgba: PILImage.Image, *, key, shadow_color_rgb=(0, 0, 0), shadow_alpha: int = 140) -> PILImage.Image:
    """Return a same-size RGBA shadow silhouette image (no offset baked in)."""
    sh = _TEMP_ICON_SHADOW_CACHE.get(key)
    if sh is not None:
        return sh
    alpha = icon_rgba.getchannel('A')
    sc = _rgba_shadow_color(shadow_color_rgb, shadow_alpha)
    sh = PILImage.new('RGBA', icon_rgba.size, (sc[0], sc[1], sc[2], 0))

    # Scale icon alpha to shadow alpha
    sa = alpha.point(lambda p: int(p * sc[3] / 255))
    sh.putalpha(sa)
    _TEMP_ICON_SHADOW_CACHE[key] = sh
    return sh

def _get_font_cached(font_path: Optional[Path], size: int) -> ImageFont.FreeTypeFont:
    """Load a FreeTypeFont for (font_path, size).

    Note: Caching FreeTypeFont objects can trigger RecursionError in some Streamlit
    execution paths. We therefore avoid caching here to maximize robustness.
    """
    try:
        if font_path is None:
            return ImageFont.load_default()
        p = Path(font_path)
        if p.exists():
            return ImageFont.truetype(str(p), int(size))
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


# ============================================================
# Overlay render throttling (decouple overlay updates from video fps)
# ============================================================
LAYOUT_AB_OVERLAY_FPS = 15   # Layout A & B overlay update fps
LAYOUT_C_OVERLAY_FPS  = 10   # Layout C overlay update fps
LAYOUT_D_OVERLAY_FPS  = 10   # Layout D overlay update fps

# Internal overlay cache (per layout): {"tq": float, "size": (w,h), "overlay": PIL.Image}
_OVERLAY_CACHE = None  # disabled to avoid recursion issues with PIL objects in some environments

# ============================================================
# Font paths
# ============================================================

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

# Layout D Temperature icon (place the PNG at: assets/icons/thermo.png)
LAYOUT_D_TEMP_ICON_REL = Path("icons") / "thermo.png"
# Layout D Heart Rate icon (place the PNG at: assets/icons/heart_rate_icon_2.png)
LAYOUT_D_HR_ICON_REL = Path("icons") / "heart_rate_icon_2.png"

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
    """Unified font loader for the project default font (RobotoCondensedBold.ttf) with caching."""
    return _get_font_cached(FONT_PATH, int(size))

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

        "shadow_enable": True, # original = True
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

    window_top: int = 1570  # original = 1600
    window_bottom_margin: int = 50  # original = 80
    px_per_m: int = 17              # original = 20

    fade_enable: bool = True
    fade_margin_px: int = 100    # original = 70
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


# ============================================================
# Layout C - Depth scale / fade caches (performance)
# ============================================================

def _layout_c_depth_scale_cache_key(
    *,
    cfg: "LayoutCDepthConfig",
    depth_max_display: int,
    W: int,
    H: int,
    pad_top: int,
    pad_bot: int,
    font_path: Optional[str],
) -> tuple:
    """
    Build a stable, hashable cache key for the Layout C depth scale (ticks + numbers).

    The scale itself is static for a given config / resolution; only the vertical offset
    changes per frame. Caching avoids re-drawing 100+ ticks and labels per frame.
    """
    fp = str(font_path) if font_path else ""
    return (
        int(W), int(H), int(pad_top), int(pad_bot),
        int(cfg.depth_min_m), int(depth_max_display),
        int(cfg.px_per_m), int(getattr(cfg, "scale_y_offset", 0)),
        int(cfg.scale_x), int(getattr(cfg, "scale_x_offset", 0)),
        # tick geometry
        int(cfg.tick_len_10m), int(cfg.tick_len_5m), int(cfg.tick_len_1m), int(cfg.tick_len_max),
        int(cfg.tick_w_10m), int(cfg.tick_w_5m), int(cfg.tick_w_1m), int(cfg.tick_w_max),
        tuple(cfg.tick_color),
        # number style / placement
        int(cfg.num_left_margin), int(getattr(cfg, "num_offset_x", 0)), int(getattr(cfg, "num_offset_y", 0)),
        int(cfg.num_font_size), tuple(cfg.num_color),
        int(getattr(cfg, "zero_num_offset_x", 0)), int(getattr(cfg, "zero_num_offset_y", 0)),
        int(getattr(cfg, "max_num_offset_x", 0)), int(getattr(cfg, "max_num_offset_y", 0)),
        fp,
    )

@lru_cache(maxsize=16)
def _get_layout_c_depth_scale_moving_cached(cache_key: tuple) -> PILImage.Image:
    """
    Return a pre-rendered RGBA image containing the full moving depth scale
    (ticks + numbers) for Layout C. The caller should treat the returned image
    as read-only.
    """
    (
        W, H, pad_top, pad_bot,
        depth_min_m, depth_max_display,
        px_per_m, scale_y_offset,
        scale_x, scale_x_offset,
        tick_len_10m, tick_len_5m, tick_len_1m, tick_len_max,
        tick_w_10m, tick_w_5m, tick_w_1m, tick_w_max,
        tick_color,
        num_left_margin, num_offset_x, num_offset_y,
        num_font_size, num_color,
        zero_num_offset_x, zero_num_offset_y,
        max_num_offset_x, max_num_offset_y,
        font_path_str,
    ) = cache_key

    moving_h = int(H) + int(pad_top) + int(pad_bot)
    moving = PILImage.new("RGBA", (int(W), int(moving_h)), (0, 0, 0, 0))
    d = ImageDraw.Draw(moving)

    # Font for scale numbers
    if font_path_str:
        try:
            num_font = _get_font_cached(Path(font_path_str), int(num_font_size))
        except Exception:
            num_font = ImageFont.load_default()
    else:
        num_font = ImageFont.load_default()

    center_x = int(scale_x) + int(scale_x_offset)

    # Draw ticks + numbers onto moving canvas
    for m in range(int(depth_min_m), int(depth_max_display) + 1):
        y = int(pad_top) + int(m) * int(px_per_m) + int(scale_y_offset)
        is_max = (m == int(depth_max_display))

        if is_max:
            w, L = int(tick_w_max), int(tick_len_max)
        elif (m % 10) == 0:
            w, L = int(tick_w_10m), int(tick_len_10m)
        elif (m % 5) == 0:
            w, L = int(tick_w_5m), int(tick_len_5m)
        else:
            w, L = int(tick_w_1m), int(tick_len_1m)

        x1 = center_x - L // 2
        x2 = center_x + L // 2
        d.line([(x1, y), (x2, y)], fill=tick_color, width=w)

        if ((m % 10) == 0) or is_max:
            txt = str(m)
            num_x = int(num_left_margin) + int(num_offset_x)
            num_y_center = y + int(num_offset_y)

            if m == 0:
                num_x += int(zero_num_offset_x)
                num_y_center += int(zero_num_offset_y)

            if is_max:
                num_x += int(max_num_offset_x)
                num_y_center += int(max_num_offset_y)

            d.text(
                (int(num_x), int(num_y_center)),
                txt,
                font=num_font,
                fill=num_color,
                anchor="lm",
            )

    return moving

@lru_cache(maxsize=32)
def _get_layout_c_fade_mask_cached(*, W: int, win_h: int, fade: int, edge_transparency: float) -> PILImage.Image:
    """
    Build and cache an L-mode alpha multiplier mask for fading the top/bottom edges.
    Mask values are in [0..255] where 255 keeps alpha and smaller values attenuate alpha.
    """
    W = int(W)
    win_h = int(win_h)
    fade = int(fade)
    edge_transparency = float(edge_transparency)

    # edge_transparency is "how transparent the edge should be"
    # so the alpha multiplier at the edges is (1 - edge_transparency)
    edge_opacity = max(0.0, min(1.0, 1.0 - edge_transparency))

    # 1D factors
    factors = np.ones((win_h,), dtype=np.float32)
    if fade > 0:
        top = np.linspace(edge_opacity, 1.0, fade, dtype=np.float32)
        factors[:fade] = top
        bot = np.linspace(1.0, edge_opacity, fade, dtype=np.float32)
        factors[win_h - fade:] = bot

    vals = np.clip(np.round(factors * 255.0), 0, 255).astype(np.uint8)

    # Build mask as (W x win_h)
    mask = np.repeat(vals[:, None], W, axis=1)
    return PILImage.fromarray(mask, mode="L")


def render_layout_c_depth_module(
    base_img: Image.Image,
    current_depth_m: float,
    cfg: LayoutCDepthConfig,
    font_path: Optional[str] = None,
    max_depth_m: Optional[float] = None,
) -> Image.Image:
    """Render Layout C depth module onto base_img (RGBA PIL Image) with shadow."""
    if not cfg.enabled:
        return base_img

    # Ensure RGBA
    out = base_img.copy().convert("RGBA")
    W, H = out.size

    y0 = int(cfg.window_top)
    y1 = int(H - cfg.window_bottom_margin)
    indicator_y = (y0 + y1) // 2

    pad_top = int(getattr(cfg, "scale_pad_top", 0))
    pad_bot = int(getattr(cfg, "scale_pad_bottom", 0))

    # Fonts (cached)
    if font_path:
        num_font = _get_font_cached(Path(font_path), int(cfg.num_font_size))
        if LAYOUT_C_VALUE_FONT_PATH.exists():
            value_font = _get_font_cached(LAYOUT_C_VALUE_FONT_PATH, int(cfg.depth_value_font_size))
            unit_font = _get_font_cached(LAYOUT_C_VALUE_FONT_PATH, int(cfg.depth_unit_font_size))
        else:
            fp = Path(font_path)
            value_font = _get_font_cached(fp, int(cfg.depth_value_font_size))
            unit_font = _get_font_cached(fp, int(cfg.depth_unit_font_size))
    else:
        num_font = ImageFont.load_default()
        value_font = ImageFont.load_default()
        unit_font = ImageFont.load_default()

    # Depth scale range
    if max_depth_m is not None and np.isfinite(max_depth_m):
        depth_max_display = int(np.ceil(float(max_depth_m)))
        depth_max_display = max(cfg.depth_min_m, depth_max_display)
    else:
        depth_max_display = int(cfg.depth_max_m)


    # Build (and cache) the static depth scale layer (ticks + numbers).
    # This is expensive to draw and does not change frame-to-frame, only the vertical offset changes.
    depth_scale_key = _layout_c_depth_scale_cache_key(
        cfg=cfg,
        depth_max_display=depth_max_display,
        W=W,
        H=H,
        pad_top=pad_top,
        pad_bot=pad_bot,
        font_path=font_path,
    )
    moving = _get_layout_c_depth_scale_moving_cached(depth_scale_key)

    # Move scale so current depth aligns to indicator
    offset_y = int(round(indicator_y - (pad_top + current_depth_m * cfg.px_per_m)))

    moved = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    moved.alpha_composite(moving, (0, offset_y))
    clipped = moved.crop((0, y0, W, y1))  # RGBA

    # Fade edges (on clipped) - cache the mask to avoid per-frame numpy work
    if cfg.fade_enable and cfg.fade_margin_px > 0 and 0.0 <= cfg.fade_edge_transparency < 1.0:
        win_h = clipped.size[1]
        fade = int(cfg.fade_margin_px)
        fade = max(0, min(fade, win_h // 2))

        if fade > 0:
            mask = _get_layout_c_fade_mask_cached(
                W=W,
                win_h=win_h,
                fade=fade,
                edge_transparency=float(cfg.fade_edge_transparency),
            )
            a = clipped.getchannel("A")
            a2 = ImageChops.multiply(a, mask)  # alpha * (mask/255)
            clipped.putalpha(a2)

    # --- Build a clean module layer (scale window + arrow + value + unit) ---
    layer = PILImage.new("RGBA", out.size, (0, 0, 0, 0))
    layer.alpha_composite(clipped, (0, y0))

    ld = ImageDraw.Draw(layer)

    # Value text + arrow
    depth_txt = f"{current_depth_m:.1f}"
    vb = ld.textbbox((0, 0), depth_txt, font=value_font)
    v_w = vb[2] - vb[0]
    v_h = vb[3] - vb[1]

    value_center_y = indicator_y + cfg.value_y_offset
    arrow_cy = value_center_y + cfg.arrow_y_offset
    arrow_right_x = cfg.value_x - cfg.arrow_to_value_gap

    tri = [
        (arrow_right_x, arrow_cy - cfg.arrow_h // 2),
        (arrow_right_x, arrow_cy + cfg.arrow_h // 2),
        (arrow_right_x - cfg.arrow_w, arrow_cy),
    ]
    ld.polygon(tri, fill=cfg.arrow_color)
    ld.text(
        (int(cfg.value_x), int(value_center_y - v_h // 2)),
        depth_txt,
        font=value_font,
        fill=cfg.depth_value_color,
    )

    # Unit
    unit_txt = "m"
    ub = ld.textbbox((0, 0), unit_txt, font=unit_font)
    u_h = ub[3] - ub[1]

    if getattr(cfg, "unit_follow_value", True):
        unit_x = cfg.value_x + v_w + cfg.unit_gap_px
    else:
        if getattr(cfg, "unit_x_fixed", 0) > 0:
            unit_x = cfg.unit_x_fixed
        else:
            v_max_bbox = ld.textbbox((0, 0), "88.8", font=value_font)
            v_max_w = v_max_bbox[2] - v_max_bbox[0]
            unit_x = cfg.value_x + v_max_w + cfg.unit_gap_px

    unit_x = int(unit_x + cfg.unit_offset_x)
    unit_y = int((value_center_y - u_h // 2) + cfg.unit_offset_y)

    ld.text((unit_x, unit_y), "m", font=unit_font, fill=cfg.depth_value_color)

    # Shadow on the module layer
    layer = _apply_shadow_layer(
        layer,
        offset=(5, 6),
        blur_radius=8,
        shadow_color=(0, 0, 0, 100),
    )

    # Composite back
    out.alpha_composite(layer)
    return out

# ============================================================
# Layout C - Rate Module
# ============================================================

@dataclass
class LayoutCRateConfig:
    enabled: bool = True  # original = True

    global_x: int = 760
    global_y: int = 125   # original = 60

    label_font_size: int = 46  # original = 52
    value_font_size: int = 98  # original = 118
    unit_font_size: int = 48  # original = 58

    decimals: int = 2

    label_color: tuple = (115, 255, 157, 255)   # Descent label
    value_color: tuple = (255, 255, 255, 255)   # original = (170, 210, 230, 255)
    unit_color: tuple = (255, 255, 255, 255)
    arrow_color: tuple = (255, 255, 255, 255)

    label_ox: int = 35  # original = 70
    label_oy: int = 0
    arrow_ox: int = 0  # original = 20
    arrow_oy: int = 6  # original = 0
    value_ox: int = 30
    value_oy: int = 20  # original = 65
    unit_ox: int = -3  # original = 10
    unit_oy: int = 46  # original = 120

    unit_follow_value: bool = True
    unit_gap_px: int = 10

    arrow_w: int = 20
    arrow_h: int = 24



@dataclass
class LayoutCHeartRateConfig:
    enabled: bool = True


    # Global anchor (top-left)
    global_x: int = 40
    global_y: int = 120

    # Icon
    icon_size: int = 85  # original = 100

    # Value text
    value_font_size: int = 100  # original = 120
    value_color: tuple = (255, 255, 255, 255)

    # Offsets (relative to global)
    icon_ox: int = 0
    icon_oy: int = 0
    value_ox: int = 85  # original = 110
    value_oy: int = 10

    # Animation
    pulse_amp: float = 0.08




@dataclass
class LayoutCTimeConfig:
    enabled: bool = True

    # Global anchor (top-left of time text)
    global_x: int = 205
    global_y: int = 1740

    # Text style
    font_size: int = 62
    color: tuple = (255, 255, 255, 255)

    # Colon uses base font due to Nereus lacking ":"
    colon_font_size: int = 44

    # Optional fine-tune offsets
    ox: int = 0
    oy: int = 0
    part_gap_px: int = 2  # extra gap between parts (mm ":" ss)

# ==========================================================
# Layout C - Centralized manual tuning block
#   Adjust Layout C (Depth / Rate / Heart Rate / Shadow) here
# ==========================================================
@dataclass
class LayoutCTempConfig:
    """Layout C temperature module configuration (reuses Layout D design defaults)."""
    enabled: bool = True

    # Positioning
    # If x/y are -1, the module is anchored to bottom-right using margin_right/margin_bottom.
    x: int = 860   # original = 860
    y: int = -1  # original = -1
    global_x: int = 0
    global_y: int = -40
    margin_right: int = 60
    margin_bottom: int = 80

    # Update rate for temperature value (Hz). Overlay can render at higher FPS; this throttles
    # only the temperature sampling/refresh. Recommended options: 1 / 2 / 5 / 15.
    temp_update_fps: int = 15

    # Icon + text
    icon_size: int = 100
    gap_px: int = 14

    # Font
    # Value (temperature number)
    value_font_size: int = 60
    decimals: int = 1
    value_ox: int = -20
    value_oy: int = -25

    # Unit ("°C")
    unit_font_size: int = 40
    unit_gap_px: int = 6
    unit_ox: int = 0
    unit_oy: int = -11

    # Degree symbol rendering
    # If True, draw degree as a small hollow dot (circle) instead of the "°" glyph.
    deg_as_dot: bool = True
    deg_dot_r_px: int = 4
    deg_dot_line_px: int = 3
    deg_dot_dx: int = 0
    deg_dot_dy: int = 5

    # Color (RGBA)
    color: tuple = (255, 255, 255, 255)

    # Hard shadow (no blur)
    shadow_enable: bool = True
    shadow_dx: int = 2
    shadow_dy: int = 2
    shadow_alpha: int = 80
    shadow_color_rgb: tuple = (0, 0, 0)

    # Unit: show as "°C" where:
    # - "°" uses base font (to avoid missing glyph in nereus)
    # - "C" uses nereus-bold
    deg_symbol: str = "°"
    unit_c: str = "C"
    
    temp_time_shift_s: float = -12.0  # e.g. 10.0 or 15.0

    # Optional: if input temperature looks like Kelvin, auto-convert to Celsius.
    auto_kelvin_to_c: bool = True
    kelvin_threshold: float = 100.0  # values above this are treated as Kelvin

    # Column name candidates (first match wins)
    temp_col_candidates: Tuple[str, ...] = (
        "water_temp_c",
        "water_temperature_c",
        "temperature_c",
        "temp_c",
        "water_temp",
        "water_temperature",
        "temperature",
        "temp",
    )

# ===================
# Layout D (Heart rate module)
# ===================

# Cache for (icon_path, icon_h) -> (outline_rgba, inside_mask_L)
_HR_ICON_CACHE: dict = {}



# Layout C - Temperature module (reuse Layout D design)


@dataclass
class LayoutCShadowConfig:
    """Layout C shadow control.

    shadow_mode:
      - "off": no shadow
      - "solid": solid translucent shadow with dx/dy offset (no blur)
    """
    shadow_mode: str = "solid"       # "off" | "solid"
    enabled: bool = True             # legacy switch; ignored when shadow_mode="off"
    offset: tuple = (4, 4)           # (dx, dy) pixels
    blur_radius: int = 0             # kept for backward-compat; not used in solid mode
    color: tuple = (80, 80, 80, 120) # RGBA (solid gray translucent)

@dataclass
class LayoutCAllConfig:
    depth: "LayoutCDepthConfig"
    rate: "LayoutCRateConfig"
    hr: "LayoutCHeartRateConfig"
    time: "LayoutCTimeConfig"
    temp: "LayoutCTempConfig"
    shadow: LayoutCShadowConfig

def get_layout_c_config() -> LayoutCAllConfig:
    """Single source of truth for Layout C defaults.

    Important: Layout C defaults should be independently tunable via the
    corresponding LayoutC*Config dataclasses. Therefore, do NOT hard-code
    overrides here—instantiate each config with its own defaults.
    """
    shadow_cfg = LayoutCShadowConfig()
    depth_cfg = LayoutCDepthConfig()
    rate_cfg = LayoutCRateConfig()
    hr_cfg = LayoutCHeartRateConfig()
    time_cfg = LayoutCTimeConfig()
    temp_cfg = LayoutCTempConfig()

    return LayoutCAllConfig(
        depth=depth_cfg,
        rate=rate_cfg,
        hr=hr_cfg,
        time=time_cfg,
        temp=temp_cfg,
        shadow=shadow_cfg,
    )

# Default Layout C shadow parameters (used by all Layout C modules)
_layout_c_shadow_defaults = get_layout_c_config().shadow
LAYOUT_C_SHADOW_ENABLED = bool(_layout_c_shadow_defaults.enabled)
LAYOUT_C_SHADOW_OFFSET = tuple(_layout_c_shadow_defaults.offset)
LAYOUT_C_SHADOW_BLUR = int(_layout_c_shadow_defaults.blur_radius)
LAYOUT_C_SHADOW_COLOR = tuple(_layout_c_shadow_defaults.color)


def _load_font_path(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return _get_font_cached(Path(path) if path else None, int(size))

# ==========================================================
# Shadow helper for Layout C modules (BBox optimized)
# ==========================================================
from PIL import ImageFilter

def _apply_shadow_layer(
    img: PILImage.Image,
    offset: tuple = (4, 4),
    blur_radius: int = 6,
    shadow_color=(0, 0, 0, 120),
) -> PILImage.Image:
    """
    Apply a drop shadow to the non-transparent area of img.

    Optimized:
    - Only process the bounding box of non-zero alpha instead of the full frame.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        return img  # nothing to shadow

    dx, dy = int(offset[0]), int(offset[1])
    br = int(blur_radius) if blur_radius is not None else 0

    # Padding to contain shadow region
    pad = max(abs(dx), abs(dy)) + (br * 2) + 2

    x0 = max(0, bbox[0] - pad)
    y0 = max(0, bbox[1] - pad)
    x1 = min(img.size[0], bbox[2] + pad)
    y1 = min(img.size[1], bbox[3] + pad)
    crop_box = (x0, y0, x1, y1)

    # Crop to small region
    img_c = img.crop(crop_box)
    alpha_c = img_c.split()[-1]

    # Build shadow only on cropped region
    shadow_c = PILImage.new("RGBA", img_c.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_c)
    shadow_draw.bitmap((0, 0), alpha_c, fill=shadow_color)

    if br > 0:
        shadow_c = shadow_c.filter(ImageFilter.GaussianBlur(br))

    # Compose shadow + original in cropped space
    base_c = PILImage.new("RGBA", img_c.size, (0, 0, 0, 0))
    base_c.paste(shadow_c, (dx, dy), shadow_c)
    base_c.paste(img_c, (0, 0), img_c)

    # Paste back to full size canvas
    out = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(base_c, (x0, y0), base_c)
    return out


def _try_render_textbbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """
    Safe wrapper for text bbox across Pillow versions.
    Returns (x0, y0, x1, y1).
    """
    try:
        return draw.textbbox((0, 0), text, font=font)
    except Exception:
        try:
            w, h = font.getsize(text)
        except Exception:
            w, h = (0, 0)
        return (0, 0, w, h)

# Layout C Heart Rate module

def render_layout_c_rate_module(
    base_img: PILImage.Image,
    speed_mps_signed: float,
    cfg: LayoutCRateConfig,
    *,
    is_descent_override: Optional[bool] = None,
) -> PILImage.Image:
    """
    Layout C Rate module:
      - Label: Klavika (Descent Rate / Ascent Rate)
      - Value: Nereus (same as depth numeric)
      - Unit: base font (RobotoCondensedBold) so "/" renders and size is controllable
      - Direction: arrow + label (value displayed as absolute)
    """
    if not cfg.enabled:
        return base_img

    # --- Draw on a dedicated transparent layer so shadow is clean ---
    layer = PILImage.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    if is_descent_override is not None:
        is_descent = bool(is_descent_override)
    else:
        is_descent = float(speed_mps_signed) >= 0.0

    label_txt = "Descent Rate" if is_descent else "Ascent Rate"
    label_color = cfg.label_color if is_descent else (255, 184, 166, 255)  # #FFB8A6

    v = abs(float(speed_mps_signed))
    value_txt = f"{v:.{int(cfg.decimals)}f}"
    unit_txt = "m/s"

    label_font = _load_font_path(LAYOUT_C_RATE_FONT_PATH, int(cfg.label_font_size))
    value_font = _load_font_path(LAYOUT_C_VALUE_FONT_PATH, int(cfg.value_font_size))
    unit_font = load_font(int(cfg.unit_font_size))  # base font => "/" OK and size controllable

    gx = int(cfg.global_x)
    gy = int(cfg.global_y)

    # Arrow
    ax = gx + int(cfg.arrow_ox)
    ay = gy + int(cfg.arrow_oy)
    aw = int(cfg.arrow_w)
    ah = int(cfg.arrow_h)

    if is_descent:
        tri = [(ax, ay), (ax + aw, ay), (ax + aw // 2, ay + ah)]          # down
    else:
        tri = [(ax + aw // 2, ay), (ax, ay + ah), (ax + aw, ay + ah)]    # up

    draw.polygon(tri, fill=cfg.arrow_color)

    # Label
    lx = gx + int(cfg.label_ox)
    ly = gy + int(cfg.label_oy)
    draw.text((int(lx), int(ly)), label_txt, font=label_font, fill=label_color)

    # Value
    vx = gx + int(cfg.value_ox)
    vy = gy + int(cfg.value_oy)
    draw.text((int(vx), int(vy)), value_txt, font=value_font, fill=cfg.value_color)

    # Unit
    vb = _try_render_textbbox(draw, value_txt, value_font)
    value_w = vb[2] - vb[0]

    if bool(cfg.unit_follow_value):
        ux = vx + value_w + int(cfg.unit_gap_px) + int(cfg.unit_ox)
    else:
        ux = vx + int(cfg.unit_ox)

    uy = gy + int(cfg.unit_oy)
    draw.text((int(ux), int(uy)), unit_txt, font=unit_font, fill=cfg.unit_color)

    # Shadow on the module layer (arrow + label + value + unit)
    if bool(LAYOUT_C_SHADOW_ENABLED):
        layer = _apply_shadow_layer(
            layer,
            offset=LAYOUT_C_SHADOW_OFFSET,
            blur_radius=int(LAYOUT_C_SHADOW_BLUR),
            shadow_color=LAYOUT_C_SHADOW_COLOR,
        )

    # Composite back onto base image
    out = base_img.copy().convert("RGBA")
    out.alpha_composite(layer)
    return out

# ============================================================


def _remove_dark_bg_to_alpha(img: PILImage.Image, threshold: int = 10) -> PILImage.Image:
    """Convert near-black pixels to transparent alpha while preserving white parts.

    Used for icons that are white-on-black (e.g., thermo.png). Threshold is on max(R,G,B).
    """
    rgba = img.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    rgb = arr[..., :3]
    alpha = arr[..., 3].copy()
    mask_dark = (rgb.max(axis=2) <= int(threshold))
    alpha[mask_dark] = 0
    arr[..., 3] = alpha
    return PILImage.fromarray(arr, mode="RGBA")

def _resize_icon(base_icon: PILImage.Image, scale: float, size: int) -> PILImage.Image:
    s = max(1, int(round(float(size) * float(scale))))
    try:
        return base_icon.resize((s, s), resample=PILImage.BICUBIC)
    except Exception:
        return base_icon.resize((s, s))

def _resize_icon_keep_aspect(base_icon: PILImage.Image, box: int) -> PILImage.Image:
    """Resize icon to fit within (box x box) while keeping aspect ratio."""
    box = max(1, int(box))
    try:
        im = base_icon.copy()
        im.thumbnail((box, box), resample=PILImage.LANCZOS)
        return im
    except Exception:
        # Fallback: keep square resize
        try:
            return base_icon.resize((box, box), resample=PILImage.BICUBIC)
        except Exception:
            return base_icon.resize((box, box))


# ==========================================================
# HR icon resize cache (Layout C)
#   Avoid per-frame resize when pulse_scale produces same target size.
#   Keyed by (icon_path, target_px).
# ==========================================================
from functools import lru_cache

@lru_cache(maxsize=256)
def _get_icon_resized_cached(icon_key: str, target_px: int) -> PILImage.Image:
    """Return resized RGBA icon cached by (icon_key, target_px).

    icon_key is typically str(path_to_icon). The function will:
    - Ensure the base icon is loaded and stored in _ICON_BASE_CACHE
    - Resize to (target_px, target_px) with high-quality resampling
    """
    target_px = max(1, int(target_px))

    # Ensure base icon is available
    base = _ICON_BASE_CACHE.get(icon_key)
    if base is None:
        try:
            p = Path(icon_key)
            if not p.exists():
                raise FileNotFoundError(icon_key)
            _im = PILImage.open(str(p)).convert("RGBA")
            _im = _remove_dark_bg_to_alpha(_im, threshold=10)
            base = _im
            _ICON_BASE_CACHE[icon_key] = base
        except Exception:
            # In worst case, return a 1x1 transparent image to avoid crash
            return PILImage.new("RGBA", (1, 1), (0, 0, 0, 0))

    try:
        return base.resize((target_px, target_px), resample=PILImage.LANCZOS)
    except Exception:
        return base.resize((target_px, target_px))




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
    if not cfg.enabled:
        return base_img
    if not show_icon:
        return base_img

    icon_path = (Path(assets_dir) / LAYOUT_C_HR_ICON_REL)
    if not icon_path.exists():
        return base_img
    # Heart icon (cached decode + optional bg removal)
    icon_key = str(icon_path)
    icon_base = _ICON_BASE_CACHE.get(icon_key)
    if icon_base is None:
        try:
            _im = PILImage.open(str(icon_path)).convert("RGBA")
            # remove any near-black background pixels if present (safe for transparent PNGs)
            _im = _remove_dark_bg_to_alpha(_im, threshold=10)
            icon_base = _im
            _ICON_BASE_CACHE[icon_key] = icon_base
        except Exception:
            return base_img

    # --- Draw HR module onto its own transparent layer (so shadow is clean) ---
    layer = PILImage.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    gx = int(cfg.global_x)
    gy = int(cfg.global_y)
    icon_size = int(cfg.icon_size)

    # Icon (animated scale)
    target_px = max(1, int(round(float(icon_size) * float(pulse_scale))))
    icon_scaled = _get_icon_resized_cached(icon_key, target_px)
    ox = (icon_size - icon_scaled.size[0]) // 2
    oy = (icon_size - icon_scaled.size[1]) // 2
    ix = gx + int(cfg.icon_ox) + ox
    iy = gy + int(cfg.icon_oy) + oy
    layer.paste(icon_scaled, (int(ix), int(iy)), icon_scaled)

    # Value text
    if show_value:
        # use base font for "--", Nereus for bpm
        if str(hr_text).strip() == "":     # original = "--"
            value_font = _load_font_path(FONT_PATH, cfg.value_font_size)
        else:
            value_font = _load_font_path(LAYOUT_C_VALUE_FONT_PATH, cfg.value_font_size)

        vx = gx + int(cfg.value_ox)
        vy = gy + int(cfg.value_oy)
        draw.text((int(vx), int(vy)), str(hr_text), font=value_font, fill=cfg.value_color)

    # Shadow on the module layer (icon + text)
    if LAYOUT_C_SHADOW_ENABLED:
        layer = _apply_shadow_layer(
            layer,
            offset=LAYOUT_C_SHADOW_OFFSET,
            blur_radius=LAYOUT_C_SHADOW_BLUR,
            shadow_color=LAYOUT_C_SHADOW_COLOR,
        )

    # Composite back onto base image
    out = base_img.copy().convert("RGBA")
    out.alpha_composite(layer)
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
    """Render Layout C dive time (mm:ss). Uses Nereus for digits; ':' uses base font."""
    if not cfg.enabled:
        return base_img

    text_mmss = _format_mmss(time_s)
    mm_txt, ss_txt = text_mmss.split(":")
    colon_txt = ":"

    # Fonts
    nereus_font = _load_font_path(LAYOUT_C_VALUE_FONT_PATH, int(cfg.font_size))
    colon_size = int(cfg.colon_font_size) if cfg.colon_font_size else int(cfg.font_size)
    colon_font = _load_font_path(FONT_PATH, colon_size)  # base font so ":" renders

    # Dedicated layer for clean shadow
    layer = PILImage.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    x0 = int(cfg.global_x) + int(cfg.ox)
    y0 = int(cfg.global_y) + int(cfg.oy)

    # Measure parts
    b_mm = _try_render_textbbox(draw, mm_txt, nereus_font)
    w_mm = b_mm[2] - b_mm[0]

    b_col = _try_render_textbbox(draw, colon_txt, colon_font)
    w_col = b_col[2] - b_col[0]

    # Draw mm
    draw.text((x0, y0), mm_txt, font=nereus_font, fill=cfg.color)

    # Draw colon
    x1 = x0 + w_mm + int(cfg.part_gap_px)
    draw.text((x1, y0), colon_txt, font=colon_font, fill=cfg.color)

    # Draw ss
    x2 = x1 + w_col + int(cfg.part_gap_px)
    draw.text((x2, y0), ss_txt, font=nereus_font, fill=cfg.color)

    # Shadow (same global params as other Layout C modules)
    if bool(LAYOUT_C_SHADOW_ENABLED):
        layer = _apply_shadow_layer(
            layer,
            offset=LAYOUT_C_SHADOW_OFFSET,
            blur_radius=int(LAYOUT_C_SHADOW_BLUR),
            shadow_color=LAYOUT_C_SHADOW_COLOR,
        )

    out = base_img.copy().convert("RGBA")
    out.alpha_composite(layer)
    return out


# Layout A: bottom bar
# ============================================================


# ===================
# Layout D (Depth Module only - v0)
# ===================

@dataclass
class LayoutDDepthConfig:
    enabled: bool = True

    # Position (top-left) of the 250x250 module on the output frame
    x: int = 60
    y: int = 240
    global_x: int = -20
    global_y: int = -90    # original = 0

    # Backplate / chart size
    plate_w: int = 250
    plate_h: int = 220
    chart_w: int = 250
    chart_h: int = 190  # curve area height (top-aligned to plate)
    radius: int = 15

    # Static visuals (merged plate + curve) - user asked to render once
    plate_color_hex: str = "#BFBFBF"  # original = #D9D9D9
    plate_alpha: int = 70  # 20% opacity

    curve_color: tuple = (255, 255, 255, 188)  # 50% opacity (merged object request)
    curve_width: int = 6

    fill_color_hex: str = "#00FFFF"  #original = #4E95D9
    fill_alpha: int = 158  # 50% opacity (merged object request)
    # Gradient fill (top more transparent)
    fill_gradient_enabled: bool = True
    fill_alpha_bottom: int = 255   # bottom alpha (more opaque)
    fill_alpha_top: int = 10       # top alpha (more transparent)
    fill_gradient_gamma: float = 1.6

    gap_px: int = 13  # blank gap between fill and curve, along curve (implemented as y-shift)

    # Progressive curve reveal (show only left of indicator)
    progressive_curve_enabled: bool = True
    progressive_reveal_extra_px: int = 0  # +px to reveal slightly ahead of indicator
    # Indicator
    dot_diameter: int = 18
    dot_color: tuple = (255, 255, 255, 255)
    vline_width: int = 2
    vline_color: tuple = (255, 255, 255, 255)

    # Depth text
    value_font_size: int = 70
    unit_font_size: int = 50
    decimals: int = 1
    text_color: tuple = (255, 255, 255, 255)

    # Depth value hard shadow (numbers only)
    depth_value_shadow_enable: bool = True  # original = False
    depth_value_shadow_dx: int = 2
    depth_value_shadow_dy: int = 2
    depth_value_shadow_alpha: int = 80
    depth_value_shadow_color_rgb: tuple = (0, 0, 0)

    # Text anchor (within plate)
    text_x: int = 130
    text_y: int = 20
    unit_gap_px: int = 4  # gap between value and unit (if not using bbox alignment)

    # Keep unit bottom aligned to value bottom
    unit_bottom_align: bool = True


    # Text alignment for value+unit block within plate
    # - 'left': text_x is left edge
    # - 'center': text_x is center anchor
    text_align: str = "center"

def _hex_to_rgba(hex_str: str, alpha: int) -> tuple:
    s = str(hex_str).strip().lstrip("#")
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b, int(alpha))


def _clip_int(v: float, lo: int, hi: int) -> int:
    try:
        return int(max(lo, min(hi, int(round(v)))))
    except Exception:
        return lo


@dataclass
class LayoutDTimeConfig:
    enabled: bool = True

    # Anchor: relative to the depth module plate (default = directly below the plate)
    x_offset: int = 0
    y_offset: int = 0
    below_gap_px: int = 16  # vertical gap below depth plate

    # Color (RGBA)
    # Legacy 'color' kept for backward-compat (fallback for value_color)
    color: tuple = (0, 255, 255, 255)
    label_color: tuple = (255, 255, 255, 255)
    value_color: tuple = (0, 255, 255, 255)

    # Hard shadow (no blur)
    shadow_enable: bool = True
    shadow_dx: int = 2
    shadow_dy: int = 2
    shadow_alpha: int = 80
    shadow_color_rgb: tuple = (0, 0, 0)

    # ------------------------------------------------------------
    # Split elements: label / colon / minutes / apostrophe / seconds
    # All positions are offsets from the time module origin:
    #   origin = (depth_plate_x + x_offset, depth_plate_y + plate_h + below_gap_px + y_offset)
    # ------------------------------------------------------------

    # 1) Label: "Dive Time"
    label_text: str = "Dive Time"
    label_font_size: int = 36
    label_ox: int = 0
    label_oy: int = 0

    stacked: bool = True
    time_row_oy: int = 20          # vertical offset of time row relative to label row (px)
    time_row_align_with_label: bool = True  # True: time row starts at label_x; False: starts at module origin x0


    # 2) Colon (between label and time), e.g. " : "
    colon_text: str = ""   # original = ":"
    colon_font_size: int = 30
    colon_ox: int = 0  # original = 0
    colon_oy: int = 0  # original = 0

    # 3) Minutes (MM)
    mm_font_size: int = 65
    mm_ox: int = 0  # original = 0
    mm_oy: int = 0  # original = 0

    # 4) Apostrophe (between MM and SS), e.g. " ' "
    apostrophe_text: str = "'"
    apostrophe_font_size: int = 36
    apostrophe_ox: int = 0  # original = 0
    apostrophe_oy: int = 10  # original = 0

    # 5) Seconds (SS)
    ss_font_size: int = 65
    ss_ox: int = 0
    ss_oy: int = 0

    # Optional fine spacing when you want auto-flow (used only if the *_ox fields are all 0)
    auto_flow_gap_px: int = 6



# ===================
# Layout D (Speed module)
# ===================

@dataclass
class LayoutDSpeedConfig:
    enabled: bool = True

    # Global module offset (applied to all elements)
    global_x: int = 0
    global_y: int = 20

    # Vertical placement: under the Layout D Time module time-row
    below_time_gap_px: int = 18

    # Color (RGBA)
    # Legacy 'color' kept for backward-compat (fallback for value_color)
    color: tuple = (0, 255, 255, 255)
    label_color: tuple = (255, 255, 255, 255)
    value_color: tuple = (0, 255, 255, 255)

    # Hard shadow (no blur)
    shadow_enable: bool = True
    shadow_dx: int = 2
    shadow_dy: int = 2
    shadow_alpha: int = 80
    shadow_color_rgb: tuple = (0, 0, 0)

    # Label: "Speed"
    label_text: str = "Speed"
    label_font_size: int = 36
    label_ox: int = 0
    label_oy: int = 0

    # Value (absolute, no +/-)
    value_font_size: int = 65
    decimals: int = 1
    value_row_oy: int = 20  # vertical offset of value row relative to label row
    value_ox: int = 7
    value_oy: int = 0

    # Unit: "m/s" (draw the slash manually as a line)
    unit_font_size: int = 50
    unit_text_left: str = "m"
    unit_text_right: str = "s"
    unit_gap_px: int = 10  # gap between value and unit block
    unit_ox: int = 0
    unit_oy: int = -3

    # Unit vertical alignment relative to value
    unit_bottom_align: bool = True
    unit_bottom_extra_dy: int = 0

    # Manual slash parameters (drawn as a line)
    slash_dx: int = 6          # +x from start to end
    slash_dy: int = -22         # -y from start to end (negative = up)
    slash_thickness_px: int = 4
    slash_gap_px: int = 4       # gap between 'm' and slash and 's'
    slash_ox: int = 0
    slash_oy: int = 25

# ===================
# Layout D (Temperature module)
# ===================

@dataclass
class LayoutDTempConfig:
    enabled: bool = True

    # Positioning
    # If x/y are -1, the module is anchored to bottom-right using margin_right/margin_bottom.
    x: int = 860   # original = -1
    y: int = -1
    global_x: int = 0
    global_y: int = -15
    margin_right: int = 60
    margin_bottom: int = 80

    # Update rate for temperature value (Hz). Overlay can render at higher FPS; this throttles
    # only the temperature sampling/refresh. Recommended options: 1 / 2 / 5 / 15.
    temp_update_fps: int = 15

    # Icon + text
    icon_size: int = 100
    gap_px: int = 14

    # Font
    # Value (temperature number)
    value_font_size: int = 56
    decimals: int = 1
    value_ox: int = -20
    value_oy: int = -25

    # Unit ("°C")
    unit_font_size: int = 40
    unit_gap_px: int = 6
    unit_ox: int = 0
    unit_oy: int = -11

    # Degree symbol rendering
    # If True, draw degree as a small hollow dot (circle) instead of the "°" glyph.
    deg_as_dot: bool = True
    deg_dot_r_px: int = 4
    deg_dot_line_px: int = 3
    deg_dot_dx: int = 0
    deg_dot_dy: int = 5

    # Color (RGBA)
    color: tuple = (255, 255, 255, 255)

    # Hard shadow (no blur)
    shadow_enable: bool = True
    shadow_dx: int = 2
    shadow_dy: int = 2
    shadow_alpha: int = 80
    shadow_color_rgb: tuple = (0, 0, 0)

    # Unit: show as "°C" where:
    # - "°" uses base font (to avoid missing glyph in nereus)
    # - "C" uses nereus-bold
    deg_symbol: str = "°"
    unit_c: str = "C"
    
    temp_time_shift_s: float = -12.0  # e.g. 10.0 or 15.0

    # Optional: if input temperature looks like Kelvin, auto-convert to Celsius.
    auto_kelvin_to_c: bool = True
    kelvin_threshold: float = 100.0  # values above this are treated as Kelvin

    # Column name candidates (first match wins)
    temp_col_candidates: Tuple[str, ...] = (
        "water_temp_c",
        "water_temperature_c",
        "temperature_c",
        "temp_c",
        "water_temp",
        "water_temperature",
        "temperature",
        "temp",
    )

# ===================
# Layout D (Heart rate module)
# ===================

# Cache for (icon_path, icon_h) -> (outline_rgba, inside_mask_L)
_HR_ICON_CACHE: dict = {}



# Layout C - Temperature module (reuse Layout D design)

@dataclass
class LayoutDHeartRateConfig:
    enabled: bool = True

    # Anchor: top-right (ICON is anchored to the right edge using margin_right)
    margin_right: int = 120
    margin_top: int = 150
    global_x: int = 0
    global_y: int = 0

    # Icon
    icon_rel: Path = LAYOUT_D_HR_ICON_REL
    icon_h: int = 40
    gap_px: int = 14

    # Value text
    value_font_size: int = 70
    color: tuple = (255, 255, 255, 255)
    value_ox: int = 0
    value_oy: int = -20

    # Animation: fill alpha (no scale)
    pulse_amp: float = 0.06
    pulse_period_s: float = 1.0

    # Follow real heart rate for pulse speed
    pulse_follow_hr: bool = True

    # Clamp alpha range: 0%~90%  => 255~25
    fill_alpha_min: int = 25     # most transparent (90% transparent)
    fill_alpha_max: int = 255    # fully filled (0% transparent)

    # (NEW) Phase-integrator state (Layout C style) to avoid jitter when bpm changes
    _anim_phase: float = 0.0
    _anim_bpm_active: Optional[float] = None
    _anim_bpm_pending: Optional[float] = None
    _anim_switch_pending: bool = False
    _anim_t_prev: Optional[float] = None

    # Hard shadow (no blur)
    shadow_enable: bool = True
    shadow_dx: int = 2
    shadow_dy: int = 2
    shadow_alpha: int = 120
    shadow_color_rgb: tuple = (80, 80, 80)


def _load_layout_d_hr_icon_assets(icon_path: Path, icon_h: int) -> Optional[tuple]:
    """
    Load HR outline icon, remove dark bg to alpha, resize by height (keep aspect),
    and compute an "inside" mask for fill animation.

    Returns:
      (outline_rgba, inside_mask_L) or None
    """
    key = (str(icon_path), int(icon_h))
    cached = _HR_ICON_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        base = PILImage.open(str(icon_path)).convert("RGBA")
    except Exception:
        return None

    # Remove dark background (supports icons with black background)
    try:
        base = _remove_dark_bg_to_alpha(base, threshold=10)
    except Exception:
        pass

    # Resize by height (keep aspect)
    try:
        outline = _resize_icon_keep_aspect(base, int(icon_h))
        outline = outline.convert("RGBA")
    except Exception:
        return None

    # Compute inside mask:
    # - boundary = alpha>0 (outline stroke)
    # - background = alpha==0
    # - flood fill background from edges => outside
    # - inside = background - outside
    try:
        a = np.array(outline.getchannel("A"), dtype=np.uint8)
        boundary = a > 0
        bg = ~boundary

        h, w = a.shape
        outside = np.zeros((h, w), dtype=bool)

        from collections import deque
        q = deque()

        # seed edges where bg is true
        for x in range(w):
            if bg[0, x] and not outside[0, x]:
                outside[0, x] = True
                q.append((0, x))
            if bg[h - 1, x] and not outside[h - 1, x]:
                outside[h - 1, x] = True
                q.append((h - 1, x))
        for y in range(h):
            if bg[y, 0] and not outside[y, 0]:
                outside[y, 0] = True
                q.append((y, 0))
            if bg[y, w - 1] and not outside[y, w - 1]:
                outside[y, w - 1] = True
                q.append((y, w - 1))

        # BFS 4-neighborhood
        while q:
            y, x = q.popleft()
            if y > 0 and bg[y - 1, x] and not outside[y - 1, x]:
                outside[y - 1, x] = True
                q.append((y - 1, x))
            if y < h - 1 and bg[y + 1, x] and not outside[y + 1, x]:
                outside[y + 1, x] = True
                q.append((y + 1, x))
            if x > 0 and bg[y, x - 1] and not outside[y, x - 1]:
                outside[y, x - 1] = True
                q.append((y, x - 1))
            if x < w - 1 and bg[y, x + 1] and not outside[y, x + 1]:
                outside[y, x + 1] = True
                q.append((y, x + 1))

        inside = bg & (~outside)
        inside_mask = PILImage.fromarray((inside.astype(np.uint8) * 255), mode="L")
    except Exception:
        # If inside mask fails, fallback to "no fill" behavior
        inside_mask = PILImage.new("L", outline.size, 0)

    _HR_ICON_CACHE[key] = (outline, inside_mask)
    return _HR_ICON_CACHE[key]


def render_layout_d_heart_rate_module(
    base_img: PILImage.Image,
    *,
    t_global_s: float,
    hr_value: Optional[float],
    cfg: LayoutDHeartRateConfig,
    assets_dir: Path,
    nereus_font_path: Path,
) -> PILImage.Image:
    """
    Layout D HR:
    - Top-right.
    - Icon has fixed size (no scaling animation).
    - Animation is "fill" inside the outline by varying alpha.
    - Value text position is fixed and LEFT-aligned; it does NOT shift when digits change.
    - Uses Layout C style phase-integrator to avoid jitter when bpm changes.
    """
    if not getattr(cfg, "enabled", True):
        return base_img

    if hr_value is None or (not np.isfinite(float(hr_value))):
        # If can't read HR => hide whole module
        return base_img

    # Resolve icon path (IMPORTANT: defines icon_path)
    icon_rel = getattr(cfg, "icon_rel", LAYOUT_D_HR_ICON_REL)
    icon_path = Path(assets_dir) / Path(icon_rel)
    if not icon_path.exists():
        alt = Path(assets_dir) / LAYOUT_C_HR_ICON_REL
        if alt.exists():
            icon_path = alt
        else:
            return base_img

    # Load icon assets (outline + inside mask)
    icon_h = int(getattr(cfg, "icon_h", 40))
    icon_assets = _load_layout_d_hr_icon_assets(icon_path, icon_h)
    if icon_assets is None:
        return base_img
    outline_icon, inside_mask = icon_assets

    # ===== Pulse -> fill alpha (CLAMPED RANGE, follow HR, phase-integrator) =====
    amp = float(getattr(cfg, "pulse_amp", 0.0))
    amin = int(getattr(cfg, "fill_alpha_min", 0))
    amax = int(getattr(cfg, "fill_alpha_max", 255))
    amin = max(0, min(255, amin))
    amax = max(0, min(255, amax))

    # Decide target bpm for animation
    target_bpm = None
    try:
        if getattr(cfg, "pulse_follow_hr", True) and float(hr_value) > 1.0:
            target_bpm = float(hr_value)
    except Exception:
        target_bpm = None

    # If not following HR, use fixed period -> equivalent bpm
    if target_bpm is None:
        per = float(getattr(cfg, "pulse_period_s", 1.0))
        if np.isfinite(per) and per > 1e-6:
            target_bpm = 60.0 / per
        else:
            target_bpm = 60.0  # safe fallback

    # Initialize state
    if getattr(cfg, "_anim_bpm_active", None) is None:
        cfg._anim_bpm_active = float(target_bpm)
        cfg._anim_bpm_pending = None
        cfg._anim_switch_pending = False
        cfg._anim_phase = 0.0
        cfg._anim_t_prev = None

    # Queue bpm change, but only switch at beat boundary (Layout C behavior)
    try:
        bpm_active = float(cfg._anim_bpm_active) if cfg._anim_bpm_active is not None else float(target_bpm)
        if abs(float(target_bpm) - bpm_active) > 0.1 and (not bool(getattr(cfg, "_anim_switch_pending", False))):
            cfg._anim_bpm_pending = float(target_bpm)
            cfg._anim_switch_pending = True
    except Exception:
        pass

    # Integrate phase with real dt to keep continuity
    t_now = float(t_global_s)
    t_prev = getattr(cfg, "_anim_t_prev", None)
    dt_real = 0.0 if (t_prev is None) else max(0.0, t_now - float(t_prev))
    cfg._anim_t_prev = t_now

    bpm_for_phase = float(cfg._anim_bpm_active or target_bpm or 0.0)
    if (not np.isfinite(bpm_for_phase)) or bpm_for_phase <= 0.0:
        bpm_for_phase = 60.0

    cfg._anim_phase = float(getattr(cfg, "_anim_phase", 0.0)) + (2.0 * math.pi * (bpm_for_phase / 60.0) * dt_real)

    # Wrap phase; switch bpm at boundary
    if cfg._anim_phase >= 2.0 * math.pi:
        # handle large dt as well
        cfg._anim_phase = math.fmod(cfg._anim_phase, 2.0 * math.pi)
        if bool(getattr(cfg, "_anim_switch_pending", False)) and (getattr(cfg, "_anim_bpm_pending", None) is not None):
            cfg._anim_bpm_active = float(cfg._anim_bpm_pending)
            cfg._anim_switch_pending = False
            cfg._anim_bpm_pending = None

    # Use sine of continuous phase
    if (not np.isfinite(amp)) or amp <= 0.0:
        fill_alpha = amax
    else:
        pulse_scale = 1.0 + amp * math.sin(float(cfg._anim_phase))

        # normalize to 0~1 based on [1-amp, 1+amp]
        denom = max(1e-9, 2.0 * amp)
        u = (pulse_scale - (1.0 - amp)) / denom
        u = max(0.0, min(1.0, u))

        # map to [amin, amax]
        fill_alpha = int(round(amin + (amax - amin) * u))
        fill_alpha = max(0, min(255, fill_alpha))

    # Build filled layer (white) using inside mask
    fill_layer = PILImage.new("RGBA", outline_icon.size, (255, 255, 255, 0))
    if fill_alpha > 0:
        a = inside_mask.point(lambda p: int(p * fill_alpha / 255))
        fill_layer.putalpha(a)

    # Compose final icon: fill under the outline stroke
    icon_img = fill_layer
    icon_img.alpha_composite(outline_icon)

    # Output image and draw helper
    out = base_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)

    # Fonts
    value_font_size = int(getattr(cfg, "value_font_size", 70))
    nereus_font = _get_font_cached(Path(nereus_font_path) if nereus_font_path else None, int(value_font_size))

    # HR text (integer bpm)
    try:
        hr_i = int(round(float(hr_value)))
    except Exception:
        hr_i = None
    if hr_i is None:
        return base_img
    hr_txt = f"{hr_i:d}"

    label_color = getattr(cfg, "label_color", (255, 255, 255, 255))
    color = getattr(cfg, "value_color", getattr(cfg, "color", (255, 255, 255, 255)))

    # Layout / positioning
    W, H = out.size
    margin_right = int(getattr(cfg, "margin_right", 120))
    margin_top = int(getattr(cfg, "margin_top", 150))
    gap_px = int(getattr(cfg, "gap_px", 14))
    gx = int(getattr(cfg, "global_x", 0))
    gy = int(getattr(cfg, "global_y", 0))

    # Text bbox (only for height / baseline alignment)
    tb = draw.textbbox((0, 0), hr_txt, font=nereus_font)
    th = (tb[3] - tb[1])

    # Icon size
    iw, ih = icon_img.size

    # ICON anchored to top-right; VALUE is fixed and LEFT-aligned next to icon
    icon_x = int((W - margin_right - iw) + gx)
    icon_y = int(margin_top + gy)

    text_x = int(icon_x + iw + gap_px + int(getattr(cfg, "value_ox", 0)) + gx)
    text_y = int(icon_y + ih - th + int(getattr(cfg, "value_oy", 0)))

    # Shadow params
    _sh_enable = bool(getattr(cfg, "shadow_enable", False))
    _sh_dx = int(getattr(cfg, "shadow_dx", 2))
    _sh_dy = int(getattr(cfg, "shadow_dy", 2))
    _sh_alpha = int(getattr(cfg, "shadow_alpha", 120))
    _sh_rgb = getattr(cfg, "shadow_color_rgb", (80, 80, 80))

    # Draw icon with hard shadow
    if _sh_enable:
        _sc = _rgba_shadow_color(_sh_rgb, _sh_alpha)
        _a = icon_img.getchannel("A")
        _sa = _a.point(lambda p: int(p * _sc[3] / 255))
        _sh_icon = PILImage.new("RGBA", icon_img.size, (_sc[0], _sc[1], _sc[2], 0))
        _sh_icon.putalpha(_sa)
        out.alpha_composite(_sh_icon, dest=(icon_x + _sh_dx, icon_y + _sh_dy))
    out.alpha_composite(icon_img, dest=(icon_x, icon_y))

    # Draw value text with hard shadow
    _draw_text_hard_shadow(
        draw,
        (text_x, text_y),
        hr_txt,
        font=nereus_font,
        fill=color,
        shadow_enable=_sh_enable,
        shadow_dx=_sh_dx,
        shadow_dy=_sh_dy,
        shadow_color_rgb=_sh_rgb,
        shadow_alpha=_sh_alpha,
    )

    return out


def _format_dive_time_mmss_tenths(t_s: float) -> tuple:
    """Return (mm, ss, d) with rounding to nearest 0.1s."""
    try:
        tt = float(t_s)
    except Exception:
        tt = 0.0
    if not np.isfinite(tt):
        tt = 0.0
    tt = max(0.0, tt)
    tenths_total = int(round(tt * 10.0))
    mm = tenths_total // 600
    ss = (tenths_total // 10) % 60
    d = tenths_total % 10
    return mm, ss, d


def render_layout_d_time_module(
    base_img: Image.Image,
    t_global_s: float,
    depth_cfg: "LayoutDDepthConfig",
    cfg: "LayoutDTimeConfig",
    nereus_font_path: Optional[Path],
    base_font_path: Optional[Path],
) -> Image.Image:
    """Draw Layout D time module (split elements).

    Elements (independently configurable):
      1) Label: "Dive Time"
      2) Colon: ": "
      3) Minutes: MM
      4) Apostrophe: "'"
      5) Seconds: SS

    Font rule:
      - English letters & digits: Nereus Bold
      - Punctuation (':', "'") uses base font (fallback).
    """
    if not getattr(cfg, "enabled", True):
        return base_img

    out = base_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)


    # Hard shadow (fast path): draw shadow text at offset, then normal text
    _sh_enable = bool(getattr(cfg, 'shadow_enable', False))
    _sh_dx = int(getattr(cfg, 'shadow_dx', 4))
    _sh_dy = int(getattr(cfg, 'shadow_dy', 4))
    _sh_alpha = int(getattr(cfg, 'shadow_alpha', 140))
    _sh_rgb = getattr(cfg, 'shadow_color_rgb', (0, 0, 0))

    def _dt(xy, s, *, font, fill):
        _draw_text_hard_shadow(draw, xy, s, font=font, fill=fill,
                              shadow_enable=_sh_enable, shadow_dx=_sh_dx, shadow_dy=_sh_dy,
                              shadow_color_rgb=_sh_rgb, shadow_alpha=_sh_alpha)

    # Base origin: directly below the depth plate
    ox = int(depth_cfg.x + depth_cfg.global_x)
    oy = int(depth_cfg.y + depth_cfg.global_y)
    x0 = ox + int(getattr(cfg, "x_offset", 0))
    y0 = oy + int(getattr(depth_cfg, "plate_h", 0)) + int(getattr(cfg, "below_gap_px", 0)) + int(getattr(cfg, "y_offset", 0))

    # Time: mm:ss (no tenths) -> we will draw MM and SS separately
    try:
        tt = float(t_global_s)
    except Exception:
        tt = 0.0
    if not np.isfinite(tt):
        tt = 0.0
    tt = max(0.0, tt)
    mm = int(tt // 60)
    ss = int(tt % 60)
    mm_txt = f"{mm:02d}"
    ss_txt = f"{ss:02d}"

    label_color = getattr(cfg, "label_color", (255, 255, 255, 255))

    color = getattr(cfg, "value_color", getattr(cfg, "color", (255, 255, 255, 255)))

    # Load fonts (independent sizes)
    def _safe_font(path: Optional[Path], size: int):
        return _get_font_cached(Path(path) if path else None, int(size))

    nereus_label = _safe_font(nereus_font_path, int(getattr(cfg, "label_font_size", 50)))
    base_colon   = _safe_font(base_font_path, int(getattr(cfg, "colon_font_size", 40)))
    nereus_mm    = _safe_font(nereus_font_path, int(getattr(cfg, "mm_font_size", 50)))
    base_apos    = _safe_font(base_font_path, int(getattr(cfg, "apostrophe_font_size", 40)))
    nereus_ss    = _safe_font(nereus_font_path, int(getattr(cfg, "ss_font_size", 50)))

    # Text strings
    label_txt = str(getattr(cfg, "label_text", "Dive Time"))
    colon_txt = str(getattr(cfg, "colon_text", ": "))
    apos_txt  = str(getattr(cfg, "apostrophe_text", "'"))

    # If user has not set explicit positions (all ox = 0), we auto-flow left-to-right.
    auto_flow = (
        int(getattr(cfg, "colon_ox", 0)) == 0 and int(getattr(cfg, "mm_ox", 0)) == 0 and
        int(getattr(cfg, "apostrophe_ox", 0)) == 0 and int(getattr(cfg, "ss_ox", 0)) == 0
    )
    gap = int(getattr(cfg, "auto_flow_gap_px", 6))

    stacked = bool(getattr(cfg, "stacked", False))
    time_row_oy = int(getattr(cfg, "time_row_oy", 0))
    align_with_label = bool(getattr(cfg, "time_row_align_with_label", True))

    # Auto-flow is for the TIME row only (colon/mm/apos/ss)
    auto_flow_time = (
        int(getattr(cfg, "colon_ox", 0)) == 0 and int(getattr(cfg, "mm_ox", 0)) == 0 and
        int(getattr(cfg, "apostrophe_ox", 0)) == 0 and int(getattr(cfg, "ss_ox", 0)) == 0
    )
    gap = int(getattr(cfg, "auto_flow_gap_px", 6))

    # --- Row 1: Label (always drawn at label_ox/label_oy) ---
    label_x = x0 + int(getattr(cfg, "label_ox", 0))
    label_y = y0 + int(getattr(cfg, "label_oy", 0))
    _dt((label_x, label_y), label_txt, font=nereus_label, fill=label_color)

    if stacked:
        # --- Row 2: Time row origin ---
        # Option A: align time row start with label_x (common in UI typography)
        # Option B: start from module origin x0
        time_x0 = label_x if align_with_label else x0
        time_y0 = label_y + time_row_oy

        if auto_flow_time:
            # Auto-flow within time row: colon -> mm -> apostrophe -> ss
            cx = time_x0 + int(getattr(cfg, "colon_ox", 0))  # still 0 in auto-flow mode
            cy = time_y0 + int(getattr(cfg, "colon_oy", 0))
            _dt((cx, cy), colon_txt, font=base_colon, fill=color)
            cw = draw.textbbox((0, 0), colon_txt, font=base_colon)[2]

            mx = cx + cw + gap
            my = time_y0 + int(getattr(cfg, "mm_oy", 0))
            _dt((mx, my), mm_txt, font=nereus_mm, fill=color)
            mw = draw.textbbox((0, 0), mm_txt, font=nereus_mm)[2]

            ax = mx + mw + gap
            ay = time_y0 + int(getattr(cfg, "apostrophe_oy", 0))
            _dt((ax, ay), apos_txt, font=base_apos, fill=color)
            aw = draw.textbbox((0, 0), apos_txt, font=base_apos)[2]

            sx = ax + aw + gap
            sy = time_y0 + int(getattr(cfg, "ss_oy", 0))
            _dt((sx, sy), ss_txt, font=nereus_ss, fill=color)
        else:
            # Explicit positioning within time row (offsets relative to time row origin)
            cx = time_x0 + int(getattr(cfg, "colon_ox", 0))
            cy = time_y0 + int(getattr(cfg, "colon_oy", 0))
            _dt((cx, cy), colon_txt, font=base_colon, fill=color)

            mx = time_x0 + int(getattr(cfg, "mm_ox", 0))
            my = time_y0 + int(getattr(cfg, "mm_oy", 0))
            _dt((mx, my), mm_txt, font=nereus_mm, fill=color)

            ax = time_x0 + int(getattr(cfg, "apostrophe_ox", 0))
            ay = time_y0 + int(getattr(cfg, "apostrophe_oy", 0))
            _dt((ax, ay), apos_txt, font=base_apos, fill=color)

            sx = time_x0 + int(getattr(cfg, "ss_ox", 0))
            sy = time_y0 + int(getattr(cfg, "ss_oy", 0))
            _dt((sx, sy), ss_txt, font=nereus_ss, fill=color)

    else:
        # --- Original single-row behavior (legacy) ---
        # Keep your original flow: label -> colon -> mm -> apostrophe -> ss
        # This preserves backward compatibility if you ever set stacked=False.

        if auto_flow_time:
            lx = x0 + int(getattr(cfg, "label_ox", 0))
            ly = y0 + int(getattr(cfg, "label_oy", 0))
            # (label already drawn above, but we redraw here for parity)
            _dt((lx, ly), label_txt, font=nereus_label, fill=label_color)
            lw = draw.textbbox((0, 0), label_txt, font=nereus_label)[2]

            cx = lx + lw + gap
            cy = y0 + int(getattr(cfg, "colon_oy", 0))
            _dt((cx, cy), colon_txt, font=base_colon, fill=color)
            cw = draw.textbbox((0, 0), colon_txt, font=base_colon)[2]

            mx = cx + cw + gap
            my = y0 + int(getattr(cfg, "mm_oy", 0))
            _dt((mx, my), mm_txt, font=nereus_mm, fill=color)
            mw = draw.textbbox((0, 0), mm_txt, font=nereus_mm)[2]

            ax = mx + mw + gap
            ay = y0 + int(getattr(cfg, "apostrophe_oy", 0))
            _dt((ax, ay), apos_txt, font=base_apos, fill=color)
            aw = draw.textbbox((0, 0), apos_txt, font=base_apos)[2]

            sx = ax + aw + gap
            sy = y0 + int(getattr(cfg, "ss_oy", 0))
            _dt((sx, sy), ss_txt, font=nereus_ss, fill=color)
        else:
            # Explicit positioning on one row (original)
            cx = x0 + int(getattr(cfg, "colon_ox", 0))
            cy = y0 + int(getattr(cfg, "colon_oy", 0))
            _dt((cx, cy), colon_txt, font=base_colon, fill=color)

            mx = x0 + int(getattr(cfg, "mm_ox", 0))
            my = y0 + int(getattr(cfg, "mm_oy", 0))
            _dt((mx, my), mm_txt, font=nereus_mm, fill=color)

            ax = x0 + int(getattr(cfg, "apostrophe_ox", 0))
            ay = y0 + int(getattr(cfg, "apostrophe_oy", 0))
            _dt((ax, ay), apos_txt, font=base_apos, fill=color)

            sx = x0 + int(getattr(cfg, "ss_ox", 0))
            sy = y0 + int(getattr(cfg, "ss_oy", 0))
            _dt((sx, sy), ss_txt, font=nereus_ss, fill=color)

        # Explicit positioning: each element uses its own offset from module origin.
        lx = x0 + int(getattr(cfg, "label_ox", 0))
        ly = y0 + int(getattr(cfg, "label_oy", 0))
        _dt((lx, ly), label_txt, font=nereus_label, fill=label_color)

        cx = x0 + int(getattr(cfg, "colon_ox", 0))
        cy = y0 + int(getattr(cfg, "colon_oy", 0))
        _dt((cx, cy), colon_txt, font=base_colon, fill=color)

        mx = x0 + int(getattr(cfg, "mm_ox", 0))
        my = y0 + int(getattr(cfg, "mm_oy", 0))
        _dt((mx, my), mm_txt, font=nereus_mm, fill=color)

        ax = x0 + int(getattr(cfg, "apostrophe_ox", 0))
        ay = y0 + int(getattr(cfg, "apostrophe_oy", 0))
        _dt((ax, ay), apos_txt, font=base_apos, fill=color)

        sx = x0 + int(getattr(cfg, "ss_ox", 0))
        sy = y0 + int(getattr(cfg, "ss_oy", 0))
        _dt((sx, sy), ss_txt, font=nereus_ss, fill=color)

    return out


def render_layout_d_speed_module(
    base_img: Image.Image,
    speed_mps: float,
    depth_cfg: "LayoutDDepthConfig",
    time_cfg: "LayoutDTimeConfig",
    cfg: "LayoutDSpeedConfig",
    nereus_font_path: Optional[Path],
    base_font_path: Optional[Path],
) -> Image.Image:
    """Draw Layout D speed module.

    Requirements:
      - Label "Speed" under the time VALUE row
      - Value uses absolute speed with 1 decimal (configurable)
      - Unit rendered as m/s, with a manually drawn slash line (configurable)
      - Left align label to time label ("Dive Time")
      - Left align value to time value (MM)
      - Unit placed to the right of value, bottom-aligned to value bottom
    """
    if not getattr(cfg, "enabled", True):
        return base_img

    out = base_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)


    # Hard shadow (fast path): draw shadow first, then normal
    _sh_enable = bool(getattr(cfg, 'shadow_enable', False))
    _sh_dx = int(getattr(cfg, 'shadow_dx', 4))
    _sh_dy = int(getattr(cfg, 'shadow_dy', 4))
    _sh_alpha = int(getattr(cfg, 'shadow_alpha', 140))
    _sh_rgb = getattr(cfg, 'shadow_color_rgb', (0, 0, 0))

    def _dt(xy, s, *, font, fill):
        _draw_text_hard_shadow(draw, xy, s, font=font, fill=fill,
                              shadow_enable=_sh_enable, shadow_dx=_sh_dx, shadow_dy=_sh_dy,
                              shadow_color_rgb=_sh_rgb, shadow_alpha=_sh_alpha)

    def _dl(p0, p1, *, fill, width):
        _draw_line_hard_shadow(draw, p0, p1, fill=fill, width=width,
                              shadow_enable=_sh_enable, shadow_dx=_sh_dx, shadow_dy=_sh_dy,
                              shadow_color_rgb=_sh_rgb, shadow_alpha=_sh_alpha)

    # ---------- Helpers ----------
    def _safe_font(path: Optional[Path], size: int):
        return _get_font_cached(Path(path) if path else None, int(size))

    def _bbox_wh(font: ImageFont.FreeTypeFont, s: str) -> tuple:
        try:
            b = font.getbbox(s)
            return (b[2] - b[0], b[3] - b[1])
        except Exception:
            return (0, 0)

    # ---------- Recompute Layout D time anchors (to align with it) ----------
    # Time module origin is below the depth plate
    ox = int(depth_cfg.x + depth_cfg.global_x)
    oy = int(depth_cfg.y + depth_cfg.global_y)
    x0 = ox + int(getattr(time_cfg, "x_offset", 0))
    y0 = oy + int(getattr(depth_cfg, "plate_h", 0)) + int(getattr(time_cfg, "below_gap_px", 0)) + int(getattr(time_cfg, "y_offset", 0))

    # Time label position
    time_label_x = x0 + int(getattr(time_cfg, "label_ox", 0))
    time_label_y = y0 + int(getattr(time_cfg, "label_oy", 0))

    # Time row origin
    stacked = bool(getattr(time_cfg, "stacked", False))
    time_row_oy = int(getattr(time_cfg, "time_row_oy", 0))
    align_with_label = bool(getattr(time_cfg, "time_row_align_with_label", True))

    # When stacked: time row starts at label_x by default
    time_x0 = (time_label_x if align_with_label else x0) if stacked else (time_label_x if align_with_label else x0)
    time_y0 = (time_label_y + time_row_oy) if stacked else time_label_y

    # Time MM start x (for alignment). We mirror the time renderer's default behavior.
    mm_x = time_x0 + int(getattr(time_cfg, "mm_ox", 0))
    mm_y = time_y0 + int(getattr(time_cfg, "mm_oy", 0))

    # Time row height (for placing speed below time value row)
    nereus_mm = _safe_font(nereus_font_path, int(getattr(time_cfg, "mm_font_size", 65)))
    nereus_ss = _safe_font(nereus_font_path, int(getattr(time_cfg, "ss_font_size", 65)))
    base_colon = _safe_font(base_font_path, int(getattr(time_cfg, "colon_font_size", 30)))
    base_apos = _safe_font(base_font_path, int(getattr(time_cfg, "apostrophe_font_size", 36)))

    colon_txt = str(getattr(time_cfg, "colon_text", ": "))
    apos_txt = str(getattr(time_cfg, "apostrophe_text", "'"))

    # Use representative glyphs for height estimation
    h_parts = [
        _bbox_wh(nereus_mm, "88")[1],
        _bbox_wh(nereus_ss, "88")[1],
        _bbox_wh(base_colon, colon_txt or ":")[1],
        _bbox_wh(base_apos, apos_txt or "'")[1],
    ]
    time_row_h = int(max(h_parts) if h_parts else 0)

    # ---------- Speed content ----------
    try:
        v = float(speed_mps)
    except Exception:
        v = 0.0
    if not np.isfinite(v):
        v = 0.0
    v = abs(v)

    decimals = int(getattr(cfg, "decimals", 1))
    value_txt = f"{v:.{decimals}f}"

    color = getattr(cfg, "color", (255, 255, 255, 255))

    font_label = _safe_font(nereus_font_path, int(getattr(cfg, "label_font_size", 36)))
    font_value = _safe_font(nereus_font_path, int(getattr(cfg, "value_font_size", 65)))
    font_unit  = _safe_font(nereus_font_path, int(getattr(cfg, "unit_font_size", 36)))

    gx = int(getattr(cfg, "global_x", 0))
    gy = int(getattr(cfg, "global_y", 0))

    # Label position: under time value row, left-aligned to time label
    label_x = time_label_x + int(getattr(cfg, "label_ox", 0)) + gx
    label_y = time_y0 + time_row_h + int(getattr(cfg, "below_time_gap_px", 18)) + int(getattr(cfg, "label_oy", 0)) + gy

    _dt((label_x, label_y), str(getattr(cfg, "label_text", "Speed")), font=font_label, fill=label_color)

    # Value position: under label, left-aligned to MM start
    value_x = mm_x + int(getattr(cfg, "value_ox", 0)) + gx
    value_y = label_y + int(getattr(cfg, "value_row_oy", 20)) + int(getattr(cfg, "value_oy", 0))

    _dt((value_x, value_y), value_txt, font=font_value, fill=color)

    # Value bbox for unit placement
    v_w, v_h = _bbox_wh(font_value, value_txt)

    # Unit block position: right of value
    unit_x = value_x + v_w + int(getattr(cfg, "unit_gap_px", 10)) + int(getattr(cfg, "unit_ox", 0))

    # Bottom align unit to value bottom if enabled
    unit_y = value_y + int(getattr(cfg, "unit_oy", 0))
    if bool(getattr(cfg, "unit_bottom_align", True)):
        # Estimate unit height with representative text "ms"
        u_h = _bbox_wh(font_unit, "ms")[1]
        unit_y = (value_y + v_h - u_h) + int(getattr(cfg, "unit_bottom_extra_dy", 0)) + int(getattr(cfg, "unit_oy", 0))

    # Draw unit "m/s" with manual slash
    left_txt = str(getattr(cfg, "unit_text_left", "m"))
    right_txt = str(getattr(cfg, "unit_text_right", "s"))

    # left part
    _dt((unit_x, unit_y), left_txt, font=font_unit, fill=color)
    left_w, left_h = _bbox_wh(font_unit, left_txt)

    slash_gap = int(getattr(cfg, "slash_gap_px", 4))
    sx = unit_x + left_w + slash_gap + int(getattr(cfg, "slash_ox", 0))
    sy = unit_y + int(max(0, (left_h * 0.75))) + int(getattr(cfg, "slash_oy", 0))

    dx = int(getattr(cfg, "slash_dx", 18))
    dy = int(getattr(cfg, "slash_dy", -28))
    thickness = int(getattr(cfg, "slash_thickness_px", 4))

    # Draw the slash line (as a thick line)
    _dl((sx, sy), (sx + dx, sy + dy), fill=color, width=max(1, thickness))

    # right part
    right_x = sx + dx + slash_gap
    # Align right part top with unit_y (simple, adjustable via unit_oy / slash_oy)
    _dt((right_x, unit_y), right_txt, font=font_unit, fill=color)

    return out



def render_layout_d_temp_module(
    base_img: PILImage.Image,
    *,
    temp_c: Optional[float],
    cfg: LayoutDTempConfig,
    assets_dir: Path,
    nereus_font_path: Path,
    base_font_path: Path,
) -> PILImage.Image:
    if not cfg.enabled:
        return base_img

    # Resolve icon path
    icon_path = (Path(assets_dir) / LAYOUT_D_TEMP_ICON_REL)
    if not icon_path.exists():
        alt = Path(assets_dir) / "thermo.png"
        if alt.exists():
            icon_path = alt
        else:
            return base_img

    # Icon cache: decode + remove dark bg + resize only once per (path, size)
    icon_size = int(getattr(cfg, "icon_size", 64))
    icon_key = (str(icon_path), int(icon_size))
    icon_img = _TEMP_ICON_CACHE.get(icon_key)
    if icon_img is None:
        try:
            icon_base = PILImage.open(str(icon_path)).convert("RGBA")
            icon_base = _remove_dark_bg_to_alpha(icon_base, threshold=10)
            # Keep original aspect ratio (avoid squash)
            icon_img = _resize_icon_keep_aspect(icon_base, icon_size)
            _TEMP_ICON_CACHE[icon_key] = icon_img
        except Exception:
            return base_img

    # Fonts (cached)
    value_font_size = int(getattr(cfg, "value_font_size", 60))
    unit_font_size = int(getattr(cfg, "unit_font_size", value_font_size))

    nereus_value_font = _get_font_cached(Path(nereus_font_path) if nereus_font_path else None, int(value_font_size))
    nereus_unit_font  = _get_font_cached(Path(nereus_font_path) if nereus_font_path else None, int(unit_font_size))
    base_unit_font    = _get_font_cached(Path(base_font_path) if base_font_path else None, int(unit_font_size))

    # Determine temperature value to display
    if temp_c is None or not np.isfinite(float(temp_c)):
        # No value: caller should already provide first/last hold; fall back to blank
        temp_str = ""
    else:
        decimals = int(getattr(cfg, "decimals", 1))
        temp_str = f"{float(temp_c):.{decimals}f}"

    # Text measurement
    W, H = base_img.size
    overlay = base_img.copy()
    draw = ImageDraw.Draw(overlay)


    # Hard shadow (fast path)
    _sh_enable = bool(getattr(cfg, 'shadow_enable', False))
    _sh_dx = int(getattr(cfg, 'shadow_dx', 4))
    _sh_dy = int(getattr(cfg, 'shadow_dy', 4))
    _sh_alpha = int(getattr(cfg, 'shadow_alpha', 140))
    _sh_rgb = getattr(cfg, 'shadow_color_rgb', (0, 0, 0))

    def _dt(xy, s, *, font, fill):
        _draw_text_hard_shadow(draw, xy, s, font=font, fill=fill,
                              shadow_enable=_sh_enable, shadow_dx=_sh_dx, shadow_dy=_sh_dy,
                              shadow_color_rgb=_sh_rgb, shadow_alpha=_sh_alpha)

    def _ellipse(bbox, *, outline, width):
        # Low-level ellipse draw (no recursion)
        draw.ellipse(bbox, outline=outline, width=width)

    def _de(bbox, *, outline, width):
        # ellipse outline with hard shadow
        if _sh_enable:
            sc = _rgba_shadow_color(_sh_rgb, _sh_alpha)
            x0, y0, x1, y1 = bbox
            _ellipse((x0 + _sh_dx, y0 + _sh_dy, x1 + _sh_dx, y1 + _sh_dy), outline=sc, width=width)
        _ellipse(bbox, outline=outline, width=width)

    # Measure value
    if temp_str:
        bbox_val = draw.textbbox((0, 0), temp_str, font=nereus_value_font)
        w_val = int(bbox_val[2] - bbox_val[0])
        h_val = int(bbox_val[3] - bbox_val[1])
    else:
        w_val = 0
        h_val = 0

    # Measure unit pieces
    # NOTE: degree may be drawn as dot; we reserve width accordingly.
    bbox_deg = draw.textbbox((0, 0), getattr(cfg, "deg_symbol", "°"), font=base_unit_font)
    w_deg_glyph = int(bbox_deg[2] - bbox_deg[0])
    h_deg = int(bbox_deg[3] - bbox_deg[1])

    bbox_c = draw.textbbox((0, 0), getattr(cfg, "unit_c", "C"), font=nereus_unit_font)
    w_c = int(bbox_c[2] - bbox_c[0])
    h_c = int(bbox_c[3] - bbox_c[1])

    w_icon, h_icon = icon_img.size
    gap = int(getattr(cfg, "gap_px", 14))
    unit_gap = int(getattr(cfg, "unit_gap_px", 6))

    # Degree dot width
    use_dot = bool(getattr(cfg, "deg_as_dot", True))
    r = int(getattr(cfg, "deg_dot_r_px", 4))
    w_deg = (2 * r) if use_dot else w_deg_glyph

    # Module size
    module_w = int(w_icon + gap + max(0, w_val) + (unit_gap if temp_str else 0) + (w_deg + w_c if temp_str else 0))
    module_h = int(max(h_icon, h_val if h_val else h_icon, h_deg, h_c))

    # Anchor bottom-right if x/y are -1
    if int(getattr(cfg, "x", -1)) < 0:
        x0 = int(W - int(getattr(cfg, "margin_right", 60)) - module_w)
    else:
        x0 = int(getattr(cfg, "x", 0))

    if int(getattr(cfg, "y", -1)) < 0:
        y0 = int(H - int(getattr(cfg, "margin_bottom", 80)) - module_h)
    else:
        y0 = int(getattr(cfg, "y", 0))

    x0 += int(getattr(cfg, "global_x", 0))
    y0 += int(getattr(cfg, "global_y", 0))

    # Icon placement (vertically centered in module)
    ix = x0
    iy = y0 + (module_h - h_icon) // 2
    if _sh_enable:
        # Fast per-frame hard shadow for icon (small icon, avoid global cache to prevent recursion issues)
        _sc = _rgba_shadow_color(_sh_rgb, _sh_alpha)
        _a = icon_img.getchannel("A")
        _sa = _a.point(lambda p: int(p * _sc[3] / 255))
        _sh_icon = PILImage.new("RGBA", icon_img.size, (_sc[0], _sc[1], _sc[2], 0))
        _sh_icon.putalpha(_sa)
        overlay.alpha_composite(_sh_icon, dest=(ix + _sh_dx, iy + _sh_dy))
    overlay.alpha_composite(icon_img, dest=(ix, iy))

    # If no temperature string, still show icon only
    if not temp_str:
        return overlay

    # Text origins
    value_ox = int(getattr(cfg, "value_ox", 0))
    value_oy = int(getattr(cfg, "value_oy", 0))
    unit_ox = int(getattr(cfg, "unit_ox", 0))
    unit_oy = int(getattr(cfg, "unit_oy", 0))

    tx_val = ix + w_icon + gap + value_ox
    ty_val = y0 + (module_h - h_val) // 2 + value_oy

    _dt((tx_val, ty_val), temp_str, font=nereus_value_font, fill=getattr(cfg, "color", (255, 255, 255, 255)))

    # Unit start (after value)
    tx_unit = tx_val + w_val + unit_gap + unit_ox
    unit_h = max(h_deg, h_c)
    ty_unit = y0 + (module_h - unit_h) // 2 + unit_oy

    # Draw degree
    if use_dot:
        r = int(getattr(cfg, "deg_dot_r_px", 4))
        line = int(getattr(cfg, "deg_dot_line_px", 2))
        dx = int(getattr(cfg, "deg_dot_dx", 0))
        dy = int(getattr(cfg, "deg_dot_dy", 5))
        # Guard: avoid line swallowing the hole
        line = max(1, min(line, max(1, r - 1)))

        cx0 = tx_unit + dx
        cy0 = ty_unit + dy
        _de(
            (cx0, cy0, cx0 + 2 * r, cy0 + 2 * r),
            outline=getattr(cfg, "color", (255, 255, 255, 255)),
            width=line,
        )
        tx_after_deg = tx_unit + 2 * r
    else:
        _dt((tx_unit, ty_unit), getattr(cfg, "deg_symbol", "°"), font=base_unit_font, fill=getattr(cfg, "color", (255, 255, 255, 255)))
        tx_after_deg = tx_unit + w_deg_glyph

    # Draw 'C'
    _dt((tx_after_deg, ty_unit), getattr(cfg, "unit_c", "C"), font=nereus_unit_font, fill=getattr(cfg, "color", (255, 255, 255, 255)))

    return overlay

def build_layout_d_depth_static(
    times_s: np.ndarray,
    depths_m: np.ndarray,
    cfg: LayoutDDepthConfig,
    max_depth_m: float,
)-> tuple[Image.Image, Image.Image, float]:
    """
    Build a single RGBA image for the static part of Layout D depth module:
      - rounded backplate
      - curve fill ABOVE the curve (with a gap)
      - curve polyline
    Returns:
      (plate_img_rgba, curve_fill_img_rgba, t_max)
    """
    Wp, Hp = int(cfg.plate_w), int(cfg.plate_h)
    Wc, Hc = int(cfg.chart_w), int(cfg.chart_h)

    # Defensive
    if times_s is None or len(times_s) == 0:
        t_max = 1.0
    else:
        t_max = float(np.nanmax(times_s))
        if not np.isfinite(t_max) or t_max <= 1e-6:
            t_max = 1.0

    md = float(max_depth_m) if (max_depth_m is not None and np.isfinite(max_depth_m) and max_depth_m > 1e-6) else 1.0

    # Base image
    plate_img = Image.new("RGBA", (Wp, Hp), (0, 0, 0, 0))
    d = ImageDraw.Draw(plate_img)

    # Rounded backplate
    plate_rgba = _hex_to_rgba(cfg.plate_color_hex, cfg.plate_alpha)
    d.rounded_rectangle([0, 0, Wp, Hp], radius=int(cfg.radius), fill=plate_rgba)
    # Init curve+fill layer placeholder
    curve_fill_img = Image.new("RGBA", (Wp, Hp), (0, 0, 0, 0))


    # Build curve points (top-aligned in plate)
    pts = []
    for t, dep in zip(times_s, depths_m):
        if not (np.isfinite(t) and np.isfinite(dep)):
            continue
        x = (float(t) / t_max) * (Wc - 1)
        y = (float(dep) / md) * (Hc - 1)
        pts.append((x, y))
    if len(pts) < 2:
        curve_fill_img = Image.new("RGBA", (Wp, Hp), (0, 0, 0, 0))
        return plate_img, curve_fill_img, t_max

    # Clamp & quantize
    pts_i = [( _clip_int(x, 0, Wc - 1), _clip_int(y, 0, Hc - 1) ) for x, y in pts]

    # Fill polygon ABOVE the curve, using a shifted curve to create a blank gap
    gap = int(cfg.gap_px)
    pts_shift = [(x, max(0, y - gap)) for x, y in pts_i]

    # --- Fill polygon ABOVE the curve on a separate layer (gradient supported) ---
    fill_img = Image.new("RGBA", (Wp, Hp), (0, 0, 0, 0))
    df = ImageDraw.Draw(fill_img)

    # Draw fill with full alpha; we will apply gradient/uniform alpha afterwards.
    fill_rgb_full = _hex_to_rgba(cfg.fill_color_hex, 255)

    # Polygon: top-left -> top-right -> follow shifted curve reversed -> back
    poly = [(0, 0), (Wc - 1, 0)] + list(reversed(pts_shift)) + [(0, 0)]
    df.polygon(poly, fill=fill_rgb_full)

    # Apply vertical alpha (top more transparent, bottom more opaque)
    if getattr(cfg, "fill_gradient_enabled", True):
        grad = Image.new("L", (1, Hc))
        a_bot = int(getattr(cfg, "fill_alpha_bottom", int(getattr(cfg, "fill_alpha", 128))))
        a_top = int(getattr(cfg, "fill_alpha_top", max(0, a_bot // 4)))
        gamma = float(getattr(cfg, "fill_gradient_gamma", 1.0))
        for y in range(Hc):
            t = y / max(1, Hc - 1)  # 0 at top, 1 at bottom
            a = int(a_top + (a_bot - a_top) * (t ** gamma))
            grad.putpixel((0, y), max(0, min(255, a)))
        grad = grad.resize((Wp, Hp))

        r, g, b, a0 = fill_img.split()
        a = ImageChops.multiply(a0, grad)
        fill_img = Image.merge("RGBA", (r, g, b, a))
    else:
        # Fallback to uniform alpha using cfg.fill_alpha
        r, g, b, a0 = fill_img.split()
        uni = Image.new("L", (Wp, Hp), int(getattr(cfg, "fill_alpha", 128)))
        a = ImageChops.multiply(a0, uni)
        fill_img = Image.merge("RGBA", (r, g, b, a))

    # Composite fill under the curve
    # (do not composite fill into plate_img; curve/fill layer is returned separately)

    # Curve line (ensure drawn last, on top of fill)
    curve_fill_img = Image.new("RGBA", (Wp, Hp), (0, 0, 0, 0))
    # paste fill and curve onto curve_fill_img
    curve_fill_img.alpha_composite(fill_img, dest=(0, 0))
    dc = ImageDraw.Draw(curve_fill_img)
    dc.line(pts_i, fill=cfg.curve_color, width=int(cfg.curve_width))

    # Clip curve/fill to the same rounded-corner shape as the backplate,
    # so the chart never protrudes outside the backplate corners.
    if int(cfg.radius) > 0:
        mask = Image.new("L", (Wp, Hp), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle([0, 0, Wp - 1, Hp - 1], radius=int(cfg.radius), fill=255)
        plate_img.putalpha(ImageChops.multiply(plate_img.getchannel("A"), mask))
        curve_fill_img.putalpha(ImageChops.multiply(curve_fill_img.getchannel("A"), mask))

    return plate_img, curve_fill_img, t_max

def render_layout_d_depth_module(
    base_img: Image.Image,
    t_global_s: float,
    current_depth_m: float,
    static_img: Image.Image,
    curve_fill_img: Optional[Image.Image],
    t_max: float,
    max_depth_m: float,
    cfg: LayoutDDepthConfig,
    font_path: Optional[str] = None,
) -> Image.Image:
    """
    Per-frame overlay for Layout D depth module:
      - paste pre-rendered backplate; overlay curve+fill separately
      - draw vertical line + dot at current (t, depth)
      - draw depth value + unit
    """
    if not cfg.enabled:
        return base_img

    out = base_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)

    # Module origin
    ox = int(cfg.x + cfg.global_x)
    oy = int(cfg.y + cfg.global_y)

    # Map t/depth to chart coords
    Wc, Hc = int(cfg.chart_w), int(cfg.chart_h)
    md = float(max_depth_m) if (max_depth_m is not None and np.isfinite(max_depth_m) and max_depth_m > 1e-6) else 1.0

    tt = float(t_global_s)
    if not np.isfinite(tt):
        tt = 0.0
    tt = max(0.0, min(float(t_max), tt))
    x = (tt / float(t_max)) * (Wc - 1)

    dep = float(current_depth_m) if np.isfinite(current_depth_m) else 0.0
    dep = max(0.0, min(md, dep))
    y = (dep / md) * (Hc - 1)

    xi = _clip_int(x, 0, Wc - 1)
    yi = _clip_int(y, 0, Hc - 1)

    # Paste backplate (always fully visible)
    out.alpha_composite(static_img, dest=(ox, oy))

    # Paste curve+fill (optionally progressive reveal only left of indicator)
    if curve_fill_img is not None:
        if getattr(cfg, "progressive_curve_enabled", False):
            reveal_x = int(xi + int(getattr(cfg, "progressive_reveal_extra_px", 0)))
            reveal_x = _clip_int(reveal_x, 0, int(cfg.plate_w))
            mask = Image.new("L", curve_fill_img.size, 0)
            dm = ImageDraw.Draw(mask)
            dm.rectangle([0, 0, reveal_x, curve_fill_img.size[1]], fill=255)
            curve_visible = Image.new("RGBA", curve_fill_img.size, (0, 0, 0, 0))
            curve_visible.paste(curve_fill_img, (0, 0), mask)
            out.alpha_composite(curve_visible, dest=(ox, oy))
        else:
            out.alpha_composite(curve_fill_img, dest=(ox, oy))
    # Vertical line (top to bottom of plate)
    x_abs = ox + xi
    y_top = oy + 0
    y_bot = oy + int(cfg.plate_h)
    draw.line([(x_abs, y_top), (x_abs, y_bot)], fill=cfg.vline_color, width=int(cfg.vline_width))

    # Dot
    r = int(cfg.dot_diameter // 2)
    cx = x_abs
    cy = oy + yi
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cfg.dot_color)

    # Fonts (cached)
    if font_path:
        fp = Path(font_path)
        value_font = _get_font_cached(fp, int(cfg.value_font_size))
        unit_font  = _get_font_cached(fp, int(cfg.unit_font_size))
    else:
        value_font = ImageFont.load_default()
        unit_font  = ImageFont.load_default()

    # Depth text
    fmt = "{:." + str(int(cfg.decimals)) + "f}"
    val_txt = fmt.format(float(current_depth_m))
    unit_txt = "m"

    # Position in plate
    ty = oy + int(cfg.text_y)

# value bbox
    vb = draw.textbbox((0, 0), val_txt, font=value_font)
    v_w = vb[2] - vb[0]
    v_h = vb[3] - vb[1]

    # unit bbox
    ub = draw.textbbox((0, 0), unit_txt, font=unit_font)
    u_w = ub[2] - ub[0]
    u_h = ub[3] - ub[1]

    

    total_w = int(v_w) + int(cfg.unit_gap_px) + int(u_w)
    if str(getattr(cfg, "text_align", "left")).lower().startswith("cent"):
        # text_x is the center anchor within the plate
        tx = (ox + int(cfg.text_x)) - (total_w // 2)
    else:
        # text_x is the left edge
        tx = ox + int(cfg.text_x)
    # Depth value (optional hard shadow)
    if bool(getattr(cfg, 'depth_value_shadow_enable', False)):
        _draw_text_hard_shadow(draw, (tx, ty), val_txt, font=value_font, fill=cfg.text_color,
                              shadow_enable=True,
                              shadow_dx=int(getattr(cfg, 'depth_value_shadow_dx', 4)),
                              shadow_dy=int(getattr(cfg, 'depth_value_shadow_dy', 4)),
                              shadow_color_rgb=getattr(cfg, 'depth_value_shadow_color_rgb', (0, 0, 0)),
                              shadow_alpha=int(getattr(cfg, 'depth_value_shadow_alpha', 140)))
    else:
        draw.text((tx, ty), val_txt, font=value_font, fill=cfg.text_color)

    if cfg.unit_bottom_align:
        # Align bottoms
        uy = ty + (v_h - u_h)
    else:
        uy = ty

    ux = tx + v_w + int(cfg.unit_gap_px)
    _draw_text_hard_shadow(
        draw, (ux, uy), unit_txt,
        font=unit_font, fill=cfg.text_color,
        shadow_enable=bool(getattr(cfg, "depth_value_shadow_enable", False)),
        shadow_dx=int(getattr(cfg, "depth_value_shadow_dx", 4)),
        shadow_dy=int(getattr(cfg, "depth_value_shadow_dy", 4)),
        shadow_color_rgb=getattr(cfg, "depth_value_shadow_color_rgb", (0, 0, 0)),
        shadow_alpha=int(getattr(cfg, "depth_value_shadow_alpha", 140)),
    )

    return out


def get_layout_d_config() -> LayoutDDepthConfig:
    """Default Layout D depth config (depth module only)."""
    return LayoutDDepthConfig()


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

    if bool(cfg.get("shadow_enable", True)): # original = True
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
        return _get_font_cached(Path(base_font_path) if base_font_path else None, int(size))

    fs = getattr(cfg, "font_sizes", {})
    
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
        _fkey = (str(flags_dir / f"{code3.lower()}.png"), int(target_w), int(target_h))
        flag_resized = _FLAG_RESIZE_CACHE.get(_fkey)
        if flag_resized is None:
            flag_resized = flag_img.resize((target_w, target_h), PILImage.LANCZOS)
            _FLAG_RESIZE_CACHE[_fkey] = flag_resized
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
    k = str(path)
    img = _FLAG_BASE_CACHE.get(k)
    if img is not None:
        return img
    try:
        img = Image.open(path).convert("RGBA")
        _FLAG_BASE_CACHE[k] = img
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
                _fkey = (str(flags_dir / f"{code3.lower()}.png"), int(target_w), int(target_h))
                flag_resized = _FLAG_RESIZE_CACHE.get(_fkey)
                if flag_resized is None:
                    flag_resized = flag_img.resize((target_w, target_h), PILImage.LANCZOS)
                    _FLAG_RESIZE_CACHE[_fkey] = flag_resized
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
# =================# ==========================================================
# Web version: input video normalization (1080p portrait)
#   Goals:
#     - Enforce 1080x1920 for predictable overlay coordinates
#     - Fix DJI Mimo HEVC 10-bit + rotation metadata cases (avoid side black bars)
#     - Skip normalization when the source is already a clean 1080x1920 portrait
# ==========================================================

def _run_cmd(cmd: list) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "Command failed")

def _ffprobe_stream_info(video_path: Path) -> dict:
    """Return primary video stream info: width/height, codec, pix_fmt, rotation, SAR."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,codec_name,pix_fmt,sample_aspect_ratio:stream_tags=rotate:side_data_list=rotation",
        "-of", "json",
        str(video_path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "ffprobe failed")
    data = json.loads(p.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        return {}
    st = streams[0]
    # Rotation can appear in tags.rotate (string) or side_data_list.rotation (number)
    rot = None
    try:
        rot = int((st.get("tags") or {}).get("rotate"))
    except Exception:
        rot = None
    if rot is None:
        sdl = st.get("side_data_list") or []
        for sd in sdl:
            if "rotation" in sd:
                try:
                    rot = int(sd.get("rotation"))
                    break
                except Exception:
                    pass
    sar = st.get("sample_aspect_ratio")
    if sar in (None, "", "N/A"):
        sar = None
    info = {
        "width": int(st.get("width") or 0),
        "height": int(st.get("height") or 0),
        "codec": (st.get("codec_name") or "").lower(),
        "pix_fmt": (st.get("pix_fmt") or "").lower(),
        "rotation": rot,
        "sar": sar,
    }
    return info

def _effective_wh(w: int, h: int, rot: Optional[int]) -> tuple:
    if rot in (90, 270, -90, -270):
        return (h, w)
    return (w, h)

def _is_clean_portrait_1080(info: dict, target_wh=(1080, 1920)) -> bool:
    """Treat missing SAR/rotation as OK. Only reject if an explicit non-1:1 SAR exists."""
    tw, th = int(target_wh[0]), int(target_wh[1])
    w, h = int(info.get("width") or 0), int(info.get("height") or 0)
    rot = info.get("rotation")
    eff_w, eff_h = _effective_wh(w, h, rot)
    if (eff_w, eff_h) != (tw, th):
        return False
    # If already portrait pixels (w<h) but rot says 90/270, treat as stale and ignore rot
    if w < h and rot in (90, 270, -90, -270):
        rot = 0
        eff_w, eff_h = _effective_wh(w, h, rot)
        if (eff_w, eff_h) != (tw, th):
            return False
    sar = info.get("sar")
    if sar is None:
        return True
    return sar in ("1:1", "1/1")

def _is_dji_mimo_like(info: dict) -> bool:
    """Heuristic for DJI Mimo exports that trigger DAR/SAR quirks in ffmpeg scale+pad."""
    codec = (info.get("codec") or "").lower()
    pix = (info.get("pix_fmt") or "").lower()
    rot = info.get("rotation")
    return (codec == "hevc") and ("10" in pix) and (rot in (90, 270, -90, -270))

def _norm_cache_path(src: Path, tag: str) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "depthrender_norm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    st = src.stat()
    key = f"{src.resolve()}|{st.st_size}|{int(st.st_mtime)}|{tag}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{src.stem}_{tag}_{h}.mp4"


def _is_norm_cache_valid(path: Path, target_wh=(1080, 1920)) -> bool:
    """Validate cached normalized video matches target dimensions and has no rotation."""
    try:
        info = _ffprobe_stream_info(path)
        if not info:
            return False
        tw, th = int(target_wh[0]), int(target_wh[1])
        w, h = int(info.get("width", 0)), int(info.get("height", 0))
        rot = int(info.get("rotation") or 0)
        return (w == tw and h == th and rot == 0)
    except Exception:
        return False

def normalize_video_to_portrait_1080(
    src_path: Path,
    *,
    target_wh=(1080, 1920),
    rotate_landscape: str = "left",  # "left" default
) -> Path:
    """Return a path to a normalized mp4. May return src_path if no normalization is needed."""
    info = _ffprobe_stream_info(src_path)
    if not info:
        return src_path

    # Fast-path: already clean 1080x1920 portrait => skip normalization
    if _is_clean_portrait_1080(info, target_wh=target_wh):
        return src_path

    tw, th = int(target_wh[0]), int(target_wh[1])
    w, h = int(info["width"]), int(info["height"])
    rot = info.get("rotation") or 0

    # Special-case: DJI Mimo-like (HEVC 10-bit + Display Matrix rotation)
    # Use fill + crop to avoid side black bars caused by DAR/SAR derivation in scale+pad.
    is_mimo = _is_dji_mimo_like(info)

    tag = "mimo_v7" if is_mimo else "norm_v7"
    out_path = _norm_cache_path(src_path, tag)
    # Reuse cache only when it matches target (protect against older buggy cache files)
    if out_path.exists() and out_path.stat().st_size > 1024 * 1024 and _is_norm_cache_valid(out_path, target_wh=target_wh):
        return out_path

    # Build filter chain
    filters = []

    # 1) Bake in metadata rotation when present
    if rot in (90, -270):
        filters.append("transpose=1")  # clockwise 90
    elif rot in (270, -90):
        filters.append("transpose=2")  # counter-clockwise 90
    elif rot in (180, -180):
        filters.append("transpose=2,transpose=2")

    # 2) If still landscape (coded or after baking), rotate to portrait (default left)
    #    We infer from effective dimensions after baking metadata.
    eff_w, eff_h = _effective_wh(w, h, rot)
    # After baking, the stream no longer has rotation; but we still use the pre-bake effective size to decide.
    if eff_w > eff_h:
        if rotate_landscape == "left":
            filters.append("transpose=2")
        elif rotate_landscape == "right":
            filters.append("transpose=1")
        else:
            # no rotation
            pass

    # 3) Force square pixels and 9:16 display aspect for stability
    filters.append("setsar=1")
    filters.append("setdar=9/16")

    # 4) Resize to target
    if is_mimo:
        # Fill then crop (no black bars)
        filters.append(f"scale={tw}:{th}:force_original_aspect_ratio=increase")
        filters.append(f"crop={tw}:{th}")
    else:
        # Fit then pad (preserve full frame)
        filters.append(f"scale={tw}:{th}:force_original_aspect_ratio=decrease")
        filters.append(f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2")

    vf = ",".join(filters)

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-noautorotate",
        "-i", str(src_path),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    ]
    _run_cmd(cmd)
    return out_path

# ===========================================
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

    # Timing model (shared by ALL layouts):
    # - t_video   : MoviePy timeline time (seconds)
    # - t_global  : data timeline time = t_video + effective_offset
    # - elapsed   : in-dive elapsed time (seconds), clamped to [0, dive_end_s - dive_start_s]
    #
    # IMPORTANT: Any *displayed* dive time should use `elapsed_dive_time(...)`.
    #            Using `t_global` directly will cause the time to keep counting after surfacing.


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

    # --- Web: FORCE 1080x1920 portrait for all inputs (web version policy) ---
    output_resolution = (1080, 1920)

    # --- Web normalization to 1080x1920 (portrait) ---
    # Skip transcode when the source is already a clean 1080x1920 portrait.
    # DJI Mimo (HEVC 10-bit + rotation metadata) is handled with a special fill+crop path to avoid side black bars.
    try:
        norm_path = normalize_video_to_portrait_1080(
            Path(video_path),
            target_wh=output_resolution,
            rotate_landscape="left",
        )
    except Exception as _e:
        # If ffmpeg/ffprobe is not available, fall back to original path
        print(f"[render_video] 影片正規化失敗，改用原始影片：{_e}")
        norm_path = Path(video_path)

    clip = VideoFileClip(str(norm_path))

    W, H = output_resolution
    # Avoid expensive resize transform when already matching the target
    try:
        if int(getattr(clip, "w", 0)) != int(W) or int(getattr(clip, "h", 0)) != int(H):
            clip = clip.resize((W, H))
    except Exception:
        clip = clip.resize((W, H))

    # Use the (possibly resized) clip as the frame source for make_frame
    src_clip = clip  # frames source

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

    # =========================
    # Layout D config (Depth module)
    # Define cfg BEFORE any Layout D build/render usage.
    # Optional overrides:
    #   layout_params["layout_d_depth_cfg"] = {"x": 80, "y": 980, ...}
    # =========================
    if layout_params is None:
        layout_params = {}
    layout_d_depth_cfg = LayoutDDepthConfig()
    try:
        _d_overrides = layout_params.get("layout_d_depth_cfg", {})
        if isinstance(_d_overrides, dict):
            for _k, _v in _d_overrides.items():
                if hasattr(layout_d_depth_cfg, _k):
                    setattr(layout_d_depth_cfg, _k, _v)
    except Exception:
        pass

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

    
    # =========================
    
    # =========================
    # Layout D config (Time module)
    # Optional overrides:
    #   layout_params["layout_d_time_cfg"] = {"below_gap_px": 16, "x_offset": 0, ...}
    # =========================
    layout_d_time_cfg = LayoutDTimeConfig()
    try:
        _t_overrides = layout_params.get("layout_d_time_cfg", {})
        if isinstance(_t_overrides, dict):
            for _k, _v in _t_overrides.items():
                if hasattr(layout_d_time_cfg, _k):
                    setattr(layout_d_time_cfg, _k, _v)
    except Exception:
        pass


    # =========================
    

    # =========================
    # Layout D config (Speed module)
    # Optional overrides:
    #   layout_params["layout_d_speed_cfg"] = {"below_time_gap_px": 18, "value_font_size": 65, ...}
    # =========================
    layout_d_speed_cfg = LayoutDSpeedConfig()
    try:
        _s_overrides = layout_params.get("layout_d_speed_cfg", {})
        if isinstance(_s_overrides, dict):
            for _k, _v in _s_overrides.items():
                if hasattr(layout_d_speed_cfg, _k):
                    setattr(layout_d_speed_cfg, _k, _v)
    except Exception:
        pass

# Layout D config (Temperature module)
    # Optional overrides:
    #   layout_params["layout_d_temp_cfg"] = {"global_x": -20, "margin_bottom": 90, "icon_size": 70, ...}
    # =========================
    layout_d_temp_cfg = LayoutDTempConfig()
    try:
        _temp_overrides = layout_params.get("layout_d_temp_cfg", {})
        if isinstance(_temp_overrides, dict):
            for _k, _v in _temp_overrides.items():
                if hasattr(layout_d_temp_cfg, _k):
                    setattr(layout_d_temp_cfg, _k, _v)
    except Exception:
        pass

# Layout D config (Heart rate module)
    # Optional overrides:
    #   layout_params["layout_d_hr_cfg"] = {"margin_right": 80, "margin_top": 60, "icon_h": 80, ...}
    # =========================
    layout_d_hr_cfg = LayoutDHeartRateConfig()
    try:
        _hr_overrides = layout_params.get("layout_d_hr_cfg", {})
        if isinstance(_hr_overrides, dict):
            for _k, _v in _hr_overrides.items():
                if hasattr(layout_d_hr_cfg, _k):
                    setattr(layout_d_hr_cfg, _k, _v)
    except Exception:
        pass


# Layout D static build (backplate + curve)
    # =========================
    layout_d_plate = None
    layout_d_curve_fill = None
    layout_d_tmax = None
    if str(layout).upper() == "D":
        try:
            layout_d_plate, layout_d_curve_fill, layout_d_tmax = build_layout_d_depth_static(
                times_s=times_d,
                depths_m=depths_d,
                cfg=layout_d_depth_cfg,
                max_depth_m=max_depth_raw,
            )
        except Exception:
            layout_d_plate = None
            layout_d_curve_fill = None
            layout_d_tmax = None

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

    # =========================
    # Temperature data (optional): infer column and build interpolation arrays
    # =========================
    _temp_col = None
    for _c in getattr(layout_d_temp_cfg, "temp_col_candidates", ()):
        if _c in dive_df.columns:
            _temp_col = _c
            break

    if _temp_col is not None:
        temps_d = pd.to_numeric(dive_df[_temp_col], errors="coerce").to_numpy(dtype=float)
        # Auto Kelvin->C for whole series if needed
        if getattr(layout_d_temp_cfg, "auto_kelvin_to_c", True):
            try:
                _med = float(np.nanmedian(temps_d)) if np.isfinite(np.nanmedian(temps_d)) else np.nan
                if np.isfinite(_med) and _med > float(getattr(layout_d_temp_cfg, "kelvin_threshold", 100.0)):
                    temps_d = temps_d - 273.15
            except Exception:
                pass
    else:
        temps_d = np.array([], dtype=float)

    # Put this flag OUTSIDE temp_at() (one line), near where temp_at is defined:
    _temp_debug_printed = False
    _temp_dbg = {"printed": False}

    def temp_at(t_video: float) -> Optional[float]:
        """
        Return temperature (C) at video time t_video, with:
        - use finite samples only
        - hold first/last valid value beyond range
        - print debug once
        """
        # ---- debug print ONCE ----
        if not _temp_dbg["printed"]:
            _temp_dbg["printed"] = True
            try:
                import numpy as np
                print("[TEMP DEBUG] times_d size:", getattr(times_d, "size", None))
                print("[TEMP DEBUG] temps_d size:", getattr(temps_d, "size", None))
    
                if getattr(temps_d, "size", 0) > 0 and getattr(times_d, "size", 0) > 0:
                    m = np.isfinite(times_d) & np.isfinite(temps_d)
                    print("[TEMP DEBUG] finite pair count:", int(m.sum()))
                    if m.any():
                        tt0 = float(times_d[m][0])
                        vv0 = float(temps_d[m][0])
                        tt1 = float(times_d[m][-1])
                        vv1 = float(temps_d[m][-1])
                        print("[TEMP DEBUG] first t/temp:", tt0, vv0)
                        print("[TEMP DEBUG] last  t/temp:", tt1, vv1)
                        vv = temps_d[m]
                        print("[TEMP DEBUG] temp min/max:", float(np.nanmin(vv)), float(np.nanmax(vv)))
            except Exception as e:
                print("[TEMP DEBUG] exception:", repr(e))
    
        # ---- normal logic ----
        if getattr(temps_d, "size", 0) == 0 or getattr(times_d, "size", 0) == 0:
            return None
    
        import numpy as np
        mask = np.isfinite(times_d) & np.isfinite(temps_d)
        if not np.any(mask):
            return None
    
        tt = times_d[mask]
        vv = temps_d[mask]
    
        t = t_video + effective_offset
        if t <= float(tt[0]):
            return float(vv[0])
        if t >= float(tt[-1]):
            return float(vv[-1])
        return float(np.interp(t, tt, vv))



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

        rising=True:  find first crossing from <thr to >=thr (start of descent)
        rising=False: find last  crossing from >thr to <=thr (end of ascent)
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
    layout_c_temp_cfg = layout_c_all.temp
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
    global LAYOUT_C_SHADOW_MODE, LAYOUT_C_SHADOW_ENABLED, LAYOUT_C_SHADOW_OFFSET, LAYOUT_C_SHADOW_BLUR, LAYOUT_C_SHADOW_COLOR
    LAYOUT_C_SHADOW_MODE = str(getattr(shadow_cfg, "shadow_mode", "solid")).strip().lower()
    if LAYOUT_C_SHADOW_MODE == "off":
        LAYOUT_C_SHADOW_ENABLED = False
    else:
        # "solid" (default) or any unknown value falls back to solid
        LAYOUT_C_SHADOW_ENABLED = bool(getattr(shadow_cfg, "enabled", True))

    LAYOUT_C_SHADOW_OFFSET = tuple(getattr(shadow_cfg, "offset", (4, 4)))
    # Solid shadow: no blur. Keep blur_radius only for backward-compat.
    LAYOUT_C_SHADOW_BLUR = 0 if LAYOUT_C_SHADOW_MODE == "solid" else int(getattr(shadow_cfg, "blur_radius", 0))
    LAYOUT_C_SHADOW_COLOR = tuple(getattr(shadow_cfg, "color", (80, 80, 80, 120)))

# =========================
    # Heart rate prep (Layout C)
    # =========================
    hr_available = False
    hr_times = None
    hr_values = None
    hr_last_value = {"v": None}
    hr_anim = {"phase": 0.0, "bpm_active": None, "bpm_pending": None, "switch_pending": False, "t_prev": None}
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

    def make_frame(t):
        if duration > 0:
            frac = max(0.0, min(1.0, t / duration))
            p = 0.12 + 0.86 * frac
            if p - last_p["value"] >= 0.01:
                last_p["value"] = p
                update_progress(p, "產生疊加畫面中...")

        frame = src_clip.get_frame(t)
        img = PILImage.fromarray(frame).convert("RGBA")
        img_w, img_h = img.size

        # ------------------------------------------------------------
        # Overlay throttling: update overlays at a fixed fps per layout
        # ------------------------------------------------------------
        layout_u = str(layout).upper()
        overlay_fps = None
        if layout_u in ("A", "B"):
            overlay_fps = float(LAYOUT_AB_OVERLAY_FPS)
        elif layout_u == "C":
            overlay_fps = float(LAYOUT_C_OVERLAY_FPS)
        elif layout_u == "D":
            overlay_fps = float(LAYOUT_D_OVERLAY_FPS)

        tq = float(t)
        if overlay_fps is not None and overlay_fps > 0:
            step = 1.0 / overlay_fps
            tq = math.floor(float(t) / step) * step

        cache = None
        if _OVERLAY_CACHE is not None:
            cache = _OVERLAY_CACHE.get(layout_u)
            if cache is not None and cache.get("tq") == tq and cache.get("size") == img.size and cache.get("overlay") is not None:
                overlay = cache["overlay"]
                composed = PILImage.alpha_composite(img, overlay).convert("RGB")
                return np.array(composed)

        # Cache miss: render overlay at quantized time tq
        overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        t_use = tq

        t_global = t_use + effective_offset
        depth_val = depth_at(t_use)

        # Layout B (abs, from df_rate)
        rate_val_abs_raw = rate_at(t_use)

        # Layout C (signed, Layout B-aligned: magnitude from df_rate + sign from depth trend)
        rate_val_signed_raw = rate_c_signed_like_layout_b(t_use)

        # Heart rate (Layout C only)
        hr_text = ""    # original = "--"
        show_hr_module = False
        show_hr_value = False
        pulse_scale = 1.0

        # Heart rate value for Layout D (and potential reuse)
        hr_value = None
        if hr_available:
            try:
                _hv = hr_at(float(t_global))
                if _hv is not None and np.isfinite(float(_hv)):
                    hr_value = float(_hv)
            except Exception:
                hr_value = None

        show_hr_module_d = (layout.upper() == "D" and getattr(layout_d_hr_cfg, "enabled", True) and hr_value is not None)


        if layout.upper() == "C" and hr_cfg.enabled and hr_available:
            t_data = float(t_global)
            data_start = float(hr_times[0]) if (hr_times is not None and len(hr_times) > 0) else 0.0
            data_end = float(hr_times[-1]) if (hr_times is not None and len(hr_times) > 0) else (float(duration) if duration else 0.0)


            if t_data < data_start:
                hr_text = ""    # original = "--"
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

                    # Use real dt based on t_data to keep HR animation in sync even when overlay updates are decoupled
                    t_now = float(t_data)
                    t_prev = hr_anim.get("t_prev")
                    dt_real = 0.0 if (t_prev is None) else max(0.0, t_now - float(t_prev))
                    hr_anim["t_prev"] = t_now

                    bpm_for_phase = float(hr_anim["bpm_active"] or 0.0)
                    hr_anim["phase"] += 2.0 * math.pi * (bpm_for_phase / 60.0) * dt_real
                    if hr_anim["phase"] >= 2.0 * math.pi:
                        hr_anim["phase"] -= 2.0 * math.pi
                        if hr_anim["switch_pending"]:
                            hr_anim["bpm_active"] = hr_anim["bpm_pending"]
                            hr_anim["switch_pending"] = False

                    pulse_scale = 1.0 + float(hr_cfg.pulse_amp) * math.sin(float(hr_anim["phase"]))
                else:
                    hr_text = ""    # original = "--"
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
        direction_is_descent = bool(is_descent_at(t_use)) if in_dive else True

        # Unified elapsed-time logic (all layouts):
        # - starts at dive_start_s
        # - pauses at/after dive_end_s (surfacing)
        # - unaffected by per-layout rendering
        elapsed = elapsed_dive_time(t)
        # (time_text assigned below from time_disp_s)
        # Display dive time base (start at 0 at video/data t_global=0; stop at dive_end_s)
        time_disp_s = float(max(0.0, t_global))
        if dive_end_s is not None:
            time_disp_s = float(min(time_disp_s, float(dive_end_s)))
        time_text = format_dive_time(time_disp_s)

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
                dive_time_s=time_disp_s,
                depth_val=depth_disp,
                params=layout_params,
            )

        # ===== Generic info card (NOT A/B/C) =====
        if layout not in ("A", "B", "C", "D"):  # Layout D uses its own modules
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
                # IMPORTANT: Layout C time MUST use the unified elapsed time,
                # not raw t_global. This prevents the timer from continuing
                # after surfacing, and aligns behavior with Layout A/D.
                time_s=float(time_disp_s),
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

            # Temperature module (reuse Layout D design)
            try:
                _hz = float(getattr(layout_c_temp_cfg, "temp_refresh_hz", 5.0))
            except Exception:
                _hz = 5.0
            if not np.isfinite(_hz) or _hz <= 0:
                _hz = 5.0
            _tstep = 1.0 / _hz
            _t_temp = math.floor(float(time_disp_s) / _tstep) * _tstep

            # Temperature sensor lag compensation (shift temperature earlier in time).
            try:
                _t_shift = float(getattr(layout_c_temp_cfg, "temp_time_shift_s", 0.0))
            except Exception:
                _t_shift = 0.0
            if not np.isfinite(_t_shift):
                _t_shift = 0.0

            _t_temp_shifted = float(_t_temp) - float(_t_shift)

            overlay = render_layout_d_temp_module(
                base_img=overlay,
                temp_c=temp_at(_t_temp_shifted),
                cfg=layout_c_temp_cfg,
                assets_dir=assets_dir,
                nereus_font_path=LAYOUT_C_VALUE_FONT_PATH,
                base_font_path=base_font_path,
            )
        # ===== Layout D (Depth module) =====
        if layout == "D" and layout_d_plate is not None and layout_d_tmax is not None:
            overlay = render_layout_d_depth_module(
                base_img=overlay,
                t_global_s=float(time_disp_s),
                current_depth_m=float(depth_disp),
                static_img=layout_d_plate,
                curve_fill_img=layout_d_curve_fill,
                t_max=float(layout_d_tmax),
                max_depth_m=float(max_depth_raw) if np.isfinite(max_depth_raw) else float(best_depth),
                cfg=layout_d_depth_cfg,
                font_path=str(LAYOUT_C_VALUE_FONT_PATH),
            )


        # ===== Layout D (Time module) =====
        if layout == "D":
            overlay = render_layout_d_time_module(
                base_img=overlay,
                t_global_s=float(time_disp_s),
                depth_cfg=layout_d_depth_cfg,
                cfg=layout_d_time_cfg,
                nereus_font_path=LAYOUT_C_VALUE_FONT_PATH,
                base_font_path=base_font_path,
            )

        # ===== Layout D (Speed module) =====
        if layout == "D":
            overlay = render_layout_d_speed_module(
                base_img=overlay,
                speed_mps=float(rate_val_abs),
                depth_cfg=layout_d_depth_cfg,
                time_cfg=layout_d_time_cfg,
                cfg=layout_d_speed_cfg,
                nereus_font_path=LAYOUT_C_VALUE_FONT_PATH,
                base_font_path=base_font_path,
            )



        # ===== Layout D (Temperature module) =====
        if layout == "D":
            # Throttle temperature sampling independently from overlay FPS.
            # Recommended values: 1 / 2 / 5 / 15 (Hz).
            try:
                _tfps = float(getattr(layout_d_temp_cfg, "temp_update_fps", 15))
            except Exception:
                _tfps = 15.0
            if not np.isfinite(_tfps) or _tfps <= 0:
                _tfps = 15.0
            _allowed = (1.0, 2.0, 5.0, 15.0)
            _tfps = min(_allowed, key=lambda v: abs(v - _tfps))
            _tstep = 1.0 / _tfps
            _t_temp = math.floor(float(t_global) / _tstep) * _tstep
        
            # Temperature sensor lag compensation (shift temperature earlier in time).
            # Example: 10.0 or 15.0 seconds.
            try:
                _t_shift = float(getattr(layout_d_temp_cfg, "temp_time_shift_s", 0.0))
            except Exception:
                _t_shift = 0.0
            if not np.isfinite(_t_shift):
                _t_shift = 0.0
        
            _t_temp_shifted = float(_t_temp) - float(_t_shift)
        
            overlay = render_layout_d_temp_module(
                base_img=overlay,
                temp_c=temp_at(_t_temp_shifted),
                cfg=layout_d_temp_cfg,
                assets_dir=assets_dir,
                nereus_font_path=LAYOUT_C_VALUE_FONT_PATH,
                base_font_path=base_font_path,
            )

        # ===== Layout D (Heart rate module) =====
        if layout == "D" and show_hr_module_d:
            overlay = render_layout_d_heart_rate_module(
                base_img=overlay,
                t_global_s=float(t_global),
                hr_value=hr_value,
                cfg=layout_d_hr_cfg,
                assets_dir=assets_dir,
                nereus_font_path=LAYOUT_C_VALUE_FONT_PATH,
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

        # Update overlay cache (A/B/C only)
        if layout_u in ("A", "B", "C"):
            if _OVERLAY_CACHE is not None:
                _OVERLAY_CACHE[layout_u] = {"tq": tq, "size": img.size, "overlay": overlay}

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
        new_clip = VideoClip(make_frame, duration=src_clip.duration)
        new_clip = new_clip.set_fps(src_clip.fps).set_audio(src_clip.audio)

        output_path = Path(tempfile.gettempdir()) / f"dive_overlay_output_{uuid.uuid4().hex}.mp4"
        tmp_audio_path = str(Path(tempfile.gettempdir()) / f"dive_overlay_audio_{uuid.uuid4().hex}.m4a")

        new_clip.write_videofile(
            str(output_path),
            codec="libx264",
            fps=src_clip.fps,
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