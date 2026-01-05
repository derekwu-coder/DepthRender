"""
Shared dive-curve preprocessing (resample to 1 Hz) and metric computation.

Key output columns for downstream renderers:
- time_s
- depth_m
- rate_abs_mps
- rate_abs_mps_smooth
- water_temp_c (optional, if input contains any temperature column)
"""
from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

# Print debug logs only once per run to avoid flooding console
_DIVE_CURVE_DEBUG_PRINTED = False


def _first_finite(arr: np.ndarray) -> Optional[float]:
    if arr is None or arr.size == 0:
        return None
    m = np.isfinite(arr)
    if not np.any(m):
        return None
    return float(arr[m][0])


def _last_finite(arr: np.ndarray) -> Optional[float]:
    if arr is None or arr.size == 0:
        return None
    m = np.isfinite(arr)
    if not np.any(m):
        return None
    return float(arr[m][-1])


def prepare_dive_curve(
    dive_df: pd.DataFrame,
    smooth_window: int
) -> Optional[pd.DataFrame]:
    """
    Input: raw dive_df (must include time_s, depth_m; may include water temperature)
    Output (1 Hz):
        time_s: integer-second grid in float
        depth_m: interpolated depth
        rate_abs_mps: abs(diff(depth)) clipped
        rate_abs_mps_smooth: rolling mean
        water_temp_c: interpolated temperature (optional; if any temp column exists)
    """
    global _DIVE_CURVE_DEBUG_PRINTED

    if dive_df is None or len(dive_df) == 0:
        return None

    df = dive_df.sort_values("time_s").reset_index(drop=True).copy()
    if "time_s" not in df.columns or "depth_m" not in df.columns:
        return None

    # Coerce required cols
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
    df = df.dropna(subset=["time_s", "depth_m"])
    if df.empty:
        return None

    t_min = float(df["time_s"].min())
    t_max = float(df["time_s"].max())
    t0 = int(np.floor(t_min))
    t1 = int(np.ceil(t_max))
    if t1 <= t0:
        return None

    uniform_time = np.arange(t0, t1 + 1, 1.0)

    # Depth interp
    depth_interp = np.interp(
        uniform_time,
        df["time_s"].to_numpy(dtype=float),
        df["depth_m"].to_numpy(dtype=float),
    )

    # Rate
    rate_uniform = np.diff(depth_interp, prepend=depth_interp[0])
    rate_abs = np.abs(rate_uniform)
    rate_abs_clipped = np.clip(rate_abs, 0.0, 3.0)

    out = pd.DataFrame({
        "time_s": uniform_time.astype(float),
        "depth_m": depth_interp.astype(float),
        "rate_abs_mps": rate_abs_clipped.astype(float),
    })

    # ---- Temperature propagation (robust) ----
    # Accept multiple possible column names.
    temp_candidates = [
        "water_temp_c",       # preferred
        "water_temperature_c",
        "temperature_c",
        "water_temperature",
        "water_temperature_k",
        "temperature",
        "temp",
        "watertemperature",
    ]
    temp_col = next((c for c in temp_candidates if c in df.columns), None)

    if temp_col is not None:
        t_src = df["time_s"].to_numpy(dtype=float)
        v_src = pd.to_numeric(df[temp_col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(t_src) & np.isfinite(v_src)

        if np.any(m):
            tt = t_src[m]
            vv = v_src[m]

            # Heuristic: if looks like Kelvin, convert to Celsius
            # (Parsers SHOULD already convert, but this protects against raw UDDF/K data.)
            v_first = _first_finite(vv)
            if v_first is not None and v_first > 100.0:
                vv = vv - 273.15

            out["water_temp_c"] = np.interp(uniform_time, tt, vv).astype(float)
        else:
            out["water_temp_c"] = np.full_like(uniform_time, np.nan, dtype=float)

    # ---- Smoothing ----
    if smooth_window is None:
        smooth_window = 1
    try:
        smooth_window = int(smooth_window)
    except Exception:
        smooth_window = 1

    if smooth_window <= 1:
        out["rate_abs_mps_smooth"] = out["rate_abs_mps"]
    else:
        out["rate_abs_mps_smooth"] = (
            out["rate_abs_mps"]
            .rolling(window=smooth_window, center=True, min_periods=1)
            .mean()
        )

    # ---- Debug once ----
    if not _DIVE_CURVE_DEBUG_PRINTED:
        _DIVE_CURVE_DEBUG_PRINTED = True
        try:
            print("[CORE.DIVE_CURVE] loaded")
            print("[CORE.DIVE_CURVE] input columns:", list(dive_df.columns))
            print("[CORE.DIVE_CURVE] temp_col detected:", temp_col)
            if "water_temp_c" in out.columns:
                arr = out["water_temp_c"].to_numpy(dtype=float)
                m2 = np.isfinite(arr)
                print("[CORE.DIVE_CURVE] out water_temp_c size:", int(arr.size), "finite:", int(m2.sum()))
                if np.any(m2):
                    print("[CORE.DIVE_CURVE] out water_temp_c first/last:",
                          float(arr[m2][0]), float(arr[m2][-1]))
                    print("[CORE.DIVE_CURVE] out water_temp_c min/max:",
                          float(np.nanmin(arr)), float(np.nanmax(arr)))
            else:
                print("[CORE.DIVE_CURVE] out has NO water_temp_c column")
        except Exception as e:
            print("[CORE.DIVE_CURVE] debug exception:", repr(e))

    return out


def compute_dive_metrics(
    df_rate: pd.DataFrame,
    dive_df_raw: Optional[pd.DataFrame],
    ff_start_depth_m: float,
) -> Dict[str, Any]:
    """
    Compute dive metrics from rate-smoothed dataframe.
    Returns:
      - descent_avg: average descent rate (m/s) before bottom (negative direction ignored; uses abs)
      - ascent_avg:  average ascent rate (m/s) after bottom (abs)
      - ff_avg:      avg rate between free-fall start and (bottom - 1s)
    """
    result: Dict[str, Any] = {
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

    df = df_rate.copy()
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
    df["rate_abs_mps_smooth"] = pd.to_numeric(df["rate_abs_mps_smooth"], errors="coerce")
    df = df.dropna(subset=["time_s", "depth_m", "rate_abs_mps_smooth"])
    if df.empty:
        return result

    # Bottom time from raw depth
    raw = dive_df_raw.copy()
    raw["time_s"] = pd.to_numeric(raw["time_s"], errors="coerce")
    raw["depth_m"] = pd.to_numeric(raw["depth_m"], errors="coerce")
    raw = raw.dropna(subset=["time_s", "depth_m"])
    if raw.empty:
        return result

    idx_bottom = int(raw["depth_m"].idxmax())
    t_bottom = float(raw.loc[idx_bottom, "time_s"])

    # descent: from start to bottom
    seg_desc = df[df["time_s"] <= t_bottom]
    if not seg_desc.empty:
        result["descent_avg"] = float(seg_desc["rate_abs_mps_smooth"].mean())

    # ascent: from bottom to end
    seg_asc = df[df["time_s"] >= t_bottom]
    if not seg_asc.empty:
        result["ascent_avg"] = float(seg_asc["rate_abs_mps_smooth"].mean())

    # free-fall avg: from first time depth>=ff_start_depth_m to (bottom-1s)
    within_dive = df.copy()
    if ff_start_depth_m is not None and float(ff_start_depth_m) > 0:
        ff_candidates = within_dive[within_dive["depth_m"] >= float(ff_start_depth_m)]
        if not ff_candidates.empty:
            t_ff_start = float(ff_candidates["time_s"].iloc[0])
            ff_end = t_bottom - 1.0
            if ff_end > t_ff_start:
                seg_ff = within_dive[(within_dive["time_s"] >= t_ff_start) & (within_dive["time_s"] <= ff_end)]
                if not seg_ff.empty:
                    result["ff_avg"] = float(seg_ff["rate_abs_mps_smooth"].mean())

    return result
