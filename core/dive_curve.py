"""
Shared dive-curve preprocessing (resample to 1 Hz) and metric computation.
"""
from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


def resample_series_to_1hz(time_s, values, t_start, t_end):
    """
    Resample an irregular time series onto 1 Hz integer seconds.

    Parameters
    ----------
    time_s : array-like
        time in seconds (float or int)
    values : array-like
        samples aligned with time_s
    t_start, t_end : int
        target range inclusive/exclusive (recommend: [t_start, t_end])

    Returns
    -------
    pd.DataFrame with columns ["time_s", "value"]
    """
    # 1) 建立整數秒時間軸
    t = np.arange(int(t_start), int(t_end) + 1, 1)

    # 2) 以 np.interp 做線性插值（或你想用前值填補也可以）
    x = np.asarray(time_s, dtype=float)
    y = np.asarray(values, dtype=float)

    # 避免 nan/空資料
    if len(x) == 0:
        return pd.DataFrame({"time_s": t, "value": np.nan})

    v = np.interp(t, x, y)

    return pd.DataFrame({"time_s": t, "value": v})

def prepare_dive_curve(
    dive_df: pd.DataFrame,
    smooth_window: int
) -> Optional[pd.DataFrame]:

    """
    輸入原始 dive_df（需含 time_s, depth_m）
    回傳：
        time_s: 每秒一個點（整數秒）
        depth_m: 線性插值後深度
        rate_abs_mps: 每秒深度變化量的絕對值
        rate_abs_mps_smooth: 平滑後速率
    """
    if dive_df is None or len(dive_df) == 0:
        return None

    df = dive_df.sort_values("time_s").reset_index(drop=True).copy()
    if "time_s" not in df.columns or "depth_m" not in df.columns:
        return None

    t_min = float(df["time_s"].min())
    t_max = float(df["time_s"].max())
    t0 = int(np.floor(t_min))
    t1 = int(np.ceil(t_max))

    if t1 <= t0:
        return None

    uniform_time = np.arange(t0, t1 + 1, 1.0)

    depth_interp = np.interp(
        uniform_time,
        df["time_s"].to_numpy(),
        df["depth_m"].to_numpy()
    )

    rate_uniform = np.diff(depth_interp, prepend=depth_interp[0])
    rate_abs = np.abs(rate_uniform)
    rate_abs_clipped = np.clip(rate_abs, 0.0, 3.0)

    out = pd.DataFrame({
        "time_s": uniform_time.astype(float),
        "depth_m": depth_interp,
        "rate_abs_mps": rate_abs_clipped,
    })

    if smooth_window <= 1:
        out["rate_abs_mps_smooth"] = out["rate_abs_mps"]
    else:
        out["rate_abs_mps_smooth"] = (
            out["rate_abs_mps"]
            .rolling(window=smooth_window, center=True, min_periods=1)
            .mean()
        )

    return out

def compute_dive_metrics(
    df_rate: pd.DataFrame,
    dive_df_raw: Optional[pd.DataFrame],
    ff_start_depth_m: float,
) -> dict:
    """
    根據重採樣後的 df_rate（time_s, depth_m, rate_abs_mps_smooth）
    與原始 dive_df_raw（time_s, depth_m），計算：
      - descent_avg: 下潛平均速率（扣除開頭與底部各 1 秒）
      - ascent_avg: 上升平均速率（扣除一開始 1 秒）
      - ff_avg: Free Fall 開始後到最低點前 1 秒的平均速率
    回傳 dict，若無法計算則為 None。
    """
    result = {
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

    raw = dive_df_raw.sort_values("time_s").reset_index(drop=True).copy()

    # 1) 找 Dive start / end（跟 Overlay tab 一樣邏輯）
    start_rows = raw[raw["depth_m"] >= 0.7]
    if start_rows.empty:
        return result

    t_start = float(start_rows["time_s"].iloc[0])
    after = raw[raw["time_s"] >= t_start]
    end_candidates = after[after["depth_m"] <= 0.05]
    if not end_candidates.empty:
        t_end = float(end_candidates["time_s"].iloc[-1])
    else:
        t_end = float(after["time_s"].iloc[-1])

    if t_end <= t_start:
        return result

    # 2) 找到底點時間 t_bottom（只看 t_start ~ t_end 區間）
    within_dive = after[after["time_s"] <= t_end]
    if within_dive.empty:
        return result

    idx_bottom = within_dive["depth_m"].idxmax()
    t_bottom = float(within_dive.loc[idx_bottom, "time_s"])

    # 3) 在 df_rate 上切出對應區段
    df = df_rate.sort_values("time_s").reset_index(drop=True)

    # ---- 下潛平均速率：從 t_start + 1 到 t_bottom - 1 ----
    desc_start = t_start + 1.0
    desc_end = t_bottom - 1.0
    if desc_end > desc_start:
        mask_desc = (df["time_s"] >= desc_start) & (df["time_s"] <= desc_end)
        seg_desc = df.loc[mask_desc]
        if not seg_desc.empty:
            result["descent_avg"] = float(seg_desc["rate_abs_mps_smooth"].mean())

    # ---- 上升平均速率：從 t_bottom + 1 到 t_end ----
    asc_start = t_bottom + 1.0
    asc_end = t_end
    if asc_end > asc_start:
        mask_asc = (df["time_s"] >= asc_start) & (df["time_s"] <= asc_end)
        seg_asc = df.loc[mask_asc]
        if not seg_asc.empty:
            result["ascent_avg"] = float(seg_asc["rate_abs_mps_smooth"].mean())

    # ---- Free Fall 段平均速率 ----
    # 從指定 FF 深度開始，到 t_bottom - 1
    max_depth = float(within_dive["depth_m"].max())
    if ff_start_depth_m > 0.0 and ff_start_depth_m < max_depth:
        # 找 raw 裡面第一次達到 FF 深度的時間（只看下潛區間）
        ff_zone = within_dive[
            (within_dive["time_s"] >= t_start) &
            (within_dive["time_s"] <= t_bottom)
        ]
        ff_candidates = ff_zone[ff_zone["depth_m"] >= ff_start_depth_m]
        if not ff_candidates.empty:
            t_ff_start = float(ff_candidates["time_s"].iloc[0])
            ff_end = t_bottom - 1.0
            if ff_end > t_ff_start:
                mask_ff = (df["time_s"] >= t_ff_start) & (df["time_s"] <= ff_end)
                seg_ff = df.loc[mask_ff]
                if not seg_ff.empty:
                    result["ff_avg"] = float(seg_ff["rate_abs_mps_smooth"].mean())

    return result


