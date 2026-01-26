# core/parser_garmin.py

from io import BytesIO
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
from fitparse import FitFile


def _load_fit_records(file_like) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Garmin .fit messages into two DataFrames:

    depth_df columns:
      - timestamp (datetime)
      - depth_m (meters, positive)
      - water_temp_c (celsius; may be NaN)

    hr_df columns:
      - timestamp (datetime)
      - heart_rate (bpm)
    """
    # Ensure we have a binary stream for FitFile
    if hasattr(file_like, "read"):
        raw = file_like.read()
        bio = BytesIO(raw)
    else:
        # path-like
        with open(file_like, "rb") as f:
            bio = BytesIO(f.read())

    fit = FitFile(bio)

    depth_rows: List[Dict[str, Any]] = []
    hr_rows: List[Dict[str, Any]] = []

    # Garmin FIT typically stores per-second samples in "record" messages
    for msg in fit.get_messages("record"):
        ts = None
        depth = None
        temp = None
        hr = None

        for d in msg:
            name = d.name
            val = d.value

            if name == "timestamp":
                ts = val

            # Depth fields vary by device/profile
            elif name in ("depth", "enhanced_depth"):
                depth = val

            # Temperature fields vary
            elif name in ("temperature", "water_temperature"):
                temp = val

            elif name in ("heart_rate", "hr", "heartrate"):
                hr = val

        if ts is None:
            continue

        ts_dt = pd.to_datetime(ts)

        if depth is not None:
            try:
                depth_val = float(depth)
                temp_val = float(temp) if temp is not None else None  # Garmin is typically already °C
                depth_rows.append(
                    {
                        "timestamp": ts_dt,
                        "depth_m": depth_val,
                        "water_temp_c": temp_val,
                    }
                )
            except (TypeError, ValueError):
                pass

        if hr is not None:
            try:
                hr_val = float(hr)
                hr_rows.append({"timestamp": ts_dt, "heart_rate": hr_val})
            except (TypeError, ValueError):
                pass

    depth_df = (
        pd.DataFrame(depth_rows)
        if depth_rows
        else pd.DataFrame(columns=["timestamp", "depth_m", "water_temp_c"])
    )
    hr_df = (
        pd.DataFrame(hr_rows)
        if hr_rows
        else pd.DataFrame(columns=["timestamp", "heart_rate"])
    )

    if not depth_df.empty:
        depth_df = depth_df.sort_values("timestamp").reset_index(drop=True)
        depth_df["depth_m"] = pd.to_numeric(depth_df["depth_m"], errors="coerce")
        depth_df["water_temp_c"] = pd.to_numeric(depth_df["water_temp_c"], errors="coerce")

    if not hr_df.empty:
        hr_df = hr_df.sort_values("timestamp").reset_index(drop=True)
        hr_df["heart_rate"] = pd.to_numeric(hr_df["heart_rate"], errors="coerce")

    return depth_df, hr_df


def parse_garmin_fit_to_dives(
    file_like,
    start_threshold: float = 1.1, # start count dive when depth_m > 1.0 m
    end_threshold: float = 0.3,
    min_dive_duration_s: float = 10.0,  # when time_s[-1] >= 10.0 count as a dive
) -> Dict[str, Any]:
    """
    Split Garmin FIT records into dives by depth threshold.
    Returns:
      {
        "dives": [DataFrame(time_s, depth_m, water_temp_c?) ...],
        "heart_rate": [DataFrame(time_s, heart_rate) or None ...]
      }
    """
    depth_df, hr_df = _load_fit_records(file_like)

    if depth_df is None or depth_df.empty:
        return {"dives": [], "heart_rate": []}

    df = depth_df.copy()
    df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
    df["water_temp_c"] = pd.to_numeric(df.get("water_temp_c"), errors="coerce")
    df = df.dropna(subset=["timestamp", "depth_m"]).sort_values("timestamp").reset_index(drop=True)

    dives: List[pd.DataFrame] = []
    hr_list: List[Optional[pd.DataFrame]] = []

    in_dive = False
    dive_start_idx: Optional[int] = None

    def finalize(depth_seg: pd.DataFrame):
        if depth_seg is None or depth_seg.empty:
            return

        depth_seg = depth_seg.reset_index(drop=True)
        t0 = depth_seg["timestamp"].iloc[0]
        t1 = depth_seg["timestamp"].iloc[-1]

        depth_seg["time_s"] = (depth_seg["timestamp"] - t0).dt.total_seconds().astype(float)
        if float(depth_seg["time_s"].iloc[-1]) < float(min_dive_duration_s):
            return

        # ---- dive curve (KEEP water_temp_c if present) ----
        cols = ["time_s", "depth_m"]
        if "water_temp_c" in depth_seg.columns:
            cols.append("water_temp_c")
        dives.append(depth_seg[cols].copy())

        # ---- HR curve aligned by timestamp range ----
        if hr_df is None or hr_df.empty:
            hr_list.append(None)
            return

        seg_hr = hr_df[(hr_df["timestamp"] >= t0) & (hr_df["timestamp"] <= t1)].copy()
        if seg_hr.empty:
            hr_list.append(None)
            return

        seg_hr["time_s"] = (seg_hr["timestamp"] - t0).dt.total_seconds().astype(float)
        seg_hr["heart_rate"] = pd.to_numeric(seg_hr["heart_rate"], errors="coerce")
        seg_hr = seg_hr.dropna(subset=["time_s", "heart_rate"])

        if seg_hr.empty:
            hr_list.append(None)
        else:
            hr_list.append(seg_hr[["time_s", "heart_rate"]].sort_values("time_s").reset_index(drop=True))

    # ---- detect dive segments ----
    for i, row in df.iterrows():
        depth = float(row["depth_m"])

        if not in_dive:
            if depth > float(start_threshold):
                in_dive = True
                dive_start_idx = i
        else:
            if depth < float(end_threshold):
                in_dive = False
                dive_end_idx = i
                finalize(df.loc[dive_start_idx:dive_end_idx].copy())
                dive_start_idx = None

    if in_dive and dive_start_idx is not None:
        finalize(df.loc[dive_start_idx:].copy())

    return {"dives": dives, "heart_rate": hr_list}

def parse_garmin_fit_to_dives_with_hr(
    file_like,
    start_threshold: float = 1.1,  # start count as a dive if depth > 1.1m
    end_threshold: float = 0.1,  # original = 0.3
    min_dive_duration_s: float = 5.0,  # count as a dive# when dive time > 10sec, original = 10.0
):
    """
    Backward-compatible wrapper for app.py.
    Returns the same dict format: {"dives": [...], "heart_rate": [...]}
    """
    return parse_garmin_fit_to_dives(
        file_like,
        start_threshold=start_threshold,
        end_threshold=end_threshold,
        min_dive_duration_s=min_dive_duration_s,
    )

