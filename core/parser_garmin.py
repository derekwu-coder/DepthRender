# core/parser_garmin.py

from io import BytesIO
from typing import List, Optional, Dict, Any, Tuple
from fitparse import FitFile
import pandas as pd


def _load_fit_records(file_like) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Garmin .fit messages into two DataFrames:

    depth_df:
      - timestamp (datetime)
      - depth_m   (meters, positive)

    hr_df:
      - timestamp (datetime)
      - heart_rate (bpm)

    Important:
    - Depth and HR do NOT have to appear in the same record.
    - We keep HR records even if depth is missing, so later we can align HR to dives by timestamp.
    """
    if isinstance(file_like, BytesIO):
        file_like.seek(0)
        fit = FitFile(file_like)
    else:
        fit = FitFile(file_like)

    depth_rows = []
    hr_rows = []

    for record in fit.get_messages("record"):
        ts = None
        depth = None
        hr = None

        for d in record:
            if d.name == "timestamp":
                ts = d.value
            elif d.name in ("depth", "enhanced_depth"):
                depth = d.value
            elif d.name in ("heart_rate", "hr", "heartrate"):
                hr = d.value

        if ts is None:
            continue

        ts_dt = pd.to_datetime(ts)

        if depth is not None:
            try:
                depth_val = float(depth)
                depth_rows.append({"timestamp": ts_dt, "depth_m": abs(depth_val)})
            except (TypeError, ValueError):
                pass

        if hr is not None:
            try:
                hr_val = float(hr)
                hr_rows.append({"timestamp": ts_dt, "heart_rate": hr_val})
            except (TypeError, ValueError):
                pass

    depth_df = pd.DataFrame(depth_rows) if depth_rows else pd.DataFrame(columns=["timestamp", "depth_m"])
    hr_df = pd.DataFrame(hr_rows) if hr_rows else pd.DataFrame(columns=["timestamp", "heart_rate"])

    if not depth_df.empty:
        depth_df = depth_df.sort_values("timestamp").reset_index(drop=True)
    if not hr_df.empty:
        hr_df = hr_df.sort_values("timestamp").reset_index(drop=True)

    return depth_df, hr_df


def _segment_dives_by_depth(
    df: pd.DataFrame,
    start_threshold: float = 1.0,
    end_threshold: float = 0.8,
    min_dive_duration_s: float = 10.0,
) -> List[pd.DataFrame]:
    """
    Segment continuous dive by depth threshold.

    Returns list of dive_df with columns:
      - time_s
      - depth_m
    """
    if df is None or df.empty:
        return []

    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    dives = []
    in_dive = False
    dive_start_idx = None

    for i, row in df.iterrows():
        depth = float(row["depth_m"])

        if not in_dive:
            if depth > start_threshold:
                in_dive = True
                dive_start_idx = i
        else:
            if depth < end_threshold:
                in_dive = False
                dive_end_idx = i

                seg = df.loc[dive_start_idx:dive_end_idx].reset_index(drop=True)
                t0 = seg["timestamp"].iloc[0]
                seg["time_s"] = (seg["timestamp"] - t0).dt.total_seconds().astype(float)

                if float(seg["time_s"].iloc[-1]) >= float(min_dive_duration_s):
                    dives.append(seg[["time_s", "depth_m"]].copy())

                dive_start_idx = None

    if in_dive and dive_start_idx is not None:
        seg = df.loc[dive_start_idx:].reset_index(drop=True)
        t0 = seg["timestamp"].iloc[0]
        seg["time_s"] = (seg["timestamp"] - t0).dt.total_seconds().astype(float)

        if float(seg["time_s"].iloc[-1]) >= float(min_dive_duration_s):
            dives.append(seg[["time_s", "depth_m"]].copy())

    return dives


def parse_garmin_fit_to_dives(file_like) -> List[pd.DataFrame]:
    """
    Original interface: return dive list (time_s, depth_m).
    """
    depth_df, _ = _load_fit_records(file_like)
    if depth_df is None or depth_df.empty:
        return []
    return _segment_dives_by_depth(depth_df)


def parse_garmin_fit_to_dives_with_hr(file_like) -> Dict[str, Any]:
    """
    Return dives and per-dive HR list aligned to each dive's time_s.

    Returns:
      {
        "dives": List[pd.DataFrame],                  # time_s, depth_m
        "heart_rate": List[Optional[pd.DataFrame]]    # time_s, heart_rate (per dive) or None
      }
    """
    depth_df, hr_stream_df = _load_fit_records(file_like)
    if depth_df is None or depth_df.empty:
        return {"dives": [], "heart_rate": []}

    # Use same segmentation policy
    start_threshold = 1.0
    end_threshold = 0.8
    min_dive_duration_s = 10.0

    df = depth_df.copy().sort_values("timestamp").reset_index(drop=True)

    dives: List[pd.DataFrame] = []
    hr_list: List[Optional[pd.DataFrame]] = []

    in_dive = False
    dive_start_idx = None

    def finalize(depth_seg: pd.DataFrame):
        if depth_seg is None or depth_seg.empty:
            return

        depth_seg = depth_seg.reset_index(drop=True)
        t0 = depth_seg["timestamp"].iloc[0]
        t1 = depth_seg["timestamp"].iloc[-1]

        depth_seg["time_s"] = (depth_seg["timestamp"] - t0).dt.total_seconds().astype(float)
        if float(depth_seg["time_s"].iloc[-1]) < float(min_dive_duration_s):
            return

        # dive curve
        dives.append(depth_seg[["time_s", "depth_m"]].copy())

        # HR curve aligned by timestamp range
        if hr_stream_df is None or hr_stream_df.empty:
            hr_list.append(None)
            return

        seg_hr = hr_stream_df[(hr_stream_df["timestamp"] >= t0) & (hr_stream_df["timestamp"] <= t1)].copy()
        if seg_hr.empty:
            hr_list.append(None)
            return

        seg_hr["time_s"] = (seg_hr["timestamp"] - t0).dt.total_seconds().astype(float)
        seg_hr["heart_rate"] = pd.to_numeric(seg_hr["heart_rate"], errors="coerce")
        seg_hr = seg_hr.dropna(subset=["time_s", "heart_rate"])

        if seg_hr.empty:
            hr_list.append(None)
        else:
            hr_list.append(
                seg_hr[["time_s", "heart_rate"]].sort_values("time_s").reset_index(drop=True)
            )

    for i, row in df.iterrows():
        depth = float(row["depth_m"])

        if not in_dive:
            if depth > start_threshold:
                in_dive = True
                dive_start_idx = i
        else:
            if depth < end_threshold:
                in_dive = False
                dive_end_idx = i
                finalize(df.loc[dive_start_idx:dive_end_idx].copy())
                dive_start_idx = None

    if in_dive and dive_start_idx is not None:
        finalize(df.loc[dive_start_idx:].copy())

    return {"dives": dives, "heart_rate": hr_list}
