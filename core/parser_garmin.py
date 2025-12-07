# core/parser_garmin.py

from io import BytesIO
from typing import List
from fitparse import FitFile
import pandas as pd


def _load_fit_records(file_like) -> pd.DataFrame:
    """
    讀取 Garmin .fit 檔裡的 record 訊息，抽出 timestamp + depth。
    回傳一個 DataFrame，欄位：
      - timestamp: pandas.Timestamp
      - depth_m:   float（正值，單位：公尺）
    """
    if isinstance(file_like, BytesIO):
        file_like.seek(0)
        fit = FitFile(file_like)
    else:
        fit = FitFile(file_like)

    rows = []

    # Garmin 的潛水紀錄通常在 "record" 訊息裡，深度欄位可能叫 depth 或 enhanced_depth
    for record in fit.get_messages("record"):
        ts = None
        depth = None

        for d in record:
            if d.name == "timestamp":
                ts = d.value
            elif d.name in ("depth", "enhanced_depth"):
                depth = d.value

        if ts is not None and depth is not None:
            try:
                depth_val = float(depth)
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "timestamp": pd.to_datetime(ts),
                    "depth_m": abs(depth_val),  # 統一用正值深度
                }
            )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "depth_m"])

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def _segment_dives_by_depth(df: pd.DataFrame,
                            start_threshold: float = 1.0,
                            end_threshold: float = 0.8,
                            min_dive_duration_s: float = 10.0) -> List[pd.DataFrame]:
    """
    用簡單的「深度門檻」把連續潛水分段：
      - 當深度 > start_threshold 視為進入一潛
      - 當深度 < end_threshold 視為回到水面，結束一潛
      - 只有總時長超過 min_dive_duration_s 的才算有效一潛
    回傳一個 dives list，每個元素是單一潛水的 DataFrame，欄位：
      - time_s：從該潛開始算的時間（秒）
      - depth_m：深度（公尺）
    """

    if df.empty:
        return []

    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    dives = []
    in_dive = False
    dive_start_idx = None

    for i, row in df.iterrows():
        depth = row["depth_m"]

        if not in_dive:
            # 還沒進入潛水，看到深度大於門檻就當作一潛開始
            if depth > start_threshold:
                in_dive = True
                dive_start_idx = i
        else:
            # 已經在一潛之中，如果深度回到 end_threshold 以下就結束一潛
            if depth < end_threshold:
                in_dive = False
                dive_end_idx = i

                dive_df = df.loc[dive_start_idx:dive_end_idx].reset_index(drop=True)

                # 用 timestamp 換算該潛的 time_s（從 0 開始）
                t0 = dive_df["timestamp"].iloc[0]
                dive_df["time_s"] = (
                    dive_df["timestamp"] - t0
                ).dt.total_seconds().astype(float)

                # 濾掉太短的潛水
                if dive_df["time_s"].iloc[-1] >= min_dive_duration_s:
                    dives.append(dive_df[["time_s", "depth_m"]])

                dive_start_idx = None

    # 如果最後還在水下卻沒回到 surface 門檻，也當作一潛收尾
    if in_dive and dive_start_idx is not None:
        dive_df = df.loc[dive_start_idx:].reset_index(drop=True)
        t0 = dive_df["timestamp"].iloc[0]
        dive_df["time_s"] = (
            dive_df["timestamp"] - t0
        ).dt.total_seconds().astype(float)

        if dive_df["time_s"].iloc[-1] >= min_dive_duration_s:
            dives.append(dive_df[["time_s", "depth_m"]])

    return dives


def parse_garmin_fit_to_dives(file_like) -> List[pd.DataFrame]:
    """
    高階介面：
      - 讀取 Garmin .fit 檔
      - 抽出 depth vs timestamp
      - 自動切成多潛
      - 回傳 List[pd.DataFrame]，每個 DataFrame 具有欄位：
          - time_s
          - depth_m
    """
    base_df = _load_fit_records(file_like)

    if base_df.empty:
        return []

    dives = _segment_dives_by_depth(base_df)
    return dives
