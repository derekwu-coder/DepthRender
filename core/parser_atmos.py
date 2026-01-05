# core/parser_atmos.py
import pandas as pd
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime

def parse_atmos_uddf(file_like) -> pd.DataFrame:
    """
    讀入 ATMOS 的 UDDF 檔（BytesIO 或檔案物件），
    回傳一個 DataFrame，至少包含：
      - time_s：從潛水開始起算的秒數（float）
      - depth_m：深度（正值，單位公尺）
    """
    # 1. 讀 XML
    if isinstance(file_like, BytesIO):
        file_like.seek(0)
        tree = ET.parse(file_like)
    else:
        tree = ET.parse(file_like)

    root = tree.getroot()

    # 2. 找出所有「採樣點」節點
    # 不同廠商的 UDDF 會用不同名稱，先嘗試幾個常見的
    samples = []
    for tag_name in ["sample", "waypoint", "wpt", "realtime"]:
        samples = root.findall(f".//{tag_name}")
        if samples:
            break

    if not samples:
        # 找不到任何 sample，就回傳空表
        return pd.DataFrame(columns=["time_s", "depth_m", "water_temp_c"])

    times = []
    depths = []
    temps_c = []  # water temperature in Celsius (may contain None)

    # 小工具：把 ISO 時間字串轉成 datetime（如果 UDDF 用絕對時間）
    def parse_iso_time(t_str: str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(t_str, fmt)
            except ValueError:
                continue
        return None

    absolute_times = []

    for s in samples:
        # ---- 找 depth ----
        depth_val = None

        # 可能是 <depth><value>12.3</value></depth>
        depth_node = s.find(".//depth/value")
        if depth_node is None:
            # 或 <depth>12.3</depth>
            depth_node = s.find(".//depth")
        if depth_node is None:
            # 或 <depth value="12.3" />
            depth_node = s.find(".//depth[@value]")
            if depth_node is not None and "value" in depth_node.attrib:
                try:
                    depth_val = float(depth_node.attrib["value"])
                except ValueError:
                    depth_val = None

        if depth_node is not None and depth_val is None and depth_node.text:
            try:
                depth_val = float(depth_node.text)
            except ValueError:
                depth_val = None

        # ---- 找 time / divetime ----
        time_sec = None

        # 情況1：<divetime>123.4</divetime> 直接給秒數
        divetime_node = s.find(".//divetime")
        if divetime_node is not None and divetime_node.text:
            try:
                time_sec = float(divetime_node.text)
            except ValueError:
                time_sec = None

        # 情況2：<time>2025-01-01T10:00:00</time> 這種絕對時間
        if time_sec is None:
            time_node = s.find(".//time")
            if time_node is not None and time_node.text:
                dt = parse_iso_time(time_node.text.strip())
                if dt is not None:
                    absolute_times.append(dt)

                    # 先暫存，等全部跑完再統一換成相對秒數
                    time_sec = dt  # 先暫存 datetime，後面再處理


        # ---- 找 temperature / watertemperature ----
        temp_val = None
        for _tname in ["watertemperature", "water_temperature", "temperature", "temp"]:
            tnode = s.find(f".//{_tname}")
            if tnode is not None and tnode.text:
                try:
                    temp_val = float(tnode.text)
                except ValueError:
                    temp_val = None
                if temp_val is not None:
                    break

        # UDDF sometimes stores Kelvin; auto convert if it looks like Kelvin
        if temp_val is not None and temp_val > 100.0:
            temp_val = temp_val - 273.15

        if depth_val is not None and time_sec is not None:
            depths.append(depth_val)
            times.append(time_sec)
            temps_c.append(temp_val)

    if not times or not depths:
        return pd.DataFrame(columns=["time_s", "depth_m", "water_temp_c"])

    # 如果 time_sec 是 datetime，就要換成「從第一筆開始的秒數」
    if isinstance(times[0], datetime):
        t0 = times[0]
        times_float = [(t - t0).total_seconds() for t in times]
    else:
        times_float = times

    # Ensure temps length matches
    if len(temps_c) != len(times_float):
        # pad with None
        temps_c = (temps_c + [None] * len(times_float))[:len(times_float)]

    df = pd.DataFrame({
        "time_s": times_float,
        "depth_m": depths,
        "water_temp_c": temps_c,
    })

    # 統一用正值深度
    df["depth_m"] = df["depth_m"].abs()

    # 依時間排序 & 重設 index
    df = df.sort_values("time_s").reset_index(drop=True)

    return df
