# core/fit_hr.py
# Minimal FIT decoder: extract Heart Rate (and Timestamp if present) with zero external deps.
# Safe to import in Streamlit Cloud.

import struct
import numpy as np
import pandas as pd
from pathlib import Path

_FIT_BASE_TYPES = {
    0x00: ("enum", 1),
    0x01: ("sint8", 1),
    0x02: ("uint8", 1),
    0x83: ("sint16", 2),
    0x84: ("uint16", 2),
    0x85: ("sint32", 4),
    0x86: ("uint32", 4),
    0x07: ("string", 1),
    0x88: ("float32", 4),
    0x89: ("float64", 8),
    0x0A: ("uint8z", 1),
    0x8B: ("uint16z", 2),
    0x8C: ("uint32z", 4),
    0x8E: ("sint64", 8),
    0x8F: ("uint64", 8),
    0x90: ("uint64z", 8),
    0x0D: ("byte", 1),
}

def _fit_decode_scalar(raw: bytes, base_type: int, arch: int):
    endian = "<" if arch == 0 else ">"
    t = _FIT_BASE_TYPES.get(base_type, ("unknown", 1))[0]

    if t in ("uint8", "enum", "uint8z", "byte"):
        return raw[0]
    if t == "sint8":
        return struct.unpack(endian + "b", raw)[0]
    if t in ("uint16", "uint16z"):
        return struct.unpack(endian + "H", raw)[0]
    if t == "sint16":
        return struct.unpack(endian + "h", raw)[0]
    if t in ("uint32", "uint32z"):
        return struct.unpack(endian + "I", raw)[0]
    if t == "sint32":
        return struct.unpack(endian + "i", raw)[0]
    if t in ("uint64", "uint64z"):
        return struct.unpack(endian + "Q", raw)[0]
    if t == "sint64":
        return struct.unpack(endian + "q", raw)[0]
    if t == "float32":
        return struct.unpack(endian + "f", raw)[0]
    if t == "float64":
        return struct.unpack(endian + "d", raw)[0]
    if t == "string":
        return raw.split(b"\x00")[0].decode("utf-8", errors="ignore")
    return raw

def extract_fit_heart_rate_series_from_bytes(blob: bytes) -> pd.DataFrame:
    """
    Extract HR from FIT 'record' messages (global msg #20).
    Output columns:
      - timestamp_s: FIT record timestamp (if present; raw seconds)
      - t_s: seconds from first HR sample
      - hr_bpm: heart rate
    """
    if not blob or len(blob) < 14 or blob[8:12] != b".FIT":
        return pd.DataFrame(columns=["timestamp_s", "t_s", "hr_bpm"])

    header_size = blob[0]
    data_size = struct.unpack("<I", blob[4:8])[0]
    data = blob[header_size : header_size + data_size]

    defs = {}
    out_ts = []
    out_hr = []

    off = 0
    while off < len(data):
        rec_hdr = data[off]
        off += 1

        # Compressed timestamp header
        if rec_hdr & 0x80:
            local = (rec_hdr >> 5) & 0x03
            d = defs.get(local)
            if not d:
                break
            msg = {}
            for f in d["fields"]:
                size = f["size"]
                raw = data[off : off + size]
                off += size
                msg[f["field_def"]] = (raw, f["base_type"])
            global_msg = d["global_msg"]
            arch = d["arch"]

        else:
            is_def = bool(rec_hdr & 0x40)
            has_dev_fields = bool(rec_hdr & 0x20)
            local = rec_hdr & 0x0F

            if is_def:
                off += 1  # reserved
                arch = data[off]; off += 1
                endian = "<" if arch == 0 else ">"
                global_msg = struct.unpack(endian + "H", data[off : off + 2])[0]
                off += 2

                nfields = data[off]; off += 1
                fields = []
                for _ in range(nfields):
                    field_def = data[off]
                    size = data[off + 1]
                    base_type = data[off + 2]
                    off += 3
                    fields.append({"field_def": field_def, "size": size, "base_type": base_type})

                if has_dev_fields:
                    dev_n = data[off]; off += 1
                    off += dev_n * 3

                defs[local] = {"global_msg": global_msg, "arch": arch, "fields": fields}
                continue

            d = defs.get(local)
            if not d:
                break

            global_msg = d["global_msg"]
            arch = d["arch"]

            msg = {}
            for f in d["fields"]:
                size = f["size"]
                raw = data[off : off + size]
                off += size
                msg[f["field_def"]] = (raw, f["base_type"])
            # dev field payload ignored

        if global_msg == 20:
            # timestamp field = 253 (uint32), heart_rate = 3 (uint8)
            ts = None
            hr = None
            if 253 in msg:
                raw, bt = msg[253]
                ts = _fit_decode_scalar(raw, bt, arch)
            if 3 in msg:
                raw, bt = msg[3]
                hr = _fit_decode_scalar(raw, bt, arch)

            if ts is not None and hr is not None:
                if isinstance(hr, (int, np.integer)) and 0 < int(hr) < 255:
                    out_ts.append(int(ts))
                    out_hr.append(int(hr))

    if not out_ts:
        return pd.DataFrame(columns=["timestamp_s", "t_s", "hr_bpm"])

    ts0 = out_ts[0]
    df = pd.DataFrame({
        "timestamp_s": np.array(out_ts, dtype=float),
        "t_s": np.array(out_ts, dtype=float) - float(ts0),
        "hr_bpm": np.array(out_hr, dtype=float),
    })
    df = df.drop_duplicates(subset=["timestamp_s"]).sort_values("timestamp_s").reset_index(drop=True)
    return df

def extract_fit_heart_rate_series(fit_path: str) -> pd.DataFrame:
    return extract_fit_heart_rate_series_from_bytes(Path(fit_path).read_bytes())

def merge_hr_into_dive_df(dive_df: pd.DataFrame, hr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer merge by absolute timestamp if dive_df has 'timestamp_s' or 'timestamp' column.
    Otherwise, fallback to merge by 'time_s' assuming both are relative to dive start.
    """
    if dive_df is None or len(dive_df) == 0 or hr_df is None or len(hr_df) == 0:
        return dive_df

    out = dive_df.copy()

    # Case A: absolute timestamp merge
    if "timestamp_s" in out.columns and "timestamp_s" in hr_df.columns:
        t = out["timestamp_s"].astype(float).to_numpy()
        ht = hr_df["timestamp_s"].astype(float).to_numpy()
        hv = hr_df["hr_bpm"].astype(float).to_numpy()
        hr_interp = np.interp(np.clip(t, ht[0], ht[-1]), ht, hv)
        hr_interp[(t < ht[0]) | (t > ht[-1])] = np.nan
        out["hr_bpm"] = hr_interp
        return out

    if "timestamp" in out.columns and "timestamp_s" in hr_df.columns:
        # If 'timestamp' is datetime-like, convert to unix seconds
        try:
            ts = pd.to_datetime(out["timestamp"]).astype("int64") / 1e9
            t = ts.astype(float).to_numpy()
            ht = hr_df["timestamp_s"].astype(float).to_numpy()
            hv = hr_df["hr_bpm"].astype(float).to_numpy()
            hr_interp = np.interp(np.clip(t, ht[0], ht[-1]), ht, hv)
            hr_interp[(t < ht[0]) | (t > ht[-1])] = np.nan
            out["hr_bpm"] = hr_interp
            return out
        except Exception:
            pass

    # Case B: relative time merge (single-dive FIT / already-sliced HR)
    if "time_s" in out.columns and "t_s" in hr_df.columns:
        t = out["time_s"].astype(float).to_numpy()
        ht = hr_df["t_s"].astype(float).to_numpy()
        hv = hr_df["hr_bpm"].astype(float).to_numpy()
        hr_interp = np.interp(np.clip(t, ht[0], ht[-1]), ht, hv)
        hr_interp[(t < ht[0]) | (t > ht[-1])] = np.nan
        out["hr_bpm"] = hr_interp
        return out

    return out
