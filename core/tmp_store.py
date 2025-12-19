"""
Temp-file persistence and session cleanup helpers.

Kept compatible with the original app.py behavior.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, Set

import streamlit as st

def persist_upload_to_tmp(uploaded_file) -> dict:
    suffix = ""
    if uploaded_file.name and "." in uploaded_file.name:
        suffix = "." + uploaded_file.name.split(".")[-1].lower()

    fd, path = tempfile.mkstemp(prefix="upload_", suffix=suffix)
    os.close(fd)

    with open(path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f, length=1024 * 1024)

    return {
        "path": path,
        "name": uploaded_file.name,
        "size": uploaded_file.size,
        "type": uploaded_file.type,
    }


# -------------------------------
# Runtime cleanup helpers (Render/Streamlit long-lived process)
# -------------------------------
TMP_PREFIXES = ("upload_", "dive_overlay_output_", "dive_overlay_audio_", "dive_overlay_input_")

def _safe_unlink(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def cleanup_tmp_dir(max_age_sec: int = 60 * 60 * 2,keep_paths: Optional[Set[str]] = None) -> None:
    """Delete old temp files created by this app under /tmp to avoid accumulation across sessions."""
    keep_paths = keep_paths or set()
    tmp_dir = Path(tempfile.gettempdir())
    now = time.time()
    try:
        for p in tmp_dir.iterdir():
            try:
                if not p.is_file():
                    continue
                if not any(p.name.startswith(pref) for pref in TMP_PREFIXES):
                    continue
                if str(p) in keep_paths:
                    continue
                age = now - p.stat().st_mtime
                if age >= max_age_sec:
                    p.unlink(missing_ok=True)
            except Exception:
                continue
    except Exception:
        pass

def cleanup_session_files(keys: list[str]) -> None:
    """Remove files referenced by session_state keys and delete the keys."""
    for k in keys:
        v = st.session_state.get(k)
        if isinstance(v, dict) and "path" in v:
            _safe_unlink(v.get("path"))
        elif isinstance(v, (str, Path)):
            _safe_unlink(str(v))
        st.session_state.pop(k, None)

def ensure_clean_before_new_job():
    """
    Ensure previous overlay job resources are cleaned
    before starting a new upload/render job.
    """
    if st.session_state.get("ov_job_state") in ("done", "error"):
        # Reset session-state pointers and delete per-job temp files
        reset_overlay_job()
        # Best-effort sweep: remove older /tmp artifacts from prior sessions
        # (do NOT keep any paths here because new uploads have not been persisted yet)
        cleanup_tmp_dir(max_age_sec=60 * 60 * 2, keep_paths=set())

def reset_overlay_job() -> None:
    cleanup_session_files([
        "ov_video_meta",
        "ov_output_path",
        "ov_tmp_audio_path",
        "ov_render_error",
    ])
    # Also clear uploader widgets to allow re-upload
    for k in ["overlay_video_file", "overlay_watch_file"]:
        st.session_state.pop(k, None)
    st.session_state["ov_job_state"] = "idle"
    try:
        import gc
        gc.collect()
    except Exception:
        pass

