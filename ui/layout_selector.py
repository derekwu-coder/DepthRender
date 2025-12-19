"""
Reusable layout selector UI (preview cards).
Extracted from app.py. No side-effects at import time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

def render_layout_selector(layouts_config, assets_dir, tr, key="overlay_layout_id"):
    """
    Render 2-column layout selector with preview cards.
    Returns selected layout id.
    """
    from pathlib import Path
    import base64
    import streamlit as st

    LAYOUTS_DIR = Path(assets_dir) / "layouts"

    # init
    if key not in st.session_state:
        st.session_state[key] = layouts_config[0]["id"]
    selected_id = st.session_state[key]

    def _img_to_base64_png(path: Path) -> str:
        if not path.exists():
            return ""
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    cols = st.columns(2, gap="small")

    for idx, cfg in enumerate(layouts_config):
        col = cols[idx % 2]
        layout_id = cfg["id"]
        label = tr(cfg["label_key"])
        img_path = LAYOUTS_DIR / cfg["filename"]
        img_b64 = _img_to_base64_png(img_path)

        is_selected = (layout_id == selected_id)
        card_state_class = "selected" if is_selected else "dimmed"

        # ✅ 勾勾（只在 selected 顯示）
        check_html = '<div class="layout-check"><span class="layout-checkmark">✓</span></div>' if is_selected else ""

        card_html = f"""
        <div class="layout-card {card_state_class}">
            <img src="data:image/png;base64,{img_b64}" />
            <div class="layout-footer">
                {check_html}
                <div class="layout-title">{label}</div>
            </div>
        </div>
        """

        with col:
            st.markdown(card_html, unsafe_allow_html=True)

            if st.button(
                tr("select_layout_btn"),              # <<< 你現在已經修好翻譯了，用這個 key
                key=f"select_layout_btn_{layout_id}",
                use_container_width=True,
            ):
                st.session_state[key] = layout_id
                st.rerun()

    return st.session_state[key]

# ==================================
# 全局 CSS：讓畫面更像 App
# ==================================
