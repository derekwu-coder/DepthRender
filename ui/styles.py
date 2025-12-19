"""
CSS injection for the Streamlit app UI.
"""
import streamlit as st

APP_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ===== 版面置中並限制最大寬度 ===== */
.main > div {
    display: flex;
    justify-content: center;
}

.main > div > div {
    max-width: 1200px;
}

/* ===== 主內容往下推一點，騰出 header 空間 ===== */
.block-container {
    padding-top: 112px;   /* header + tabs 的總高度，大約 100~120 之間自己可以微調 */
}

/* ===== 頂部品牌列：包成一個 fixed header ===== */
.app-header-row {
    position: fixed;         /* 原本是 sticky，改成 fixed 綁在視窗 */
    top: 0;
    left: 0;
    right: 0;
    z-index: 100;
    padding: 0.25rem 0.1rem 0.35rem 0.1rem;
    backdrop-filter: blur(10px);
    background: rgba(248,250,252,0.96);  /* 淺色模式淡底 */
}

/* 深色模式下 header 背景 */
@media (prefers-color-scheme: dark) {
    .app-header-row {
        background: rgba(15,23,42,0.98);
    }
}

/* 內層品牌列內容 */
.app-top-bar {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.2rem 0.6rem 0.4rem;
}

.app-top-icon {
    font-size: 1.6rem;
}

.app-title-text {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.9rem;
}

.app-title-sub {
    font-size: 1.0rem;
    opacity: 0.8;
}

/* ⭐ 手機版品牌標題縮小 */
@media (max-width: 600px) {
    .app-title-text {
        font-size: 1.45rem !important;
        line-height: 1.45rem !important;
    }
    .app-title-sub {
        font-size: 0.9rem !important;
    }
}

/* ===== app-card（白底卡片） ===== */
.app-card {
    background-color: rgba(255,255,255,0.90);
    border-radius: 15px;
    padding: 1rem 1.2rem 1.4rem 1.2rem;
    box-shadow: 0 8px 20px rgba(15,23,42,0.10);
}

/* 深色模式 */
@media (prefers-color-scheme: dark) {
    .app-card {
        background-color: rgba(15,23,42,0.90);
        box-shadow: 0 8px 20px rgba(0,0,0,0.60);
    }
}

/* ===== 標題縮小 ===== */
h3 {
    font-size: 1.05rem !important;
    margin-top: 0.6rem;
    margin-bottom: 0.2rem;
}


/* ===== Align time block: force 100% width on mobile ===== */
.align-time-block { width: 100%; }
@media (max-width: 600px){
  .align-time-block { width: 100% !important; max-width: 100% !important; }
  .align-time-block div[data-testid="stHorizontalBlock"]{ width: 100% !important; }
}

/* ======================================================
   🌑 Tabs 外觀：背景融入 + 保留膠囊造型
   ====================================================== */

/* 讓 stTabs 整塊本身不要多餘底線/陰影 */
div[data-testid="stTabs"] {
    border-bottom: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* Tabs 的 tablist：上面那條長條所在的區域 */
div[data-testid="stTabs"] div[role="tablist"] {
    position: fixed;
    top: 60px;  /* 往上靠一點，讓長條更貼近 header 底部 */
    left: 0;
    right: 0;
    z-index: 90;

    /* 上方 padding 改為 0，避免標籤長條上面還有一層空隙 */
    padding: 0 0.4rem 0.20rem 0.4rem !important;
    margin-bottom: 0 !important;

    background: #f8fafc !important;
    border-bottom: none !important;
    box-shadow: none !important;
}



/* 深色模式：改成你實際量到的 #0E1117 */
@media (prefers-color-scheme: dark) {
    div[data-testid="stTabs"] div[role="tablist"] {
        background: #0E1117 !important;
    }
}

/* 移除 tablist 可能附加的裝飾 bar（避免多一層亮條）*/
div[data-testid="stTabs"] div[role="tablist"]::before,
div[data-testid="stTabs"] div[role="tablist"]::after {
    content: none !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* 👉 移動中的 pill / highlight：直接關掉整個元素 */
div[data-baseweb="tab-highlight"] {
    display: none !important;          /* 最直接：整條不畫 */
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    height: 0 !important;
    opacity: 0 !important;
}

/* 深色模式保險再蓋一次 */
@media (prefers-color-scheme: dark) {
    div[data-baseweb="tab-highlight"] {
        display: none !important;
        background: transparent !important;
        opacity: 0 !important;
    }
}

/* 這個通常是 Tabs 底部那條長 bar，用同色把它「蓋掉」 */
div[data-baseweb="tab-border"] {
    background: #f8fafc !important;
    box-shadow: none !important;
    border: none !important;
    height: 0.10rem !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* 深色模式下，底部 bar 也改成 #0E1117（跟背景完全融在一起） */
@media (prefers-color-scheme: dark) {
    div[data-baseweb="tab-border"] {
        background: #0E1117 !important;
    }
}

/* ⭐ 真正的膠囊 tab 按鈕樣式（這一段是你現在少掉的，所以膠囊會消失） */
div[data-testid="stTabs"] button[role="tab"] {
    border-radius: 999px !important;        /* 膠囊形狀 */
    padding: 0.18rem 0.9rem !important;
    margin-right: 0.45rem !important;
    border: 1px solid rgba(148,163,184,0.7) !important;  /* gray-ish 邊框 */
    background-color: #f3f4f6 !important;   /* 淺灰 */
    color: #111827 !important;              /* 深字 */
    font-size: 0.88rem !important;          /* 稍微小一點，手機不會太霸佔 */
    font-weight: 500 !important;
    box-shadow: none !important;
}

/* 取消 Streamlit 原本的 underline */
div[data-testid="stTabs"] button[role="tab"]::after {
    content: none !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* 被選中的 tab（淺色模式）：淡藍色膠囊 */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background-color: #dbeafe !important;    /* light blue */
    border-color: #38bdf8 !important;        /* cyan-ish */
    color: #0f172a !important;               /* slate-900 */
}

/* 深色模式下 tabs 的顏色配置 */
@media (prefers-color-scheme: dark) {

    /* 未選取：深灰膠囊 */
    div[data-testid="stTabs"] button[role="tab"] {
        background-color: #111827 !important;
        border-color: rgba(55,65,81,0.9) !important;
        color: #e5e7eb !important;
    }

    /* 已選取：稍亮一點的藍灰膠囊 */
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background-color: #1f2937 !important;   /* 深藍灰 */
        border-color: #38bdf8 !important;
        color: #e5f2ff !important;
    }
}

/* Tabs 底部與內文的距離再縮一點 */
div[data-testid="stTabs"] + div {
    margin-top: 0.20rem !important;
}

/* ===== Layout Grid Selector ===== */
.layout-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

/* 卡片本體 */

.layout-card{
  position: relative;
  border-radius: 18px;
  overflow: hidden;
  box-sizing: border-box;
}

.layout-card img{
  display:block;
  width:100%;
  height:auto;
}

/* 未選取：更灰、更暗 */
.layout-card.dimmed img{
  filter: grayscale(100%) saturate(0%) contrast(85%) brightness(70%);
  opacity: 0.80;
}

.layout-card.selected .layout-title{
  color: #111;
}

.layout-card.dimmed .layout-title{
  color: #666;
}

/* footer：預設（dark mode 下也好看） */
.layout-footer{
  background: #f2f2f2;   /* 白天模式不融入背景 */
  height: 40px;          /* footer 高度（你之前太大就改小） */
  display:flex;
  align-items:center;
  justify-content:center;
  gap: 10px;
}

.layout-check{
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #4CAF50;          /* 綠色圓底 */
  color: #ffffff;               /* 白色勾勾 */
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 700;
  line-height: 1;
  flex-shrink: 0;
}


.layout-title{
  font-size: 18px;       /* 標題大小 */
  font-weight: 800;
  color: #111;
}

/* ======================================================
   🌟 手機優化區（以下 100% 保證效果正確） 
   ====================================================== */
@media (max-width: 768px) {

    .app-card {
        padding: 0.8rem 0.9rem 1.1rem 0.9rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(15,23,42,0.15);
    }

    h3 {
        font-size: 0.95rem !important;
    }

    .stButton>button,
    .stDownloadButton>button {
        width: 100%;
    }

    /* ==========================================================
       ① 全站預設：所有 st.columns 手機上「左右並排」(50/50)
       ========================================================== */
    div[data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: flex-start;
    }

    div[data-testid="stHorizontalBlock"] > div {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        max-width: 50% !important;
        padding-left: 0.35rem;
        padding-right: 0.35rem;
        box-sizing: border-box;
    }

    div[data-testid="stHorizontalBlock"] > div > div {
        max-width: 100% !important;
    }

    /* ==========================================================
       ② 在「疊加影片產生器 tab」裡把 st.columns 改回上下排列
          （避免深度圖 / 速率圖在手機端被擠成兩欄）
       ========================================================== */

    /* 疊加影片頁面的 wrapper */
    .overlay-stack-mobile div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        flex-wrap: nowrap !important;
    }

    /* 每欄吃滿 100% */
    .overlay-stack-mobile div[data-testid="stHorizontalBlock"] > div {
        max-width: 100% !important;
        width: 100% !important;
    }
}

</style>
"""


def inject_app_css() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
