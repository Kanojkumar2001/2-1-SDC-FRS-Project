"""
app.py  –  FRS (Face Recognition System) — Streamlit front-end
Run with:  streamlit run app.py
"""

import os
import sys
import threading

import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode

# ── Make sure imports from config / src work regardless of CWD ──────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SAMPLES_PER_PERSON, TRAINER_FILE, DATASET_DIR, CONFIDENCE_THRESH
from src.capture_faces import capture_from_frame, finalize_capture
from src.database import load_students, delete_student
from src.recognize import load_recognizer, recognize_frame
from src.train_model import train
from src.utils import ensure_cascade

# ── Ensure cascade XML is present before anything else ──────────────────────
ensure_cascade()


@st.cache_data(ttl=3600, show_spinner=False)
def build_rtc_configuration():
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    cloudflare_key_id = os.getenv("CLOUDFLARE_TURN_KEY_ID", "").strip()
    cloudflare_api_token = os.getenv("CLOUDFLARE_TURN_KEY_API_TOKEN", "").strip()

    if cloudflare_key_id and cloudflare_api_token:
        try:
            response = requests.post(
                "https://rtc.live.cloudflare.com/v1/turn/keys/"
                f"{cloudflare_key_id}/credentials/generate-ice-servers",
                headers={"Authorization": f"Bearer {cloudflare_api_token}"},
                json={"ttl": 7200},
                timeout=10,
            )
            response.raise_for_status()
            cloudflare_servers = response.json().get("iceServers", [])
            if cloudflare_servers:
                return {"iceServers": cloudflare_servers}
        except (requests.RequestException, ValueError) as exc:
            print(f"[WARNING] Could not load Cloudflare TURN servers: {exc}")

    turn_urls = [
        url.strip()
        for url in os.getenv("TURN_SERVER_URLS", "").split(",")
        if url.strip()
    ]
    turn_username = os.getenv("TURN_USERNAME", "").strip()
    turn_credential = os.getenv("TURN_CREDENTIAL", "").strip()

    if turn_urls and turn_username and turn_credential:
        ice_servers.append({
            "urls": turn_urls,
            "username": turn_username,
            "credential": turn_credential,
        })

    return {"iceServers": ice_servers}


# ════════════════════════════════════════════════════════════════════════════
#  Page config
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FRS – Face Recognition System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

RTC_CONFIGURATION = build_rtc_configuration()

# ════════════════════════════════════════════════════════════════════════════
#  CSS – premium dark theme
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ══ BACKGROUND ══ */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ══ GLOBAL TEXT — ensure all content is light on dark ══ */
    .main .block-container,
    .main .block-container p,
    .main .block-container span,
    .main .block-container div,
    .main .block-container label {
        color: #e2e8f0;
    }
    h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }

    /* ══ SIDEBAR ══ */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.10);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* ══ CARDS ══ */
    .frs-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
        color: #e2e8f0 !important;
        transition: box-shadow .2s ease;
    }
    .frs-card:hover { box-shadow: 0 8px 32px rgba(99,102,241,.35); }
    .frs-card b, .frs-card strong { color: #f8fafc !important; }
    .frs-card span { color: #cbd5e1 !important; }

    /* ══ HERO TITLE ══ */
    .frs-hero { text-align: center; padding: 24px 0 8px; }
    .frs-hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #818cf8, #c084fc, #fb7185);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .frs-hero p { color: #cbd5e1 !important; font-size: 1.05rem; }

    /* ══ STAT BADGE ══ */
    .stat-badge {
        display: inline-block;
        background: rgba(99,102,241,.25);
        border: 1px solid rgba(99,102,241,.5);
        border-radius: 999px;
        padding: 4px 14px;
        font-size: .85rem;
        color: #e0e7ff;
        margin: 2px 4px;
    }

    /* ══ SECTION HEADER ══ */
    .section-header {
        color: #a5b4fc !important;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(165,180,252,.4);
        padding-bottom: 6px;
    }

    /* ══ METRIC WIDGET — fix invisible labels & values ══ */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(165,180,252,0.2);
        border-radius: 14px;
        padding: 16px 20px !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] div {
        color: #a5b4fc !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] > div {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        opacity: 1 !important;
    }

    /* ══ FORM CONTAINER ══ */
    [data-testid="stForm"] {
        border: 1px solid rgba(165,180,252,0.2) !important;
        border-radius: 14px !important;
        padding: 20px !important;
        background: rgba(255,255,255,0.04) !important;
    }

    /* ══ INPUT LABELS ══ */
    .stTextInput label,
    .stTextInput label p,
    .stSelectbox label,
    .stSelectbox label p,
    .stTextInput > label,
    .stSelectbox > label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        opacity: 1 !important;
    }

    /* ══ TEXT INPUT BOX ══ */
    .stTextInput > div > div > input {
        background: rgba(15, 12, 41, 0.85) !important;
        border: 1.5px solid rgba(165,180,252,0.5) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8 !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 2px rgba(129,140,248,0.25) !important;
    }

    /* ══ SELECT BOX ══ */
    .stSelectbox > div > div,
    .stSelectbox [data-baseweb="select"] > div {
        background: rgba(15, 12, 41, 0.85) !important;
        border: 1.5px solid rgba(165,180,252,0.5) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div,
    .stSelectbox > div > div > div {
        color: #f1f5f9 !important;
    }

    /* ══ RADIO BUTTONS & SLIDERS ══ */
    .stRadio label,
    .stRadio label p,
    .stRadio label span,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] label span,
    .stSlider label,
    .stSlider label p,
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] label p {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    /* ══ BUTTONS ══ */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 700 !important;
        font-size: .95rem !important;
        transition: all .2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99,102,241,.5);
    }
    .stButton > button p,
    .stButton > button span { color: #ffffff !important; }

    /* ══ ALERT BOXES — force readable text ══ */
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] div,
    div[data-testid="stNotification"] p,
    div[data-testid="stNotification"] span {
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
    }
    .stInfo {
        background: rgba(99,102,241,0.22) !important;
        border-left: 4px solid #818cf8 !important;
        border-radius: 10px !important;
    }
    .stSuccess {
        background: rgba(16,185,129,0.22) !important;
        border-left: 4px solid #10b981 !important;
        border-radius: 10px !important;
    }
    .stWarning {
        background: rgba(251,191,36,0.22) !important;
        border-left: 4px solid #fbbf24 !important;
        border-radius: 10px !important;
    }
    .stError {
        background: rgba(239,68,68,0.22) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 10px !important;
    }
    .stInfo p, .stSuccess p, .stWarning p, .stError p,
    .stInfo span, .stSuccess span, .stWarning span, .stError span,
    .stInfo div, .stSuccess div, .stWarning div, .stError div {
        color: #f1f5f9 !important;
    }

    /* ══ TABLES / DATAFRAME ══ */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ══ PROGRESS BAR ══ */
    .stProgress > div > div > div {
        background: linear-gradient(90deg,#6366f1,#c084fc) !important;
    }
    [data-testid="stProgressBar"] ~ div,
    .stProgress + div p { color: #e2e8f0 !important; }

    /* ══ SPINNER ══ */
    .stSpinner p, .stSpinner > div { color: #a5b4fc !important; }

    /* ══ MARKDOWN TEXT ══ */
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span { color: #e2e8f0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
#  Hero header
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="frs-hero">
        <h1>🎓 Face Recognition System</h1>
        <p>AI-powered student attendance &amp; identity verification</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
#  Sidebar navigation
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🗂️ Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Dashboard", "📷 Register Student", "🧠 Train Model",
         "🔍 Recognize Face", "📋 Student List"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    students = load_students()
    trained  = os.path.exists(TRAINER_FILE)
    st.markdown(f'<span class="stat-badge">👤 {len(students)} students</span>',
                unsafe_allow_html=True)
    st.markdown(
        f'<span class="stat-badge">🧠 Model: {"✅ Ready" if trained else "❌ Not trained"}</span>',
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 – Dashboard
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    from config import DATASET_DIR

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="frs-card">', unsafe_allow_html=True)
        st.metric("👤 Registered Students", len(students))
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="frs-card">', unsafe_allow_html=True)
        st.metric("🧠 Model Status", "Trained ✅" if trained else "Not Trained ❌")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        total_imgs = 0
        if os.path.exists(DATASET_DIR):
            for d in os.listdir(DATASET_DIR):
                dpath = os.path.join(DATASET_DIR, d)
                if os.path.isdir(dpath):
                    total_imgs += len(os.listdir(dpath))
        st.markdown('<div class="frs-card">', unsafe_allow_html=True)
        st.metric("🖼️ Total Face Samples", total_imgs)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    for icon, title, desc in [
        ("1️⃣", "Register Student",
         "Go to **Register Student** → enter details → capture 30 face samples via your webcam."),
        ("2️⃣", "Train Model",
         "Go to **Train Model** → click Train. The LBPH model will be built from captured images."),
        ("3️⃣", "Recognize Faces",
         "Go to **Recognize Face** → your webcam shows live recognition with name & roll overlays."),
    ]:
        st.markdown(
            f'<div class="frs-card"><b>{icon} {title}</b><br>'
            f'<span style="color:#94a3b8">{desc}</span></div>',
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 – Register Student (Capture Faces)
# ════════════════════════════════════════════════════════════════════════════
elif page == "📷 Register Student":
    st.markdown('<p class="section-header">📷 Register New Student</p>',
                unsafe_allow_html=True)

    with st.form("register_form"):
        c1, c2, c3 = st.columns(3)
        roll   = c1.text_input("Roll Number", placeholder="e.g. 101")
        name   = c2.text_input("Full Name",   placeholder="e.g. Kanoj Kumar")
        gender = c3.selectbox("Gender", ["Male", "Female"])
        submitted = st.form_submit_button("▶️ Start Capture")

    if submitted:
        if not roll.strip() or not name.strip():
            st.error("Please fill in all fields.")
        else:
            st.session_state["cap_roll"]   = roll.strip()
            st.session_state["cap_name"]   = name.strip()
            st.session_state["cap_gender"] = gender
            st.session_state["cap_count"]  = 0
            st.session_state["cap_done"]   = False

    # ── WebRTC capture ──────────────────────────────────────────────────────
    if st.session_state.get("cap_roll") and not st.session_state.get("cap_done", True):
        cap_roll   = st.session_state["cap_roll"]
        cap_name   = st.session_state["cap_name"]
        cap_gender = st.session_state["cap_gender"]

        st.info(
            f"📸 Capturing for **{cap_name}** (Roll: {cap_roll}) — "
            f"target: **{SAMPLES_PER_PERSON}** samples. "
            "Stream will stop automatically when done."
        )
        prog_ph = st.empty()

        class CaptureProcessor(VideoProcessorBase):
            lock  = threading.Lock()
            count: int = 0
            done:  bool = False

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                with self.lock:
                    if self.done or self.count >= SAMPLES_PER_PERSON:
                        self.done = True
                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                    img, self.count = capture_from_frame(
                        img, cap_roll, cap_name, cap_gender, self.count
                    )
                    if self.count >= SAMPLES_PER_PERSON:
                        self.done = True
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(
            key="capture",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=CaptureProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            with ctx.video_processor.lock:
                count = ctx.video_processor.count
                done  = ctx.video_processor.done

            prog_ph.progress(
                min(count / SAMPLES_PER_PERSON, 1.0),
                text=f"Samples captured: {count}/{SAMPLES_PER_PERSON}",
            )

            if done and not st.session_state.get("cap_done"):
                msg = finalize_capture(cap_roll, cap_name, cap_gender, count)
                st.session_state["cap_done"] = True
                st.success(msg)
                st.balloons()

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 – Train Model
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Train Model":
    st.markdown('<p class="section-header">🧠 Train Face Recognition Model</p>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="frs-card">Train the <b>LBPH (Local Binary Patterns Histogram)</b> model '
        "on all captured face samples. This may take a few seconds.</div>",
        unsafe_allow_html=True,
    )

    if trained:
        st.info("ℹ️ A trained model already exists. Clicking Train will rebuild it using the latest dataset.")

    if st.button("🚀 Train Model Now", type="primary"):
        with st.spinner("Training in progress …"):
            result = train()
        if result.startswith("✅"):
            load_recognizer(force=True)
            st.success(result)
            st.balloons()
        else:
            st.error(result)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 4 – Recognize Faces
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Recognize Face":
    st.markdown('<p class="section-header">🔍 Live Face Recognition</p>',
                unsafe_allow_html=True)

    rec = load_recognizer()
    if not trained or rec is None:
        st.warning("⚠️ No trained model found. Please go to **Train Model** first.")
    else:
        st.info(
            "👇 Click **START**, allow camera access, and face the webcam. "
            "Real-time recognition overlays your name, roll, and confidence score."
        )

        ctrl_col1, ctrl_col2 = st.columns([3, 1])
        with ctrl_col1:
            conf_threshold = st.slider(
                "🎯 Recognition Threshold (LBPH Distance)",
                min_value=50,
                max_value=160,
                value=int(CONFIDENCE_THRESH),
                step=5,
                help=(
                    "In LBPH, lower score = higher similarity. "
                    "Typical webcam match is 75–110. "
                    "If recognized as Unknown, raise this threshold."
                ),
            )
        with ctrl_col2:
            st.write("")
            st.write("")
            if st.button("🔄 Reload Model"):
                load_recognizer(force=True)
                st.success("Model refreshed!")

        class RecognizeProcessor(VideoProcessorBase):
            current_thresh: float = conf_threshold

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                img, _ = recognize_frame(img, conf_thresh=self.current_thresh)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        RecognizeProcessor.current_thresh = conf_threshold

        webrtc_streamer(
            key="recognize",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=RecognizeProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 5 – Student List
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Student List":
    st.markdown('<p class="section-header">📋 Registered Students & Status</p>',
                unsafe_allow_html=True)

    students = load_students()

    if not students:
        st.info("No students registered yet. Go to **Register Student** to add one.")
    else:
        import pandas as pd

        # Check sample count for each student directory
        sample_counts = {}
        if os.path.exists(DATASET_DIR):
            for folder in os.listdir(DATASET_DIR):
                folder_path = os.path.join(DATASET_DIR, folder)
                if os.path.isdir(folder_path):
                    try:
                        roll_key = str(folder.split("_")[0])
                        sample_counts[roll_key] = len(os.listdir(folder_path))
                    except Exception:
                        pass

        rows = []
        for roll_str, info in students.items():
            count = sample_counts.get(roll_str, 0)
            is_ready = count >= 10
            status_label = "Ready ✅" if is_ready else "Needs Face Capture ⚠️"
            rows.append({
                "Roll": roll_str,
                "Name": info.get("name", "Unknown"),
                "Gender": info.get("gender", "-"),
                "Samples": f"{count} sample(s)" if count > 0 else "0 (No faces)",
                "Status": status_label,
            })

        df = pd.DataFrame(rows)

        # Proper numeric ordering by roll number
        try:
            df["_sort"] = df["Roll"].astype(int)
        except ValueError:
            df["_sort"] = df["Roll"]
        df = df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

        ready_count = sum(1 for r in rows if "Ready" in r["Status"])
        missing_count = len(rows) - ready_count

        # Summary KPIs
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("👤 Total Registered", len(students))
        kpi2.metric("✅ Ready for Recognition", ready_count)
        kpi3.metric("⚠️ Needs Capture", missing_count)

        # Filters and Search
        f_col1, f_col2 = st.columns([2, 2])
        with f_col1:
            search = st.text_input("🔍 Search Student", placeholder="Search by name or roll...")
        with f_col2:
            status_filter = st.radio(
                "Filter by Status",
                ["All", "Ready ✅", "Needs Face Capture ⚠️"],
                horizontal=True,
            )

        filtered_df = df.copy()
        if search.strip():
            query = search.strip().lower()
            mask = (
                filtered_df["Roll"].str.lower().contains(query) |
                filtered_df["Name"].str.lower().contains(query)
            )
            filtered_df = filtered_df[mask]

        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["Status"] == status_filter]

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        # Warning / actionable alert for incomplete students
        missing_names = [r["Name"] for r in rows if "Needs Face Capture" in r["Status"]]
        if missing_names:
            st.warning(
                f"⚠️ **{len(missing_names)} student(s) have no captured faces**: "
                f"{', '.join(missing_names)}.\n\n"
                "To recognize them, go to **📷 Register Student** to capture their webcam samples, "
                "then click **Train Model Now**."
            )

        st.markdown("---")
        # Student Management: Delete student
        with st.expander("⚙️ Manage Students (Delete Record)"):
            st.markdown("Select a student to delete from the database and remove their face dataset.")
            del_options = {f"{r['Roll']} - {r['Name']}": r["Roll"] for r in rows}
            selected_label = st.selectbox("Select Student to Remove", list(del_options.keys()))
            if st.button("🗑️ Delete Selected Student", type="secondary"):
                roll_to_delete = del_options[selected_label]
                if delete_student(roll_to_delete):
                    st.success(f"Student {selected_label} deleted successfully.")
                    st.rerun()
                else:
                    st.error("Failed to delete student.")
