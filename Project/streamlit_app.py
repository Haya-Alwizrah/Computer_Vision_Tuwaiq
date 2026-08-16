from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from ultralytics import YOLO


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE = BASE_DIR / "database" / "emergency.db"
CSS_PATH = BASE_DIR / "static" / "css" / "style2.css"

LANES = ["North", "East", "South", "West",]
SIGNAL_ORDER = ["North", "East", "South", "West",]
VIDEO_PATHS = {
    "North": BASE_DIR / "static" / "videos" / "north.mp4",
    "East": BASE_DIR / "static" / "videos" / "east.mp4",
    "South": BASE_DIR / "static" / "videos" / "south.mp4",
    "West": BASE_DIR / "static" / "videos" / "west.mp4",
}

AMBULANCE_MODEL_PATH = BASE_DIR / "models" / "ambulance_model1.pt"
SIREN_MODEL_PATH = BASE_DIR / "models" / "siren_classifier.keras"
SIREN_CONFIG_PATH = BASE_DIR / "models" / "siren_model_config.json"

GREEN_DURATION = 7
YELLOW_DURATION = 2
AMBULANCE_MISSING_LIMIT = 3

# PAGE CONFIG

st.set_page_config(
    page_title="Emergency Traffic Control",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# LOAD EXTERNAL CSS
def load_css():
    if not CSS_PATH.exists():
        st.warning(f"CSS file not found: {CSS_PATH}")
        return
    
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        css = f.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )

load_css()

# DATABASE
def get_db():
    DATABASE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        str(DATABASE),
        check_same_thread=False,
        timeout=30,
    )
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS emergency_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            approach TEXT NOT NULL,
            ambulance_confidence REAL,
            siren TEXT,
            siren_confidence REAL,
            signal_closed TEXT,
            duration TEXT
        )
        """
    )
    conn.commit()
    conn.close()

# MODEL LOADING
@st.cache_resource(show_spinner=False)
def load_models():
    print("[MODEL] Loading ambulance model...")

    if not AMBULANCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Ambulance model not found: "
            f"{AMBULANCE_MODEL_PATH}"
        )

    ambulance_model = YOLO(str(AMBULANCE_MODEL_PATH))
    print("[MODEL] Ambulance model loaded.")

    print("[MODEL] Loading YAMNet...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("[MODEL] YAMNet loaded.")

    print("[MODEL] Loading siren classifier...")
    if not SIREN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Siren classifier not found: {SIREN_MODEL_PATH}")

    siren_model = tf.keras.models.load_model(str(SIREN_MODEL_PATH))
    print("[MODEL] Siren classifier loaded.")

    if SIREN_CONFIG_PATH.exists():
        with open(SIREN_CONFIG_PATH, "r", encoding="utf-8",) as f:
            siren_config = json.load(f)
    else:
        siren_config = {}

    sample_rate = siren_config.get("sample_rate", 16000)
    siren_threshold = siren_config.get("threshold", 0.5)

    return (
        ambulance_model,
        yamnet_model,
        siren_model,
        sample_rate,
        siren_threshold,
    )

# SYSTEM STATE
state_lock = threading.RLock()

signal_state = {
    "North": "GREEN",
    "East": "RED",
    "South": "RED",
    "West": "RED",
}

emergency_active = False
priority_lane = None
active_event_id = None
previous_green_signal = None

simulation_running = False
simulation_id = 0

normal_signal_index = 0
normal_signal_phase = "GREEN"
normal_phase_start = time.time()

ambulance_missing_count = {
    lane: 0
    for lane in LANES
}

video_threads = []

# STATE SNAPSHOT
def get_state_snapshot():
    with state_lock:
        return {
            "signals": signal_state.copy(),
            "emergency_active": emergency_active,
            "priority_lane": priority_lane,
            "simulation_running": simulation_running,
        }

# AUDIO
def extract_audio_from_video(video_path, sample_rate, duration=3,):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()

    command = [
        "ffmpeg",
        "-y",
        "-sseof",
        f"-{duration}",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-t",
        str(duration),
        temp_file.name,
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return temp_file.name

    except Exception as e:
        print(f"[AUDIO ERROR] {e}")
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except OSError:
                pass
        return None

def predict_siren(audio_path, yamnet_model, siren_model, sample_rate, siren_threshold,):
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        audio = audio.astype(np.float32)
        scores, embeddings, spectrogram = yamnet_model(audio)

        embeddings = embeddings.numpy()
        frame_probabilities = siren_model.predict(embeddings, verbose=0).flatten()
        probability = float(np.mean(frame_probabilities))

        if probability >= siren_threshold:
            label = "Siren"
            confidence = probability
        else:
            label = "No Siren"
            confidence = 1 - probability

        return {
            "label": label,
            "probability": probability,
            "confidence": confidence,
        }

    except Exception as e:
        print(f"[SIREN ERROR] {e}")
        return {
            "label": "No Siren",
            "probability": 0.0,
            "confidence": 0.0,
        }

    finally:
        if (audio_path and os.path.exists(audio_path)):
            try:
                os.remove(audio_path)
            except OSError:
                pass

# YOLO
def detect_ambulance(frame, ambulance_model,):
    results = ambulance_model.predict(source=frame, conf=0.50, verbose=False)
    highest_confidence = 0.0
    ambulance_detected = False

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = (
                ambulance_model.names.get(
                    class_id,
                    str(class_id),
                )
            )

            print(
                f"[YOLO] class={class_name} "
                f"id={class_id} "
                f"conf={confidence:.2%}"
            )

            if (class_name.lower() == "ambulance"):
                if (confidence > highest_confidence):
                    highest_confidence = confidence
                    ambulance_detected = True

    return ambulance_detected,  highest_confidence,

# NORMAL SIGNAL CONTROL
def update_normal_signals():
    global normal_signal_index
    global normal_signal_phase
    global normal_phase_start

    now = time.time()

    elapsed = now - normal_phase_start
    current_lane = SIGNAL_ORDER[normal_signal_index]

    with state_lock:
        if emergency_active:
            return

        if (normal_signal_phase == "GREEN"):
            for lane in LANES:
                signal_state[lane] = "RED"

            signal_state[current_lane] = "GREEN"

            if elapsed >= GREEN_DURATION:
                normal_signal_phase = "YELLOW"
                normal_phase_start = now

        elif (normal_signal_phase == "YELLOW"):
            for lane in LANES:
                signal_state[lane] = "RED"

            signal_state[current_lane] = "YELLOW"

            if elapsed >= YELLOW_DURATION:
                normal_signal_index = (normal_signal_index + 1) % len(SIGNAL_ORDER)
                normal_signal_phase = "GREEN"
                normal_phase_start = now

# EMERGENCY SIGNAL
def open_signal(approach,):
    global previous_green_signal

    with state_lock:
        previous_green_signal = None
        for lane in LANES:
            if (signal_state[lane] == "GREEN"):
                previous_green_signal = lane
                break

        for lane in LANES:
            signal_state[lane] = "RED"

        signal_state[approach] = "GREEN"

    print(
        f"[EMERGENCY SIGNAL] "
        f"{approach}=GREEN "
        f"Previous GREEN="
        f"{previous_green_signal}"
    )

    return previous_green_signal

def close_signal(approach,):
    with state_lock:
        for lane in LANES:
            signal_state[lane] = "RED"

    print(
        f"[EMERGENCY SIGNAL] "
        f"{approach}=CLOSED"
    )

# VALIDATION
def simulation_is_valid(thread_simulation_id):
    with state_lock:
        return (simulation_running and simulation_id == thread_simulation_id)

# REGISTER DETECTION
def register_detection(approach, ambulance_confidence, siren_label, siren_confidence, thread_simulation_id,):
    global emergency_active
    global priority_lane
    global active_event_id

    if not simulation_is_valid(thread_simulation_id):
        return

    now = datetime.now()
    date = now.strftime("%d/%m/%Y")
    start_time = now.strftime("%H:%M:%S")

    # Ambulance + Siren
    if siren_label == "Siren":
        with state_lock:
            if (not simulation_running or simulation_id != thread_simulation_id):
                return

            if emergency_active:
                return

            emergency_active = True
            priority_lane = approach

        closed_signal = open_signal(approach)

        conn = get_db()
        cursor = conn.execute(
            """
            INSERT INTO emergency_history (
                date,
                start_time,
                approach,
                ambulance_confidence,
                siren,
                siren_confidence,
                signal_closed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date,
                start_time,
                approach,
                ambulance_confidence,
                "Siren",
                siren_confidence,
                closed_signal,
            ),
        )

        active_event_id = cursor.lastrowid
        conn.commit()
        conn.close()

        print("====================================")
        print("EMERGENCY DETECTED")
        print(f"Approach: {approach}")
        print(f"Ambulance: {ambulance_confidence:.2%}")
        print(f"Siren: {siren_confidence:.2%}")
        print(f"Signal closed: {closed_signal}")
        print("====================================")
        return

    # Ambulance without siren
    if not simulation_is_valid(thread_simulation_id):
        return

    conn = get_db()
    conn.execute(
        """
        INSERT INTO emergency_history (
            date,
            start_time,
            approach,
            ambulance_confidence,
            siren,
            siren_confidence,
            signal_closed,
            duration
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            date,
            start_time,
            approach,
            ambulance_confidence,
            "No Siren",
            siren_confidence,
            None,
            None,
        ),
    )

    conn.commit()
    conn.close()

    print(
        f"[{approach}] Ambulance "
        f"detected without siren. "
        f"Traffic unchanged."
    )

# CLOSE EMERGENCY
def close_emergency():
    global emergency_active
    global priority_lane
    global active_event_id
    global previous_green_signal

    with state_lock:
        event_id = active_event_id
        approach = priority_lane

    if event_id is None:
        with state_lock:
            emergency_active = False
            priority_lane = None
            previous_green_signal = None
        return

    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")

    conn = get_db()
    event = conn.execute(
        """
        SELECT start_time
        FROM emergency_history
        WHERE id = ?
        """,
        (event_id,),
    ).fetchone()

    duration = None
    if event:
        try:
            start = datetime.strptime(event["start_time"], "%H:%M:%S")
            end = datetime.strptime( end_time, "%H:%M:%S")
            seconds = int((end - start).total_seconds())
            if seconds < 0:
                seconds += 86400

            duration = (f"{seconds // 60:02d}:{seconds % 60:02d}")

        except Exception as e:
            print(f"[DURATION ERROR] {e}")

    if approach:
        close_signal(approach)

    conn.execute(
        """
        UPDATE emergency_history
        SET end_time = ?,
            duration = ?
        WHERE id = ?
        """,
        (
            end_time,
            duration,
            event_id,
        ),
    )

    conn.commit()
    conn.close()

    with state_lock:
        emergency_active = False
        priority_lane = None
        active_event_id = None
        previous_green_signal = None

        for lane in LANES:
            ambulance_missing_count[lane] = 0

    print(
        f"[EMERGENCY CLOSED] "
        f"{approach}"
    )

    print(
        f"[EMERGENCY DURATION] "
        f"{duration}"
    )

# VIDEO PROCESSING
def process_video(
    approach,
    video_path,
    thread_simulation_id,
    ambulance_model,
    yamnet_model,
    siren_model,
    sample_rate,
    siren_threshold,
):

    print(
        f"[{approach}] Monitoring "
        f"{video_path}"
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"[ERROR] Cannot open "
            f"{video_path}"
        )

        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    frame_interval = max(1, int(fps))
    frame_counter = 0

    while True:
        if not simulation_is_valid(thread_simulation_id):
            cap.release()
            return

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
            continue

        frame_counter += 1

        # Analyze approximately
        # one frame per second
        if (frame_counter % frame_interval != 0):
            continue

        # YOLO
        ambulance_detected, ambulance_confidence = detect_ambulance(frame, ambulance_model)

        if not simulation_is_valid(thread_simulation_id):
            cap.release()
            return

        print(
            f"[{approach}] "
            f"Ambulance="
            f"{ambulance_detected} "
            f"Confidence="
            f"{ambulance_confidence:.2%}"
        )

        with state_lock:
            current_emergency = emergency_active
            current_priority = priority_lane

        # EMERGENCY ALREADY ACTIVE

        if current_emergency:
            if current_priority == approach:
                if ambulance_detected:
                    ambulance_missing_count[approach] = 0
                else:
                    ambulance_missing_count[approach] += 1
                    if (ambulance_missing_count[approach] >= AMBULANCE_MISSING_LIMIT):
                        if simulation_is_valid(thread_simulation_id):
                            close_emergency()

                        ambulance_missing_count[approach] = 0
            continue

        # AMBULANCE DETECTED
        if ambulance_detected:
            if not simulation_is_valid(thread_simulation_id):
                cap.release()
                return

            print(
                f"[{approach}] "
                f"Checking siren..."
            )

            audio_path = (extract_audio_from_video(video_path, sample_rate, duration=3))

            if not simulation_is_valid(thread_simulation_id):
                if (audio_path and os.path.exists(audio_path)):
                    try:
                        os.remove(audio_path)
                    except OSError:
                        pass

                cap.release()
                return

            if audio_path is None:
                register_detection(approach, ambulance_confidence, "No Siren", 0.0, thread_simulation_id)
                continue

            siren_result = predict_siren(audio_path, yamnet_model, siren_model, sample_rate, siren_threshold)

            if not simulation_is_valid(thread_simulation_id):
                cap.release()
                return

            register_detection(approach, ambulance_confidence, siren_result["label"], siren_result["confidence"], thread_simulation_id)


# SIGNAL CONTROLLER
def signal_controller():
    print("[SIGNAL] Controller started.")

    while True:
        with state_lock:
            emergency = emergency_active
            running = simulation_running

        if running and not emergency:
            update_normal_signals()

        time.sleep(0.1)

# START SIMULATION
def start_simulation(ambulance_model, yamnet_model, siren_model, sample_rate, siren_threshold,):
    global simulation_running
    global normal_signal_index
    global normal_signal_phase
    global normal_phase_start
    global simulation_id
    global emergency_active
    global priority_lane
    global active_event_id
    global previous_green_signal
    global video_threads

    with state_lock:
        if simulation_running:
            return False

        simulation_id += 1
        current_id = simulation_id
        simulation_running = True
        normal_signal_index = 0
        normal_signal_phase = "GREEN"
        normal_phase_start = time.time()
        emergency_active = False
        priority_lane = None
        active_event_id = None
        previous_green_signal = None

        for lane in LANES:
            signal_state[lane] = "RED"
            ambulance_missing_count[lane] = 0

        signal_state["North"] = "GREEN"

    video_threads = []

    for approach, video_path in (VIDEO_PATHS.items()):
        if not video_path.exists():
            print(
                f"[WARNING] Missing video: "
                f"{video_path}"
            )
            continue

        thread = threading.Thread(
            target=process_video,
            args=(
                approach,
                video_path,
                current_id,
                ambulance_model,
                yamnet_model,
                siren_model,
                sample_rate,
                siren_threshold,
            ),
            daemon=True,
            name=(f"video-{approach.lower()}"),
        )

        video_threads.append(thread)
        thread.start()

    return True

# STOP SIMULATION
def stop_simulation():
    global simulation_running
    global simulation_id
    global emergency_active
    global priority_lane
    global active_event_id
    global previous_green_signal
    global normal_signal_index
    global normal_signal_phase
    global normal_phase_start

    with state_lock:

        simulation_running = False
        simulation_id += 1
        emergency_active = False
        priority_lane = None
        active_event_id = None
        previous_green_signal = None
        normal_signal_index = 0
        normal_signal_phase = "GREEN"
        normal_phase_start = time.time()

        for lane in LANES:
            ambulance_missing_count[lane] = 0
            signal_state[lane] = "RED"

        signal_state["North"] = "GREEN"

# HISTORY
def get_history(filter_type="all",):
    conn = get_db()

    if filter_type == "emergency":
        rows = conn.execute(
            """
            SELECT *
            FROM emergency_history
            WHERE siren = 'Siren'
            ORDER BY id DESC
            """
        ).fetchall()

    elif (filter_type== "without_siren"):
        rows = conn.execute(
            """
            SELECT *
            FROM emergency_history
            WHERE siren = 'No Siren'
            ORDER BY id DESC
            """
        ).fetchall()

    else:
        rows = conn.execute(
            """
            SELECT *
            FROM emergency_history
            ORDER BY id DESC
            """
        ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


# ------------------------------------------ interface ----------------------------------
# HEADER
def render_header():
    st.markdown(
        """
        <div class="app-header">
            <div class="app-header-inner">
                <div class="brand">
                    <div class="logo-box">ETC</div>
                    <div>
                        <div class="app-title">Emergency Traffic Control</div>
                        <div class="app-subtitle">Intelligent Emergency Vehicle Priority System</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# NAVIGATION
def render_navigation():
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"

    col1, col2, col3 = st.columns([1, 1, 7])

    with col1:
        if st.button("Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    with col2:
        if st.button("Emergency History", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()

# INFO BAR
def render_info_bar():
    now = datetime.now()
    snapshot = get_state_snapshot()
    status = (
        "RUNNING"
        if snapshot["simulation_running"]
        else "OFF"
    )

    st.markdown(
        f"""
        <div class="info-bar">
            <div class="info-item">
                <div class="info-label">Street Name</div>
                <div class="info-value">King Abdulaziz Road</div>
            </div>
            <div class="info-item">
                <div class="info-label">Date</div>
                <div class="info-value">{now.strftime("%d/%m/%Y")}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Time</div>
                <div class="info-value">{now.strftime("%H:%M:%S")}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return status

# SIMULATION CONTROLS
def render_simulation_controls(ambulance_model, yamnet_model, siren_model, sample_rate, siren_threshold,):
    snapshot = get_state_snapshot()
    col1, col2, col3 = st.columns([1, 1, 6])

    with col1:
        if st.button(
            "START SIMULATION",
            type = "primary",
            use_container_width = True,
            disabled = snapshot["simulation_running"],
        ):
            started = start_simulation(
                ambulance_model,
                yamnet_model,
                siren_model,
                sample_rate,
                siren_threshold,
            )

            if started:
                st.rerun()

    with col2:
        if st.button(
            "STOP SIMULATION",
            use_container_width = True,
            disabled = not snapshot["simulation_running"],
        ):

            stop_simulation()
            st.rerun()

# CAMERA CARD
def render_camera(number, lane, video_path,):
    st.markdown(
        f"""
        <div class="camera-card">
            <div class="camera-header">
                <span class="camera-label">CAMERA {number:02d}</span>
                <span class="camera-live">LIVE</span>
                <div class="camera-name">{lane} Approach</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if video_path.exists():
        st.video(
            str(video_path),
            format="video/mp4",
            start_time=0,
            autoplay=True,
            muted=True,
            loop=True,
        )

    else:
        st.error(f"Video not found: {video_path}")

# TRAFFIC SIGNAL
def render_single_signal(lane, state,):
    red_active = "active" if state == "RED" else ""
    yellow_active = "active" if state == "YELLOW" else ""
    green_active = "active" if state == "GREEN" else ""

    state_class = state.lower()

    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-name">{lane}</div>
            <div class="traffic-light">
                <div class="light red {red_active}"></div>
                <div class=" light yellow {yellow_active}"></div>
                <div class=" light green {green_active}"></div>
            </div>
            <div class="signal-status signal-{state_class}">{state}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# LIVE STATUS FRAGMENT
@st.fragment(run_every=1)
def live_status():
    snapshot = get_state_snapshot()

    # Emergency / System Banner
    if snapshot["emergency_active"]:
        lane = (snapshot["priority_lane"] or "Unknown")
        st.markdown(
            f"""
            <div class=" system-banner emergency-banner">
                EMERGENCY ACTIVE — PRIORITY: {lane.upper()}
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif snapshot["simulation_running"]:
        st.markdown(
            """
            <div class=" system-banner normal-banner">
                AI SYSTEM ACTIVE — NORMAL TRAFFIC MODE
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div class=" system-banner ready-banner">
                SYSTEM READY — SIMULATION OFF
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Traffic Signals
    st.markdown(
        """<div class="section-title">TRAFFIC SIGNAL STATUS</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="signal-container">""",
        unsafe_allow_html=True,
    )

    signal_cols = st.columns(4)
    for index, lane in enumerate(LANES):
        with signal_cols[index]:
            render_single_signal(
                lane,
                snapshot["signals"][lane],
            )

    st.markdown(
        """</div>""",
        unsafe_allow_html=True,
    )

# DASHBOARD
def render_dashboard(
    ambulance_model,
    yamnet_model,
    siren_model,
    sample_rate,
    siren_threshold,
):

    render_info_bar()

    render_simulation_controls(
        ambulance_model,
        yamnet_model,
        siren_model,
        sample_rate,
        siren_threshold,
    )

    st.markdown(
        "<div class='dashboard-gap'></div>",
        unsafe_allow_html=True,
    )

    # Cameras
    st.markdown(
        """<div class="section-title">TRAFFIC CAMERAS</div>""",
        unsafe_allow_html=True,
    )

    row1_col1, row1_col2 = st.columns(2, gap="small")

    with row1_col1:
        render_camera(1, "North", VIDEO_PATHS["North"])

    with row1_col2:
        render_camera(2, "East", VIDEO_PATHS["East"])

    row2_col1, row2_col2 = st.columns(2, gap="small")

    with row2_col1:
        render_camera(3, "South", VIDEO_PATHS["South"])

    with row2_col2:
        render_camera(4, "West", VIDEO_PATHS["West"])

    st.markdown(
        "<div class='dashboard-gap'></div>",
        unsafe_allow_html=True,
    )

    # Live status
    live_status()


# HISTORY PAGE
def render_history():
    st.markdown(
        """<div class="section-title">EMERGENCY HISTORY</div>""",
        unsafe_allow_html=True,
    )

    filter_options = {
        "All": "all",
        "Emergency / Siren": "emergency",
        "Ambulance without Siren": "without_siren",
    }

    selected_filter = st.selectbox(
        "Filter",
        list(filter_options.keys()),
    )

    rows = get_history(filter_options[selected_filter])

    if rows:
        display_rows = []

        for row in rows:
            display_rows.append(
                {
                    "Date": row["date"],
                    "Start Time": row["start_time"],
                    "End Time": row["end_time"] or "-",
                    "Approach": row["approach"],
                    "Ambulance":
                        (
                            f"{row['ambulance_confidence']:.1%}"
                            if row["ambulance_confidence"] is not None
                            else "-"
                        ),
                    "Siren": row["siren"] or "-",
                    "Siren Confidence":
                        (
                            f"{row['siren_confidence']:.1%}"
                            if row["siren_confidence"] is not None
                            else "-"
                        ),
                    "Signal Closed": row["signal_closed"] or "-",
                    "Duration": row["duration"] or "-",
                }
            )

        st.dataframe(display_rows, use_container_width=True, hide_index=True,)

    else:
        st.info("No history records found.")

# INITIALIZATION
init_db()

# LOAD MODELS
try:
    ambulance_model, yamnet_model, siren_model, SAMPLE_RATE, SIREN_THRESHOLD = load_models()

except Exception as e:
    st.error("AI models could not be loaded.")
    st.code(str(e))
    st.info(
        "Make sure the model files exist "
        "and FFmpeg is installed."
    )
    st.stop()

# START SIGNAL CONTROLLER
if ("signal_controller_started" not in st.session_state):
    st.session_state["signal_controller_started"] = False

if not st.session_state["signal_controller_started"]:
    signal_thread = threading.Thread(
        target=signal_controller,
        daemon=True,
        name="signal-controller",
    )

    signal_thread.start()
    st.session_state["signal_controller_started"] = True

# HEADER
render_header()

# NAVIGATION
render_navigation()

# PAGE
if (st.session_state.page == "dashboard"):
    render_dashboard(
        ambulance_model,
        yamnet_model,
        siren_model,
        SAMPLE_RATE,
        SIREN_THRESHOLD,
    )
else:
    render_history()