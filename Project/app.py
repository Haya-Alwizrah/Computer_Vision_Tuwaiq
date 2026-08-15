from flask import Flask, render_template, jsonify
import sqlite3
from datetime import datetime
from ultralytics import YOLO
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import json
import os
import cv2
import threading
import time
import subprocess
import tempfile

app = Flask(__name__)

# ------------------------------------------------------------- CONFIGURATION -------------------------------------------
DATABASE = "database/emergency.db"
LANES = ["North", "East", "South", "West"]
VIDEO_PATHS = {
    "North": "static/videos/north.mp4",
    "East": "static/videos/east.mp4",
    "South": "static/videos/south.mp4",
    "West": "static/videos/west.mp4"
}

AMBULANCE_MODEL_PATH = "models/ambulance_model1.pt"
SIREN_MODEL_PATH = "models/siren_classifier.keras"
SIREN_CONFIG_PATH = "models/siren_model_config.json"

# ------------------------------------------------------------- LOAD MODELS----------------------------------------

print("Loading ambulance model...")
ambulance_model = YOLO(AMBULANCE_MODEL_PATH)
print("Ambulance model loaded.")

print("Loading YAMNet...")
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(YAMNET_URL)
print("YAMNet loaded.")

print("Loading siren classifier...")
siren_model = tf.keras.models.load_model(SIREN_MODEL_PATH)
print("Siren classifier loaded.")

with open(SIREN_CONFIG_PATH, "r") as f:
    siren_config = json.load(f)

SAMPLE_RATE = siren_config.get("sample_rate", 16000)
SIREN_THRESHOLD = siren_config.get("threshold", 0.5)

# SYSTEM STATE
signal_state = {
    "North": "RED",
    "East": "GREEN",
    "South": "RED",
    "West": "RED"
}

emergency_active = False        # Is emergency currently active?
priority_lane = None            # Which lane currently has priority?
active_event_id = None          # Current database event
simulation_running = False      # Is the whole simulation running?
state_lock = threading.Lock()   # Lock shared state between threads

# NORMAL TRAFFIC SETTINGS 
GREEN_DURATION = 7
YELLOW_DURATION = 2

signal_order = ["North", "East", "South", "West"]
normal_signal_index = 0
normal_signal_phase = "GREEN"
normal_phase_start = time.time()

# AMBULANCE SETTINGS
AMBULANCE_MISSING_LIMIT = 3
ambulance_missing_count = 0     # Number of consecutive frames without ambulance

# THREAD MANAGEMENT
signal_thread = None
video_threads = []

# -------------------------------------------------------------------------- DATABASE ----------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs("database", exist_ok=True)
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS emergency_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            approach TEXT NOT NULL,
            ambulance_confidence REAL,
            siren_confidence REAL,
            signal_opened TEXT,
            signal_closed TEXT,
            duration TEXT
        )
    """)
    conn.commit()
    conn.close()

# -------------------------------------------------------------------------- DASHBOARD --------------------------------------------------------------------------

@app.route("/")
def dashboard():
    return render_template("dashboard.html", street_name="King Abdulaziz Road")

# -------------------------------------------------------------------------- HISTORY --------------------------------------------------------------------------
@app.route("/history")
def history():
    conn = get_db()
    events = conn.execute("""
        SELECT *
        FROM emergency_history
        ORDER BY id DESC
    """).fetchall()
    conn.close()

    return render_template("history.html", events=events)

# -------------------------------------------------------------------------- CURRENT SYSTEM STATUS --------------------------------------------------------------------------
@app.route("/api/status")
def status():
    with state_lock:
        return jsonify({
            "signals": signal_state.copy(),
            "emergency_active": emergency_active,
            "priority_lane": priority_lane,
            "simulation_running": simulation_running
        })

# -------------------------------------------------------------------------- CURRENT TIME --------------------------------------------------------------------------
@app.route("/api/current-time")
def current_time():
    now = datetime.now()
    return jsonify({
        "date": now.strftime("%d/%m/%Y"),
        "time": now.strftime("%H:%M:%S")
    })

# -------------------------------------------------------------------------- START SIMULATION --------------------------------------------------------------------------
@app.route("/api/start", methods=["POST"])
def start_simulation():
    global simulation_running
    global normal_phase_start

    with state_lock:
        if simulation_running:
            return jsonify({
                "success": True,
                "running": True,
                "message": "Simulation is already running"
            })

        simulation_running = True
        normal_phase_start = time.time()    # Restart normal signal timing

    print("====================================")
    print("[SYSTEM] SIMULATION STARTED")
    print("====================================")

    start_video_monitors()
    return jsonify({
        "success": True,
        "running": True
    })

# -------------------------------------------------------------------------- STOP SIMULATION --------------------------------------------------------------------------
@app.route("/api/stop", methods=["POST"])
def stop_simulation():
    global simulation_running
    global ambulance_missing_count
    global normal_signal_index
    global normal_signal_phase
    global normal_phase_start

    with state_lock:
        was_emergency_active = emergency_active

    if was_emergency_active:     # If emergency is active, close it properly
        close_emergency()

    with state_lock:
        simulation_running = False

        normal_signal_index = 0
        normal_signal_phase = "GREEN"
        normal_phase_start = time.time()

        signal_state["North"] = "GREEN"
        signal_state["East"] = "RED"
        signal_state["South"] = "RED"
        signal_state["West"] = "RED"

    ambulance_missing_count = 0

    print("====================================")
    print("[SYSTEM] SIMULATION STOPPED")
    print("[SYSTEM] NORMAL TRAFFIC MODE")
    print("====================================")

    return jsonify({
        "success": True,
        "running": False
    })

# -------------------------------------------------------------------------- AUDIO FUNCTIONS --------------------------------------------------------------------------
def extract_audio_from_video(video_path, duration=3):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    command = [
        "ffmpeg",
        "-y",
        "-sseof",
        f"-{duration}",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-t",
        str(duration),
        temp_file.name
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        return temp_file.name

    except Exception as e:
        print(
            f"[AUDIO ERROR] "
            f"Could not extract audio: {e}"
        )

        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

        return None

# -------------------------------------------------------------------------- SIREN PREDICTION --------------------------------------------------------------------------
def predict_siren(audio_path):

    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)
        scores, embeddings, spectrogram = yamnet_model(audio)

        embeddings = embeddings.numpy()
        frame_probabilities = siren_model.predict(embeddings, verbose=0).flatten()
        probability = float(np.mean(frame_probabilities))

        if probability >= SIREN_THRESHOLD:
            label = "Siren"
            confidence = probability
        else:
            label = "No Siren"
            confidence = 1 - probability

        return {
            "label": label,
            "probability": probability,
            "confidence": confidence
        }

    except Exception as e:
        print(f"[SIREN ERROR] {e}")
        return {
            "label": "No Siren",
            "probability": 0.0,
            "confidence": 0.0
        }

    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

# -------------------------------------------------------------------------- YOLO AMBULANCE DETECTION --------------------------------------------------------------------------
def detect_ambulance(frame):
    results = ambulance_model.predict(source=frame, conf=0.25, verbose=False)
    highest_confidence = 0.0
    ambulance_detected = False

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = ambulance_model.names.get(class_id, str(class_id))

            print(
                f"[YOLO] "
                f"class={class_name} "
                f"id={class_id} "
                f"conf={confidence:.2%}"
            )

            if class_name.lower() == "ambulance":
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    ambulance_detected = True

    return (
        ambulance_detected,
        highest_confidence
    )

# -------------------------------------------------------------------------- NORMAL SIGNAL CONTROL --------------------------------------------------------------------------
def update_normal_signals():
    global normal_signal_index
    global normal_signal_phase
    global normal_phase_start

    now = time.time()
    elapsed = now - normal_phase_start
    current_lane = signal_order[normal_signal_index]

    with state_lock:
        if normal_signal_phase == "GREEN":
            for lane in LANES:
                signal_state[lane] = "RED"

            signal_state[current_lane] = "GREEN"
            if elapsed >= GREEN_DURATION:
                normal_signal_phase = "YELLOW"
                normal_phase_start = now

        elif normal_signal_phase == "YELLOW":
            for lane in LANES:
                signal_state[lane] = "RED"

            signal_state[current_lane] = "YELLOW"
            if elapsed >= YELLOW_DURATION:
                normal_signal_index = (normal_signal_index + 1) % len(signal_order)
                normal_signal_phase = "GREEN"
                normal_phase_start = now

# -------------------------------------------------------------------------- OPEN EMERGENCY SIGNAL --------------------------------------------------------------------------
def open_signal(approach):
    with state_lock:
        for lane in LANES:
            signal_state[lane] = "RED"
        signal_state[approach] = "GREEN"

    print(
        f"[EMERGENCY SIGNAL] "
        f"{approach} = GREEN"
    )

# -------------------------------------------------------------------------- CLOSE SIGNAL --------------------------------------------------------------------------
def close_signal(approach):
    with state_lock:
        for lane in LANES:
            signal_state[lane] = "RED"

    print(
        f"[EMERGENCY SIGNAL] "
        f"{approach} = CLOSED"
    )

# -------------------------------------------------------------------------- REGISTER EMERGENCY --------------------------------------------------------------------------
def register_emergency(approach, ambulance_confidence, siren_confidence):
    global emergency_active
    global priority_lane
    global active_event_id
    global ambulance_missing_count

    ambulance_missing_count = 0

    with state_lock:
        if emergency_active:
            return

        emergency_active = True
        priority_lane = approach

    now = datetime.now()
    date = now.strftime("%d/%m/%Y")
    start_time = now.strftime("%H:%M:%S")
    open_signal(approach)
    conn = get_db()
    cursor = conn.execute("""
        INSERT INTO emergency_history (
            date,
            start_time,
            approach,
            ambulance_confidence,
            siren_confidence,
            signal_opened
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        date,
        start_time,
        approach,
        ambulance_confidence,
        siren_confidence,
        approach
    ))

    active_event_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print("====================================")
    print("EMERGENCY DETECTED")
    print(
        f"Approach: {approach}"
    )
    print(
        f"Ambulance: "
        f"{ambulance_confidence:.2%}"
    )
    print(
        f"Siren: "
        f"{siren_confidence:.2%}"
    )
    print("====================================")

# -------------------------------------------------------------------------- CLOSE EMERGENCY --------------------------------------------------------------------------
def close_emergency():
    global emergency_active
    global priority_lane
    global active_event_id
    global ambulance_missing_count

    with state_lock:
        event_id = active_event_id
        approach = priority_lane

    if event_id is None:
        with state_lock:
            emergency_active = False
            priority_lane = None
        return

    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")
    conn = get_db()
    event = conn.execute("""
        SELECT start_time
        FROM emergency_history
        WHERE id = ?
    """, (event_id,)).fetchone()

    duration = None
    if event:
        try:
            start = datetime.strptime(event["start_time"], "%H:%M:%S")
            end = datetime.strptime(end_time, "%H:%M:%S")
            seconds = int((end - start).total_seconds())
            if seconds < 0:
                seconds += 86400

            duration = (
                f"{seconds // 60:02d}:"
                f"{seconds % 60:02d}"
            )

        except Exception as e:
            print(f"[DURATION ERROR] {e}")
            duration = None

    if approach:
        close_signal(approach)

    conn.execute("""
        UPDATE emergency_history
        SET
            end_time = ?,
            signal_closed = ?,
            duration = ?
        WHERE id = ?
    """, (
        end_time,
        approach,
        duration,
        event_id
    ))

    conn.commit()
    conn.close()

    with state_lock:
        emergency_active = False
        priority_lane = None
        active_event_id = None

    ambulance_missing_count = 0
    print(
        f"[EMERGENCY CLOSED] "
        f"{approach}"
    )

# -------------------------------------------------------------------------- PROCESS ONE VIDEO --------------------------------------------------------------------------
def process_video(approach, video_path):
    global ambulance_missing_count

    print("\n====================================")
    print(f"[{approach}]")
    print(f"Path: {os.path.abspath(video_path)}")
    print(f"Exists: {os.path.exists(video_path)}")
    print(f"[START] Monitoring {approach}")

    # Open video
    cap = cv2.VideoCapture(os.path.abspath(video_path))

    print(f"Opened: {cap.isOpened()}")

    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    if fps <= 0:
        fps = 25

    frame_interval = max(1, int(fps))
    frame_counter = 0

    while True:
        with state_lock:
            running = simulation_running

        if not running:
            cap.release()
            print(f"[STOP] Monitoring {approach}")
            return

        ret, frame = cap.read()
        if not ret:
            print(f"[END] {approach} video ended")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
            continue

        frame_counter += 1

        if (frame_counter % frame_interval != 0):
            continue

        # STEP 1: YOLO
        ambulance_detected, ambulance_confidence = detect_ambulance(frame)
        print(
            f"[{approach}] "
            f"Ambulance: {ambulance_detected} "
            f"Confidence: {ambulance_confidence:.2%}"
        )

        # Read current emergency state
        with state_lock:
            current_emergency = emergency_active
            current_priority = priority_lane
            running = simulation_running

        # If simulation stopped
        if not running:
            continue

        # Emergency is already active
        if current_emergency:
            # Only monitor the ambulance's lane
            if current_priority == approach:
                if ambulance_detected:
                    ambulance_missing_count = 0
                    print(f"[{approach}] Ambulance still present")
                else:
                    ambulance_missing_count += 1
                    print(
                        f"[{approach}] "
                        f"Ambulance missing "
                        f"("
                        f"{ambulance_missing_count}/"
                        f"{AMBULANCE_MISSING_LIMIT}"
                        f")"
                    )
                    if (ambulance_missing_count >= AMBULANCE_MISSING_LIMIT):
                        close_emergency()
                        ambulance_missing_count = 0
            continue

        # Ambulance detected
        if ambulance_detected:
            print(f"[{approach}] AMBULANCE DETECTED!")

            # STEP 2: CHECK SIREN
            print(
                f"[{approach}] "
                f"Checking siren..."
            )

            audio_path = extract_audio_from_video(video_path, duration=3)
            if audio_path is None:
                print(f"[{approach}] Could not extract audio.")
                print(f"[{approach}] No emergency priority.")
                continue

            siren_result = predict_siren(audio_path)
            print(
                f"[{approach}] "
                f"Siren: {siren_result['label']}"
                f"Siren confidence: {siren_result['confidence']:.2%}"
            )

            # Ambulance + Siren
            if siren_result["label"] == "Siren":
                print(
                    f"[{approach}] "
                    f"AMBULANCE + SIREN CONFIRMED"
                )

                register_emergency(approach, ambulance_confidence, siren_result["confidence"])

            # Ambulance WITHOUT Siren
            else:
                print(
                    f"[{approach}] "
                    f"Ambulance detected "
                    f"but NO SIREN."
                )

                print(
                    f"[{approach}] "
                    f"No emergency priority."
                )

# -------------------------------------------------------------------------- SIGNAL CONTROLLER --------------------------------------------------------------------------
def signal_controller():
    print(
        "[SIGNAL] "
        "Traffic controller thread started"
    )

    while True:
        with state_lock:
            emergency = emergency_active

        if not emergency:
            update_normal_signals()
        time.sleep(0.1)

# -------------------------------------------------------------------------- START VIDEO MONITORS --------------------------------------------------------------------------
def start_video_monitors():
    global signal_thread
    global video_threads

    # Start signal controller only once
    if (signal_thread is None or not signal_thread.is_alive()):
        signal_thread = threading.Thread(
            target=signal_controller,
            daemon=True
        )

        signal_thread.start()

    # Start video monitors
    video_threads = []

    for approach, video_path in VIDEO_PATHS.items():
        thread = threading.Thread(
            target=process_video,
            args=(approach, video_path),
            daemon=True
        )

        video_threads.append(thread)
        thread.start()

# -------------------------------------------------------------------------- RUN FLASK --------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()

    print("====================================")
    print("SYSTEM READY")
    print("Simulation is OFF.")
    print("Press START from the dashboard.")
    print("====================================")

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        use_reloader=False
    )