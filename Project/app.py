from flask import Flask, render_template, jsonify, request
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
import queue

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
simulation_id = 0               # Used to invalidate old threads/tasks

# NORMAL TRAFFIC SETTINGS 
GREEN_DURATION = 8
YELLOW_DURATION = 2

signal_order = ["North", "East", "South", "West"]
normal_signal_index = 0
normal_signal_phase = "GREEN"
normal_phase_start = time.time()

# AMBULANCE SETTINGS
AMBULANCE_MISSING_LIMIT = 3
ambulance_missing_count = {
    "North": 0,
    "East": 0,
    "South": 0,
    "West": 0
}

# THREAD MANAGEMENT
signal_thread = None
video_threads = []

# AUDIO WORKER
audio_queue = queue.Queue(maxsize=10)

audio_check_pending = {
    "North": False,
    "East": False,
    "South": False,
    "West": False
}
audio_cooldown = {
    "North": 0,
    "East": 0,
    "South": 0,
    "West": 0
}
AUDIO_COOLDOWN_SECONDS = 5

# -------------------------------------------------------------------------- DATABASE ----------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
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
            siren TEXT,
            siren_confidence REAL,
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
    filter_type = request.args.get("filter", "all")

    conn = get_db()
    if filter_type == "emergency":
        events = conn.execute("""
            SELECT *
            FROM emergency_history
            WHERE siren = 'Siren'
            ORDER BY id DESC
        """).fetchall()

    elif filter_type == "without_siren":
        events = conn.execute("""
            SELECT *
            FROM emergency_history
            WHERE siren = 'No Siren'
            ORDER BY id DESC
        """).fetchall()

    else:
        events = conn.execute("""
            SELECT *
            FROM emergency_history
            ORDER BY id DESC
        """).fetchall()

    conn.close()

    return render_template("history.html", events=events, current_filter=filter_type)

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
    global normal_signal_index
    global normal_signal_phase
    global normal_phase_start
    global simulation_id
    global emergency_active
    global priority_lane
    global active_event_id

    with state_lock:
        if simulation_running:
            return jsonify({
                "success": True,
                "running": True,
                "message": "Simulation is already running"
            })

        simulation_id += 1
        current_simulation_id = simulation_id
        simulation_running = True

        normal_signal_index = 0
        normal_signal_phase = "GREEN"
        normal_phase_start = time.time()    # Restart normal signal timing

        emergency_active = False
        priority_lane = None
        active_event_id = None

        # Reset signals
        signal_state["North"] = "RED"
        signal_state["East"] = "GREEN"
        signal_state["South"] = "RED"
        signal_state["West"] = "RED"

        # Reset ambulance counters
        for lane in LANES:
            ambulance_missing_count[lane] = 0
            audio_check_pending[lane] = False
            audio_cooldown[lane] = 0

    print("====================================")
    print("[SYSTEM] AI SIMULATION STARTED")
    print(f"[SYSTEM] Simulation ID: {current_simulation_id}")
    print("[SYSTEM] Ambulance detection ACTIVE")
    print("[SYSTEM] Siren detection ACTIVE")
    print("====================================")

    start_video_monitors(current_simulation_id)

    return jsonify({
        "success": True,
        "running": True
    })

# -------------------------------------------------------------------------- STOP SIMULATION --------------------------------------------------------------------------
@app.route("/api/stop", methods=["POST"])
def stop_simulation():
    global simulation_running
    global normal_signal_index
    global normal_signal_phase
    global normal_phase_start
    global simulation_id
    global emergency_active
    global priority_lane
    global active_event_id

    with state_lock:
        simulation_running = False
        simulation_id += 1

        # Reset emergency state
        emergency_active = False
        priority_lane = None
        active_event_id = None

        # Reset normal traffic
        normal_signal_index = 0
        normal_signal_phase = "GREEN"
        normal_phase_start = time.time()

        # Reset traffic lights
        signal_state["North"] = "RED"
        signal_state["East"] = "GREEN"
        signal_state["South"] = "RED"
        signal_state["West"] = "RED"

        # Reset ambulance counters
        for lane in LANES:
            ambulance_missing_count[lane] = 0
            audio_check_pending[lane] = False
            audio_cooldown[lane] = 0

    print("====================================")
    print("[SYSTEM] SIMULATION STOPPED")
    print("[SYSTEM] Old detection threads invalidated")
    print("[SYSTEM] NORMAL TRAFFIC MODE")
    print("====================================")

    return jsonify({
        "success": True,
        "running": False,
        "emergency_active": False,
        "priority_lane": None,
        "signals": signal_state.copy()
    })


def simulation_is_valid(thread_simulation_id):
    with state_lock:
        return (simulation_running and simulation_id == thread_simulation_id)
    
# -------------------------------------------------------------------------- AUDIO FUNCTIONS --------------------------------------------------------------------------
def extract_audio_from_video(video_path, timestamp, duration=3):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    start_time = max(0, timestamp - duration)

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
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

        print(
            f"[AUDIO] Extracted "
            f"{start_time:.2f}s -> {timestamp:.2f}s"
        )
        return temp_file.name

    except Exception as e:
        print(f"[AUDIO ERROR] Could not extract audio: {e}")

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

def audio_worker():
    print("[AUDIO] Audio worker started")

    while True:
        task = audio_queue.get()

        if task is None:
            audio_queue.task_done()
            break

        approach = task["approach"]
        video_path = task["video_path"]
        timestamp = task["timestamp"]
        ambulance_confidence = task["ambulance_confidence"]
        thread_simulation_id = task["simulation_id"]
        detected_green_signal = task["detected_green_signal"]

        try:
            print(f"[AUDIO] Processing {approach} at {timestamp:.2f}s")
            print(
                f"[AUDIO] Signal GREEN "
                f"when ambulance detected: {detected_green_signal}"
            )

            # Check simulation
            if not simulation_is_valid(thread_simulation_id):
                print(f"[AUDIO] Ignored {approach} - simulation stopped")
                continue

            audio_path = extract_audio_from_video(
                video_path=video_path,
                timestamp=timestamp,
                duration=3
            )

            if audio_path is None:
                print(f"[{approach}] Audio extraction failed")

                register_detection(
                    approach,
                    ambulance_confidence,
                    "No Siren",
                    0.0,
                    thread_simulation_id,
                    detected_green_signal
                )
                continue

            # Check simulation again
            if not simulation_is_valid(thread_simulation_id):
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                continue

            # Siren model
            siren_result = predict_siren(audio_path)

            print(
                f"[{approach}] "
                f"Siren: {siren_result['label']} "
                f"Confidence: {siren_result['confidence']:.2%}"
            )

            if not simulation_is_valid(thread_simulation_id):
                continue

            register_detection(
                approach,
                ambulance_confidence,
                siren_result["label"],
                siren_result["confidence"],
                thread_simulation_id,
                detected_green_signal
            )

            audio_cooldown[approach] = time.time() + AUDIO_COOLDOWN_SECONDS

        except Exception as e:
            print(
                f"[AUDIO WORKER ERROR] "
                f"{approach}: {e}"
            )

        finally:
            audio_check_pending[approach] = False
            audio_queue.task_done()

# -------------------------------------------------------------------------- YOLO AMBULANCE DETECTION --------------------------------------------------------------------------
def detect_ambulance(frame):
    results = ambulance_model.predict(source=frame, conf=0.50, verbose=False)
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

    return ambulance_detected, highest_confidence

# -------------------------------------------------------------------------- GET CURRENT GREEN SIGNAL --------------------------------------------------------------------------
def get_current_green_signal():
    with state_lock:
        for lane in LANES:
            if signal_state[lane] == "GREEN":
                return lane
    return None

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

# -------------------------------------------------------------------------- OPEN / CLOSE EMERGENCY SIGNAL --------------------------------------------------------------------------
def open_signal(approach):
    with state_lock:
        for lane in LANES:
            signal_state[lane] = "RED"

        signal_state[approach] = "GREEN"

    print(
        f"[EMERGENCY SIGNAL] "
        f"{approach} = GREEN"
    )

def close_signal(approach):
    with state_lock:
        for lane in LANES:
            signal_state[lane] = "RED"

    print(
        f"[EMERGENCY SIGNAL] "
        f"{approach} = CLOSED"
    )

# -------------------------------------------------------------------------- REGISTER EMERGENCY --------------------------------------------------------------------------
def register_detection(approach, ambulance_confidence, siren_label, siren_confidence, thread_simulation_id, detected_green_signal):
    global emergency_active
    global priority_lane
    global active_event_id

    if not simulation_is_valid(thread_simulation_id):
        print(
            f"[{approach}] "
            f"Detection ignored - simulation stopped."
        )
        return
    
    now = datetime.now()
    date = now.strftime("%d/%m/%Y")
    start_time = now.strftime("%H:%M:%S")

    # CASE 1: AMBULANCE + SIREN
    if siren_label == "Siren":
        with state_lock:
            # Check again
            if (not simulation_running or simulation_id != thread_simulation_id):
                print(
                    f"[{approach}] "
                    f"Emergency ignored - simulation stopped."
                )
                return

            if emergency_active:
                print(f"[{approach}] Emergency already active.")
                return

            emergency_active = True
            priority_lane = approach

        open_signal(approach)

        conn = get_db()
        cursor = conn.execute("""
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
        """, (
            date,
            start_time,
            approach,
            ambulance_confidence,
            "Siren",
            siren_confidence,
            detected_green_signal
        ))

        active_event_id = cursor.lastrowid
        conn.commit()
        conn.close()

        print("====================================")
        print("EMERGENCY DETECTED")
        print(f"Approach: {approach}")
        print(f"Ambulance: {ambulance_confidence:.2%}")
        print(f"Siren: {siren_confidence:.2%}")
        print(f"Signal closed: {detected_green_signal}")
        print("====================================")
        return

    # CASE 2: AMBULANCE WITHOUT SIREN
    if not simulation_is_valid(thread_simulation_id):
        return

    conn = get_db()
    conn.execute("""
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
    """, (
        date,
        start_time,
        approach,
        ambulance_confidence,
        "No Siren",
        siren_confidence,
        None,
        None
    ))

    conn.commit()
    conn.close()

    print("====================================")
    print("AMBULANCE DETECTED - NO EMERGENCY")
    print(f"Approach: {approach}")
    print(f"Ambulance: {ambulance_confidence:.2%}")
    print(f"Siren: {siren_confidence:.2%}")
    print("Traffic signal unchanged.")
    print("====================================")

# -------------------------------------------------------------------------- CLOSE EMERGENCY --------------------------------------------------------------------------
def close_emergency():
    global emergency_active
    global priority_lane
    global active_event_id

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

            duration = (f"{seconds // 60:02d}:{seconds % 60:02d}")

        except Exception as e:
            print(f"[DURATION ERROR] {e}")
            duration = None

    if approach:
        close_signal(approach)

    conn.execute("""
        UPDATE emergency_history
        SET
            end_time = ?,
            duration = ?
        WHERE id = ?
    """, (
        end_time,
        duration,
        event_id
    ))

    conn.commit()
    conn.close()

    with state_lock:
        emergency_active = False
        priority_lane = None
        active_event_id = None

    for lane in LANES:
        ambulance_missing_count[lane] = 0

    print(f"[EMERGENCY CLOSED] {approach}")
    print(f"[EMERGENCY DURATION] {duration}")

# -------------------------------------------------------------------------- PROCESS ONE VIDEO --------------------------------------------------------------------------
def process_video(approach, video_path, thread_simulation_id):
    print("\n====================================")
    print(f"[{approach}]")
    print(f"Path: {os.path.abspath(video_path)}")
    print(f"Exists: {os.path.exists(video_path)}")
    print(f"[START] Monitoring {approach}")
    print(f"Simulation ID: {thread_simulation_id}")

    cap = cv2.VideoCapture(os.path.abspath(video_path))
    print(f"Opened: {cap.isOpened()}")

    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    print(f"FPS: {fps}")

    # YOLO runs approximately 5 times per second
    YOLO_FPS = 5
    frame_interval = max(1, int(fps / YOLO_FPS))
    frame_counter = 0
    ambulance_was_present = False       # Used to know when ambulance disappears

    while True:
        # Check simulation
        if not simulation_is_valid(thread_simulation_id):
            cap.release()
            print(f"[STOP] {approach} thread invalidated")
            return

        # Read next frame
        ret, frame = cap.read()
        if not ret:
            print(f"[END] {approach} video ended - restarting")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0

            ambulance_was_present = False       # Reset detection state when video loops
            audio_check_pending[approach] = False
            audio_cooldown[approach] = 0
            continue

        frame_counter += 1

        # Skip frames to control YOLO FPS
        if frame_counter % frame_interval != 0:
            continue

        # Current video timestamp
        video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # STEP 1: YOLO
        ambulance_detected, ambulance_confidence = detect_ambulance(frame)

        if not simulation_is_valid(thread_simulation_id):
            cap.release()
            print(f"[STOP] {approach} detection cancelled")
            return

        print(
            f"[{approach}] "
            f"Time: {video_timestamp:.2f}s | "
            f"Ambulance: {ambulance_detected} | "
            f"Confidence: {ambulance_confidence:.2%}"
        )

        # Read emergency state
        with state_lock:
            current_emergency = emergency_active
            current_priority = priority_lane

        # CASE 1: EMERGENCY IS ALREADY ACTIVE
        if current_emergency:

            # Only monitor the ambulance that caused the current emergency
            if current_priority == approach:
                if ambulance_detected:
                    ambulance_missing_count[approach] = 0
                    print(f"[{approach}] Ambulance still present")

                else:
                    ambulance_missing_count[approach] += 1
                    print(
                        f"[{approach}] "
                        f"Ambulance missing "
                        f"({ambulance_missing_count[approach]}/{AMBULANCE_MISSING_LIMIT})"
                    )

                    if (ambulance_missing_count[approach] >= AMBULANCE_MISSING_LIMIT):
                        if simulation_is_valid(thread_simulation_id):
                            close_emergency()

                        ambulance_missing_count[approach] = 0
            continue

        # CASE 2: NO AMBULANCE
        if not ambulance_detected:
            # Remember that ambulance disappeared
            if ambulance_was_present:
                print(f"[{approach}] Ambulance disappeared")

            ambulance_was_present = False

            # Allow next ambulance to trigger audio
            audio_check_pending[approach] = False
            audio_cooldown[approach] = 0

            continue

        # CASE 3: AMBULANCE DETECTED
        ambulance_was_present = True
        print(f"[{approach}] AMBULANCE DETECTED!")

        detected_green_signal = get_current_green_signal()
        print(
            f"[{approach}] Signal GREEN before "
            f"emergency: {detected_green_signal}"
        )

        # Check if audio analysis is already running
        if audio_check_pending[approach]:
            print(f"[{approach}] Audio check already pending")
            continue

        # Check cooldown
        if time.time() < audio_cooldown[approach]:
            print(f"[{approach}] Audio cooldown active")
            continue

        # Check simulation
        if not simulation_is_valid(thread_simulation_id):
            cap.release()
            return

        # Send audio task to worker
        audio_task = {
            "approach": approach,
            "video_path": video_path,
            "timestamp": video_timestamp,
            "ambulance_confidence": ambulance_confidence,
            "simulation_id": thread_simulation_id,
            "detected_green_signal": detected_green_signal
        }

        try:
            audio_queue.put_nowait(audio_task)
            audio_check_pending[approach] = True
            print(f"[{approach}] Audio task queued at {video_timestamp:.2f}s")

        except queue.Full:
            print(f"[{approach}] Audio queue is full")

# -------------------------------------------------------------------------- SIGNAL CONTROLLER --------------------------------------------------------------------------
def signal_controller():
    print("[SIGNAL] Traffic controller thread started")

    while True:
        with state_lock:
            emergency = emergency_active

        if not emergency:
            update_normal_signals()
        time.sleep(0.1)

# -------------------------------------------------------------------------- START VIDEO MONITORS --------------------------------------------------------------------------
def start_video_monitors(current_simulation_id):
    global video_threads
    video_threads = []

    for approach, video_path in VIDEO_PATHS.items():
        thread = threading.Thread(
            target=process_video,
            args=(approach, video_path, current_simulation_id),
            daemon=True
        )

        video_threads.append(thread)
        thread.start()

# -------------------------------------------------------------------------- RUN FLASK --------------------------------------------------------------------------
init_db()

# SIGNAL CONTROLLER
signal_thread = threading.Thread(
    target=signal_controller,
    daemon=True
)
signal_thread.start()

# AUDIO WORKER
audio_thread = threading.Thread(
    target=audio_worker,
    daemon=True
)

audio_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("====================================")
    print("SYSTEM READY")
    print("Traffic lights are running normally.")
    print("AI simulation is OFF.")
    print("Press START to activate AI.")
    print("====================================")

    app.run(host="0.0.0.0", port=port)