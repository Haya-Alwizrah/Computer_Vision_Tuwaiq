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
# ------------------------------- CONFIGURATION --------------------------------------------
DATABASE = "database/emergency.db"
LANES = ["North", "East", "South", "West"]
VIDEO_PATHS = {
    "North": "videos/north.mp4",
    "East": "videos/east.mp4",
    "South": "videos/south.mp4",
    "West": "videos/west.mp4"
}
AMBULANCE_MODEL_PATH = "models/ambulance_model.pt"
SIREN_MODEL_PATH = "models/siren_classifier.keras"
SIREN_CONFIG_PATH = "models/siren_model_config.json"

# -------------------------- LOAD MODELS -----------------------------------------------------

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

# --------------------- SYSTEM STATE ------------------------------------------
signal_state = {
    "North": "RED",
    "East": "GREEN",
    "South": "RED",
    "West": "RED"
}

emergency_active = False
priority_lane = None
active_event_id = None
state_lock = threading.Lock()

# ---------------------------------------------- DATABASE ----------------------------------------
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

# ------------------------------------ DASHBOARD -----------------------------------------------------
@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        street_name="King Abdulaziz Road"
    )

# -------------------------------------------- HISTORY ------------------------------------------------
@app.route("/history")
def history():
    conn = get_db()
    events = conn.execute("""
        SELECT *
        FROM emergency_history
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return render_template(
        "history.html",
        events=events
    )

# -------------------------------------- CURRENT SIGNAL STATUS -----------------------------------------
@app.route("/api/status")
def status():
    with state_lock:
        return jsonify({
            "signals": signal_state.copy(),
            "emergency_active": emergency_active,
            "priority_lane": priority_lane
        })

# ------------------------------- CURRENT TIME ---------------------------------------
@app.route("/api/current-time")
def current_time():
    now = datetime.now()
    return jsonify({
        "date": now.strftime("%d/%m/%Y"),
        "time": now.strftime("%H:%M:%S")
    })

# ----------------------------------- AUDIO FUNCTIONS ------------------------------------
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

    except Exception:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        return None


def predict_siren(audio_path):
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

# ---------------------------------------------------------- YOLO - AMBULANCE DETECTION----------------------------------------------
def detect_ambulance(frame):
    results = ambulance_model.predict(source=frame, conf=0.50, verbose=False)
    highest_confidence = 0.0
    ambulance_detected = False

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > highest_confidence:
                highest_confidence = confidence
                ambulance_detected = True

    return ambulance_detected, highest_confidence

# ------------------------------------------------------- SIGNAL CONTROL -------------------------------------------
def open_signal(approach):
    global signal_state

    with state_lock:
        for lane in LANES:
            signal_state[lane] = "RED"
        signal_state[approach] = "GREEN"

    print(f"[SIGNAL] {approach} OPEN")

def close_signal(approach):
    global signal_state

    with state_lock:
        signal_state = {
            "North": "RED",
            "East": "GREEN",
            "South": "RED",
            "West": "RED"
        }

    print(f"[SIGNAL] {approach} CLOSED")

# --------------------------------------------------- REGISTER EMERGENCY-------------------------------------
def register_emergency(approach, ambulance_confidence, siren_confidence):
    global emergency_active
    global priority_lane
    global active_event_id

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

    with state_lock:
        emergency_active = True
        priority_lane = approach

    print("====================================")
    print("EMERGENCY DETECTED")
    print(f"Approach: {approach}")
    print(f"Ambulance: {ambulance_confidence:.2%}")
    print(f"Siren: {siren_confidence:.2%}")
    print("====================================")

# CLOSE EMERGENCY
def close_emergency():
    global emergency_active
    global priority_lane
    global active_event_id

    if active_event_id is None:
        return

    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")
    approach = priority_lane

    conn = get_db()
    event = conn.execute("""
        SELECT start_time
        FROM emergency_history
        WHERE id = ?

    """, (active_event_id,)).fetchone()
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

        except Exception:
            duration = None

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
        active_event_id
    ))
    conn.commit()
    conn.close()

    with state_lock:
        emergency_active = False
        priority_lane = None
        active_event_id = None

    print(f"[EMERGENCY CLOSED] {approach}")

# --------------------------------------------------------------------------------- PROCESS ONE VIDEO -----------------------------------------------
def process_video(approach, video_path):
    print(f"[START] Monitoring {approach}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    frame_interval = int(fps)
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[END] {approach} video ended")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_counter += 1

        if frame_counter % frame_interval != 0:
            continue

        # STEP 1: YOLO
        ambulance_detected, ambulance_confidence = detect_ambulance(frame)
        print(
            f"[{approach}] "
            f"Ambulance: "
            f"{ambulance_detected} "
            f"({ambulance_confidence:.2%})"
        )

        if not ambulance_detected:
            continue

        # STEP 2: AMBULANCE DETECTED ->> NOW check audio.
        print(f"[{approach}] Ambulance detected!")
        print(f"[{approach}] Checking siren...")

        audio_path = extract_audio_from_video(video_path, duration=3)
        if audio_path is None:
            print(
                f"[{approach}] "
                "Could not extract audio."
            )
            continue

        try:
            siren_result = predict_siren(audio_path)

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        print(
            f"[{approach}] "
            f"Siren: "
            f"{siren_result['label']} "
            f"({siren_result['confidence']:.2%})"
        )

        # STEP 3: FINAL DECISION
        if (siren_result["label"] == "Siren"):
            with state_lock:
                already_active = emergency_active

            if not already_active:
                register_emergency(approach, ambulance_confidence, siren_result["confidence"])
        else:
            print(
                f"[{approach}] "
                "Ambulance detected but "
                "NO SIREN."
            )

# -------------------------------- START ALL VIDEO MONITORS------------------------------------------
def start_video_monitors():
    for approach, video_path in VIDEO_PATHS.items():
        thread = threading.Thread(
            target=process_video,
            args=(approach, video_path),
            daemon=True
        )
        thread.start()

# ---------------------------------- RUN FLASK----------------------------------------
if __name__ == "__main__":
    init_db()
    start_video_monitors()

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        use_reloader=False
    )