import cv2
import time
import base64
import sqlite3
import json
import requests
import os

# Config
USE_REMOTE_API = True  # Set False if you run Ollama locally
LOCAL_MODEL = "qwen2.5vl:3b"
REMOTE_API_URL = "http://192.168.14.10:11434/api/generate"
REMOTE_MODEL = "qwen2.5vl:72b"
VIDEO_FILENAME = "surveillance_clip.mp4"
CLIP_DURATION_SECONDS = 10
FRAME_RATE = 20.0
CONTEXT_LIMIT = 15  # Number of recent entries to feed
ALERT_KEYWORDS = ["violence", "fight", "shouting", "altercation"]

# Database setup
DB_FILE = "context_storage.db"
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    analysis TEXT
)""")
conn.commit()

def save_context_to_db(analysis):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    cur.execute("INSERT INTO context (timestamp, analysis) VALUES (?, ?)", (timestamp, analysis))
    conn.commit()

def get_recent_contexts(limit=CONTEXT_LIMIT):
    cur.execute("SELECT analysis FROM context ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [row[0] for row in reversed(rows)]  # Oldest first

def record_video_clip(filename=VIDEO_FILENAME, duration=CLIP_DURATION_SECONDS, fps=FRAME_RATE):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Recording... Press q to stop early', frame)
        if time.time() - start_time > duration:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return True

def get_keyframes_from_video(filename=VIDEO_FILENAME, interval_sec=2):
    cap = cv2.VideoCapture(filename)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval_sec)
    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def frame_to_base64(frame):
    _, buf = cv2.imencode('.jpg', frame)
    jpg_bytes = buf.tobytes()
    b64_bytes = base64.b64encode(jpg_bytes)
    return b64_bytes.decode('utf-8')

def compose_prompt(contexts):
    if contexts:
        base_prompt = (
            "You are analyzing a short video clip from classroom surveillance for "
            "signs of violence or abnormal behavior. Based on the following context from previous clips and current clip images, "
            "describe what is happening. Mention anything suspicious or violent.\n\n"
        )
        base_prompt += "Previous context:\n"
        for i, ctx in enumerate(contexts):
            base_prompt += f"{i + 1}. {ctx}\n"
        base_prompt += "\nCurrent clip imagery to be analyzed.\n"
    else:
        base_prompt = (
            "You are analyzing a short video clip from classroom surveillance for "
            "signs of violence or abnormal behavior. "
            "There is no previous context. Analyze the current clip imagery alone and describe what is happening.\n\n"
        )
    return base_prompt

def alert_if_necessary(analysis):
    lower = analysis.lower()
    for kw in ALERT_KEYWORDS:
        if kw in lower:
            print("[ALERT] Physical altercation detected!")
            # Here you can add sending email/SMS or log alert
            return True
    return False

def analyze_clip_locally(prompt, images_b64):
    import ollama
    messages = [
        {"role": "system", "content": "You are a helpful assistant for video security analysis."},
        {"role": "user", "content": prompt, "images": images_b64}
    ]
    response = ollama.chat(
        model=LOCAL_MODEL,
        messages=messages
    )
    return response.get('message', {}).get('content', 'No response from model.')

def analyze_clip_remote_api(prompt, images_b64):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": REMOTE_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for video security analysis."},
            {"role": "user", "content": prompt, "images": images_b64}
        ]
    }
    resp = requests.post(REMOTE_API_URL, json=data, headers=headers)
    resp.raise_for_status()
    result = resp.json()
    return result.get("message", {}).get("content", "No response from remote model.")

def main():
    print("Recording 10-second video clip...")
    if not record_video_clip():
        print("Failed to record video. Exiting.")
        return
    print("Extracting keyframes...")
    frames = get_keyframes_from_video()
    images_b64 = [frame_to_base64(f) for f in frames if f is not None]
    contexts = get_recent_contexts()
    prompt = compose_prompt(contexts)
    print("Analyzing clip with Ollama model...")
    if USE_REMOTE_API:
        analysis = analyze_clip_remote_api(prompt, images_b64)
    else:
        analysis = analyze_clip_locally(prompt, images_b64)
    print("Analysis result:")
    print(analysis)
    saved = alert_if_necessary(analysis)
    save_context_to_db(analysis)
    print(f"Video saved as {VIDEO_FILENAME}. Context saved to DB.")
    if saved:
        print("Alert action taken!")

if __name__ == "__main__":
    main()