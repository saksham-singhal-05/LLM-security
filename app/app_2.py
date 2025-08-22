import cv2
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import threading
import time
import yaml
import psycopg2
import base64
import requests
from requests.exceptions import Timeout
import os
from datetime import datetime
import json

CONFIG_PATH = "config.yaml"
CLIP_DURATION = 5
FPS = 2
NUM_FRAMES = 10
OLLAMA_ENDPOINT = "http://192.168.14.10:11434/api/generate"
MODEL_NAME = "gemma3:27b"

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def connect_db(cfg):
    return psycopg2.connect(
        host=cfg['postgres']['host'],
        port=cfg['postgres']['port'],
        dbname=cfg['postgres']['database'],
        user=cfg['postgres']['user'],
        password=cfg['postgres']['password']
    )

def extract_frames(video_path, num_frames):
    """Extracts 'num_frames' evenly spaced frames from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        raise ValueError("Not enough frames in the video.")
    step = total_frames // num_frames
    frames_b64 = []
    for i in range(num_frames):
        frame_index = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        frames_b64.append(img_b64)
    cap.release()
    return frames_b64

def send_to_ollama(images_b64, prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": images_b64,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get('response', '')
    except Timeout:
        raise Timeout("Connection timed out while contacting the LLM endpoint.")
    except Exception as e:
        raise Exception(f"Request failed: {str(e)}")

def store_event(conn, category, headcount, reasoning, video_path):
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO events (timestamp, category, headcount, reasoning, video_path)
               VALUES (%s, %s, %s, %s, %s)""",
            (datetime.now(), category, headcount, reasoning, video_path)
        )
    conn.commit()

def record_clip(out_path, camera_id=0, duration=5):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return False
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Unable to read frame from camera")
        cap.release()
        return False
    height, width = frame.shape[:2]
    size = (int(width), int(height))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, 10, size)
    start = time.time()
    while cap.isOpened() and time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    return True

# === GUI LOGIC ===
class SurveillanceApp:
    def __init__(self, master):
        self.master = master
        self.is_running = False
        master.title("Simple LLM Surveillance")
        self.start_btn = ttk.Button(master, text="Start Surveillance", command=self.toggle_surveillance)
        self.start_btn.pack(pady=10)
        self.status = tk.StringVar(value="Idle")
        self.status_lbl = ttk.Label(master, textvariable=self.status)
        self.status_lbl.pack(pady=5)
        self.last_frame_img = None
        self.panel = tk.Label(master)
        self.panel.pack()
        self.result_text = ScrolledText(master, width=60, height=10, wrap=tk.WORD)
        self.result_text.pack(pady=10)

        self.config = load_config()
        self.conn = connect_db(self.config)
        self.surv_thread = None

    def toggle_surveillance(self):
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(text="Stop Surveillance")
            self.status.set("Surveillance running...")
            self.surv_thread = threading.Thread(target=self.surveillance_loop)
            self.surv_thread.start()
        else:
            self.is_running = False
            self.start_btn.config(text="Start Surveillance")
            self.status.set("Stopped")

    def surveillance_loop(self):
        while self.is_running:
            try:
                self.status.set("Capturing video...")
                video_path = "captured_clip.mp4"
                success = record_clip(video_path)
                if not success:
                    self.status.set("Failed to capture video; retrying...")
                    time.sleep(2)
                    continue

                self.status.set("Extracting frames...")
                frames_b64 = extract_frames(video_path, NUM_FRAMES)

                self.status.set("Sending to LLM...")
                prompt = """Act as an advanced security LLM responsible for actively monitoring a classroom environment in real time. Your objective is to analyze input events and behaviors, assess their risk, and deliver a structured JSON report with an explicit headcount. Your outputs must be grounded in objective reasoning and clear evidence—you should not simply agree with any user input, but must independently assess all information.
Begin by analyzing all available class data, observations, or event logs.
Classify the situation using one of four categories, based on the observed risk level:
"normal"
"pre_alert"
"alert"
"human_intervention_needed"
Determine and return the headcount as an integer, representing the number of people present.
For each output, include a reasoning field explaining, step by step, why the particular category is assigned. If user input is incomplete or suspicious, flag this in your reasoning. Never simply agree—always evaluate evidence.
Return your findings in a structured JSON format:
json

"category": "",
"headcount": ,
"reasoning": ""

Apply clear criteria for each category (examples):
normal: All observed behaviors and values are within safe, expected bounds.
pre_alert: Small deviations or minor issues that may require monitoring but are not immediately dangerous.
alert: Significant anomaly/potential risk requiring prompt attention.
human_intervention_needed: Immediate action needed due to active threat, malfunction, or unresolved escalation.
For every input, reason independently. Challenge subjective descriptions and verify with the actual event/context. If ambiguity exists, state this in reasoning and default to the lower-risk category unless clear evidence suggests escalation.
Example Output:
json

"category": "pre_alert",
"headcount": 28,
"reasoning": "Detected mild argument between two students. No physical risk, but monitoring is advised. Room headcount verified against entry logs."

Task is complete when every entry is classified with a validated headcount and explicit reasoning, and no information is accepted at face value without substantiation."""

                response = send_to_ollama(frames_b64, prompt)

                # Show full JSON response (with pretty formatting) in Text box
                self.result_text.delete(1.0, tk.END)
                raw_json_str = ""
                try:
                    if isinstance(response, str):
                        resp_dict = json.loads(response)
                        raw_json_str = json.dumps(resp_dict, indent=2)
                    else:
                        resp_dict = response
                        raw_json_str = json.dumps(resp_dict, indent=2)
                    category = resp_dict.get("category", "unknown")
                    headcount = int(resp_dict.get("headcount", 0))
                    reasoning = resp_dict.get("reasoning", "")
                except Exception:
                    category, headcount, reasoning = "unknown", 0, "Invalid response format"
                    raw_json_str = str(response) if response else "Invalid response format"
                self.result_text.insert(tk.END, raw_json_str)

                # Store event to DB
                store_event(self.conn, category, headcount, reasoning, video_path)

                # Update status with summary info
                status_msg = f"Detected: {category} | Headcount: {headcount}"
                self.status.set(status_msg)

            except Timeout:
                self.status.set("Error: Connection timed out. Retrying shortly...")
            except Exception as e:
                self.status.set(f"Error: {str(e)}. Retrying...")
            time.sleep(5) # delay before next cycle

root = tk.Tk()
app = SurveillanceApp(root)
root.mainloop()
