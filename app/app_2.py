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
import re

CONFIG_PATH = "config.yaml"
CLIP_DURATION = 5
FPS = 2
NUM_FRAMES = 10
OLLAMA_ENDPOINT = "http://192.168.14.10:11434/api/generate"
MODEL_NAME = "qwen2.5vl:72b"

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
        resp = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=None)
        resp.raise_for_status()
        return resp.json()
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
        master.title("LLM Surveillance")
        
        self.start_btn = ttk.Button(master, text="Start Surveillance", command=self.toggle_surveillance)
        self.start_btn.pack(pady=10)
        
        self.status = tk.StringVar(value="Idle")
        self.status_lbl = ttk.Label(master, textvariable=self.status)
        self.status_lbl.pack(pady=5)
        
        self.last_frame_img = None
        self.panel = tk.Label(master)
        self.panel.pack()
        
        # Add ScrolledText widget for displaying full JSON response
        self.result_text = ScrolledText(master, width=80, height=15, wrap=tk.WORD)
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
                prompt = """<|im_start|>system
                You are an advanced security AI monitoring a classroom environment in real-time. Your task is to analyze visual input and provide structured security assessments. You must be objective, evidence-based, and never accept information at face value without verification.

                Classification Categories:
                - "normal": Standard classroom activities, no concerns
                - "pre_alert": Minor issues requiring monitoring but not immediately dangerous  
                - "alert": Significant anomalies requiring prompt attention
                - "human_intervention_needed": Critical situations requiring immediate action

                Response Format: Always respond with valid JSON only, no additional text or markdown formatting.
                <|im_end|>
                <|im_start|>user
                Analyze the provided classroom images and assess the security situation. Count all visible people and classify the risk level based on observed behaviors and conditions.

                Provide your assessment in this exact JSON format:
                {
                "category": "",
                "headcount": 0,
                "reasoning": ""
                }

                Evidence Requirements:
                - Base headcount on actual visible people only
                - Identify specific behaviors or conditions that support your classification
                - If uncertain, default to lower risk category and explain why
                - Never invent details not visible in the images
                <|im_end|>
                <|im_start|>assistant"""

                response = send_to_ollama(frames_b64, prompt)

                # Parse response exactly like video_test.py
                self.result_text.delete(1.0, tk.END)
                raw_response = response.get("response", response)
                
                # Display the full response in the text widget
                self.result_text.insert(tk.END, f"Full Response:\n{raw_response}\n\n")
                
                try:
                    # Extract JSON from the response (handling ```
                    if isinstance(raw_response, str):
                        # Look for JSON block within ```json ```
                        json_match = re.search(r'```json\s*(\{.*?\})\s*```',raw_response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            resp_dict = json.loads(json_str)
                        else:
                            # Try to parse the entire response as JSON
                            resp_dict = json.loads(raw_response)
                    else:
                        resp_dict = raw_response
                    
                    # Display parsed JSON
                    formatted_json = json.dumps(resp_dict, indent=2)
                    self.result_text.insert(tk.END, f"Parsed JSON:\n{formatted_json}")
                    
                    category = resp_dict.get("category", "unknown")
                    headcount = int(resp_dict.get("headcount", 0))
                    reasoning = resp_dict.get("reasoning", "")
                    
                except Exception as parse_error:
                    category, headcount, reasoning = "unknown", 0, f"Parse error: {str(parse_error)}"
                    self.result_text.insert(tk.END, f"Parse Error: {str(parse_error)}")

                # Store event to DB
                store_event(self.conn, category, headcount, reasoning, video_path)

                # Update status with summary info
                status_msg = f"Detected: {category} | Headcount: {headcount}"
                self.status.set(status_msg)

            except Timeout:
                self.status.set("Error: Connection timed out. Retrying shortly...")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Error: Connection timed out")
            except Exception as e:
                self.status.set(f"Error: {str(e)}. Retrying...")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Error: {str(e)}")
            
            time.sleep(0.5) 
root = tk.Tk()
app = SurveillanceApp(root)
root.mainloop()
