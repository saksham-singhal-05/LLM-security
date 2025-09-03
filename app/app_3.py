import cv2
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
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
import subprocess
import sys

CONFIG_PATH = "config.yaml"
CLIP_DURATION = 5
FPS = 2
NUM_FRAMES = 10
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_NAME = "security_manager:latest"

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

# === ENHANCED GUI LOGIC ===

class SurveillanceControlApp:
    def __init__(self, master):
        self.master = master
        self.is_running = False
        self.surveillance_process = None
        
        master.title("🏫 LLM Surveillance Control Center")
        master.geometry("1000x700")
        master.configure(bg='#f0f0f0')
        
        # Create main frame
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        
        self.setup_header()
        self.setup_control_panel()
        self.setup_status_panel()
        self.setup_live_feed()
        self.setup_logs()
        
        # Initialize variables
        self.config = None
        self.conn = None
        self.surv_thread = None
        
        # Load configuration
        self.load_configuration()
    
    def setup_header(self):
        """Setup header section"""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        title_label = ttk.Label(
            header_frame, 
            text="🏫 LLM Classroom Surveillance System", 
            font=("Arial", 18, "bold")
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            header_frame, 
            text="Real-time AI-powered classroom monitoring and analytics",
            font=("Arial", 10)
        )
        subtitle_label.pack()
    
    def setup_control_panel(self):
        """Setup control buttons panel"""
        control_frame = ttk.LabelFrame(self.main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Surveillance controls
        self.start_btn = ttk.Button(
            control_frame, 
            text="🚀 Start Surveillance", 
            command=self.toggle_surveillance,
            style="Accent.TButton"
        )
        self.start_btn.pack(fill=tk.X, pady=5)
        
        # Dashboard button
        self.dashboard_btn = ttk.Button(
            control_frame, 
            text="📊 Open Analytics Dashboard", 
            command=self.open_dashboard
        )
        self.dashboard_btn.pack(fill=tk.X, pady=5)
        
        # Settings button
        self.settings_btn = ttk.Button(
            control_frame, 
            text="⚙️ Settings", 
            command=self.open_settings
        )
        self.settings_btn.pack(fill=tk.X, pady=5)
        
        # Test connection button
        self.test_btn = ttk.Button(
            control_frame, 
            text="🔍 Test Connections", 
            command=self.test_connections
        )
        self.test_btn.pack(fill=tk.X, pady=5)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Configuration status
        config_label = ttk.Label(control_frame, text="Configuration:", font=("Arial", 9, "bold"))
        config_label.pack(anchor=tk.W)
        
        self.config_status = tk.StringVar(value="Loading...")
        config_status_label = ttk.Label(control_frame, textvariable=self.config_status, font=("Arial", 8))
        config_status_label.pack(anchor=tk.W)
        
        # Database status
        db_label = ttk.Label(control_frame, text="Database:", font=("Arial", 9, "bold"))
        db_label.pack(anchor=tk.W, pady=(10, 0))
        
        self.db_status = tk.StringVar(value="Not connected")
        db_status_label = ttk.Label(control_frame, textvariable=self.db_status, font=("Arial", 8))
        db_status_label.pack(anchor=tk.W)
        
        # LLM status
        llm_label = ttk.Label(control_frame, text="LLM Service:", font=("Arial", 9, "bold"))
        llm_label.pack(anchor=tk.W, pady=(10, 0))
        
        self.llm_status = tk.StringVar(value="Not tested")
        llm_status_label = ttk.Label(control_frame, textvariable=self.llm_status, font=("Arial", 8))
        llm_status_label.pack(anchor=tk.W)
    
    def setup_status_panel(self):
        """Setup status and metrics panel"""
        status_frame = ttk.LabelFrame(self.main_frame, text="System Status", padding="10")
        status_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        
        # Current status
        self.status = tk.StringVar(value="System Idle")
        status_label = ttk.Label(status_frame, textvariable=self.status, font=("Arial", 12, "bold"))
        status_label.pack(pady=(0, 10))
        
        # Metrics frame
        metrics_frame = ttk.Frame(status_frame)
        metrics_frame.pack(fill=tk.X)
        
        # Events processed
        events_label = ttk.Label(metrics_frame, text="Events Processed:", font=("Arial", 9))
        events_label.grid(row=0, column=0, sticky=tk.W)
        
        self.events_count = tk.StringVar(value="0")
        events_value = ttk.Label(metrics_frame, textvariable=self.events_count, font=("Arial", 9, "bold"))
        events_value.grid(row=0, column=1, sticky=tk.E)
        
        # Last detection
        last_label = ttk.Label(metrics_frame, text="Last Detection:", font=("Arial", 9))
        last_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        self.last_detection = tk.StringVar(value="None")
        last_value = ttk.Label(metrics_frame, textvariable=self.last_detection, font=("Arial", 9))
        last_value.grid(row=1, column=1, sticky=tk.E, pady=(5, 0))
        
        # Configure grid weights
        metrics_frame.columnconfigure(1, weight=1)
    
    def setup_live_feed(self):
        """Setup live camera feed display"""
        feed_frame = ttk.LabelFrame(self.main_frame, text="Live Feed", padding="10")
        feed_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Camera display
        self.camera_label = ttk.Label(feed_frame, text="Camera feed will appear here", anchor=tk.CENTER)
        self.camera_label.pack(expand=True)
        
        # Feed controls
        feed_controls = ttk.Frame(feed_frame)
        feed_controls.pack(fill=tk.X, pady=(10, 0))
        
        self.camera_id = tk.StringVar(value="0")
        camera_label = ttk.Label(feed_controls, text="Camera ID:")
        camera_label.pack(side=tk.LEFT)
        
        camera_entry = ttk.Entry(feed_controls, textvariable=self.camera_id, width=5)
        camera_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        test_camera_btn = ttk.Button(feed_controls, text="Test Camera", command=self.test_camera)
        test_camera_btn.pack(side=tk.LEFT)
    
    def setup_logs(self):
        """Setup logs and response display"""
        logs_frame = ttk.LabelFrame(self.main_frame, text="System Logs & LLM Responses", padding="10")
        logs_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(logs_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # System logs tab
        logs_tab = ttk.Frame(notebook)
        notebook.add(logs_tab, text="System Logs")
        
        self.logs_text = ScrolledText(logs_tab, height=8, wrap=tk.WORD)
        self.logs_text.pack(fill=tk.BOTH, expand=True)
        
        # LLM responses tab
        response_tab = ttk.Frame(notebook)
        notebook.add(response_tab, text="LLM Responses")
        
        self.response_text = ScrolledText(response_tab, height=8, wrap=tk.WORD)
        self.response_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for expansion
        self.main_frame.rowconfigure(3, weight=1)
    
    def load_configuration(self):
        """Load system configuration"""
        try:
            self.config = load_config()
            self.config_status.set("✅ Loaded successfully")
            self.log_message("Configuration loaded successfully")
            
            # Try to connect to database
            try:
                self.conn = connect_db(self.config)
                self.db_status.set("✅ Connected")
                self.log_message("Database connected successfully")
            except Exception as e:
                self.db_status.set(f"❌ Error: {str(e)[:30]}...")
                self.log_message(f"Database connection failed: {e}")
                
        except Exception as e:
            self.config_status.set(f"❌ Error: {str(e)[:30]}...")
            self.log_message(f"Configuration load failed: {e}")
    
    def log_message(self, message):
        """Add message to logs"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.logs_text.insert(tk.END, log_entry)
        self.logs_text.see(tk.END)
    
    def toggle_surveillance(self):
        """Toggle surveillance on/off"""
        if not self.is_running:
            if not self.config or not self.conn:
                messagebox.showerror("Error", "Please ensure configuration and database connection are working")
                return
            
            self.is_running = True
            self.start_btn.config(text="⏹️ Stop Surveillance")
            self.status.set("🔴 Surveillance Active")
            self.log_message("Surveillance started")
            
            self.surv_thread = threading.Thread(target=self.surveillance_loop)
            self.surv_thread.daemon = True
            self.surv_thread.start()
        else:
            self.is_running = False
            self.start_btn.config(text="🚀 Start Surveillance")
            self.status.set("⚫ System Idle")
            self.log_message("Surveillance stopped")
    
    def surveillance_loop(self):
        """Main surveillance loop"""
        events_processed = 0
        
        while self.is_running:
            try:
                self.status.set("📹 Capturing video...")
                video_path = "captured_clip.mp4"
                success = record_clip(video_path, int(self.camera_id.get()))
                
                if not success:
                    self.status.set("❌ Failed to capture video")
                    self.log_message("Video capture failed, retrying...")
                    time.sleep(2)
                    continue
                
                self.status.set("🖼️ Extracting frames...")
                frames_b64 = extract_frames(video_path, NUM_FRAMES)
                
                self.status.set("🤖 Analyzing with LLM...")
                prompt = """<|im_start|>system
You are an advanced security AI monitoring a classroom environment in real-time. Your task is to analyze visual input and provide structured security assessments.

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
<|im_end|>

<|im_start|>assistant"""
                
                response = send_to_ollama(frames_b64, prompt)
                
                # Display response
                raw_response = response.get("response", response)
                self.response_text.delete(1.0, tk.END)
                self.response_text.insert(tk.END, f"Timestamp: {datetime.now()}\n")
                self.response_text.insert(tk.END, f"Raw Response:\n{raw_response}\n\n")
                
                # Parse response
                try:
                    if isinstance(raw_response, str):
                        json_match = re.search(r'``````', raw_response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            resp_dict = json.loads(json_str)
                        else:
                            resp_dict = json.loads(raw_response)
                    else:
                        resp_dict = raw_response
                    
                    category = resp_dict.get("category", "unknown")
                    headcount = int(resp_dict.get("headcount", 0))
                    reasoning = resp_dict.get("reasoning", "")
                    
                    # Display parsed results
                    formatted_json = json.dumps(resp_dict, indent=2)
                    self.response_text.insert(tk.END, f"Parsed Results:\n{formatted_json}\n")
                    self.response_text.insert(tk.END, "-" * 50 + "\n")
                    
                except Exception as parse_error:
                    category, headcount, reasoning = "unknown", 0, f"Parse error: {str(parse_error)}"
                    self.response_text.insert(tk.END, f"Parse Error: {str(parse_error)}\n")
                
                # Store event
                store_event(self.conn, category, headcount, reasoning, video_path)
                
                events_processed += 1
                self.events_count.set(str(events_processed))
                self.last_detection.set(f"{category} ({headcount})")
                
                # Update status
                status_msg = f"✅ Detected: {category} | Count: {headcount}"
                self.status.set(status_msg)
                self.log_message(f"Event processed: {category}, headcount: {headcount}")
                
            except Exception as e:
                error_msg = f"Error in surveillance loop: {str(e)}"
                self.status.set("❌ Processing Error")
                self.log_message(error_msg)
            
            time.sleep(5)  # Wait before next cycle
    
    def open_dashboard(self):
        """Launch the Streamlit dashboard"""
        self.log_message("Launching analytics dashboard...")
        try:
            # Launch streamlit in a separate process
            subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "dashboard.py",
                "--server.headless", "true",
                "--server.port", "8501"
            ])
            self.log_message("Dashboard launched successfully at http://localhost:8501")
            messagebox.showinfo(
                "Dashboard Launched", 
                "Analytics dashboard is starting...\n\nIt will open in your browser at:\nhttp://localhost:8501"
            )
        except Exception as e:
            error_msg = f"Failed to launch dashboard: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def open_settings(self):
        """Open settings dialog"""
        settings_window = tk.Toplevel(self.master)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.grab_set()  # Make modal
        
        # Settings content
        ttk.Label(settings_window, text="Settings", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Model selection
        model_frame = ttk.LabelFrame(settings_window, text="LLM Model", padding="10")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.selected_model = tk.StringVar(value=MODEL_NAME)
        models = ["security_manager:latest", "qwen2.5vl:3b"]
        
        for model in models:
            ttk.Radiobutton(
                model_frame, 
                text=model, 
                variable=self.selected_model, 
                value=model
            ).pack(anchor=tk.W)
        
        # Camera settings
        camera_frame = ttk.LabelFrame(settings_window, text="Camera Settings", padding="10")
        camera_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(camera_frame, text="Clip Duration (seconds):").pack(anchor=tk.W)
        duration_var = tk.StringVar(value=str(CLIP_DURATION))
        ttk.Entry(camera_frame, textvariable=duration_var).pack(fill=tk.X, pady=2)
        
        ttk.Label(camera_frame, text="Number of Frames to Extract:").pack(anchor=tk.W, pady=(10, 0))
        frames_var = tk.StringVar(value=str(NUM_FRAMES))
        ttk.Entry(camera_frame, textvariable=frames_var).pack(fill=tk.X, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(settings_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Apply", command=settings_window.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT)
    
    def test_connections(self):
        """Test all system connections"""
        self.log_message("Testing system connections...")
        
        # Test database
        try:
            if self.conn:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1")
                self.db_status.set("✅ Connected")
                self.log_message("Database: OK")
            else:
                self.db_status.set("❌ Not connected")
                self.log_message("Database: Failed")
        except Exception as e:
            self.db_status.set("❌ Error")
            self.log_message(f"Database test failed: {e}")
        
        # Test LLM
        try:
            test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if test_response.status_code == 200:
                self.llm_status.set("✅ Connected")
                self.log_message("LLM Service: OK")
            else:
                self.llm_status.set("❌ Error")
                self.log_message("LLM Service: Failed")
        except Exception as e:
            self.llm_status.set("❌ Not available")
            self.log_message(f"LLM test failed: {e}")
    
    def test_camera(self):
        """Test camera connection"""
        try:
            camera_id = int(self.camera_id.get())
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.log_message(f"Camera {camera_id}: OK")
                    messagebox.showinfo("Camera Test", f"Camera {camera_id} is working!")
                else:
                    self.log_message(f"Camera {camera_id}: No frame")
                    messagebox.showerror("Camera Test", f"Camera {camera_id} cannot capture frames")
            else:
                self.log_message(f"Camera {camera_id}: Cannot open")
                messagebox.showerror("Camera Test", f"Cannot open camera {camera_id}")
            cap.release()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid camera ID number")
        except Exception as e:
            self.log_message(f"Camera test error: {e}")
            messagebox.showerror("Camera Test", f"Camera test failed: {e}")

def main():
    """Main application entry point"""
    root = tk.Tk()
    
    # Configure ttk style
    style = ttk.Style()
    style.theme_use('clam')  # Use a modern theme
    
    # Create and run the application
    app = SurveillanceControlApp(root)
    
    # Handle window closing
    def on_closing():
        if app.is_running:
            app.is_running = False
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
