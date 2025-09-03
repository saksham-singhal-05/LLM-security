
import cv2
import base64
import requests
import os
import time
from datetime import datetime

# === CONFIG ===
VIDEO_DIR = '.'  # Directory containing the videos
VIDEO_NAMES = ['1.mp4', '2.mp4', '3.mp4', '4.mp4']  # List of video filenames
OLLAMA_ENDPOINT = "http://192.168.14.10:11434/api/generate"

# Model configurations
MODEL_GEMMA = "gemma3:27b"
MODEL_QWEN = "qwen2.5vl:72b"

PROMPT_GEMMA = """Act as an advanced security LLM responsible for actively monitoring a classroom environment in real time. Your objective is to analyze input events and behaviors, assess their risk, and deliver a structured JSON report with an explicit headcount. Your outputs must be grounded in objective reasoning and clear evidence—you should not simply agree with any user input, but must independently assess all information.

            Begin by analyzing all available class data, observations, or event logs. **All input frames must be considered in sequential order before delivering a single, comprehensive assessment. Do not respond to each frame individually—only give one final response after evaluating the complete input.**

            Classify the situation using one of four categories, based on the observed risk level:

            - "normal"
            - "pre_alert"
            - "alert"
            - "human_intervention_needed"

            Determine and return the headcount as an integer, representing the number of people present.

            For each output, include a reasoning field explaining, step by step, why the particular category is assigned. If user input is incomplete or suspicious, flag this in your reasoning. Never simply agree—always evaluate evidence.

            Return your findings in a structured JSON format:

            json
            {
            "category": "<one of: normal | pre_alert | alert | human_intervention_needed>",
            "headcount": <integer>,
            "reasoning": "<detailed objective explanation of assessment>"
            }

            Apply clear criteria for each category (examples):

            - normal: All observed behaviors and values are within safe, expected bounds.
            - pre_alert: Small deviations or minor issues that may require monitoring but are not immediately dangerous.
            - alert: Significant anomaly/potential risk requiring prompt attention.
            - human_intervention_needed: Immediate action needed due to active threat, malfunction, or unresolved escalation.

            For every input, reason independently. Challenge subjective descriptions and verify with the actual event/context. If ambiguity exists, state this in reasoning and default to the lower-risk category unless clear evidence suggests escalation.

            Example Output:

            json
            {
            "category": "pre_alert",
            "headcount": 28,
            "reasoning": "Detected mild argument between two students. No physical risk, but monitoring is advised. Room headcount verified against entry logs."
            }

            Task is complete when every entry is classified with a validated headcount and explicit reasoning, and no information is accepted at face value without substantiation."""


PROMPT_QWEN = """<|im_start|>system
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



def extract_frames_equally_spaced(video_path, num_frames):
    """Extracts 'num_frames' equally spaced frames from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        cap.release()
        raise ValueError(f"Not enough frames in {video_path}. Total frames: {total_frames}")
    
    step = total_frames // num_frames
    frames_b64 = []
    
    for i in range(num_frames):
        frame_index = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        
        if not success:
            continue
            
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        frames_b64.append(img_b64)
    
    cap.release()
    return frames_b64

def extract_frames_20_from_5sec_video(video_path):
    """Extracts up to 20 frames equally spaced from a 5 second video (4 fps)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"    Video duration: {duration:.2f}s, FPS: {fps:.2f}, Total frames: {total_frames}")
    
    # For a 5-second video at any FPS, extract up to 20 frames equally spaced
    num_frames_to_extract = min(20, total_frames)
    
    if total_frames < num_frames_to_extract:
        cap.release()
        raise ValueError(f"Not enough frames in {video_path}. Total frames: {total_frames}")
    
    step = total_frames // num_frames_to_extract
    frames_b64 = []
    
    for i in range(num_frames_to_extract):
        frame_index = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        
        if not success:
            continue
            
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        frames_b64.append(img_b64)
    
    cap.release()
    print(f"    Extracted {len(frames_b64)} frames (up to 20 max)")
    return frames_b64

def send_video_to_qwen(model_name, prompt, video_path):
    """Sends direct video file to Qwen model."""
    # Encode video as base64 for direct video input
    with open(video_path, 'rb') as video_file:
        video_b64 = base64.b64encode(video_file.read()).decode('utf-8')
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [],
        "video": video_b64,  # Direct video input
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=300)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def send_to_ollama(model_name, prompt, images_b64):
    """Sends the base64-encoded images to Ollama and returns the response."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": images_b64,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=300)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def process_single_video(video_name, video_path):
    """Process a single video with both models and all modes, measuring time taken."""
    print(f"\nProcessing {video_name}...")
    
    results = {
        "video_name": video_name,
        "timestamp": datetime.now().isoformat()
    }
    
    # Extract 5 equally spaced frames
    try:
        frames_5 = extract_frames_equally_spaced(video_path, 5)
        print(f"  Extracted {len(frames_5)} equally spaced frames (5 frames)")
    except Exception as e:
        frames_5 = []
        print(f"  Error extracting 5 frames: {e}")
    
    # Extract up to 20 frames from 5 second video (4 fps)
    try:
        frames_20 = extract_frames_20_from_5sec_video(video_path)
    except Exception as e:
        frames_20 = []
        print(f"  Error extracting 20 frames: {e}")
    
    # === GEMMA MODEL (2 MODES) ===
    print("  Processing with Gemma model...")
    
    # Gemma Mode 1: 5 equally spaced frames
    if frames_5:
        print("    - 5 equally spaced frames mode")
        start_time = time.time()
        results["gemma_5_frames"] = send_to_ollama(MODEL_GEMMA, PROMPT_GEMMA, frames_5)
        results["gemma_5_frames_time"] = round(time.time() - start_time, 2)
        print(f"      Time taken: {results['gemma_5_frames_time']}s")
    else:
        results["gemma_5_frames"] = {"error": "No 5 frames available"}
        results["gemma_5_frames_time"] = None
    
    # Gemma Mode 2: Up to 20 frames from 5 second video
    if frames_20:
        print("    - Up to 20 frames mode")
        start_time = time.time()
        results["gemma_20_frames"] = send_to_ollama(MODEL_GEMMA, PROMPT_GEMMA, frames_20)
        results["gemma_20_frames_time"] = round(time.time() - start_time, 2)
        print(f"      Time taken: {results['gemma_20_frames_time']}s")
    else:
        results["gemma_20_frames"] = {"error": "No 20 frames available"}
        results["gemma_20_frames_time"] = None
    
    # === QWEN MODEL (3 MODES) ===
    print("  Processing with Qwen model...")
    
    # Qwen Mode 1: 5 equally spaced frames
    if frames_5:
        print("    - 5 equally spaced frames mode")
        start_time = time.time()
        results["qwen_5_frames"] = send_to_ollama(MODEL_QWEN, PROMPT_QWEN, frames_5)
        results["qwen_5_frames_time"] = round(time.time() - start_time, 2)
        print(f"      Time taken: {results['qwen_5_frames_time']}s")
    else:
        results["qwen_5_frames"] = {"error": "No 5 frames available"}
        results["qwen_5_frames_time"] = None
    
    # Qwen Mode 2: Up to 20 frames from 5 second video
    if frames_20:
        print("    - Up to 20 frames mode")
        start_time = time.time()
        results["qwen_20_frames"] = send_to_ollama(MODEL_QWEN, PROMPT_QWEN, frames_20)
        results["qwen_20_frames_time"] = round(time.time() - start_time, 2)
        print(f"      Time taken: {results['qwen_20_frames_time']}s")
    else:
        results["qwen_20_frames"] = {"error": "No 20 frames available"}
        results["qwen_20_frames_time"] = None
    
    # Qwen Mode 3: Direct video input
    print("    - Direct video input mode")
    start_time = time.time()
    results["qwen_full_video"] = send_video_to_qwen(MODEL_QWEN, PROMPT_QWEN, video_path)
    results["qwen_full_video_time"] = round(time.time() - start_time, 2)
    print(f"      Time taken: {results['qwen_full_video_time']}s")
    
    return results

def save_results_to_file(all_results, filename="video_analysis_results.txt"):
    """Save all results to a properly formatted text file including response times."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("VIDEO ANALYSIS RESULTS WITH RESPONSE TIMES\n")
        f.write("GEMMA: 2 modes (5 frames + up to 20 frames)\n")
        f.write("QWEN: 3 modes (5 frames + up to 20 frames + full video)\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for video_results in all_results:
            f.write(f"VIDEO: {video_results['video_name']}\n")
            f.write(f"Processed at: {video_results['timestamp']}\n")
            f.write("-" * 60 + "\n\n")
            
            # Gemma results
            f.write("GEMMA3:27B MODEL RESULTS:\n")
            f.write("~" * 30 + "\n")
            
            f.write("5 Equally Spaced Frames Mode:\n")
            f.write(f"Response Time: {video_results.get('gemma_5_frames_time')} seconds\n")
            f.write(str(video_results['gemma_5_frames']) + "\n\n")
            
            f.write("Up to 20 Frames Mode:\n")
            f.write(f"Response Time: {video_results.get('gemma_20_frames_time')} seconds\n")
            f.write(str(video_results['gemma_20_frames']) + "\n\n")
            
            # Qwen results
            f.write("QWEN2.5VL:72B MODEL RESULTS:\n")
            f.write("~" * 30 + "\n")
            
            f.write("5 Equally Spaced Frames Mode:\n")
            f.write(f"Response Time: {video_results.get('qwen_5_frames_time')} seconds\n")
            f.write(str(video_results['qwen_5_frames']) + "\n\n")
            
            f.write("Up to 20 Frames Mode:\n")
            f.write(f"Response Time: {video_results.get('qwen_20_frames_time')} seconds\n")
            f.write(str(video_results['qwen_20_frames']) + "\n\n")
            
            f.write("Direct Video Input Mode:\n")
            f.write(f"Response Time: {video_results.get('qwen_full_video_time')} seconds\n")
            f.write(str(video_results['qwen_full_video']) + "\n\n")
            
            f.write("=" * 80 + "\n\n")

def main():
    """Main function to process all videos."""
    print("Starting video analysis with response time tracking...")
    print("GEMMA MODES: 5 equally spaced frames + up to 20 frames")
    print("QWEN MODES: 5 equally spaced frames + up to 20 frames + full video")
    print(f"Processing {len(VIDEO_NAMES)} videos...")
    
    all_results = []
    total_start_time = time.time()
    
    for video_name in VIDEO_NAMES:
        video_path = os.path.join(VIDEO_DIR, video_name)
        
        if not os.path.exists(video_path):
            print(f"Warning: Video {video_name} not found in {VIDEO_DIR}")
            continue
        
        # Process single video
        video_results = process_single_video(video_name, video_path)
        all_results.append(video_results)
    
    total_time = round(time.time() - total_start_time, 2)
    
    # Save all results
    output_filename = f"video_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    save_results_to_file(all_results, output_filename)
    
    print(f"\n" + "=" * 50)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_filename}")
    print(f"Processed {len(all_results)} videos successfully.")
    print(f"Total processing time: {total_time} seconds")
    print("=" * 50)

if __name__ == "__main__":
    main()
