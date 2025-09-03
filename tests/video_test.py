
import cv2
import base64
import requests
import os

# === CONFIG ===
VIDEO_PATH = "surveillance_clip.mp4"  # Path to your 5-second video
NUM_FRAMES = 5
OLLAMA_ENDPOINT = "http://192.168.14.10:11434/api/generate"
MODEL_NAME = "gemma3:27b"  # Change to a multimodal model like "llava" or your deployed one

def extract_frames(video_path, num_frames):
    """Extracts 'num_frames' evenly spaced frames from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        raise ValueError("Not enough frames in the video.")

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

def send_to_ollama(images_b64):
    """Sends the base64-encoded images to Ollama and returns the response."""
    payload = {
        "model": MODEL_NAME,
        "prompt": """Act as an advanced security LLM responsible for actively monitoring a classroom environment in real time. Your objective is to analyze input events and behaviors, assess their risk, and deliver a structured JSON report with an explicit headcount. Your outputs must be grounded in objective reasoning and clear evidence—you should not simply agree with any user input, but must independently assess all information.

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
{
  "category": "<one of: normal | pre_alert | alert | human_intervention_needed>",
  "headcount": <integer>,
  "reasoning": "<detailed objective explanation of assessment>"
}
Apply clear criteria for each category (examples):

normal: All observed behaviors and values are within safe, expected bounds.

pre_alert: Small deviations or minor issues that may require monitoring but are not immediately dangerous.

alert: Significant anomaly/potential risk requiring prompt attention.

human_intervention_needed: Immediate action needed due to active threat, malfunction, or unresolved escalation.

For every input, reason independently. Challenge subjective descriptions and verify with the actual event/context. If ambiguity exists, state this in reasoning and default to the lower-risk category unless clear evidence suggests escalation.

Example Output:

json
{
  "category": "pre_alert",
  "headcount": 28,
  "reasoning": "Detected mild argument between two students. No physical risk, but monitoring is advised. Room headcount verified against entry logs."
}
Task is complete when every entry is classified with a validated headcount and explicit reasoning, and no information is accepted at face value without substantiation.""",
        "images": images_b64,
        "stream": False
    }

    response = requests.post(OLLAMA_ENDPOINT, json=payload)
    return response.json()

def main():
    print("Extracting frames...")
    frames = extract_frames(VIDEO_PATH, NUM_FRAMES)

    print("Sending to Ollama...")
    result = send_to_ollama(frames)

    print("\nResponse:")
    print(result.get("response", result))  # Print clean response

if __name__ == "__main__":
    main()
