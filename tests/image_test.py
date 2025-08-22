import requests
import base64

# Load and encode image
with open("example.jpeg", "rb") as img_file:
    image_bytes = img_file.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

# Prepare payload
endpoint = "http://192.168.14.10:11434/api/generate"
payload = {
    "model": "gemma3:27b",  # Must be a multimodal model!
    "prompt": "if its a sequential flow like 5 frames from a 5 second recording? can you give me a analysis like if there is any erratic behavior in the video? ",
    #"images": [image_b64],  # <-- Only works with supported models
    "stream": False
}

# Send request
response = requests.post(endpoint, json=payload)
print("Response:")
print(response.json())
