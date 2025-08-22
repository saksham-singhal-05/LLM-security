import requests

endpoint = "http://192.168.14.10:11434/api/generate"

payload = {
    "model": "gemma3:27b",  # make sure this matches your model's exact name
    "prompt": "why did you flag last request as normal. what do you think was in the video? ",
    "stream": False
}

try:
    response = requests.post(endpoint, json=payload)
    response.raise_for_status()
    print("Response from Ollama:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print("Error connecting to Ollama endpoint:", e)
