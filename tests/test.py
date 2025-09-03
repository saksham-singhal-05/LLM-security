import requests

endpoint = "http://192.168.14.10:11434/api/generate"

payload = {
    "model": "qwen2.5vl:72b",  # make sure this matches your model's exact name
    "prompt": "yo bro whats up? ",
    "stream": False
}

try:
    response = requests.post(endpoint, json=payload)
    response.raise_for_status()
    print("Response from Ollama:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print("Error connecting to Ollama endpoint:", e)
