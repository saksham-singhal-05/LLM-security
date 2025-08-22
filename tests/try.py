import requests
import json

endpoint = "http://192.168.14.10:11434/api/show"
payload = {"name": "gemma3:27b"}

try:
    response = requests.post(endpoint, json=payload)
    response.raise_for_status()
    model_info = response.json()
    
    # Pretty-print JSON to stdout
    print(json.dumps(model_info, indent=4))
    
    # Optional: save to file for easier searching
    with open("model_info.json", "w") as f:
        json.dump(model_info, f, indent=4)
    
except requests.exceptions.RequestException as e:
    print("Error:", e)
