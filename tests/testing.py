import requests
import socket

def check_ollama_connection(endpoint="http://192.168.14.10:11434", model="gemma3:27b"):
    print(f" Checking Ollama at {endpoint} with model '{model}'\n")

    # Step 1: Check if host/port is reachable
    try:
        host = endpoint.split("//")[-1].split(":")[0]
        port = int(endpoint.split(":")[-1])
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect((host, port))
        sock.close()
        print(f"Port {port} on {host} is reachable.")
    except Exception as e:
        print(f"Cannot reach {endpoint}. Error: {e}")
        return False

    # Step 2: Test if /api/tags works (lists models)
    try:
        tags_url = f"{endpoint}/api/tags"
        r = requests.get(tags_url, timeout=5)
        if r.status_code == 200:
            print("Ollama /api/tags reachable.")
            models = r.json().get("models", [])
            print("Available models:")
            for m in models:
                print(f"  - {m['name']}")
            if not any(model in m["name"] for m in models):
                print(f"Model '{model}' not found in list above.")
        else:
            print(f" /api/tags failed with {r.status_code}: {r.text}")
    except Exception as e:
        print(f" Could not query /api/tags: {e}")

    # Step 3: Try generating a response
    try:
        url = f"{endpoint}/api/generate"
        payload = {
            "model": model,
            "prompt": "Hello Gemma 3 27B, indentify yourself and tell me about your capabilities"
        }
        r = requests.post(url, json=payload, stream=True, timeout=30)
        if r.status_code == 200:
            print(" Generate API working. Response:")
            for line in r.iter_lines():
                if line:
                    print(line.decode("utf-8"))
            return True
        else:
            print(f" Generate failed: {r.status_code} {r.text}")
            return False
    except Exception as e:
        print(f" Error calling /api/generate: {e}")
        return False

if __name__ == "__main__":
    check_ollama_connection()
