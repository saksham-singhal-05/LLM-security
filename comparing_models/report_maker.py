import re
import json

def extract_json_blocks(text):
    # Handles both `````` and '{ ... }' or "... }"
    json_blocks = []
    # Match triple-backtick JSON blocks
    for match in re.finditer(r"``````", text, re.DOTALL):
        json_blocks.append(match.group(1).strip())
    # Also catch regular inline dicts/JSON if not triple-backtick
    for match in re.finditer(r"\{(?:\s*[\w\"':,.\-\s\\\/\[\]]+)+\}", text):
        block = match.group(0)
        try:
            # See if this is a valid JSON object (ignore false matches)
            if block.count(":") >= 2:
                json.loads(block.replace("'", '"'))
                json_blocks.append(block)
        except Exception:
            continue
    return json_blocks

def parse_report(text_filename):
    with open(text_filename, "r", encoding="utf-8") as f:
        text = f.read()
    
    report = []
    # Split by video section
    video_sections = re.split(r"=+\s*\nVIDEO: (.*?\.mp4)\n", text)
    if len(video_sections) < 2:
        raise ValueError("No video blocks found.")

    # skip the first entry (header), then chunk as video_name, rest, ...
    for i in range(1, len(video_sections), 2):
        video_name = video_sections[i].strip()
        section_text = video_sections[i+1]

        models = []
        # Find model entries (QWEN/GEMMA and their frame modes)
        model_blocks = re.split(r"(QWEN2\.5VL.*?MODEL RESULTS:|GEMMA.*?MODEL RESULTS:)", section_text, flags=re.DOTALL)
        for j in range(1, len(model_blocks), 2):
            model_type = "qwen2.5vl" if model_blocks[j].lower().startswith("qwen") else "gemma"
            block = model_blocks[j+1]

            # Find mode sections (e.g., 5 frames, up to 20 frames, direct...)
            mode_blocks = re.split(r"((?:5|Up to 20|Direct Video Input).*?Mode:)", block, flags=re.DOTALL|re.IGNORECASE)
            for k in range(1, len(mode_blocks), 2):
                mode = mode_blocks[k].split(":")[0].replace("Mode","").strip()
                result_text = mode_blocks[k+1][:2000]  # Reasonably sized block

                # Extract JSON blocks
                json_blocks = extract_json_blocks(result_text)
                for json_str in json_blocks:
                    try:
                        # Some blocks may have single quotes, fix for JSON parsing if needed
                        json_data = json.loads(json_str.replace("'", '"'))
                        output = {
                            "video": video_name,
                            "model": model_type,
                            "mode": mode,
                            "category": json_data.get("category"),
                            "headcount": json_data.get("headcount"),
                            "reasoning": json_data.get("reasoning")
                        }
                        models.append(output)
                    except Exception as e:
                        # Not a proper JSON block, skip
                        continue

        report.extend(models)
    return report

if __name__ == "__main__":
    report = parse_report("video_analysis_results_20250826_102628.txt")
    # Pretty print or save as json
    print(json.dumps(report, indent=2, ensure_ascii=False))
