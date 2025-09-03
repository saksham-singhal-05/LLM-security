import requests
import json
import base64
import cv2
import os
import tempfile
from pathlib import Path
from typing import List, Optional
import numpy as np

class OllamaVideoTester:
    def __init__(self, model_name="qwen2.5vl:72b", ollama_url="http://192.168.14.10:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_url = f"{ollama_url}/api/generate"
        self.chat_url = f"{ollama_url}/api/chat"
        
    def test_model_availability(self):
        """Test if the model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                if self.model_name in models:
                    print(f"✅ Model {self.model_name} is available")
                    return True
                else:
                    print(f"❌ Model {self.model_name} not found. Available models: {models}")
                    return False
        except Exception as e:
            print(f"❌ Error checking model availability: {e}")
            return False
    
    def extract_frames(self, video_path: str, num_frames: int = 8) -> List[np.ndarray]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract evenly distributed frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def frames_to_base64(self, frames: List[np.ndarray]) -> List[str]:
        """Convert frames to base64 strings"""
        base64_frames = []
        for frame in frames:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(base64_frame)
        return base64_frames
    
    def method1_direct_base64(self, video_path: str):
        """Method 1: Direct base64 encoding of entire video file"""
        print("\n🔍 Method 1: Direct Base64 Encoding of Video File")
        try:
            with open(video_path, 'rb') as video_file:
                video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
            
            payload = {
                "model": self.model_name,
                "prompt": "Describe what happens in this video",
                "images": [video_base64],
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                print(f"Response: {result.get('response', 'No response')}")
            else:
                print(f"❌ Failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def method2_frame_extraction(self, video_path: str):
        """Method 2: Extract frames and send as multiple images"""
        print("\n🔍 Method 2: Frame Extraction (Multiple Images)")
        try:
            frames = self.extract_frames(video_path, num_frames=8)
            base64_frames = self.frames_to_base64(frames)
            
            payload = {
                "model": self.model_name,
                "prompt": "Describe what happens in this video based on these key frames",
                "images": base64_frames,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                print(f"Response: {result.get('response', 'No response')}")
            else:
                print(f"❌ Failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def method3_chat_api(self, video_path: str):
        """Method 3: Using chat API with video"""
        print("\n🔍 Method 3: Chat API with Video")
        try:
            with open(video_path, 'rb') as video_file:
                video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": "Describe what happens in this video",
                        "images": [video_base64]
                    }
                ]
            }
            
            response = requests.post(self.chat_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                print(f"Response: {result.get('message', {}).get('content', 'No response')}")
            else:
                print(f"❌ Failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def method4_sequential_frames(self, video_path: str):
        """Method 4: Send frames sequentially in conversation"""
        print("\n🔍 Method 4: Sequential Frame Analysis")
        try:
            frames = self.extract_frames(video_path, num_frames=4)
            
            for i, frame in enumerate(frames):
                print(f"  Analyzing frame {i+1}/{len(frames)}")
                _, buffer = cv2.imencode('.jpg', frame)
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                
                payload = {
                    "model": self.model_name,
                    "prompt": f"Describe what you see in this frame (frame {i+1} of {len(frames)})",
                    "images": [base64_frame],
                    "stream": False
                }
                
                response = requests.post(self.api_url, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    print(f"  Frame {i+1}: {result.get('response', 'No response')[:100]}...")
                else:
                    print(f"  ❌ Frame {i+1} failed: {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def method5_video_format_variants(self, video_path: str):
        """Method 5: Try different video formats and encodings"""
        print("\n🔍 Method 5: Video Format Variants")
        
        formats = ['.mp4', '.avi', '.mov', '.webm']
        
        for fmt in formats:
            try:
                print(f"  Trying format: {fmt}")
                
                # Convert video to different format
                temp_path = f"temp_video{fmt}"
                cap = cv2.VideoCapture(video_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') if fmt == '.mp4' else cv2.VideoWriter_fourcc(*'XVID')
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
                
                # Copy first 2 seconds only to reduce size
                frame_count = 0
                max_frames = int(fps * 2)  # 2 seconds
                
                while frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    frame_count += 1
                
                cap.release()
                out.release()
                
                # Try sending this format
                if os.path.exists(temp_path):
                    with open(temp_path, 'rb') as video_file:
                        video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
                    
                    payload = {
                        "model": self.model_name,
                        "prompt": f"Describe this video in {fmt} format",
                        "images": [video_base64],
                        "stream": False
                    }
                    
                    response = requests.post(self.api_url, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        print(f"  ✅ {fmt} Success: {result.get('response', 'No response')[:100]}...")
                    else:
                        print(f"  ❌ {fmt} Failed: {response.status_code}")
                    
                    # Clean up
                    os.remove(temp_path)
                
            except Exception as e:
                print(f"  ❌ Error with {fmt}: {e}")
    
    def method6_gif_conversion(self, video_path: str):
        """Method 6: Convert to GIF and send"""
        print("\n🔍 Method 6: GIF Conversion")
        try:
            import imageio
            
            # Read video and convert to GIF
            reader = imageio.get_reader(video_path)
            frames = []
            
            # Take every 10th frame to reduce size
            for i, frame in enumerate(reader):
                if i % 10 == 0 and len(frames) < 20:  # Max 20 frames
                    frames.append(frame)
            
            # Save as GIF
            gif_path = "temp_video.gif"
            imageio.mimsave(gif_path, frames, duration=0.5)
            
            # Send GIF
            with open(gif_path, 'rb') as gif_file:
                gif_base64 = base64.b64encode(gif_file.read()).decode('utf-8')
            
            payload = {
                "model": self.model_name,
                "prompt": "Describe what happens in this animated GIF",
                "images": [gif_base64],
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                print("✅ GIF Success!")
                print(f"Response: {result.get('response', 'No response')}")
            else:
                print(f"❌ GIF Failed: {response.status_code} - {response.text}")
            
            # Clean up
            if os.path.exists(gif_path):
                os.remove(gif_path)
                
        except ImportError:
            print("❌ imageio not installed. Install with: pip install imageio[ffmpeg]")
        except Exception as e:
            print(f"❌ Error: {e}")

def create_sample_video(output_path: str = "sample_video.mp4"):
    """Create a sample video for testing"""
    print("📹 Creating sample video for testing...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (640, 480))
    
    for i in range(50):  # 5 second video at 10 fps
        # Create a simple animated frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Moving circle
        center_x = int(320 + 200 * np.sin(i * 0.2))
        center_y = 240
        cv2.circle(frame, (center_x, center_y), 30, (0, 255, 255), -1)
        
        # Add text
        cv2.putText(frame, f'Frame {i+1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Sample video created: {output_path}")
    return output_path

def main():
    print("🚀 Ollama Video Input Tester for Qwen2.5VL:72B")
    print("=" * 50)
    
    # Initialize tester
    tester = OllamaVideoTester()
    
    # Check if model is available
    if not tester.test_model_availability():
        print("\n💡 Make sure to:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull the model: ollama pull qwen2.5vl:72b")
        return
    
    # Get video path from user or create sample
    video_path = "surveillance_clip.mp4"
    
    if not video_path:
        video_path = create_sample_video()
    elif not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        video_path = create_sample_video()
    
    print(f"\n🎥 Using video: {video_path}")
    
    # Test all methods
    methods = [
        tester.method1_direct_base64,
        tester.method2_frame_extraction,
        tester.method3_chat_api,
        tester.method4_sequential_frames,
        tester.method5_video_format_variants,
        tester.method6_gif_conversion
    ]
    
    for method in methods:
        try:
            method(video_path)
        except Exception as e:
            print(f"❌ Method failed: {e}")
        print("-" * 40)
    
    print("\n🎯 Summary:")
    print("- If Method 2 (Frame Extraction) worked, use that approach")
    print("- If Method 6 (GIF) worked, convert videos to GIF first")
    print("- Sequential frame analysis (Method 4) provides detailed descriptions")
    print("- Some models may not support direct video input yet")

if __name__ == "__main__":
    main()
