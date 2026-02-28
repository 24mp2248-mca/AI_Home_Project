import cv2
import json
import os
import sys
import glob
import numpy as np
from google import genai
from dotenv import load_dotenv
import time

# Load Environment for API Key
env_path = os.path.join(os.path.dirname(__file__), 'backend', '.env')
if not os.path.exists(env_path):
    env_path = os.path.join(os.path.dirname(__file__), '.env') # Fallback
load_dotenv(env_path, override=True)

api_key = os.getenv("GEMINI_API_KEY")
client = None
model_name = None

if api_key:
    client = genai.Client(api_key=api_key)
    
    # Try to find a working vision model
    # gemini-flash-latest is our most stable free-tier model right now
    candidates = ['gemini-flash-latest', 'gemini-2.0-flash']
    
    for m_name in candidates:
        try:
            print(f"DEBUG: Testing model {m_name}...")
            
            # VALIDATE by generating simple text
            # This ensures the model name exists and API key has access
            client.models.generate_content(model=m_name, contents="Hello")
            
            model_name = m_name
            print(f"DEBUG: Selected and Validated model {m_name}")
            break
        except Exception as e:
            print(f"DEBUG: Model {m_name} failed validation: {e}")
            
    if model_name is None:
         print("WARNING: No working Gemini Vision model found. AI features disabled.")
else:
    print("WARNING: No GEMINI_API_KEY found. Room recognition will be heuristic.")

# Determine paths
# Arg 1: Processed Image (Binary mask)
# Arg 2: Output JSON Path
# Arg 3: Original Image (Color, for AI Vision)
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    candidates = sorted(glob.glob(os.path.join('processed', '*')), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise FileNotFoundError('No files found in processed/; run preprocessing first')
    IMAGE_PATH = candidates[0]

OUTPUT_LAYOUT = sys.argv[2] if len(sys.argv) > 2 else "layout.json"
ORIGINAL_PATH = sys.argv[3] if len(sys.argv) > 3 else None

# Read processed image
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Processed image not readable: {IMAGE_PATH}")

# Read Original Image for AI
original_img = None
if ORIGINAL_PATH and os.path.exists(ORIGINAL_PATH):
    original_img = cv2.imread(ORIGINAL_PATH)

# Ensure a clean binary image (use Otsu to adapt to varying brightness)
_, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# If walls are white lines and interiors are dark, invert so room interiors are white regions
white_count = int(np.sum(thresh == 255))
black_count = int(np.sum(thresh == 0))
if white_count < black_count:
    thresh = 255 - thresh

# Morphology to fill small gaps and remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

# Find contours
contours, hierarchy = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

rooms = []
if contours is None:
    contours = []

def identify_room_with_ai(crop_img):
    """
    Sends a cropped room image to Gemini Vision to identify the room type.
    """
    global client, model_name
    if client is None or model_name is None or crop_img is None:
        return None
    
    try:
        # Convert BGR to RGB
        rgb_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(rgb_crop)
        
        prompt = "Look at this floor plan room crop. Identify the room type based on furniture (bed/table/toilet) or text labels. Return ONLY one of these words: Bedroom, Bathroom, Kitchen, Living Room, Dining Room, Garage, Entrance. If unsure, say Unknown."
        
        # Determine strict list for robustness
        response = client.models.generate_content(model=model_name, contents=[prompt, pil_img])
        text = response.text.strip().replace('.','').replace('"','')
        
        valid_types = ["Bedroom", "Bathroom", "Kitchen", "Living Room", "Dining Room", "Garage", "Entrance"]
        for t in valid_types:
            if t.lower() in text.lower():
                return t
        return None
    except Exception as e:
        print(f"AI Vision Error: {e}")
        # If model not found or huge error, disable it for future calls
        if "404" in str(e) or "not found" in str(e).lower():
            model_name = None
        return None

print(f"DEBUG: Found {len(contours)} contours.")

# Extract Rooms
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)

    # Ignore noise
    if area < 2000 or area > (image.shape[0] * image.shape[1] * 0.95):
        continue

    x, y, w, h = cv2.boundingRect(contour)
    
    # --- AI RECOGNITION UPGRADE ---
    room_type = "Unknown"
    
    # Only try AI if we have the original image and API key
    if original_img is not None and client is not None and model_name is not None:
        # Crop with some padding
        pad = 10
        cy1 = max(0, y - pad)
        cy2 = min(original_img.shape[0], y + h + pad)
        cx1 = max(0, x - pad)
        cx2 = min(original_img.shape[1], x + w + pad)
        
        crop = original_img[cy1:cy2, cx1:cx2]
        
        print(f" Asking AI about Room {len(rooms)+1}...")
        ai_label = identify_room_with_ai(crop)
        if ai_label:
            room_type = ai_label
            print(f" -> AI says: {ai_label}")
        else:
            print(" -> AI unsure.")
            # If AI failed with critical error (model_name is None now), stop trying
            if model_name is None: 
                print("Stopping AI extraction due to previous errors.")
    
        # Rate limit protection (Free tier 15 RPM = 1 req / 4s)
        # We'll sleep a bit to be safe, though 1s might be enough if not many rooms
        time.sleep(1.0) 
    
    # -----------------------------

    room_data = {
        "room_id": len(rooms) + 1,
        "type": room_type, # Placeholder, will be filled by AI or Heuristic
        "approx_area": int(area),
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    }
    rooms.append(room_data)

# --- Fallback Heuristics for 'Unknown' Rooms ---
# Use the old logic to fill in gaps if AI failed or Key missing
rooms.sort(key=lambda r: r['approx_area'], reverse=True)
total_rooms = len(rooms)

# If mostly unknown, apply heuristics
unknowns = [r for r in rooms if r['type'] == "Unknown"]

if len(unknowns) > 0:
    print("Applying heuristics to unknown rooms...")
    for i, r in enumerate(rooms):
        if r['type'] != "Unknown":
            continue
            
        # Logic based on rank in size sorted list
        # We need the index 'i' to be the rank in the SORTED list
        
        if i == 0:
            r['type'] = 'Living Room'
        elif i == total_rooms - 1 and total_rooms > 2:
            r['type'] = 'Bathroom'
        elif i == 1:
             if total_rooms >= 4: r['type'] = 'Master Bedroom'
             else: r['type'] = 'Kitchen'
        else:
             r['type'] = 'Bedroom'

# Reset ID
for i, r in enumerate(rooms):
    r['room_id'] = i + 1

layout_data = {
    "total_rooms": len(rooms),
    "rooms": rooms
}

with open(OUTPUT_LAYOUT, "w") as f:
    json.dump(layout_data, f, indent=4)

print(f"Layout extraction completed. Output: {OUTPUT_LAYOUT}")
