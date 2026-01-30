import cv2
import json
import os
import sys
import glob
import numpy as np

# Determine processed image path: prefer CLI arg, else pick newest file in processed/
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    candidates = sorted(glob.glob(os.path.join('processed', '*')), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise FileNotFoundError('No files found in processed/; run preprocessing first')
    IMAGE_PATH = candidates[0]

# Determine output layout path: prefer 2nd CLI arg, else use default layout.json
OUTPUT_LAYOUT = sys.argv[2] if len(sys.argv) > 2 else "layout.json"

# Read processed image
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Processed image not readable: {IMAGE_PATH}")

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

# Find contours using CCOMP to get both external contours and holes
# NOTE: Y-axis coordinates are preserved from input image (no flipping)
contours, hierarchy = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

rooms = []

if contours is None:
    contours = []

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)

    # Ignore very small areas (noise) and extremely large (whole image)
    if area < 2000 or area > (image.shape[0] * image.shape[1] * 0.95):
        continue

    x, y, w, h = cv2.boundingRect(contour)
    room_data = {
        "room_id": len(rooms) + 1,
        "approx_area": int(area),
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    }
    rooms.append(room_data)

# --- Heuristic Room Labeling ---
# Sort rooms by area descending
rooms.sort(key=lambda r: r['approx_area'], reverse=True)

total_rooms = len(rooms)
if total_rooms > 0:
    # 1. Largest is Living Room
    rooms[0]['type'] = 'Living Room'
    
    # 2. Assign others
    for i in range(1, total_rooms):
        r = rooms[i]
        
        # Smallest usually Bathroom
        if i == total_rooms - 1 and total_rooms > 2:
            r['type'] = 'Bathroom'
            
        elif i == 1:
            # Second largest
            if total_rooms >= 4:
                r['type'] = 'Master Bedroom'
            else:
                r['type'] = 'Kitchen'
                
        elif i == total_rooms - 2 and total_rooms >= 4:
            # 2nd smallest is often Entrance/Foyer
            r['type'] = 'Entrance'
            
        else:
            # Middle rooms
            existing_types = [rm.get('type') for rm in rooms]
            if 'Kitchen' not in existing_types:
                r['type'] = 'Kitchen'
            elif 'Dining Room' not in existing_types and total_rooms >= 3:
                r['type'] = 'Dining Room'
            else:
                r['type'] = 'Bedroom'

# Reset room_ids to match new sorted order (optional, but cleaner)
for i, r in enumerate(rooms):
    r['room_id'] = i + 1

# -------------------------------

print(f"DEBUG: Found {len(contours)} contours. Filtered down to {len(rooms)} rooms.")
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area >= 2000 and area <= (image.shape[0] * image.shape[1] * 0.95):
        print(f"  - Contour {i} kept: Area {area}")
    else:
        # print(f"  - Contour {i} skipped: Area {area}")
        pass

layout_data = {
    "total_rooms": len(rooms),
    "rooms": rooms
}

# Save layout as JSON (use unique output path if provided, else default)
with open(OUTPUT_LAYOUT, "w") as f:
    json.dump(layout_data, f, indent=4)

print(f"Layout extraction completed successfully. Detected rooms: {len(rooms)}, Output: {OUTPUT_LAYOUT}")
