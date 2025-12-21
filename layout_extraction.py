import cv2
import json
import os

# Path to processed image
IMAGE_PATH = "processed/home1.png"

# Read processed image
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

# Threshold image
_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rooms = []

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    
    # Ignore very small areas (noise)
    if area > 500:
        room_data = {
            "room_id": i + 1,
            "approx_area": int(area)
        }
        rooms.append(room_data)

layout_data = {
    "total_rooms": len(rooms),
    "rooms": rooms
}

# Save layout as JSON
with open("layout.json", "w") as f:
    json.dump(layout_data, f, indent=4)

print("Layout extraction completed successfully.")
