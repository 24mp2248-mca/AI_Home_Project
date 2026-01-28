from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np

app = FastAPI()

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ FOLDERS ------------------
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ------------------ ROOT ------------------
@app.get("/")
def read_root():
    return {"message": "Backend running successfully"}

# ------------------ UPLOAD + AI PROCESS ------------------
@app.post("/upload-sketch/")
async def upload_sketch(file: UploadFile = File(...)):

    # ---- Save original image ----
    upload_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    # ---- AI PROCESSING ----
    image = cv2.imread(upload_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # ---- CONTOUR DETECTION ----
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # ---- CONVERT CONTOURS TO ROOMS (JSON) ----
    rooms = []
    room_id = 1

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > 20 and h > 50:  # ignore small noise
            rooms.append({
                "id": f"R{room_id}",
                "type": "room",
                "dimensions": {
                    "length": int(w / 10),
                    "width": int(h / 10),
                    "height": 10
                },
                "position": {
                    "x": int(x / 10),
                    "y": int(y / 10)
                }
            })
            room_id += 1

    # ---- Save processed image ----
    processed_path = os.path.join(PROCESSED_DIR, file.filename)
    cv2.imwrite(processed_path, edges)

    # ---- FINAL RESPONSE ----
    return {
        "message": "Sketch uploaded, AI processed, layout generated",
        "rooms": rooms,
        "processed_image": processed_path
    }
