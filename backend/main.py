from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import cv2
import numpy as np
import uuid
from datetime import datetime

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

    # create a unique filename to avoid overwrites
    original_name = os.path.basename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    upload_path = os.path.join(UPLOAD_DIR, unique_name)

    # ---- Save original image ----
    try:
        with open(upload_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # ---- AI PROCESSING ----
    image = cv2.imread(upload_path)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to handle variable lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological closing to join broken walls
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Edge detection (on closed image)
    edges = cv2.Canny(closed, 50, 150)

    # ---- CONTOUR DETECTION ----
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # ---- CONVERT CONTOURS TO ROOMS (JSON) ----
    rooms = []
    room_id = 1
    scale = 10.0  # pixels -> units divisor (same as frontend expectation)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > 50 and h > 50:  # ignore small noise
            length = int(w / scale)
            width = int(h / scale)
            height = 10
            rooms.append({
                "id": f"R{room_id}",
                "type": "room",
                "dimensions": {
                    "length": length,
                    "width": width,
                    "height": height
                },
                "position": {
                    "x": int(x / scale),
                    "y": int(y / scale)
                }
            })
            room_id += 1

    # ---- Compute total area and simple cost estimate ----
    total_area = 0
    for r in rooms:
        total_area += r['dimensions']['length'] * r['dimensions']['width']

    # simple cost model: cost per unit area (e.g., per sq ft)
    COST_PER_UNIT = 150  # USD per unit area (adjustable)
    estimated_cost = round(total_area * COST_PER_UNIT, 2)

    # ---- Save processed image ----
    processed_name = f"proc_{unique_name}".replace(' ', '_')
    processed_path = os.path.join(PROCESSED_DIR, processed_name)
    cv2.imwrite(processed_path, edges)

    # ---- FINAL RESPONSE ----
    processed_url = f"/processed/{processed_name}"
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Sketch uploaded, AI processed, layout generated",
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "filename": unique_name,
            "room_count": len(rooms),
            "total_area": total_area,
            "estimated_cost": estimated_cost,
            "rooms": rooms,
            "processed_image_url": processed_url
        }
    )


@app.get('/processed/{fname}')
def get_processed_image(fname: str):
    path = os.path.join(PROCESSED_DIR, fname)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='Processed image not found')
    return FileResponse(path, media_type='image/png')
