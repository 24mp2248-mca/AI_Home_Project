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

    # ---- AI PROCESSING (Computer Vision) ----
    image = cv2.imread(upload_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection (AI preprocessing)
    edges = cv2.Canny(blurred, 50, 150)

    # Save processed image
    processed_path = os.path.join(PROCESSED_DIR, file.filename)
    cv2.imwrite(processed_path, edges);

    return {
        "message": "Sketch uploaded and AI processed successfully",
        "original_image": upload_path,
        "processed_image": processed_path
    }
