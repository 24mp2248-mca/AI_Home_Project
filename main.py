from fastapi import FastAPI, File, UploadFile
import shutil
import os
import cv2
import numpy as np

app = FastAPI()

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "FastAPI is running successfully"}

@app.post("/upload-sketch/")
async def upload_sketch(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read image
    image = cv2.imread(file_path)

    if image is None:
        return {"error": "Image not read properly"}

    image = cv2.resize(image, (600, 600))

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    processed_path = os.path.join(PROCESSED_FOLDER, file.filename)
    cv2.imwrite(processed_path, edges)

    return {
        "message": "Sketch uploaded and preprocessed successfully",
        "processed_image": file.filename
    }
