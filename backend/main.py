from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Backend running"}

@app.post("/upload-sketch/")
async def upload_sketch(file: UploadFile = File(...)):
    print("UPLOAD REQUEST RECEIVED:", file.filename)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    print("Saving to:", file_path)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "message": "Sketch uploaded successfully",
        "filename": file.filename
    }

