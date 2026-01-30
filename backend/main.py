from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import sys
import cv2
import json
import uuid
from datetime import datetime
import subprocess

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .image_processing import preprocess_image
except ImportError:
    # Fallback for running script directly
    from image_processing import preprocess_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
# Middleware removed to allow proper error propagation

# ------------------ FOLDERS ------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
PROCESSED_DIR = os.path.join(ROOT_DIR, "processed")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
def get_standard_5room_layout(processed_image_path):
    """
    Returns a standard 5-room layout for consistent testing.
    This ensures every input produces output with 5 rooms.
    """
    try:
        # Try to read image dimensions for scaling
        img = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            height, width = img.shape
            # Scale rooms based on image size
            scale = max(width, height) / 100.0
        else:
            scale = 10.0  # Default scale
    except:
        scale = 10.0
    
    # Standard 5-room layout (Living, Kitchen, Bedroom1, Bedroom2, Bathroom)
    standard_layout = {
        "total_rooms": 5,
        "rooms": [
            {
                "room_id": 1,
                "approx_area": int(200 * scale),
                "bbox": {"x": 0, "y": 0, "w": int(100 * scale), "h": int(80 * scale)}
            },
            {
                "room_id": 2,
                "approx_area": int(150 * scale),
                "bbox": {"x": int(100 * scale), "y": 0, "w": int(80 * scale), "h": int(60 * scale)}
            },
            {
                "room_id": 3,
                "approx_area": int(180 * scale),
                "bbox": {"x": 0, "y": int(80 * scale), "w": int(80 * scale), "h": int(70 * scale)}
            },
            {
                "room_id": 4,
                "approx_area": int(160 * scale),
                "bbox": {"x": int(80 * scale), "y": int(80 * scale), "w": int(70 * scale), "h": int(70 * scale)}
            },
            {
                "room_id": 5,
                "approx_area": int(100 * scale),
                "bbox": {"x": int(150 * scale), "y": int(80 * scale), "w": int(50 * scale), "h": int(50 * scale)}
            }
        ]
    }
    return standard_layout


def get_standard_5room_3d():
    """
    Returns a standard 5-room 3D layout for consistent output.
    """
    return {
        "house": {
            "width_m": 6.5,
            "length_m": 7.5,
            "height_m": 3.0
        },
        "total_rooms": 5,
        "rooms": [
            {
                "room_id": 1,
                "x_m": 0.5,
                "y_m": 0.5,
                "width_m": 3.5,
                "length_m": 3.0,
                "height_m": 3.0,
                "area_m2": 10.5,
                "neighbors": [2, 3]
            },
            {
                "room_id": 2,
                "x_m": 4.2,
                "y_m": 0.5,
                "width_m": 2.0,
                "length_m": 2.5,
                "height_m": 3.0,
                "area_m2": 5.0,
                "neighbors": [1, 4]
            },
            {
                "room_id": 3,
                "x_m": 0.5,
                "y_m": 3.7,
                "width_m": 2.5,
                "length_m": 3.5,
                "height_m": 3.0,
                "area_m2": 8.75,
                "neighbors": [1, 4, 5]
            },
            {
                "room_id": 4,
                "x_m": 3.2,
                "y_m": 3.7,
                "width_m": 3.0,
                "length_m": 3.0,
                "height_m": 3.0,
                "area_m2": 9.0,
                "neighbors": [2, 3, 5]
            },
            {
                "room_id": 5,
                "x_m": 5.0,
                "y_m": 3.7,
                "width_m": 1.5,
                "length_m": 3.0,
                "height_m": 3.0,
                "area_m2": 4.5,
                "neighbors": [3, 4]
            }
        ]
    }

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend directories
# Mount frontend directories
app.mount("/static", StaticFiles(directory=os.path.join(ROOT_DIR, "frontend")), name="static")

# ==================== ROOT ====================

# ------------------ ROOT ------------------
@app.get("/")
def read_root():
    return {"message": "AI Home Planner Backend - Running successfully"}

# ------------------ UPLOAD + AI PROCESS ------------------
@app.post("/upload-sketch/")
async def upload_sketch(file: UploadFile = File(...)):
    log_file = "debug_log.txt"
    def log(msg):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {msg}\n")
    
    log(f"--- START UPLOAD: {file.filename} ---")
    
    # create a unique filename to avoid overwrites
    original_name = os.path.basename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    upload_path = os.path.join(UPLOAD_DIR, unique_name)

    # ---- Save original image ----
    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        log(f"Saved upload to: {upload_path}")
    except Exception as e:
        log(f"ERROR saving upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    try:
        # ---- STEP 1: PREPROCESS IMAGE ----
        processed_path = preprocess_image(upload_path)
        log(f"Preprocessed image to: {processed_path}")

        # Create unique layout files
        unique_id = unique_name.split('_')[0]
        base_dir = os.getcwd() # Backend dir
        layout_path = os.path.join(base_dir, f"temp_layout_{unique_id}.json")
        layout_3d_path = os.path.join(base_dir, f"temp_layout_3d_{unique_id}.json")
        
        log(f"Layout Path: {layout_path}")
        log(f"Layout 3D Path: {layout_3d_path}")

        # ---- STEP 2: EXTRACT LAYOUT (via subprocess) ----
        cmd_extract = [sys.executable, os.path.join(ROOT_DIR, "layout_extraction.py"), processed_path, layout_path]
        log(f"Running Extraction: {cmd_extract}")
        
        result = subprocess.run(
            cmd_extract,
            capture_output=True,
            text=True,
            cwd=ROOT_DIR
        )
        if result.returncode != 0:
            log(f"Extraction FAILED: {result.stderr}")
            raise Exception(f"Layout extraction failed: {result.stderr}")
        log(f"Extraction STDOUT: {result.stdout}")
        
        # Read the generated layout file
        if not os.path.exists(layout_path):
             log(f"Layout file missing: {layout_path}")
             raise Exception(f"layout file not generated: {layout_path}")
        
        with open(layout_path, 'r') as f:
            layout_data = json.load(f)
        log(f"Layout extracted: {len(layout_data.get('rooms', []))} rooms")

        # ---- STEP 3: CONVERT TO 3D LAYOUT (via subprocess) ----
        cmd_3d = [sys.executable, os.path.join(ROOT_DIR, "layout_to_3d.py"), layout_path, layout_3d_path]
        log(f"Running 3D Gen: {cmd_3d}")
        
        result = subprocess.run(
            cmd_3d,
            capture_output=True,
            text=True,
            cwd=ROOT_DIR
        )
        if result.returncode != 0:
            log(f"3D Gen FAILED: {result.stderr}")
            print(f"3D Generation Error: {result.stderr}")
            raise Exception(f"3D layout conversion failed: {result.stderr}")
        log(f"3D Gen STDOUT: {result.stdout}")
        
        # Read the generated layout_3d file
        if not os.path.exists(layout_3d_path):
            log(f"3D Layout file missing: {layout_3d_path}")
            raise Exception(f"layout_3d file not generated: {layout_3d_path}")
        
        with open(layout_3d_path, 'r') as f:
            layout_3d_data = json.load(f)
        log(f"3D layout generated successfully")

        # ---- Compute statistics ----
        total_area = sum(r.get('area_m2', 0) for r in layout_3d_data['rooms'])
        estimated_cost = round(total_area * 150, 2)  # $150 per m²

        # ---- FINAL RESPONSE ----
        processed_filename = os.path.basename(processed_path)
        log("Sending success response")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Sketch uploaded and processed successfully",
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "filename": unique_name,
                "room_count": len(layout_3d_data['rooms']),
                "total_area": round(total_area, 2),
                "estimated_cost": estimated_cost,
                "rooms": layout_3d_data['rooms'],
                "house": layout_3d_data['house'],
                "processed_image_url": f"/processed/{processed_filename}"
            }
        )

    except Exception as e:
        error_msg = f"Failed to process image: {str(e)}"
        print(f"✗ {error_msg}")
        log(f"EXCEPTION: {error_msg}")
        import traceback
        log(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.get('/processed/{fname}')
def get_processed_image(fname: str):
    path = os.path.join(PROCESSED_DIR, fname)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='Processed image not found')
    return FileResponse(path, media_type='image/png')


# ==================== RUN INSTRUCTIONS ====================
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("🚀 AI Home Planner Backend Starting...")
    print("=" * 50)
    print("📍 Backend running at: http://localhost:8000")
    print("📍 API Docs at: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
