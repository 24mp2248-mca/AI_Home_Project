# AI Home Planner - Integration Guide

## System Overview

The application now has full backend-frontend integration:

### Architecture Flow
1. **Upload Page** (`frontend/upload.html`) → User selects floor plan image
2. **Backend API** (`backend/main.py`) → Processes image, detects rooms, returns JSON
3. **Visualization Page** (`frontend/visualization.html`) → Renders 3D model from room data

## Setup Instructions

### 1. Start Backend Server

```bash
cd backend
pip install fastapi uvicorn opencv-python numpy
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at: `http://localhost:8000`

### 2. Start Frontend Server

```bash
cd frontend
python -m http.server 8080
```

Or use any static server. Visit: `http://localhost:8080`

### 3. Test the Integration

1. Go to **Upload Floor Plan** page
2. Select an image (JPG/PNG)
3. Click **Start Analysis**
4. The frontend will:
   - Upload to backend (`POST /upload-sketch/`)
   - Wait for processing
   - Receive room data as JSON
   - Pass data to visualization page
5. View the **3D Visualization** with detected rooms

## API Response Format

The backend returns:
```json
{
  "success": true,
  "message": "Sketch uploaded, AI processed, layout generated",
  "timestamp": "2026-01-28T...",
  "filename": "floor_plan.jpg",
  "room_count": 3,
  "rooms": [
    {
      "id": "R1",
      "type": "room",
      "dimensions": {
        "length": 32,
        "width": 20,
        "height": 10
      },
      "position": {
        "x": 10,
        "y": 15
      }
    }
  ],
  "processed_image_path": "processed/floor_plan.jpg"
}
```

## Key Fixes Applied

✅ **Backend (`main.py`)**
- Added proper JSON response formatting
- Added error handling with HTTPException
- Fixed CORS headers
- Returns structured room data

✅ **Upload Page (`upload.html`)**
- Removed simulated progress
- Added actual file upload to backend
- Proper error handling with user feedback
- Passes room data via sessionStorage to visualization

✅ **Visualization Page (`visualization.html`)**
- Now retrieves room data from sessionStorage
- Creates 3D models from actual room data
- Dynamically positions camera based on layout size
- Implements `createRoomsFromJSON()` properly
- Shows detected room count and total area

## Troubleshooting

### "Backend not running" error
- Make sure uvicorn is running on port 8000
- Check backend output for errors

### CORS errors
- Backend already configured with `allow_origins=["*"]`
- Ensure frontend URL matches browser origin

### Rooms not rendering
- Check browser console for errors
- Verify room data is in correct format
- Room coordinates should be positive numbers

## Future Enhancements

- [ ] Add authentication
- [ ] Store projects in database
- [ ] Export 3D models (OBJ, GLTF)
- [ ] Room property editing
- [ ] Furniture placement
- [ ] Cost estimation
