import json
import math

# Read layout data
with open("layout.json", "r") as f:
    layout = json.load(f)

rooms_3d = []

WALL_HEIGHT = 3  # meters (fixed for simplicity)

for room in layout["rooms"]:
    area = room["approx_area"]

    # Convert area to width & length (simple square assumption)
    side = math.sqrt(area)

    room_3d = {
        "room_id": room["room_id"],
        "width": round(side / 100, 2),
        "length": round(side / 100, 2),
        "height": WALL_HEIGHT
    }
    rooms_3d.append(room_3d)

layout_3d = {
    "total_rooms": layout["total_rooms"],
    "rooms": rooms_3d
}

# Save 3D-ready layout
with open("layout_3d.json", "w") as f:
    json.dump(layout_3d, f, indent=4)

print("3D layout preparation completed successfully.")
