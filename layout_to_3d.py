import json
import math
import sys


def rects_touch(a, b, tol=0.0):
    # a and b are (x,y,w,h)
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b

    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    # Expand by tol
    ax1 -= tol; ay1 -= tol; ax2 += tol; ay2 += tol
    bx1 -= tol; by1 -= tol; bx2 += tol; by2 += tol

    horiz = not (ax2 < bx1 or bx2 < ax1)
    vert = not (ay2 < by1 or by2 < ay1)
    return horiz and vert


# Read layout data
infile = 'layout.json'
if len(sys.argv) > 1:
    infile = sys.argv[1]

# Output file: prefer CLI arg, else derive from input
outfile = 'layout_3d.json'
if len(sys.argv) > 2:
    outfile = sys.argv[2]

with open(infile, 'r') as f:
    layout = json.load(f)

# Parameters
SCALE = 0.01  # meters per pixel (default: 1 px = 0.01 m)
WALL_HEIGHT = 3.0

# Build room list using bounding boxes if available
rooms = []
for room in layout.get('rooms', []):
    bbox = room.get('bbox')
    if bbox:
        x = bbox['x']
        y = bbox['y']
        w = bbox['w']
        h = bbox['h']
    else:
        # fallback: derive square from approx_area
        side = math.sqrt(room.get('approx_area', 0))
        x = 0; y = 0; w = side; h = side

    rooms.append({
        'room_id': room.get('room_id'),
        'type': room.get('type', 'Unknown'),
        'px': int(x), 'py': int(y), 'pw': int(w), 'ph': int(h)
    })

if not rooms:
    raise RuntimeError('No rooms found in layout.json')

# Compute house bounding box (union of room rects)
# Y-axis is preserved from input layout (no inversion or transformation)
min_x = min(r['px'] for r in rooms)
min_y = min(r['py'] for r in rooms)
max_x = max(r['px'] + r['pw'] for r in rooms)
max_y = max(r['py'] + r['ph'] for r in rooms)

house_w_px = max_x - min_x
house_h_px = max_y - min_y

# Convert to meters
house = {
    'width_m': round(house_w_px * SCALE, 3),
    'length_m': round(house_h_px * SCALE, 3),
    'height_m': WALL_HEIGHT
}

# Normalize rooms to house origin and compute meters
rooms_out = []
for r in rooms:
    rel_x = r['px'] - min_x
    rel_y = r['py'] - min_y
    w_m = round(r['pw'] * SCALE, 3)
    h_m = round(r['ph'] * SCALE, 3)
    rooms_out.append({
        'room_id': r['room_id'],
        'x_m': round(rel_x * SCALE, 3),
        'y_m': round(rel_y * SCALE, 3),
        'width_m': w_m,
        'length_m': h_m,
        'height_m': WALL_HEIGHT,
        'area_m2': round((r['pw'] * r['ph']) * (SCALE**2), 3),
        'type': r.get('type', 'Unknown')
    })

# Compute adjacency (touching or overlapping bboxes) with small tolerance
tol_px = 2
for ro in rooms_out:
    ro['neighbors'] = []
for i, a in enumerate(rooms):
    for j, b in enumerate(rooms):
        if i == j:
            continue
        if rects_touch((a['px'], a['py'], a['pw'], a['ph']), (b['px'], b['py'], b['pw'], b['ph']), tol=tol_px):
            aid = a['room_id']
            bid = b['room_id']
            # find output record for aid
            for ro in rooms_out:
                if ro['room_id'] == aid:
                    if bid not in ro['neighbors']:
                        ro['neighbors'].append(bid)

# Build final 3D layout
layout_3d = {
    'house': house,
    'total_rooms': len(rooms_out),
    'rooms': rooms_out
}

# Save 3D-ready layout (use unique output path if provided, else default)
with open(outfile, 'w') as f:
    json.dump(layout_3d, f, indent=4)

print(f'3D layout generation completed. House size (m): {house["width_m"]} x {house["length_m"]}, Output: {outfile}')
