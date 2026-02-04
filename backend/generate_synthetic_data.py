import cv2
import numpy as np
import os
import random

def generate_synthetic_data(num_samples=100, output_dir="backend/data/train"):
    """
    Generates synthetic floor plan images (lines) and corresponding masks (filled walls).
    """
    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic samples in {output_dir}...")
    
    for i in range(num_samples):
        # Create blank canvas (black background)
        img_size = 256
        image = np.zeros((img_size, img_size), dtype=np.uint8)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        
        # Draw random "rooms" (rectangles)
        num_rooms = random.randint(3, 8)
        
        for _ in range(num_rooms):
            # Random coordinates
            x1 = random.randint(10, img_size - 60)
            y1 = random.randint(10, img_size - 60)
            w = random.randint(40, 100)
            h = random.randint(40, 100)
            x2, y2 = x1 + w, y1 + h
            
            # Ensure within bounds
            x2 = min(x2, img_size - 10)
            y2 = min(y2, img_size - 10)
            
            # Draw walls on Image (White lines)
            # Simulate sketch style: varying thickness
            thickness = random.randint(1, 3)
            cv2.rectangle(image, (x1, y1), (x2, y2), 255, thickness)
            
            # Draw walls on Mask (Filled walls ? No, usually masks for segmentation are the pixels of the class)
            # If we want to detect WALLS, the mask should be '1' where the wall is.
            # We'll make the mask slightly thicker to be robust.
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness)
        
            # Add some "noise" lines (furniture or scribbles) to IMAGE only, not MASK
            # This teaches the AI to ignore them.
            if random.random() > 0.5:
                nx1 = random.randint(10, img_size - 10)
                ny1 = random.randint(10, img_size - 10)
                nx2 = nx1 + random.randint(-20, 20)
                ny2 = ny1 + random.randint(-20, 20)
                cv2.line(image, (nx1, ny1), (nx2, ny2), 150, 1) # Dimmer line

        # Save
        filename = f"sample_{i}.png"
        cv2.imwrite(os.path.join(img_dir, filename), image)
        cv2.imwrite(os.path.join(mask_dir, filename), mask)
        
    print("Generation Complete.")

if __name__ == "__main__":
    generate_synthetic_data()
