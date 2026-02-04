import torch
import cv2
import numpy as np
import random
import os
from dl_model import UNet

def analyze():
    print("Loading Model...")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load("backend/wall_segmentation_model.pth", map_location=DEVICE))
    model.eval()

    # 1. Generate a fresh random "test" sample on the fly
    img_size = 256
    image = np.zeros((img_size, img_size), dtype=np.uint8)
    mask_true = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Draw a "House" (Multi-room)
    # Room 1: Living
    cv2.rectangle(image, (50, 50), (150, 150), 255, 2)
    cv2.rectangle(mask_true, (50, 50), (150, 150), 255, 2)
    
    # Room 2: Bedroom (Right)
    cv2.rectangle(image, (150, 50), (220, 120), 255, 2)
    cv2.rectangle(mask_true, (150, 50), (220, 120), 255, 2)
    
    # Room 3: Kitchen (Bottom)
    cv2.rectangle(image, (50, 150), (120, 220), 255, 2)
    cv2.rectangle(mask_true, (50, 150), (120, 220), 255, 2)

    # Dictionary Label (Noise)
    cv2.putText(image, "LIVING", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150), 1)
    
    # Furniture (Noise - Table)
    cv2.circle(image, (100, 100), 20, 150, 1)

    # 2. Run Inference
    input_tensor = torch.from_numpy(image).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE) # (1, 1, 256, 256)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.squeeze().cpu().numpy()
        pred_mask = (pred > 0.5).astype(np.uint8) * 255

    # 3. Visualization
    # Stack images horizontally: [Input] [Truth] [AI Output]
    result = np.hstack((image, mask_true, pred_mask))
    
    # Add text labels
    result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.putText(result_color, "Input Sketch", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(result_color, "Ground Truth", (266, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(result_color, "AI Prediction", (522, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    output_path = "analysis_result.png"
    cv2.imwrite(output_path, result_color)
    print(f"Analysis complete. Open '{output_path}' to see the AI's performance.")

if __name__ == "__main__":
    analyze()
