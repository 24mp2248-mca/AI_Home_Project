import torch
import cv2
import numpy as np
import os
from dl_model import UNet

def segment_walls(image_path, model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "wall_segmentation_model.pth")
    """
    Uses the U-Net model to segment walls from the floor plan sketch.
    If model weights are not found, falls back to OpenCV heuristic.
    """
    
    # 1. Load Image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} not found")

    # 2. Check for DL Model
    if os.path.exists(model_path):
        print(f"Loading Deep Learning model from {model_path}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(n_channels=1, n_classes=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Preprocess for DL
        input_img = cv2.resize(img, (256, 256))
        input_tensor = torch.from_numpy(input_img).unsqueeze(0).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Postprocess
        pred = output.squeeze().cpu().numpy()
        pred_mask = (pred > 0.5).astype(np.uint8) * 255
        pred_mask = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))
        
        return pred_mask
    else:
        print("DL Model weights not found. Falling back to OpenCV image processing.")
        return None

if __name__ == "__main__":
    # Test run
    # Mock image path
    print("Testing DL Inference Module...")
    # segment_walls("test.jpg")
