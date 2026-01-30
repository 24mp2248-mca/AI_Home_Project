import cv2
import os
import numpy as np


def preprocess_image(input_path):
    """Read image, enhance contrast, denoise, detect edges with adaptive thresholds,
    close gaps with morphology and Hough lines, then save a cleaned binary edge image
    to the `processed` folder (preserves original filename and Y-AXIS ORIENTATION).

    Returns the output file path.
    
    IMPORTANT: Y-axis is preserved from input to output (no flipping or inversion).
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve local contrast (helps faint lines)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoise while preserving edges
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive Canny thresholds based on median
    v = np.median(denoised)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(denoised, lower, upper)

    # Morphological closing to join broken walls, then dilate to thicken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    # Optional: detect line segments and draw them to help close gaps
    lines = cv2.HoughLinesP(dilated, rho=1, theta=np.pi / 180, threshold=80, minLineLength=50, maxLineGap=20)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(dilated, (x1, y1), (x2, y2), 255, 3)

    # Ensure output folder exists and write image
    output_path = input_path.replace("uploads", "processed")
    out_dir = os.path.dirname(output_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save as binary PNG
    _, bin_img = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, bin_img)

    return output_path
