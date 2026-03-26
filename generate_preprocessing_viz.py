import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_viz(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read {input_path}")
        return

    # 1. Original
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Grayscale + CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. Canny Edges
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    v = np.median(denoised)
    lower = int(max(0.0, 0.66 * v))
    upper = int(min(255.0, 1.33 * v))
    edges = cv2.Canny(denoised, lower, upper)

    # 4. Final Dilated (Processed)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    
    # Optional: draw Hough lines as in image_processing.py
    lines = cv2.HoughLinesP(dilated, rho=1, theta=np.pi / 180, threshold=80, minLineLength=50, maxLineGap=20)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(dilated, (x1, y1), (x2, y2), 255, 3)

    # Create plot
    plt.figure(figsize=(15, 10))
    
    titles = ['1. Original Sketch', '2. CLAHE Contrast Enhancement', '3. Canny Edge Detection', '4. Final Morphological Stabilization']
    images = [original, enhanced, edges, dilated]
    cmaps = [None, 'gray', 'gray', 'gray']

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i], cmap=cmaps[i])
        plt.title(titles[i], fontsize=14)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    sample_img = r"d:\project\AI_Home_Project\uploads\d43ee8517c014ac18a25ea53bedd96d0_home.png"
    out_img = r"d:\project\AI_Home_Project\preprocessing_analysis.png"
    generate_viz(sample_img, out_img)
