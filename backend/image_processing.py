import cv2
import os

def preprocess_image(input_path):
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    output_path = input_path.replace("uploads", "processed")
    cv2.imwrite(output_path, edges)

    return output_path
