import os
import sys

def check_ai_status():
    print("="*60)
    print("       AI COMPONENT STATUS CHECKER")
    print("="*60)
    
    # 1. CHECK CHATBOT (Gemini)
    print("\n[1] CHATBOT (Generative AI)")
    env_path = os.path.join("backend", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            content = f.read()
            if "GEMINI_API_KEY" in content and "AIza" in content:
                print("    [OK] API Key found in backend/.env")
                print("    -> Chatbot is ACTIVE.")
            else:
                print("    [WARNING] .env found but API Key seems missing or invalid.")
    else:
        print("    [MISSING] backend/.env file not found.")

    # 2. CHECK COST ESTIMATION (Machine Learning)
    print("\n[2] COST ESTIMATION (Random Forest)")
    cost_model = "cost_model.pkl"
    if os.path.exists(cost_model):
        print(f"    [OK] Model file '{cost_model}' found.")
        print("    -> Cost AI is ACTIVE.")
    else:
        print(f"    [MISSING] '{cost_model}' not found.")
        print("    -> System will use simple math fallback ($1500 * area).")

    # 3. CHECK VISION / LAYOUT (Deep Learning)
    print("\n[3] VISION / LAYOUT (U-Net Deep Learning)")
    # Check common locations
    dl_model_name = "wall_segmentation_model.pth"
    dl_paths = [
        dl_model_name,
        os.path.join("backend", dl_model_name)
    ]
    found_dl = False
    for p in dl_paths:
        if os.path.exists(p):
            print(f"    [OK] Deep Learning weights found at '{p}'.")
            found_dl = True
            break
    
    if found_dl:
        print("    -> Vision AI is READY. (Sketch -> 3D will use Neural Network)")
    else:
        print(f"    [MISSING] '{dl_model_name}' not found.")
        print("    -> Vision AI is INACTIVE. (Sketch -> 3D is using OpenCV/Math fallback)")

    print("\n" + "="*60)

if __name__ == "__main__":
    check_ai_status()
    input("\nPress Enter to exit...")
