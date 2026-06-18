import os
import shutil
from ultralytics import YOLO

def export_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    weights_dir = os.path.join(current_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True) 

    pt_model_path = os.path.join(weights_dir, "yolov8n.pt")
    final_target_path = os.path.join(weights_dir, "yolov8n_openvino_model")

    if not os.path.exists(pt_model_path):
        print(f"[System] {pt_model_path} not found.")
        print("[System] Downloading yolov8n.pt from Ultralytics servers...")
        
        tmp_model = YOLO("yolov8n.pt") 
        
        if os.path.exists("yolov8n.pt"):
            shutil.move("yolov8n.pt", pt_model_path)
            print(f"[System] Download Complete! Saved to {pt_model_path}")
        else:
            print("[System] Model loaded successfully.")

    print("\n[System] Loading model for export...")
    model = YOLO(pt_model_path)

    print("[System] Exporting model to OpenVINO FP16 format...")
    exported_path = model.export(format="openvino", half=True, verbose=False)
    
    print(f"[System] YOLO Exported to: {exported_path}")

    if os.path.abspath(exported_path) != os.path.abspath(final_target_path):
        if os.path.exists(final_target_path):
            shutil.rmtree(final_target_path)
        shutil.move(exported_path, final_target_path)

    print(f"\n[System] SUCCESS: Everything is ready!")
    print(f"[System] Model Directory: {final_target_path}")
    print(f"[System] Inside files: {os.listdir(final_target_path)}")

if __name__ == "__main__":
    export_model()