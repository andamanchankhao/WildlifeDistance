import os
import urllib.request
from transformers import DPTForDepthEstimation, DPTImageProcessor

def download_models():
    # 1. Download yolov5s.pt if not present
    yolo_path = "yolov5s.pt"
    if not os.path.exists(yolo_path):
        print("Downloading yolov5s.pt...")
        url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
        urllib.request.urlretrieve(url, yolo_path)
        print("yolov5s.pt downloaded successfully.")
    else:
        print("yolov5s.pt already exists locally.")

    # 2. Download and save DPT model
    dpt_dir = "dpt-model"
    model_name = "Intel/dpt-hybrid-midas"
    if not os.path.exists(dpt_dir) or not os.listdir(dpt_dir):
        print(f"Downloading DPT model ({model_name})...")
        processor = DPTImageProcessor.from_pretrained(model_name)
        model = DPTForDepthEstimation.from_pretrained(model_name)
        
        print(f"Saving DPT model locally to '{dpt_dir}'...")
        os.makedirs(dpt_dir, exist_ok=True)
        processor.save_pretrained(dpt_dir)
        model.save_pretrained(dpt_dir)
        print("DPT model saved successfully.")
    else:
        print("DPT model folder already exists and is not empty.")

if __name__ == "__main__":
    download_models()
