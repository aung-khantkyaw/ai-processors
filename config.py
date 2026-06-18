import os
from dotenv import load_dotenv

load_dotenv()

MEDIAMTX_HOST = os.getenv("MEDIAMTX_HOST", "localhost")
RTSP_PORT = os.getenv("RTSP_PORT", "8554")
API_PORT = os.getenv("API_PORT", "9997")

OPENVINO_MODEL_PATH = "models/weights/yolov8n_openvino_model/"
PYTORCH_MODEL_PATH = "models/weights/yolov8n.pt"

AI_DEVICE_MODE = os.getenv("AI_DEVICE_MODE", "AUTO").upper()