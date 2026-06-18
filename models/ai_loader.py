import os
import numpy as np
import torch
from ultralytics import YOLO
from config import OPENVINO_MODEL_PATH, PYTORCH_MODEL_PATH, AI_DEVICE_MODE


class AILoader:

    @staticmethod
    def _load_nvidia_cuda():
        """NVIDIA GPU (CUDA) စနစ်ဖြင့် မော်ဒယ်ကို တင်မည့် Function"""
        if not os.path.exists(PYTORCH_MODEL_PATH):
            print(f"[System] Error: PyTorch model not found at {PYTORCH_MODEL_PATH}")
            return None, None, None
        try:
            print("[System] Loading PyTorch Model on NVIDIA GPU (CUDA)...")
            model = YOLO(PYTORCH_MODEL_PATH)

            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(dummy_img, device="cuda", verbose=False)

            print("[System] SUCCESS: NVIDIA CUDA Engine Initialized.")
            return model, "cuda", "[NVIDIA CUDA]"
        except Exception as e:
            print(f"[System] NVIDIA CUDA Initialization Failed: {e}")
            return None, None, None

    @staticmethod
    def _load_intel_openvino():
        """Intel GPU (OpenVINO) စနစ်ဖြင့် မော်ဒယ်ကို တင်မည့် Function"""
        if not os.path.exists(OPENVINO_MODEL_PATH):
            print(f"[System] Error: OpenVINO model not found at {OPENVINO_MODEL_PATH}")
            return None, None, None
        try:
            print("[System] Injecting Intel GPU Patch into OpenVINO Engine...")
            import openvino as ov

            _orig_compile_model = ov.Core.compile_model

            def patched_compile_model(self, ov_model, *args, **kwargs):
                new_kwargs = kwargs.copy()
                new_args = list(args)

                if len(new_args) > 0:
                    new_args[0] = "GPU"
                else:
                    new_kwargs["device_name"] = "GPU"

                return _orig_compile_model(self, ov_model, *new_args, **new_kwargs)

            ov.Core.compile_model = patched_compile_model

            print("[System] Loading Intel OpenVINO Engine on Intel GPU...")
            model = YOLO(OPENVINO_MODEL_PATH, task="detect")

            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(dummy_img, device="cpu", verbose=False)

            print("[System] SUCCESS: OpenVINO Engine Initialized on Intel GPU.")
            return model, "cpu", "[OpenVINO GPU]"
        except Exception as e:
            print(f"[System] Intel OpenVINO Initialization Failed: {e}")
            return None, None, None

    @staticmethod
    def _load_pytorch_cpu():
        """PyTorch CPU စနစ်ဖြင့် မော်ဒယ်ကို တင်မည့် Function"""
        if not os.path.exists(PYTORCH_MODEL_PATH):
            print(f"[System] CRITICAL ERROR: PyTorch model not found at {PYTORCH_MODEL_PATH}")
            exit()
        try:
            print("[System] Loading PyTorch CPU Model...")
            model = YOLO(PYTORCH_MODEL_PATH)
            print("[System] SUCCESS: PyTorch CPU Engine Initialized.")
            return model, "cpu", "[PyTorch CPU]"
        except Exception as e:
            print(f"[System] CRITICAL ERROR: Could not load PyTorch CPU model. {e}")
            exit()

    @staticmethod
    def initialize():
        print(f"\n[System] AI Device Mode Requested: {AI_DEVICE_MODE}")

        if AI_DEVICE_MODE == "GPU":
            if torch.cuda.is_available():
                model, device, tag = AILoader._load_nvidia_cuda()
                if model is not None:
                    return model, device, tag

            print("[System] NVIDIA GPU not available. Automatically switching to Intel GPU...")

            model, device, tag = AILoader._load_intel_openvino()
            if model is not None:
                return model, device, tag

            print("[System] CRITICAL ERROR: Both NVIDIA and Intel GPU initializations failed. Exiting...")
            exit()

        elif AI_DEVICE_MODE == "CPU":
            return AILoader._load_pytorch_cpu()

        else:
            print("[System] Auto-detecting best available hardware...")
            if torch.cuda.is_available():
                model, device, tag = AILoader._load_nvidia_cuda()
                if model is not None:
                    return model, device, tag

            model, device, tag = AILoader._load_intel_openvino()
            if model is not None:
                return model, device, tag

            print("[System] Falling back to CPU...")
            return AILoader._load_pytorch_cpu()