# AI Processors - Object Detection with YOLOv8n

This repository contains code for an AI processor that performs object detection using the YOLOv8n model.  The project includes scripts for exporting the model, loading it for inference, and configuration settings. 

## export_model.py
This script is responsible for exporting the YOLOv8n model to a format suitable for deployment. It uses the Ultralytics YOLO library to load the model and save it in the desired format.

## ai_loader.py
This script is responsible for loading the YOLOv8n model for inference. It supports loading the model on different devices such as NVIDIA GPU (CUDA) and Intel OpenVINO. 

