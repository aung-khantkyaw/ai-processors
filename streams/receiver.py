import cv2
import threading
import queue
import requests
from config import MEDIAMTX_HOST, RTSP_PORT, API_PORT

class StreamReceiver:
    def __init__(self):
        self.active_urls = {}
        self.camera_queues = {}
        self.running = True
        self.threads = []

    def discover_cameras(self):
        api_url = f"http://{MEDIAMTX_HOST}:{API_PORT}/v3/paths/list"
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                paths = response.json().get("items", [])
                for path in paths:
                    name = path.get("name")
                    if path.get("ready", False):
                        cam_title = f"{name.upper()}"
                        self.active_urls[cam_title] = f"rtsp://{MEDIAMTX_HOST}:{RTSP_PORT}/{name}"
                print(f"[Receiver] Found {len(self.active_urls)} active streams.")
            else:
                print(f"[Receiver] API Error: {response.status_code}")
        except Exception as e:
            print(f"[Receiver] Connection Error: {e}")
        return self.active_urls

    def _camera_worker(self, cam_name, rtsp_url):
        print(f"[{cam_name}] Connecting to {rtsp_url}...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera_queues[cam_name] = queue.Queue(maxsize=1)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"[{cam_name}] Stream lost.")
                break
            if self.camera_queues[cam_name].full():
                try:
                    self.camera_queues[cam_name].get_nowait()
                except queue.Empty:
                    pass
            self.camera_queues[cam_name].put(frame)
        cap.release()

    def start_threads(self):
        for cam_name, url in self.active_urls.items():
            t = threading.Thread(target=self._camera_worker, args=(cam_name, url), daemon=True)
            self.threads.append(t)
            t.start()

    def stop(self):
        self.running = False