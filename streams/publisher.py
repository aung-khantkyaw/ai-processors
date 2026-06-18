import subprocess
from config import MEDIAMTX_HOST, RTSP_PORT

class StreamPublisher:
    def __init__(self, camera_name):
        self.publish_url = f"rtsp://{MEDIAMTX_HOST}:{RTSP_PORT}/{camera_name.lower()}_ai"
        self.process = None

    def start(self, width, height, fps=15):
        command = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f"{width}x{height}", '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-bf', '0',              
            '-g', str(fps),          
            '-sc_threshold', '0',    
            '-fflags', 'nobuffer',   
            '-flags', 'low_delay',    
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp', self.publish_url
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        print(f"[Publisher] Publishing Processed Stream to: {self.publish_url}")

    def push_frame(self, frame):
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(frame.tobytes())
            except Exception as e:
                pass

    def stop(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()