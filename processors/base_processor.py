from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    def __init__(self, skip_frames=3):
        self.skip_frames = skip_frames
        self.frame_counters = {}
        self.last_known_data = {}

    @abstractmethod
    def process(self, frame, camera_id):
        pass