import cv2
from processors import BaseProcessor
from utils import Visualizer

class HumanDetector(BaseProcessor):
    def __init__(self, model, device, tag, skip_frames=3):
        super().__init__(skip_frames)
        self.model = model
        self.device = device
        self.tag = tag

    def process(self, img, camera_id):
        if camera_id not in self.frame_counters:
            self.frame_counters[camera_id] = 0
            self.last_known_data[camera_id] = []

        self.frame_counters[camera_id] += 1

        # Frame Skipping Logic
        if self.frame_counters[camera_id] % self.skip_frames != 0:
            Visualizer.draw_predictions(img, self.last_known_data[camera_id])
            self._draw_info(img, camera_id)
            return img

        # AI Prediction Logic
        results = self.model(img, stream=True, verbose=False, classes=[0], device=self.device)
        current_boxes = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf > 0.5:
                    current_boxes.append((x1, y1, x2, y2, conf))

        Visualizer.draw_predictions(img, current_boxes)
        self.last_known_data[camera_id] = current_boxes
        self._draw_info(img, camera_id)

        return img

    def _draw_info(self, img, camera_id):
        cv2.putText(img, f"{camera_id} {self.tag}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)