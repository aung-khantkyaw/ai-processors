import time
import cv2
from models import AILoader
from processors import HumanDetector
from streams import StreamReceiver, StreamPublisher

MOTION_THRESHOLD = 1500 
POST_RECORD_SECONDS = 5.0 

def trigger_recording(cam_name, action):
    print(f"[{cam_name}] {action.upper()} RECORDING...")
    pass

def main():
    model, device, tag = AILoader.initialize()
    human_detector = HumanDetector(model, device, tag)

    receiver = StreamReceiver()
    urls = receiver.discover_cameras()

    if not urls:
        print("[System] No active camera streams found. Exiting...")
        return

    receiver.start_threads()
    publishers = {}

    ai_states = {}
    bg_subtractors = {}

    print("\n[System] Smart Event-Based AI Detection Started...")

    try:
        while True:
            frame_processed = False
            
            for cam_name in list(urls.keys()):
                if cam_name in receiver.camera_queues:
                    queue = receiver.camera_queues[cam_name]
                    
                    frame = None
                    while not queue.empty():
                        frame = queue.get()
                        frame_processed = True
                    
                    if frame is not None:
                        if cam_name not in ai_states:
                            ai_states[cam_name] = {"is_active": False, "last_human_time": 0}
                            bg_subtractors[cam_name] = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

                        state = ai_states[cam_name]
                        
                        if not state["is_active"]:
                            fg_mask = bg_subtractors[cam_name].apply(frame)
                            motion_score = cv2.countNonZero(fg_mask)

                            if motion_score > MOTION_THRESHOLD:
                                print(f"[{cam_name}] Motion Detected! Waking up AI...")
                                state["is_active"] = True
                                state["last_human_time"] = time.time()
                                trigger_recording(cam_name, "start")
                                processed_frame = human_detector.process(frame, cam_name)
                            else:
                                processed_frame = frame.copy()

                        else:
                            processed_frame = human_detector.process(frame, cam_name)
                            
                            current_boxes = human_detector.last_known_data.get(cam_name, [])

                            if len(current_boxes) > 0:
                                state["last_human_time"] = time.time()
                            else:
                                time_since_last_seen = time.time() - state["last_human_time"]
                                
                                if time_since_last_seen > POST_RECORD_SECONDS:
                                    print(f"[{cam_name}] Object cleared for 5s. Sleeping AI...")
                                    state["is_active"] = False
                                    trigger_recording(cam_name, "stop")

                        if cam_name not in publishers:
                            h, w, _ = processed_frame.shape
                            publishers[cam_name] = StreamPublisher(cam_name)
                            publishers[cam_name].start(width=w, height=h)

                        publishers[cam_name].push_frame(processed_frame)

            if not frame_processed:
                time.sleep(0.001) 

    except KeyboardInterrupt:
        print("\n[System] Stopping gracefully...")
    finally:
        receiver.stop()
        for pub in publishers.values():
            pub.stop()
        print("[System] Clean exit.")

if __name__ == "__main__":
    main()