import time
from models import AILoader
from processors import HumanDetector
from streams import StreamReceiver, StreamPublisher


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

    print("\n[System] Headless AI Detection Started. Press 'Ctrl+C' in terminal to quit.")

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
                        processed_frame = human_detector.process(frame, cam_name)

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
        # cv2.destroyAllWindows()
        print("[System] Clean exit.")


if __name__ == "__main__":
    main()