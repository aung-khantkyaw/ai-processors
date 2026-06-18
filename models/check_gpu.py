import openvino as ov

def check_openvino_devices():
    core = ov.Core()

    devices = core.available_devices
    
    print("\n" + "="*40)
    print("=== OpenVINO Available Devices ===")
    print("="*40)
    
    for device in devices:
        try:
            device_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"Device: {device:<5} | Name: {device_name}")
        except Exception:
            print(f"Device: {device:<5} | (Could not fetch full name)")
            
    print("="*40 + "\n")

if __name__ == "__main__":
    check_openvino_devices()