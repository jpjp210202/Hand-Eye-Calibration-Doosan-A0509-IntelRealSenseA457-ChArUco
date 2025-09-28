"""
0_capture_intrinsics_images.py
Capture images of ChArUco board for intrinsic calibration (no robot poses needed).
Press 'c' to save image, 'q' to quit.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time

# === CONFIG ===
SAVE_DIR = r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data"
IMG_DIR = os.path.join(SAVE_DIR, "intrinsics_images")  # separate folder for intrinsics
os.makedirs(IMG_DIR, exist_ok=True)

# === INIT REALSENSE ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

print("🤖 Ready for intrinsic calibration image capture")
print("Press 'c' to capture an image, 'q' to quit.")

counter = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        cv2.imshow("Intrinsic Capture", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            img_name = f"calib_{counter:03d}.png"
            img_path = os.path.join(IMG_DIR, img_name)
            cv2.imwrite(img_path, color_image)
            print(f"✅ Saved {img_name}")
            counter += 1
            time.sleep(0.5)  # debounce

        elif key == ord('q'):
            print("👋 Exiting capture")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
