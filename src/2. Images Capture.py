"""
capture.py
Synchronize Doosan A0509 robot poses (via RoboDK) with Intel RealSense D455/D457 images
and save them for hand–eye calibration with ChArUco board overlay.

Author: JP (CMTI Projects)
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from robodk import robolink  # RoboDK API
import os
import time

# === CONFIG ===
SAVE_DIR = r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data"
IMG_DIR = os.path.join(SAVE_DIR, "images")
POSE_DIR = os.path.join(SAVE_DIR, "poses")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(POSE_DIR, exist_ok=True)

# === ChArUco board definition ===
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CHARUCO_BOARD = cv2.aruco.CharucoBoard(
    (7, 5),        # squaresX (horizontal), squaresY (vertical)
    0.03,          # squareLength in meters (30 mm)
    0.021,         # markerLength in meters (22 mm)
    ARUCO_DICT
)

# Detector parameters (tuned for reliability)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_PARAMS.adaptiveThreshWinSizeMin = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax = 23
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 10
ARUCO_PARAMS.minMarkerPerimeterRate = 0.02
ARUCO_PARAMS.maxMarkerPerimeterRate = 4.0
ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# === INIT ROBO DK ===
RDK = robolink.Robolink()
robot = RDK.Item('', robolink.ITEM_TYPE_ROBOT)

# === INIT REALSENSE ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # max quality mode
pipeline.start(config)

print("🤖 Ready! Press 'c' to capture, 'q' to quit.")

counter = 0

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # === Detect ArUco markers ===
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            # Interpolate ChArUco corners
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, CHARUCO_BOARD
            )

            if retval > 0:
                cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids)
                cv2.putText(color_image, f"ChArUco Detected ({retval} corners)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(color_image, f"AruCo Detected ({len(ids)} markers)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            cv2.putText(color_image, "No markers",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Show live feed with overlay
        cv2.imshow("RealSense + ChArUco", color_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # === GET ROBOT POSE ===
            pose_mat = robot.Pose()
            pose_array = np.array(pose_mat)  # ensure 4x4 matrix

            # === SAVE IMAGE ===
            img_name = f"img_{counter:03d}.png"
            img_path = os.path.join(IMG_DIR, img_name)
            cv2.imwrite(img_path, color_image)

            # === SAVE POSE ===
            pose_name = f"pose_{counter:03d}.txt"
            pose_path = os.path.join(POSE_DIR, pose_name)
            np.savetxt(pose_path, pose_array, fmt="%.6f")

            # === SAVE CHARUCO DETECTIONS ===
            if retval > 0:
                np.savez(os.path.join(POSE_DIR, f"charuco_{counter:03d}.npz"),
                         corners=charuco_corners, ids=charuco_ids)

            print(f"✅ Captured #{counter}: {img_name} + {pose_name}")
            counter += 1
            time.sleep(0.5)  # debounce

        elif key == ord('q'):
            print("👋 Exiting...")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
