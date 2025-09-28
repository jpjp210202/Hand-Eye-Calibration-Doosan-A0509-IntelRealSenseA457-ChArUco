"""
capture.py
Synchronize Doosan A0509 robot poses (via RoboDK) with Intel RealSense D435
and save them for hand–eye calibration with ChArUco board overlay.

Author: JP (CMTI Projects)
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from robodk import robolink
import os
import time

# === CONFIG ===
SAVE_DIR = r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data"
IMG_DIR = os.path.join(SAVE_DIR, "images")
POSE_DIR = os.path.join(SAVE_DIR, "poses")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(POSE_DIR, exist_ok=True)

# === LOAD CAMERA INTRINSICS ===
intrinsics_file = os.path.join(SAVE_DIR, "camera_intrinsics.npz")
if not os.path.exists(intrinsics_file):
    raise FileNotFoundError("❌ Run intrinsic calibration first! Missing camera_intrinsics.npz")

intrinsics = np.load(intrinsics_file)
cameraMatrix = intrinsics["cameraMatrix"]
distCoeffs = intrinsics["distCoeffs"]

# === ChArUco board definition ===
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CHARUCO_BOARD = cv2.aruco.CharucoBoard(
    (7, 5), 0.030, 0.021, ARUCO_DICT  # squaresX, squaresY, squareLength, markerLength
)

ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# === INIT ROBO DK ===
RDK = robolink.Robolink()
robot = RDK.Item('', robolink.ITEM_TYPE_ROBOT)

# === INIT REALSENSE ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

print("🤖 Ready! Press 'c' to capture, 'q' to quit.")

counter = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # === Detect ArUco markers ===
        # === Detect ArUco markers ===
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

        rvec, tvec = None, None

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            # Step 1: Interpolate ChArUco corners
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, CHARUCO_BOARD
            )

            if retval > 10:  # at least 10 corners
                # Step 2: Estimate pose using those corners
                success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, CHARUCO_BOARD, cameraMatrix, distCoeffs, None, None
                )

                if success:
                    cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids)
                    cv2.drawFrameAxes(color_image, cameraMatrix, distCoeffs, rvec, tvec, 0.05)
                    cv2.putText(color_image, f"ChArUco Pose OK ({retval} corners)",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(color_image, f"AruCo only ({len(ids)} markers)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        else:
            cv2.putText(color_image, "No markers",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Show live feed
        cv2.imshow("RealSense + ChArUco", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # === FORCE ROBO DK REFRESH ===
            RDK.Render()
            time.sleep(0.2)  # small buffer to ensure pose update

            # === GET ROBOT POSE ===
            pose_mat = robot.Pose()
            pose_array = np.array([[pose_mat[i, j] for j in range(4)] for i in range(4)])

            # === SAVE IMAGE ===
            img_name = f"img_{counter:03d}.png"
            cv2.imwrite(os.path.join(IMG_DIR, img_name), color_image)

            # === SAVE ROBOT POSE ===
            pose_path = os.path.join(POSE_DIR, f"pose_{counter:03d}.txt")
            np.savetxt(pose_path, pose_array, fmt="%.6f")

            # === SAVE BOARD POSE (if valid) ===
            if rvec is not None and tvec is not None:
                board_path = os.path.join(POSE_DIR, f"boardpose_{counter:03d}.npz")
                np.savez(board_path, rvec=rvec, tvec=tvec)
                print(f"✅ Captured #{counter}: {img_name}, {pose_path}, {board_path}")
            else:
                print(f"⚠️ Captured #{counter}: {img_name}, {pose_path} (no board pose)")

            counter += 1


        elif key == ord('q'):
            print("👋 Exiting...")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
