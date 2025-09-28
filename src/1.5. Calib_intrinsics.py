"""
1_calib_intrinsics.py
Calibrate Intel RealSense D435 intrinsics using a ChArUco board.
Outputs: camera_intrinsics.npz (cameraMatrix, distCoeffs)
"""

import cv2
import numpy as np
import os
import glob

# === PATHS ===
SAVE_DIR = r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data"
IMG_DIR = r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data\intrinsics_images"
out_file = os.path.join(SAVE_DIR, "camera_intrinsics.npz")

# === ChArUco Board Definition ===
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CHARUCO_BOARD = cv2.aruco.CharucoBoard(
    (7, 5), 0.03, 0.021, ARUCO_DICT  # (squaresX, squaresY, squareLength, markerLength)
)

# Detector Parameters
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# === Collect calibration data ===
all_corners = []
all_ids = []
imsize = None

images = glob.glob(os.path.join(IMG_DIR, "*.png"))
print(f"Found {len(images)} images for calibration")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    if ids is not None and len(ids) > 0:
        # Refine and interpolate ChArUco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, CHARUCO_BOARD
        )

        if retval > 10:  # need at least 10 corners for good calibration
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            imsize = gray.shape[::-1]

print(f"Collected {len(all_corners)} valid detections out of {len(images)} images")

if len(all_corners) < 10:
    raise RuntimeError("❌ Not enough valid ChArUco detections! Capture more images.")

# === Run calibration ===
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    all_corners, all_ids, CHARUCO_BOARD, imsize, None, None
)

print("\n✅ Calibration Successful!" if ret else "\n⚠️ Calibration Failed")
print("Camera Matrix:\n", cameraMatrix)
print("Distortion Coeffs:\n", distCoeffs.ravel())

# === Save intrinsics ===
np.savez(out_file, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
print(f"\n💾 Saved intrinsics to: {out_file}")
