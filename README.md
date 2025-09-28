# 🤖 Hand–Eye Calibration – Doosan A0509 + Intel RealSense A457/D435 + ChArUco

## 📌 Introduction

This repository documents the complete workflow for **hand–eye calibration** of the **Doosan A0509 collaborative robot** with an **Intel RealSense depth camera (D435 and A457 tested)**, using **ChArUco calibration boards**.
The project was carried out at the **Central Manufacturing Technology Institute (CMTI), Bangalore** as part of the **M.Tech Robotics & AI program at Ramaiah Institute of Technology (MSRIT)**.

* Executed individually by **JP Jaipuneeth** through self-learning (OpenCV tutorials, research papers, YouTube references).
* Completed as part of an **academic + industry collaboration portfolio project**.
* **RoboDK** was used for both simulation and real robot control.

## ⚙️ Hardware Setup

* **Robot**: Doosan A0509 collaborative robot
* **Camera**: Intel RealSense D435 and A457
* **Calibration Target**: Multiple ChArUco boards tested

  * 5×7 squares, 30 mm square length, 22 mm marker length (DICT_4X4_50)
  * Additional A4/A3 ChArUco boards also validated
* **Control**: RoboDK Python API for simulation and execution on physical robot

## 🖥️ Software Requirements

* Python **3.10**
* OpenCV contrib **4.7.0.72** (for ArUco/ChArUco support)
* NumPy **1.26.4** (NumPy 2.x causes ABI issues with OpenCV)
* SciPy (for rotation/Euler conversions)
* pyrealsense2 (Intel RealSense SDK for Python)
* robodk (RoboDK Python API)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## 📂 Repository Structure

Hand-Eye-Calibration-Doosan-A0509-IntelRealSenseA457-ChArUco/
│
├── src/ # All Python scripts
│ ├── 1.* # Camera intrinsics & value checks
│ ├── 2.* # Image capture for calibration
│ ├── 3.* # Calibration computation
│ ├── 4.* # Updated calibration, math checks, validation
│ ├── 4.6. Check_Calib.py # ✅ Final calibration verification
│ ├── 4.9. LiveBoardPose_ID_Basis.py # ✅ Final pose validation per ID
│ ├── 5.* # Additional calibration ID testing
│ └── Test_Code_jpjp.py # Experimental/test script
│
├── data/
│ ├── images/ # Captured images for calibration
│ ├── intrinsics_images/ # Images for intrinsic calibration
│ ├── poses/ # Robot base→tool poses & board poses
│ ├── board_pose.txt # Saved rvec/tvec of board in camera frame
│ ├── pose_live.txt # Current live robot base→tool pose
│ ├── camera_intrinsics.npz # Intrinsic calibration (fx, fy, cx, cy, dist)
│ └── handeye_T_tool_cam.txt # Final hand–eye transform (tool→camera)
│
├── docs/
│ ├── ChArUco_297x210_5x7_30_22_DICT_4X4.pdf
│ ├── ChArUco_420x297_6x9_30_22_DICT_4X4.pdf
│ └── Project notes & calibration references
│
├── robodk/ # RoboDK project files (.rdk)
├── cad/ # CAD models & drill files used
├── requirements.txt
└── README.md

## 📜 Python Scripts – Purpose

**1.x Series – Intrinsics & Checks**

* `1. Values Check with robodk.py` → Verify basic matrix values from RoboDK
* `1.25. Capture Cam Intrinsics.py` → Capture images for intrinsic calibration
* `1.251. Check Intrinsics Images.py` → Validate captured intrinsics images
* `1.5. Calib_intrinsics.py` → Compute camera intrinsics and save `camera_intrinsics.npz`

**2.x Series – Image Capture**

* `2. Images Capture.py` → Capture calibration images (ChArUco) with robot poses
* `2. Updated Images Capture.py` → Refined image capture with pose synchronization

**3.x Series – Calibration**

* `3. Calibration.py` → Run calibration using captured images + robot poses

**4.x Series – Updated Calibration, Math Checks, Validation**

* `4. Updated Calibration.py` → Improved calibration solver with corrections
* `4.5. Maths Check.py` → Debug script for matrix consistency checks
* `4.6. Check_Calib.py` → Final calibration verification script (loads `pose_live.txt`, `handeye_T_tool_cam.txt`, `board_pose.txt`, computes predicted [x,y,z] of ChArUco board in robot base frame, validates accuracy against ground truth)
* `4.7. Detection_SaveBoardPose.py` → Detect board & save rvec/tvec to `board_pose.txt`
* `4.8. SaveLive&BoardPose.py` → Save both robot pose (`pose_live.txt`) + board pose
* `4.9. LiveBoardPose_ID_Basis.py` → Final pose validation per ArUco ID (map ArUco IDs to keys; on key press print predicted [x,y,z] position of that ID in robot base frame)

**5.x Series – Extra Tests**

* `5. Checking Calibration IDs.py` → Extra test of calibration correctness per ID

**Test / Experimental**

* `Test_Code_jpjp.py` → Internal test script (scratchpad)

## 🚀 Workflow

1. **Capture Camera Intrinsics**

   * Run `1.25. Capture Cam Intrinsics.py` → take calibration images
   * Run `1.5. Calib_intrinsics.py` → save `camera_intrinsics.npz`

2. **Capture Robot Poses & Images**

   * Run `2. Images Capture.py` → sync images + robot poses
   * Data saved in `data/images/` and `data/poses/`

3. **Run Hand–Eye Calibration**

   * Run `3. Calibration.py` or `4. Updated Calibration.py`
   * Saves `handeye_T_tool_cam.txt`

4. **Verification – Final Stage**

   * Run `4.6. Check_Calib.py` → confirm calibration validity
   * Run `4.9. LiveBoardPose_ID_Basis.py` → test any ArUco ID, get predicted [x,y,z]

## 📊 Data Files Explained

* `camera_intrinsics.npz` → Focal lengths, principal point, distortion coefficients
* `handeye_T_tool_cam.txt` → 4×4 matrix: tool → camera transform
* `pose_live.txt` → Current robot base→tool homogeneous transform
* `board_pose.txt` → Current ChArUco board pose (from detection)
* `poses/pose_*.txt` → Saved robot poses for calibration dataset
* `poses/charuco_*.npz` → Detected ChArUco corners per image

## ✅ Final Notes

The last two scripts to run are:

* `4.6. Check_Calib.py` → Check calibration correctness
* `4.9. LiveBoardPose_ID_Basis.py` → Validate any marker ID pose

This repository is maintained as part of an **academic + industry collaboration portfolio project**, showcasing calibration of industrial cobots with vision systems.

✍️ **Author**: JP Jaipuneeth
📍 **CMTI, Bangalore — M.Tech Robotics & AI (MSRIT)**
