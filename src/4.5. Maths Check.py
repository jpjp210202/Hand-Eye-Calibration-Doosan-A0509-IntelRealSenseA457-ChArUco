import numpy as np

# Load your hand–eye result
T_tool_cam = np.loadtxt(r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data\handeye_T_tool_cam.txt")

# Print it
print("Hand–Eye Transform (Tool → Camera):\n", T_tool_cam)

# --- 1. Check orientation sanity ---
R = T_tool_cam[:3,:3]
det = np.linalg.det(R)
print("Determinant (should be ~+1):", det)

# Column norms
print("Column norms:", [np.linalg.norm(R[:,i]) for i in range(3)])

# --- 2. Show camera axes in tool frame ---
x_axis = R[:,0]   # camera X direction in tool frame
y_axis = R[:,1]   # camera Y direction in tool frame
z_axis = R[:,2]   # camera Z direction in tool frame

print("Camera X axis in tool frame:", x_axis)
print("Camera Y axis in tool frame:", y_axis)
print("Camera Z axis in tool frame:", z_axis)

# --- 3. Show translation ---
t = T_tool_cam[:3,3]
print("Camera origin wrt tool TCP (mm):", t)
