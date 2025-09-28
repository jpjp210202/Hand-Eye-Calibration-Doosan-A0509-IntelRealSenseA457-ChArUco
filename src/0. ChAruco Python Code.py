import cv2
import cv2.aruco as aruco

# --- Board parameters ---
squares_x = 9      # number of squares along X (width)
squares_y = 6      # number of squares along Y (height)
square_length = 30 # in mm
marker_length = 24 # in mm

# Use 4x4 dictionary (50 markers available)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# Create ChArUco board
board = aruco.CharucoBoard_create(
    squaresX=squares_x,
    squaresY=squares_y,
    squareLength=square_length,
    markerLength=marker_length,
    dictionary=aruco_dict
)

# Render the board at high resolution for printing
img = board.draw((2480, 3508))  # A3 size at ~300 DPI

# Save as PNG (or use cv2.imwrite for jpg)
cv2.imwrite("Charuco_A3.png", img)

# Optional: also save as PDF (needs matplotlib)
import matplotlib.pyplot as plt
plt.figure(figsize=(11.7, 16.5))  # A3 in inches at 300 DPI
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.savefig("Charuco_A3.pdf", bbox_inches='tight', pad_inches=0)
plt.close()

print("✅ ChArUco board saved as Charuco_A3.png and Charuco_A3.pdf")
