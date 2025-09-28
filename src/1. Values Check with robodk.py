from robodk import robolink
import math

# ---------- helpers ----------
def deg(a):  # rad -> deg
    return a * 180.0 / math.pi

def wrap180(a_deg):  # keep angles in [-180, 180]
    a = (a_deg + 180.0) % 360.0 - 180.0
    # Use -180 instead of +180 for consistency with RoboDK panel
    return -180.0 if abs(a - 180.0) < 1e-9 else a

def euler_zyz_from_R(R):
    """
    Compute ZYZ Euler (RotZ1, RotY, RotZ2) from a 3x3 rotation matrix.
    Matches RoboDK panel's 'Rot(Z, Y’, Z’’)' convention.
    """
    r00, r01, r02 = R[0]
    r10, r11, r12 = R[1]
    r20, r21, r22 = R[2]

    # clip for acos numerical safety
    c = max(min(r22, 1.0), -1.0)
    beta = math.acos(c)                # around Y'
    sb = math.sin(beta)

    if sb > 1e-12:  # regular case
        alpha = math.atan2(r12, r02)   # first Z
        gamma = math.atan2(r21, -r20)  # second Z
    else:
        # singular: beta ~ 0 or ~ pi
        if r22 > 0:  # beta ~ 0
            beta = 0.0
            alpha = math.atan2(r10, r00)
            gamma = 0.0
        else:        # beta ~ pi
            beta = math.pi
            # when beta ≈ π the product Rz(alpha)*Ry(π)*Rz(gamma)
            # reduces to R ≈ Rz(alpha - gamma)*Ry(π)
            # choose gamma = 0 and recover alpha from XY block with sign flip
            alpha = math.atan2(-r10, -r00)
            gamma = 0.0

    # to degrees and wrap for readable match with panel
    return list(map(wrap180, (deg(alpha), deg(beta), deg(gamma))))

def R_from_euler_zyz(z1_deg, y_deg, z2_deg):
    """Rebuild rotation matrix from ZYZ angles (deg) for verification."""
    z1 = math.radians(z1_deg); y = math.radians(y_deg); z2 = math.radians(z2_deg)
    cz1, sz1 = math.cos(z1), math.sin(z1)
    cy,  sy  = math.cos(y),  math.sin(y)
    cz2, sz2 = math.cos(z2), math.sin(z2)

    # R = Rz(z1) * Ry(y) * Rz(z2)
    Rz1 = [[cz1, -sz1, 0],
           [sz1,  cz1, 0],
           [  0,    0, 1]]
    Ry  = [[ cy, 0, sy],
           [  0, 1,  0],
           [-sy, 0, cy]]
    Rz2 = [[cz2, -sz2, 0],
           [sz2,  cz2, 0],
           [  0,    0, 1]]

    # multiply Rz1*Ry
    A = [[sum(Rz1[i][k]*Ry[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    # multiply (Rz1*Ry)*Rz2
    R = [[sum(A[i][k]*Rz2[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return R

def max_abs_diff(R1, R2):
    return max(abs(R1[i][j] - R2[i][j]) for i in range(3) for j in range(3))

# ---------- main ----------
RDK = robolink.Robolink()
robot = RDK.Item('Doosan Robotics A0509', robolink.ITEM_TYPE_ROBOT)

pose = robot.Pose()  # 4x4 homogeneous matrix (R|t)

# translation
x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]

# rotation matrix (no numpy; read elements directly)
R = [[pose[i, j] for j in range(3)] for i in range(3)]

# ZYZ Euler like the RoboDK panel
z1_deg, y_deg, z2_deg = euler_zyz_from_R(R)

# verification: rebuild R from these angles and compare to the original
R_chk = R_from_euler_zyz(z1_deg, y_deg, z2_deg)
err = max_abs_diff(R, R_chk)

print("\n--- RAW POSE MATRIX (R|t) ---")
print(pose)

print("\n--- TRANSLATION (XYZ mm) ---")
print(f"X={x:.3f}, Y={y:.3f}, Z={z:.3f}")

print("\n--- ZYZ EULER (deg)  [matches RoboDK panel Rot(Z,Y’,Z’’)] ---")
print(f"RotZ1={z1_deg:.3f}, RotY={y_deg:.3f}, RotZ2={z2_deg:.3f}")

print("\n--- VERIFICATION ---")
print(f"max |R - R(ZYZ)| = {err:.3e}  (should be ~1e-12 to 1e-9)")
