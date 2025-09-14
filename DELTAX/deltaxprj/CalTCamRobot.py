import cv2
import cv2.aruco as aruco
import numpy as np
import os
import json
import time

def save_matrix(file_path, matrix):
    data = matrix.tolist()
    with open(file_path, 'w') as f:
        json.dump({"T_camera_to_robot": data}, f, indent=4)


def create_custom_board_for_detection(marker_size=40, marker_ids=[0, 8, 16, 32], 
                                      dictionary_type=aruco.DICT_4X4_50):
    """
    Create a cv2.aruco.Board object that matches the printed custom board layout (A4 with 4 markers at corners).
    """
    # A4 paper in mm
    a4_width = 297
    a4_height = 210
    offset = 20  # margin from edges
    
    # Get the dictionary
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    
    # Marker corner layout (top-left, top-right, bottom-right, bottom-left)
    base_corners = np.array([
        [0, 0, 0],                     # top-left of marker
        [marker_size, 0, 0],           # top-right
        [marker_size, -marker_size, 0],# bottom-right (y negative)
        [0, -marker_size, 0]           # bottom-left
    ], dtype=np.float32)

    obj_points = []  # list of 4x3 arrays (each marker's 3D corners)
    ids = []         # marker ids

    # Marker positions on A4
    marker_positions = [
        (-a4_width/2 + offset,  a4_height/2 - offset),                       # top-left
        ( a4_width/2 - offset - marker_size, a4_height/2 - offset),          # top-right
        (-a4_width/2 + offset, -a4_height/2 + offset + marker_size),         # bottom-left (fix)
        ( a4_width/2 - offset - marker_size, -a4_height/2 + offset + marker_size)  # bottom-right
    ]

    for i, pos in enumerate(marker_positions):
        x, y = pos
        corner_3d = base_corners + np.array([x, y, 0])
        obj_points.append(corner_3d)
        ids.append(marker_ids[i])

    # Convert to correct format
    obj_points = [np.array(c, dtype=np.float32) for c in obj_points]
    ids = np.array(ids, dtype=np.int32)
    # sau khi obj_points tạo xong
    all_pts = np.vstack(obj_points)   # shape (16,3)
    print("centroid (x,y) =", all_pts[:,:2].mean(axis=0))
    print("min/max x,y =", all_pts[:,:2].min(axis=0), all_pts[:,:2].max(axis=0))
    
    # Create the board directly
    board = aruco.Board(objPoints=obj_points, dictionary=aruco_dict, ids=ids)
    all_pts = np.vstack(obj_points)
    print("example object points (first marker):", obj_points[0])
    print("object y min/max:", all_pts[:,1].min(), all_pts[:,1].max())
    return board, aruco_dict, ids, obj_points

def rvec_tvec_to_matrix(rvec, tvec):
    """Converts rvec, tvec into a 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    angle = np.linalg.norm(rvec) * 180 / np.pi  # độ
    print(f"Rotation angle (deg): {angle:.2f}")
    print("X axis board in camera frame:", R[:,0])
    print("Y axis board in camera frame:", R[:,1])
    print("Z axis board in camera frame:", R[:,2])
    normal = R[:,2]
    height = float(np.dot(tvec.flatten(), normal))
    print("height along board normal (mm):", height)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def invert_transform(T):
    """
    Inverts a 4x4 homogeneous transformation matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    R_inv = R.T
    t_inv = -R_inv @ t

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv

def load_transformation_matrix(json_path):
    """
    Load transformation matrix from JSON file.

    Args:
        json_path (str): Path to the JSON file containing T_board_to_robot.

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Nếu JSON lưu dưới dạng list lồng list
    matrix = np.array(data["transformation_matrix"], dtype=float)
    return matrix

def load_camera_calibration(json_path):
    """
    Load camera matrix và distortion coefficients từ file JSON.

    Args:
        json_path (str): Đường dẫn tới file JSON.

    Returns:
        tuple: (camera_matrix, dist_coeffs) dưới dạng numpy.ndarray
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)

    return camera_matrix, dist_coeffs

def camera_setup(cap, parameters):
    width, height = 1280, 1024
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Tắt auto exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)      # Giá trị exposure

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)         # Tắt auto focus
    cap.set(cv2.CAP_PROP_FOCUS, 160)           # Giá trị focus

    cap.set(cv2.CAP_PROP_GAIN, 192)            # Gain
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 220)     # Brightness

    # Một số camera không cho chỉnh sharpness qua OpenCV, nhưng nếu được:
    cap.set(cv2.CAP_PROP_SHARPNESS, 1671.4)    # Sharpness   # 0 = focus gần, tăng số = focus xa (tùy cam hỗ trợ)

    # ----- Khóa white balance -----
    cap.set(cv2.CAP_PROP_AUTO_WB, 0) 
    
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.03
    parameters.minCornerDistanceRate = 0.05
    parameters.minMarkerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    parameters.minOtsuStdDev = 5.0
    parameters.perspectiveRemovePixelPerCell = 4
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
    parameters.maxErroneousBitsInBorderRate = 0.35
    parameters.minOtsuStdDev = 5.0
    parameters.errorCorrectionRate = 0.6
    
    return cap, parameters

def average_pose(T_list):
    """
    T_list: list các ma trận 4x4 (numpy)
    return: T_averaged (4x4)
    """

    # --- 1. average translation ---
    translations = [T[:3, 3] for T in T_list]
    t_avg = np.mean(translations, axis=0)

    # --- 2. average rotation using quaternion method ---
    quats = []
    for T in T_list:
        R = T[:3,:3]
        q = rot_to_quat(R)   # convert rotation matrix to quaternion
        quats.append(q / np.linalg.norm(q))  # normalize

    quats = np.array(quats)
    # build matrix for averaging
    A = np.zeros((4,4))
    for q in quats:
        A += np.outer(q,q)
    A /= len(quats)

    # lấy eigenvector lớn nhất
    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    q_avg /= np.linalg.norm(q_avg)
    R_avg = quat_to_rot(q_avg)

    # --- 3. ghép lại ---
    T_avg = np.eye(4)
    T_avg[:3,:3] = R_avg
    T_avg[:3,3] = t_avg
    return T_avg

def rot_to_quat(R):
    """Chuyển rotation matrix sang quaternion [x,y,z,w]"""
    q = np.zeros(4)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2,1] - R[1,2]) * s
        q[1] = (R[0,2] - R[2,0]) * s
        q[2] = (R[1,0] - R[0,1]) * s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = 2.0*np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            q[3] = (R[2,1] - R[1,2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = 2.0*np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            q[3] = (R[0,2] - R[2,0]) / s
            q[0] = (R[0,1] + R[1,0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0*np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            q[3] = (R[1,0] - R[0,1]) / s
            q[0] = (R[0,2] + R[2,0]) / s
            q[1] = (R[1,2] + R[2,1]) / s
            q[2] = 0.25 * s
    return q

def quat_to_rot(q):
    """Chuyển quaternion [x,y,z,w] sang rotation matrix"""
    x,y,z,w = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])
    return R

# Hàm tính reprojection error trung bình trên tất cả corner
def compute_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    """
    object_points: (N,3) ndarray
    image_points: (N,2) ndarray
    rvec, tvec: pose từ solvePnP
    camera_matrix, dist_coeffs: calib của camera

    return: (mean_error_px, per_corner_errors)
    """
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1,2)
    errors = np.linalg.norm(projected - image_points.reshape(-1,2), axis=1)
    return float(np.mean(errors)), errors


# Hàm lọc frame xấu
def filter_good_frames(frames, max_reproj_error=2.5, min_markers=3):
    """
    frames: list chứa dict mỗi frame, phải có các key:
        'object_points', 'image_points', 'rvec', 'tvec', 'ids'

    max_reproj_error: ngưỡng px
    min_markers: số marker tối thiểu

    return: list frame tốt
    """
    kept = []
    for f in frames:
        if f.get('rvec') is None or f.get('tvec') is None: 
            continue
        if f.get('ids') is None or len(f['ids']) < min_markers: 
            continue

        mean_err, _ = compute_reprojection_error(
            f['object_points'], f['image_points'],
            f['rvec'], f['tvec'],
            f['camera_matrix'], f['dist_coeffs']
        )
        if mean_err <= max_reproj_error:
            f['reproj_error'] = mean_err
            kept.append(f)
    print(f"Giữ {len(kept)} frames trên {len(frames)} frames")
    return kept

def collect_object_and_image_points_from_corners(corners, ids, board, marker_ids, obj_points):
    """
    corners, ids : outputs from cv2.aruco.detectMarkers(gray, dictionary)
      - corners: list of marker corners; each element can be shaped (1,4,2) or (4,1,2)
      - ids: array of detected ids shape (M,1)
    board : the GridBoard / Board you created (has board.objPoints and board.ids)

    Returns:
      object_points: (K,3) numpy array of 3D points (marker corners in board coordinates)
      image_points:  (K,2) numpy array of corresponding 2D image points (pixels)
    If no matching markers -> returns (None, None).
    """
    if ids is None:
        return None, None

    objpts_all = []
    imgpts_all = []

    # flatten ids to 1D list for easy looping
    ids_flat = ids.flatten()

    # board.ids is array of ids used when creating board
    board_ids_list = marker_ids

    for corner, id0 in zip(corners, ids_flat):
        # corner may be shape (1,4,2) or (4,1,2) or (4,2)
        imgp = np.array(corner).reshape(4,2)   # now (4,2) in pixel coordinates
        # find index of this marker id in the board definition
        try:
            idx = board_ids_list.index(int(id0))
        except ValueError:
            # detected marker id not in this board (skip)
            continue
        # board.objPoints[idx] is list/ndarray shape (4,3) for that marker's corners (board coordinates)
        objp = np.array(obj_points[idx], dtype=np.float64).reshape(4,3)
        objpts_all.append(objp)
        imgpts_all.append(imgp)

    if len(objpts_all) == 0:
        return None, None

    object_points = np.vstack(objpts_all).astype(np.float64)   # shape (K,3)
    image_points  = np.vstack(imgpts_all).astype(np.float64)   # shape (K,2)
    return object_points, image_points

def main():
    # === Load calibration ===
    calib_file = "camera_calibration.json"
    board_to_robot_file = "T_board_to_robot.json"
    if not os.path.exists(calib_file):
        print(f"Calibration file not found: {calib_file}")
        return

    camera_matrix, dist_coeffs= load_camera_calibration(calib_file)

    
    
    # === Known transform: board to robot ===
    T_board_to_robot = load_transformation_matrix(board_to_robot_file)
    
    # === Camera init ===
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Camera failed to open.")
        return

    board, aruco_dict, marker_ids, obj = create_custom_board_for_detection(
    marker_size=40,
    marker_ids=[0, 8, 16, 32],
    dictionary_type=aruco.DICT_4X4_50
    )
    retval = False
    parameters = aruco.DetectorParameters()
    cap, parameters = camera_setup(cap, parameters)
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    T_list = []
    frame_list = []
    i = 0
    while i < 10:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) == 4:
            rvec = np.zeros((1, 3), dtype=np.float64)
            tvec = np.zeros((1, 3), dtype=np.float64)
            
            retval, rvec, tvec = aruco.estimatePoseBoard(
                corners, ids, board, camera_matrix, dist_coeffs, rvec, tvec)

        if retval:
            # Convert to 4x4 transformation matrix
            print("rvec:", rvec)
            print("tvec:", tvec)
            
            T_board_to_camera = rvec_tvec_to_matrix(rvec, tvec)
            T_list.append(T_board_to_camera)
        else:
            print("cannot identify the board")
            time.sleep(1)
            continue
        i += 1
    cap.release()
    T_board_to_camera = average_pose(T_list)
    T_camera_to_board = invert_transform(T_board_to_camera)
    T_camera_to_robot =  T_board_to_robot @ T_camera_to_board
    R = T_camera_to_robot[:3,:3]
    print("det(R) =", np.linalg.det(R))
    print("orthogonality error =", np.linalg.norm(R.T @ R - np.eye(3)))
    save_matrix('T_camera_to_robot.json', T_camera_to_robot)
    print("Đã lưu ma trận")
if __name__ == "__main__":
    main()
