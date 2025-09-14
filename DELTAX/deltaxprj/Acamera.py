
import numpy as np
import cv2
import cv2.aruco as aruco
import json
import random as rd
import msvcrt as ms
import math
from inference_sdk import InferenceHTTPClient
import os, json, tempfile, time
from filelock import FileLock, Timeout
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "roboflowKey.env"))

API_KEY = os.getenv("ROBOFLOW_API_KEY")
if API_KEY is None:
    raise RuntimeError("Không tìm thấy ROBOFLOW_API_KEY trong roboflowKey.env")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key= API_KEY
) 


def atomic_write_json(path, data, mode=0o644):
    """
    Ghi JSON một cách nguyên tử:
    - ghi vào file tạm cùng thư mục
    - fsync -> os.replace(tmp, path)
    """
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)  # atomic replace
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

def safe_read_json(path, lock_timeout=1.0, retries=5, backoff=0.05):
    """
    Đọc JSON an toàn: cố lấy lock để tránh read-modify-write race.
    Trả về dict hoặc raise RuntimeError nếu không đọc được sau retries.
    """
    lock = FileLock(path + ".lock")
    for attempt in range(retries):
        try:
            with lock.acquire(timeout=lock_timeout):
                # Nếu file chưa tồn tại, trả về None
                if not os.path.exists(path):
                    return None
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Timeout:
            time.sleep(backoff * (1 + attempt))
        except json.JSONDecodeError:
            # có thể file bị ghi dở (hiếm khi với atomic replace) -> đợi
            time.sleep(backoff * (1 + attempt))
    print(f"⚠️ Cảnh báo: đọc {path} thất bại sau {retries} lần, trả về None")
    return None

def safe_write_with_lock(path, data, lock_timeout=2.0):
    """
    Ghi JSON an toàn bằng lock + atomic replace.
    """
    lock = FileLock(path + ".lock")
    with lock.acquire(timeout=lock_timeout):
        atomic_write_json(path, data)

def save_board(coord_list, robot_move, game_board, file_path = "Game_board.json"):
    payload = {
        "Coord": coord_list,
        "Robot_move": robot_move,
        "Game_board": game_board.tolist(),
        "timestamp": time.time()
    }
    safe_write_with_lock(file_path, payload)

def save_flag(state_str, extra=None, file_path = "Flag.json"):
    """
    state_str: "idle" / "request_move" / "robot_busy" / "robot_done" / "error"/ "brain done"
    extra: optional dict to attach thêm dữ liệu (timestamp tự thêm)
    """
    payload = {"Flag": state_str, "timestamp": time.time()}
    if isinstance(extra, dict):
        payload.update(extra)
    safe_write_with_lock(file_path, payload)

def load_flag(file_path = "Flag.json"):
    data = safe_read_json(file_path)
    if data is None:
        print("Không đọc được dữ liệu flag")
        return None
    flag = data["Flag"]
    return flag # có thể là None nếu file chưa tồn tại
        

def test_cameras():
    """Test different camera indices to find the working one"""
    working_cameras = []
    for i in range(4):  # Test cameras 0, 1, 2, 3
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} is working")
                working_cameras.append(i)
            cap.release()
    
    if not working_cameras:
        print("No working camera found")
        return -1
    
    return working_cameras

def select_camera():
    """Let user select which camera to use"""
    working_cameras = test_cameras()
    
    if working_cameras == -1:
        return -1
    
    if len(working_cameras) == 1:
        print(f"Only one camera found. Using camera {working_cameras[0]}")
        return working_cameras[0]
    
    print("\nAvailable cameras:")
    for i, cam_idx in enumerate(working_cameras):
        print(f"{i+1}. Camera {cam_idx}")
    
    while True:
        try:
            choice = input(f"\nSelect camera (1-{len(working_cameras)}) or press Enter for camera 0: ")
            if choice == "":
                return 0
            choice = int(choice)
            if 1 <= choice <= len(working_cameras):
                selected_camera = working_cameras[choice - 1]
                print(f"Selected camera {selected_camera}")
                return selected_camera
            else:
                print(f"Please enter a number between 1 and {len(working_cameras)}")
        except ValueError:
            print("Please enter a valid number")

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

def load_calibration(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    camera_matrix = np.array(config["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(config["dist_coeffs"], dtype=np.float32)

    return camera_matrix, dist_coeffs

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
    
    # Create the board directly
    board = aruco.Board(objPoints=obj_points, dictionary=aruco_dict, ids=ids)
    all_pts = np.vstack(obj_points)
    print("example object points (first marker):", obj_points[0])
    print("object y min/max:", all_pts[:,1].min(), all_pts[:,1].max())
    return board, aruco_dict

def save_T_board_to_robot(T, file_path = "T_board_to_robot_main.json"):
    payload = {"T_board_to_robot": T.tolist(), "Timestamp": time.time()}
    safe_write_with_lock(file_path, payload)

def load_T_camera_to_robot(file_path):
    data = safe_read_json(file_path)
    if data is None:
        raise RuntimeError(f"Không load được {file_path}")
    return np.array(data['T_camera_to_robot'], dtype=np.float32)

def get_board_to_robot(corners, ids, board, camera_matrix, dist_coeffs, T_camera_to_robot):
    rvec = np.zeros((1, 3), dtype=np.float64)
    tvec = np.zeros((1, 3), dtype=np.float64)
    retval, rvec, tvec = aruco.estimatePoseBoard(
            corners, ids, board, camera_matrix, dist_coeffs, rvec, tvec)
    
    if retval:
        R, _ = cv2.Rodrigues(rvec)
        T_board_to_camera = np.eye(4)
        T_board_to_camera[:3, :3] = R
        T_board_to_camera[:3, 3] = tvec.flatten()
        print("Ma trận board to camera: ", T_board_to_camera)
        T_board_to_robot = T_camera_to_robot @ T_board_to_camera
    else:
        return None    
    return T_board_to_robot, T_board_to_camera    

def get_p1(corners, ids):
    """
    corners, ids: output từ cv2.aruco.detectMarkers
    Trả về: p1 (4 góc board theo thứ tự: top-left, top-right, bottom-right, bottom-left)
    """
    if ids is None or len(ids) == 0:
        return None

    # Gom tất cả point của các marker
    all_pts = []
    for pts in corners:
        all_pts.extend(pts[0])
    all_pts = np.array(all_pts)

    # Tính convex hull bao quanh toàn bộ board
    hull = cv2.convexHull(all_pts)

    # Chuyển hull về 2D (n x 2)
    hull = hull.reshape(-1, 2)

    # Lấy 4 điểm cực trị từ hull (trái-trên, phải-trên, phải-dưới, trái-dưới)
    s = hull.sum(axis=1)
    diff = np.diff(hull, axis=1).ravel()

    top_left = hull[np.argmin(s)]
    bottom_right = hull[np.argmax(s)]
    top_right = hull[np.argmin(diff)]
    bottom_left = hull[np.argmax(diff)]

    p1 = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    return p1

def warp(img, p1, p2, size):
    T = cv2.getPerspectiveTransform(p1,p2)
    trans_img = cv2.warpPerspective(img, T, size)
    return trans_img

def draw_projected_grid(img, T_board_to_cam, camera_matrix, dist_coeffs, n=3, cell_size=50):
    """
    Vẽ grid (n x n) dựa trên T_board_to_cam (4x4), camera_matrix, dist_coeffs.
    cell_size in same units as board (mm).
    Origin board assumed at center (same logic như get_cells).
    Trả về ảnh đã vẽ (copy).
    """
    # Lấy rvec, tvec từ ma trận quay
    R = T_board_to_cam[:3, :3].astype(np.float64)
    t = T_board_to_cam[:3, 3].astype(np.float64).reshape(3,1)
    rvec, _ = cv2.Rodrigues(R)  # rvec shape (3,1)
    tvec = t

    half = n * cell_size / 2.0
    pts3d = []
    # tạo grid góc (4 góc cho mỗi ô) hoặc chỉ grid lines
    for i in range(n+1):         # ngang lines
        for j in range(n+1):     # vertical division points (dùng cho debug)
            x = -half + j * cell_size
            y =  half - i * cell_size
            pts3d.append([x, y, 0.0])   # z=0 trên plane board

    pts3d = np.array(pts3d, dtype=np.float64)
    imgpts, _ = cv2.projectPoints(pts3d, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1,2).astype(np.int32)

    dbg = img.copy()
    # Vẽ những điểm (dùng để debug), hoặc nối lại thành đường lưới
    idx = 0
    # vẽ điểm lưới
    for i in range(n+1):
        row_pts = []
        for j in range(n+1):
            p = tuple(imgpts[idx])
            row_pts.append(p)
            idx += 1
        # nối hàng ngang
        for k in range(len(row_pts)-1):
            cv2.line(dbg, row_pts[k], row_pts[k+1], (0,255,0), 2)
    # nối cột dọc
    idx = 0
    for j in range(n+1):
        col_pts = []
        for i in range(n+1):
            col_pts.append(tuple(imgpts[i*(n+1) + j]))
        for k in range(len(col_pts)-1):
            cv2.line(dbg, col_pts[k], col_pts[k+1], (0,255,0), 2)

    return dbg

def get_game_board (img, p1, T_board_to_camera, camera_matrix, dist_coeffs, game_board, empty_cell):
    new_game_board = game_board.copy()
    board_img = img.copy()
    board_p2 = np.array([[0,0],
                         [1485,0],
                         [1485,1050],
                         [0,1050]], dtype=np.float32)
    warped_board = warp(board_img, p1, board_p2, (1485,1050))

    simulalte_img = draw_projected_grid(img, T_board_to_camera, camera_matrix, dist_coeffs)
    cv2.imwrite("simulate_grid_on_camera.jpg", simulalte_img)
    
    new_game_board = np.array(classify_board(warped_board, empty_cell))
    return new_game_board

def preprocess_cell(cell_bgr, border_crop=0.1, size=128):
    cell = cv2.resize(cell_bgr, (size, size))
    m = int(border_crop * size)
    cell = cell[m:size-m, m:size-m]
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # thử adaptive trước, nếu quá ít nét thì fallback Otsu
    bin_adap = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )
    if cv2.countNonZero(bin_adap) < 0.002 * bin_adap.size:
        _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        bin_ = bin_adap

    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, k3, iterations=1)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, k3, iterations=1)
    return bin_

def contour_features(bin_):
    cnts, hier = cv2.findContours(bin_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: 
        return None, {}
    area_min = 0.01 * bin_.size
    cnts = [c for c in cnts if cv2.contourArea(c) > area_min]
    if not cnts: 
        return None, {}

    # chọn contour lớn nhất
    idx = np.argmax([cv2.contourArea(c) for c in cnts])
    cnt = cnts[idx]

    A = cv2.contourArea(cnt)
    P = cv2.arcLength(cnt, True)
    circularity = 4*np.pi*A/(P*P) if P>0 else 0
    x,y,w,h = cv2.boundingRect(cnt)
    aspect = w/h
    rect_extent = A/(w*h)

    hull = cv2.convexHull(cnt)
    solidity = A / cv2.contourArea(hull)

    # đếm lỗ: contour con có hierarchy parent = idx?
    holes = 0
    if hier is not None:
        hier = hier[0]
        for i, hnode in enumerate(hier):
            parent = hnode[3]
            if parent == idx:
                holes += 1

    feats = dict(A=A, P=P, circularity=circularity, aspect=aspect,
                 rect_extent=rect_extent, solidity=solidity, holes=holes)
    return cnt, feats

def x_score_from_lines(bin_):
    edges = cv2.Canny(bin_, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)
    if lines is None:
        return 0
    angles = []
    for x1,y1,x2,y2 in lines[:,0,:]:
        ang = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
        angles.append(ang)
    angles = np.array(angles)
    tol = 20
    d1 = np.minimum(np.abs(angles-45), 180-np.abs(angles-45))
    d2 = np.minimum(np.abs(angles-135), 180-np.abs(angles-135))
    n1 = (d1<tol).sum()
    n2 = (d2<tol).sum()

    H, W = bin_.shape
    cx, cy = W/2, H/2
    near_center = 0
    for x1,y1,x2,y2 in lines[:,0,:]:
        num = abs((y2-y1)*cx - (x2-x1)*cy + x2*y1 - y2*x1)
        den = math.hypot(y2-y1, x2-x1)
        dist = num/den if den>0 else 1e9
        if dist < 0.1*min(H,W):  # 10% kích thước
            near_center += 1

    score = 0
    if min(n1,n2) >= 1: score += 1
    if near_center >= 1: score += 1
    return score  # 0..2

def o_score_from_circle(bin_, cnt):
    # circularity + fit circle error + holes
    A = cv2.contourArea(cnt)
    P = cv2.arcLength(cnt, True)
    circularity = 4*np.pi*A/(P*P) if P>0 else 0

    (cx, cy), R = cv2.minEnclosingCircle(cnt)
    pts = cnt.reshape(-1,2).astype(np.float32)
    err = np.abs(np.hypot(pts[:,0]-cx, pts[:,1]-cy) - R).mean()
    fit_score = 1/(1+err)  # 0..1

    # holes
    cnts, hier = cv2.findContours(bin_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hier is not None:
        hier = hier[0]
        idx_main = None
        for i,c in enumerate(cnts):
            if np.array_equal(c, cnt):
                idx_main = i; break
        if idx_main is not None:
            for i, h in enumerate(hier):
                if h[3] == idx_main:
                    holes += 1

    # tổng hợp
    score = 0.0
    score += np.clip((circularity-0.4)/0.6, 0, 1) * 0.5   # trọng số 0.5
    score += np.clip(fit_score, 0, 1) * 0.4              # trọng số 0.4
    if holes >= 1: score += 0.2                           # có lỗ: cộng điểm
    return min(score, 1.0)  # 0..1

def classify_cell_rule_based(cell_bgr):
    bin_ = preprocess_cell(cell_bgr)

    cnt, feats = contour_features(bin_)
    if cnt is None:
        return ""
    # điểm cho O và X
    o_s = o_score_from_circle(bin_, cnt)
    x_s = x_score_from_lines(bin_)

    # quyết định
    if o_s >= 0.6 and o_s > (0.3 + 0.3*x_s):
        return "O"
    if x_s >= 1 and (x_s >= 2 or o_s < 0.5):
        return "X"

    # fallback bằng circularity
    circ = feats.get("circularity", 0)
    if circ > 0.65:
        return "O"
    return ""

def map_predictions_to_board(predictions, img_w=1485, img_h=1050, board_size_mm=150):
    """
    Ánh xạ danh sách predictions (tọa độ tâm box trong ảnh warp A4)
    sang ma trận 3x3 caro.

    Parameters:
        predictions: list[dict] - mỗi phần tử có keys: "x", "y", "class"
        img_w, img_h: int - kích thước ảnh warp (px)
        board_size_mm: float - kích thước thật của board caro (mm)

    Returns:
        board: list[list[str]] - ma trận 3x3 chứa "X","O" hoặc ""
    """
    # scale px/mm từ A4 (297x210 mm warp -> 1485x1050 px)
    px_per_mm_x = img_w / 297.0
    px_per_mm_y = img_h / 210.0
    assert abs(px_per_mm_x - px_per_mm_y) < 1e-6, "Ảnh warp không đồng tỉ lệ!"

    px_per_mm = px_per_mm_x

    # kích thước board caro theo px
    board_size_px = board_size_mm * px_per_mm

    # toạ độ biên trên-trái của board trong ảnh
    x0 = (img_w - board_size_px) / 2
    y0 = (img_h - board_size_px) / 2
    cell_size = board_size_px / 3

    # khởi tạo board rỗng
    board = [["" for _ in range(3)] for _ in range(3)]

    for p in predictions:
        col = int((p["x"] - x0) // cell_size)
        row = int((p["y"] - y0) // cell_size)
        if 0 <= row < 3 and 0 <= col < 3:
            board[row][col] = p["class"]

    return board

def classify_board(board_bgr, confidence_threshold = 0.5):
    cv2.imwrite("tmp_board.jpg", board_bgr)
    try:
        result = client.run_workflow(
            workspace_name="hmq",
            workflow_id="custom-workflow",
            images={
                "image": "tmp_board.jpg"
            },
            use_cache=True # cache workflow definition for 15 minutes
        )
    except Exception as e:
        raise RuntimeError("Roboflow workflow failed:", e)

    #Kiểm tra kết quả
    if not result or not isinstance(result, list) or len(result) == 0:
        raise RuntimeError("Roboflow returned empty result")

    # lấy danh sách prediction thô
    raw_preds = result[0]["predictions"]["predictions"]
    # lọc thành dạng cần cho map_predictions_to_board
    predictions = [{"x": p["x"], "y": p["y"], "class": p["class"]}
                for p in raw_preds]
    board = map_predictions_to_board(predictions)
    return board

def choose_first_player():
    print("===Chọn người đi trước===\n")
    print("Nhấn A cho robot\n")
    print("Nhấn B cho người chơi\n")
    print("Nhấn Q để out\n")
        
    choice = ms.getch().decode().lower()
    if choice == "a":
        first_player = 'robot'
        return first_player
    elif choice == "b":
        first_player = "player"
        return first_player
    elif choice == "q":
        print("===Out game===\n")
        global game_state
        game_state = False
    else:
        print("===Chọn A hoặc B===\n")
        return choose_first_player()
    
    cv2.waitKey()
    
def find_current_player(move_counter, first_player):
    if move_counter % 2 == 0:
        return first_player
    return "robot" if first_player == "player" else "player"

def check_hori (game_board):
    for i in range(0,3):
        if game_board[i][0] == game_board[i][1] and game_board[i][0] == game_board[i][2] and game_board[i][0] != "":
            return False
    return True
        
def check_verti (game_board):
    for i in range(0,3):
        if game_board[0][i] == game_board[1][i] and game_board[0][i] == game_board[2][i] and game_board[0][i] != "":
            return False
    return True    

def check_diag (game_board):
    if game_board[0][0] == game_board[1][1] and game_board[0][0] == game_board[2][2] and game_board[0][0] != "":
         return False
    elif game_board[0][2] == game_board[1][1] and game_board[0][2] == game_board[2][0] and game_board[0][2] != "":
         return False
    return True

def choose_mode():
    print("===Choose difficulty===\n")
    print("Press E for easy mode\n")
    print("Press H for hard mode\n")
    print("Press I for impossible\n")
    print("Press esc to out")
    while True:
        choice = ms.getch().decode().lower()
        if choice == "e" or choice == "h" or choice == "i":
            return choice
        elif choice == '\x1b':
            break
        else:
            print("Chưa chọn chế độ hợp lệ\n")
            print("Vui lòng chọn lại")
            return choose_mode()

def cell_to_board(cell, cell_size = 50, n = 3):
    x = (cell[1] - (n-1)/2) * cell_size
    y = -(cell[0] - (n-1)/2) * cell_size
    return (x,y)

def easy_mode(current_player, empty_cell, first_player, game_board, move_counter):
    new_game_board = game_board.copy()
    if current_player == 'robot' and empty_cell is not None:
        x,y = rd.choice(empty_cell)
        move ="X" if current_player == first_player else "O"        
        new_game_board[x][y] = move
        move_counter += 1
        return (x,y), move, new_game_board, move_counter
    else:
        return None, None, game_board, move_counter  

def hard_mode(current_player, empty_cell, first_player, game_board, move_counter):
    if current_player == 'robot' and empty_cell is not None:
        robot ="X" if current_player == first_player else "O"
        player = "X" if robot != "X" else "O"
        
        

def setup():
    game_board = np.array([["","",""],
                           ["","",""],
                           ["","",""]])
    move_counter = 0
    game_state = True
    empty_cell = list(zip(*np.where(np.array(game_board) == "")))
    first_player = choose_first_player()
    if first_player == "player":
        flag = "robot done"
    else:
        flag = "player done"
    flag = "innit"
    save_flag(flag)
    print(flag)
    mode = choose_mode()
    return game_board, move_counter, game_state, empty_cell, first_player, mode, flag

def is_player_move(key):
    print("Bấm f khi bạn đi xong")
    if key == "f":
        return False
    return True

def main():
### SETUP ###
    board, aruco_dict = create_custom_board_for_detection(marker_size=40, marker_ids=[0, 8, 16, 32], 
                                      dictionary_type=aruco.DICT_4X4_50)
    camera_matrix, dist_coeffs= load_calibration("camera_calibration.json")
    T_camera_to_robot = load_T_camera_to_robot("T_camera_to_robot.json")

    # Let user select camera
    camera_index = select_camera()
    if camera_index == -1:
        print("No camera found. Please check your camera connection.")
        return

    print(f"Using camera index: {camera_index}")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Không mở được camera")
        return
    
    # Set camera properties for better detection
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    
    # Adjust detection parameters for better sensitivity
    cap, parameters = camera_setup(cap, parameters)
    
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    print("Press 'q' to quit, 's' to save current frame, 'c' to change camera")
    frame_count = 0
    game_board, move_counter, game_state, empty_cell, first_player, mode, flag = setup()
### LOOP ###
    while True:
        ret, img = cap.read()
        if not ret:
            print("Không đọc được khung hình")
            break
            
        frame_count += 1
        # Create a copy for drawing
        display_img = img.copy()
        
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray_image)
            
        if ids is not None and len(ids)>3:
            flag = load_flag()
            if flag == "innit":
                res = get_board_to_robot(corners, ids, board, camera_matrix,
                                            dist_coeffs, T_camera_to_robot)
                if res is None:
                    cv2.putText(display_img, "Không ước lượng được pose board - bỏ qua frame", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)                
                    print("Không ước lượng được board")
                    continue
                else:
                    T_board_to_robot, T_board_to_camera = res
                if (T_board_to_robot) is not None:
                    board_position = T_board_to_robot[:3,3]
                    save_T_board_to_robot(T_board_to_robot)
                    print("Vị trí của tâm board là:", board_position)                        
                else: 
                    print("Failed to identify the board")
                    flag = "error"
                    print(flag)
                    save_flag(flag)
                
                flag = "innit done"
                save_flag(flag)
            aruco.drawDetectedMarkers(display_img, corners, ids)
            pp_avg_x = 0
            pp_avg_y = 0
            
            for i in corners:
                itr = i[0]
                avg_x = np.mean(itr[:,0])
                avg_y = np.mean(itr[:,1])
                pp_avg_x += avg_x
                pp_avg_y += avg_y
                cv2.circle(display_img, (int(avg_x), int(avg_y)), radius=10, color=(255,0,0), thickness = 1 ) #color theo bảng BGR
            
            if len(corners) == 4:
                pp_avg_x /= 4
                pp_avg_y /= 4
                cv2.circle(display_img, (int(pp_avg_x), int(pp_avg_y)), radius=10, color=(255, 0 ,0 ), thickness= 1 ) 

            if game_state:
                flag = load_flag()
                if flag != "robot done" and flag != "innit done" and flag != "player done":
                    time.sleep(0.1)
                    print("flag flag")
                    continue
                elif flag == "error":
                    raise RuntimeError("Có lỗi flag bị chuyển thành error")
                elif flag is None:
                    raise RuntimeError("có lỗi trong hàm setup, flag trả về none")
                
                current_player = find_current_player(move_counter, first_player)
                old_game_board = game_board.copy()
                if current_player == "player" and (flag == "grid done" or flag == "innit done" or flag == "robot done"):
                    while np.array_equal(old_game_board, game_board): 
                        ret_live, live = cap.read()
                        if not ret_live:
                            print("Không đọc được khung hình trong lúc chờ người chơi")
                            time.sleep(0.01)
                            continue

                        # vẽ thông báo lên frame live
                        disp = live.copy()
                        cv2.putText(disp, "Waiting for player move - press 'f' when done", (20,40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                        cv2.imshow("Boardchecker", disp)
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord('q'):
                            print("Thoát sớm bởi nguời chơi")
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                        elif key == ord('f'):
                            # khi người chơi bấm 'f', chụp frame hiện tại để so sánh
                            ret, check_img = cap.read()
                            if not ret:
                                print("Không thấy ảnh để check gameboard")
                                raise RuntimeError
                            
                            gray_check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2GRAY)
                            check_corners, check_ids, _ = detector.detectMarkers(gray_check_img)
                            check_p1 = get_p1(check_corners, check_ids)
                            check_res = get_board_to_robot(check_corners, check_ids, board, camera_matrix,
                                                        dist_coeffs, T_camera_to_robot)
                            if check_res is None:             
                                print("Không ước lượng được board")
                                continue
                            else:
                                check_T_board_to_robot, check_T_board_to_camera = check_res
                                
                            if (check_T_board_to_robot) is not None:
                                board_position = check_T_board_to_robot[:3,3]
                                save_T_board_to_robot(check_T_board_to_robot)
                                print("Vị trí của tâm board là:", board_position)                        
                            else: 
                                print("Failed to identify the board")
                                flag = "error"
                                print(flag)
                                save_flag(flag)
                
                            game_board = get_game_board(check_img, check_p1, check_T_board_to_camera, camera_matrix, dist_coeffs, game_board, empty_cell)
                            print("game board xác định được là: \n", game_board)
                            empty_cell = list(zip(*np.where(game_board == ""))) 
                            
                            if np.array_equal(old_game_board, np.array(game_board)):
                                print("Chưa tìm được nước đi mới, tiếp tục chờ...")
                            else:
                                print("Phát hiện nước đi mới")
                                break
                    print("Thoát khỏi vòng lặp tính game board")
                    cv2.destroyAllWindows()
                    empty_cell = list(zip(*np.where(game_board == "")))
                    move_counter += 1
                    flag = "player done"
                    print(flag)
                    save_flag(flag)
                elif current_player == "robot" and (flag == "player done" or flag == "grid done"):
                    if mode == 'e':
                        coord, robot_move, game_board, move_counter = easy_mode(current_player, empty_cell, first_player, game_board, move_counter)
                        if coord is None or robot_move is None:
                            print("Không có move thỏa, hoặc không phải lượt của robot")
                            break
                        empty_cell = list(zip(*np.where(game_board == "")))
                        save_board(cell_to_board(coord), robot_move, game_board)
                        flag = "brain done"
                        save_flag(flag)
                print(game_board)    
                game_state = check_diag(game_board) and check_hori(game_board) and check_verti(game_board) and bool(empty_cell) and move_counter < 9
            else:
                print("===Game over===")
                break

        cv2.imshow("Camera", display_img)

        # Nhấn 'q' để thoát, 's' để lưu frame, 'c' để đổi camera
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"frame_{frame_count}.png", img)
            print(f"Saved frame_{frame_count}.png")
        elif key == ord('c'):
            # Change camera
            cap.release()
            cv2.destroyAllWindows()
            print("\nChanging camera...")
            return main()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
