import serial
import time
import numpy as np
import json
import os, tempfile
from filelock import FileLock, Timeout
dai_X = 25
rong_X = 21
R = 14
COM_PORT = 'COM5'

def atomic_write_json(path, data, mode=0o644):
    """
    Ghi JSON một cách nguyên tử:
    - ghi vào file tạm cùng thư mục
    - fsync -> os.replace(tmp, path)
    """
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
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

def save_flag(state_str, extra=None, file_path = "Flag.json"):
    """
    state_str: "idle" / "request_move" / "robot_busy" / "robot_done" / "error"/ "brain done
    extra: optional dict to attach thêm dữ liệu (timestamp tự thêm)
    """
    payload = {"Flag": state_str, "timestamp": time.time()}
    if isinstance(extra, dict):
        payload.update(extra)
    safe_write_with_lock(file_path, payload)

def load_flag(file_path = "Flag.json"):
    data = safe_read_json(file_path)
    if data is None:
        print("Không lấy được dữ liệu flag cho robot")
        return None
    flag = data["Flag"]
    return flag # có thể là None nếu file chưa tồn tại

def load_T_board_to_robot(file_path = "T_board_to_robot_main.json"):
    data = safe_read_json(file_path)
    if data is None:
        print("Không tìm thấy dữ liệu ma trận T_board_to_robot")
        return None
    T_board_to_robot = np.array(data["T_board_to_robot"], dtype= np.float32)
    return T_board_to_robot
    
def load_robot_move(file_path = "Game_board.json" ):
    data = safe_read_json(file_path)
    if data is None:
        print("Không tìm thấy dữ liệu nước đi")
        return None, None
    coord = list(data["Coord"])
    robot_move = data["Robot_move"]
    return coord, robot_move 
    
def send_code(ser, code, f_rate = 200):
    ser.write(f'{code} F{f_rate}\n'.encode())
    time.sleep(1)
    while True:
        response = ser.readline().decode().strip()
        print(response)
        try:
            if "ok" in response.lower():
                break
            
        except Exception as e:
            print(f"error is {e}")
            break

def nang_but(ser, Z) :
    send_code(ser, f'G01 Z{Z + 10}')

def ha_but (ser, Z):
    send_code(ser, f'G01 Z{Z}')

def draw_line(ser, start, end, Z):
    nang_but(ser, Z)
    send_code(ser, f"G01 X{start[0]} Y{start[1]} F100")
    ha_but(ser, Z)
    send_code(ser, f"G01 X{end[0]} Y{end[1]} F100")
    nang_but(ser, Z)

def draw_grid(ser, verti, hori, Z, T_board_to_robot):
    # chuyển đổi điểm từ board → robot
    verti_points = [point_board_to_robot(p, T_board_to_robot) for p in verti]
    hori_points  = [point_board_to_robot(p, T_board_to_robot) for p in hori]

    # vẽ 2 đoạn dọc
    draw_line(ser, verti_points[0], verti_points[1], Z)
    draw_line(ser, verti_points[2], verti_points[3], Z)

    # vẽ 2 đoạn ngang
    draw_line(ser, hori_points[0], hori_points[1], Z)
    draw_line(ser, hori_points[2], hori_points[3], Z)

    nang_but(ser, Z)
    send_code(ser, "G28")
    
def drawX(ser, X, Y, Z):
    
    draw_line(ser, (X+dai_X/2, Y+rong_X/2), (X-dai_X/2, Y-rong_X/2), Z)
    draw_line(ser, (X-dai_X/2, Y+rong_X/2),(X+dai_X/2, Y-rong_X/2), Z)
    
    send_code(ser, "G28")
    
def drawO(ser, X, Y, Z):
    # Nhấc bút trước khi di chuyển đến start
    nang_but(ser, Z)
    send_code(ser, f"G01 X{X+R} Y{Y} Z{Z+10} F100")  # di chuyển nhanh trên cao
    ha_but(ser, Z)  # hạ bút xuống
    
    # Vẽ tròn bằng G02
    send_code(ser, f'G02 X{X+R} Y{Y} I{-R} J0 F100')
    
    # Nhấc bút lên sau khi vẽ xong
    nang_but(ser, Z)
    
    send_code(ser, "G28")
    
def point_board_to_robot(point_board, T_board_to_robot):
    """
    Chuyển 1 điểm từ hệ tọa độ board sang hệ robot.

    Args:
        T_board_to_robot (np.ndarray): Ma trận 4x4 chuyển đổi từ board sang robot.
        point_board (tuple or list): Điểm thứ nhất trên giấy, ví dụ (x1, y1)

    Returns:
        tọa độ x,y,z của point trong hệ robot.
    """
    # Thêm chiều z=0 và 1 vào điểm trên mặt giấy để đưa về dạng 4x1
    point_board = np.array([point_board[0], point_board[1], 0, 1])

    # Nhân ma trận để chuyển sang hệ robot
    point_robot = T_board_to_robot @ point_board

    # Chỉ lấy phần tọa độ (bỏ ra phần đồng nhất)
    return point_robot[:3]

def take_input(a):
    choice = input(f"{a}").lower()
    return choice

def main():
    verti_grid_points = [(-25, 75),(-25,-75), # top left, bottom left
                        (25,75),(25,-75)]    # top right, bottom right
    
    hori_grid_points = [(-75, 25),(75, 25),    #top left, top right
                        (-75, -25),(75,-25)]  #bottom left, bottom right
    T_board_to_robot = load_T_board_to_robot()
    flag = load_flag()
    Z = -353.1
    try:
        with serial.Serial(COM_PORT, 115200, timeout = 2) as ser:
            time.sleep(2)
            send_code(ser, "G90")
            send_code(ser, "G01 F100")
            send_code(ser, "G28")
            print("kết nối thành công vào cổng\n")
            if flag == "innit done":
                flag = "robot drawing"
                save_flag(flag)
                while True:
                    choice = take_input("Nhấn D để vẽ bảng caro/Q để không vẽ\n")
                    if choice == "d":
                        retries = 10
                        while T_board_to_robot is None and retries>0:
                            print("Chưa có T_board_to_robot, chờ 0.5s...")
                            time.sleep(0.5)
                            T_board_to_robot = load_T_board_to_robot()
                            retries -= 1
                        if T_board_to_robot is None:
                            print("Không tìm được T_board_to_robot sau nhiều lần, thoát")
                            flag = "error"
                            save_flag(flag)
                            print(flag)
                            raise RuntimeError("lỗi rồi ní")
                        draw_grid(ser, verti_grid_points, hori_grid_points, Z, T_board_to_robot)
                        time.sleep(0.5)
                        flag = "grid done"
                        print(flag)
                        save_flag(flag)
                        break
                    elif choice == "q":
                        flag = "grid done"
                        print(flag)
                        save_flag(flag)
                        break
                
            while True:
                flag = load_flag()
                if flag is None:
                    print("Không thấy flag bỏ qua lần này")
                    time.sleep(0.1)
                    continue
                if flag == "brain done":
                    flag = "robot drawing"
                    print(flag)
                    save_flag(flag)
                    T_board_to_robot = load_T_board_to_robot()
                    coord, robot_move = load_robot_move()
                    if T_board_to_robot is None or coord is None:
                        print("Không thấy ma trận hoặc coord bỏ qua lần này")
                        time.sleep(0.2)
                        continue
                    X,Y,_ = point_board_to_robot(coord, T_board_to_robot)
                    time.sleep(1)
                    if robot_move == "X":
                        print(f"Robot đi X tại {coord}")
                        time.sleep(0.5)
                        drawX(ser, X, Y, Z)
                        time.sleep(0.5)
                    elif robot_move == "O":
                        print(f"Robot đi O tại {coord}")
                        time.sleep(0.5)
                        drawO(ser, X, Y, Z)
                        time.sleep(0.5)
                    else:
                        print("Không tìm được là nước X hay O")
                        time.sleep(1)
                        break
                    print("chạy xong ròi")
                    flag = "robot done"
                    print(flag)
                    save_flag(flag)
                    time.sleep(1)
        print("Đóng cổng port")
    except Exception as e:
        print(f"lỗi {e} rồi chú")

if __name__ == "__main__":
    main()