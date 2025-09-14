import numpy as np
import json

def compute_T_board_to_robot(points_board, points_robot):
    P = np.asarray(points_board, dtype=float)
    Q = np.asarray(points_robot, dtype=float)

    assert P.shape == Q.shape, "Hai tập điểm phải cùng kích thước"
    N = P.shape[0]
    if N < 3:
        raise ValueError("Cần ít nhất 3 điểm không thẳng hàng")

    # Tính centroid
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    # Dịch về gốc
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Ma trận hiệp phương sai
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T

    # Sửa lật gương nếu cần
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T

    # Vector tịnh tiến
    t = centroid_Q - R @ centroid_P

    # Ma trận đồng nhất 4x4
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # Sai số từng điểm
    P_transformed = (R @ P.T).T + t
    errors = np.linalg.norm(P_transformed - Q, axis=1)
    rmse = np.sqrt(((P_transformed - Q) ** 2).sum() / N)

    return T, rmse, errors

def save_transformation_matrix(T, filename="board_to_robot.json"):
    data = {
        "transformation_matrix": T.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved transformation matrix to {filename}")
    
# ===========================
# Thay dữ liệu thực tế vào đây
Z= -362.5
P_board = np.array([
    [0.0, 0.0, 0.0],      # P0
    [50.0, 0.0, 0.0],     # P1
    [0.0, 50.0, 0.0],     # P2
    [50.0, 50.0, 0.0],    # P3
    [-50.0, 0.0, 0.0],    # P4 (thêm điểm)
    [0.0, -50.0, 0.0],    # P5
])

P_robot = np.array([
    [-22.4, -0.8, Z],     # P0
    [27.7, 0.4, Z],       # P1
    [-24, 49.8, Z],    # P2
    [26, 51.8, Z],      # P3
    [-71.4, -2.1, Z],     # P4         
    [-20.5, -52.4, Z]      # P5

])

# Tính toán
T, rmse, errors = compute_T_board_to_robot(P_board, P_robot)

# In kết quả
print("\nMa trận T_board_to_robot:\n", T)
print("\nSai số từng điểm (mm):")
for i, e in enumerate(errors):
    print(f"  Điểm {i}: {e:.3f} mm")
print(f"\nSai số khớp (RMSE): {rmse:.3f} mm")

save_transformation_matrix(T, "T_board_to_robot.json")
