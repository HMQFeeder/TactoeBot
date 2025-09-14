import cv2
import numpy as np
import time

cam_id = 0
cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

# === Cấu hình ban đầu ===
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Tắt auto
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DirectShow: 0.25 = manual
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)

# Hàm đo độ sáng và nét
def measure(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return brightness, sharpness

# Dải quét (tùy camera)
exposure_range = np.linspace(-10, -3, 8)  # với DirectShow thường là số âm
focus_range = np.arange(0, 256, 32)       # 0–255
gain_range = np.arange(0, 256, 32)        # 0–255

best_score = -1
best_settings = None

for exp in exposure_range:
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    for focus in focus_range:
        cap.set(cv2.CAP_PROP_FOCUS, focus)
        for gain in gain_range:
            cap.set(cv2.CAP_PROP_GAIN, gain)
            time.sleep(0.2)  # chờ camera ổn định

            ret, frame = cap.read()
            if not ret:
                continue

            brightness, sharpness = measure(frame)

            # Tính điểm (ưu tiên nét và sáng hợp lý)
            score = sharpness - abs(brightness - 140) * 0.5

            if score > best_score:
                best_score = score
                best_settings = (exp, focus, gain, brightness, sharpness)

            # Hiển thị thử
            disp = frame.copy()
            cv2.putText(disp, f"Exp: {exp} Focus: {focus} Gain: {gain}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(disp, f"Bright: {brightness:.1f} Sharp: {sharpness:.1f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("Camera", disp)
            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()

if best_settings:
    exp, focus, gain, bright, sharp = best_settings
    print(f"Thông số tối ưu:")
    print(f"  Exposure: {exp}")
    print(f"  Focus: {focus}")
    print(f"  Gain: {gain}")
    print(f"  Brightness: {bright:.1f}")
    print(f"  Sharpness: {sharp:.1f}")
