import subprocess
import sys
import os

# Thư mục làm việc (ví dụ chính là thư mục chứa innit.py)
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Khởi động Acamera ngay
print("[innit] Khởi động Acamera trong work folder...")
subprocess.Popen(
    ["start", "cmd", "/k", f"{sys.executable} Acamera.py"],
    cwd=WORK_DIR,
    shell=True
)

# 2. Chờ bạn nhấn phím 'i' để mở Arobot
print("[innit] Nhấn phím 'i' rồi Enter để khởi động Arobot...")
user_input = input().strip().lower()

if user_input == "i":
    print("[innit] Khởi động Arobot trong work folder...")
    subprocess.Popen(
        ["start", "cmd", "/k", f"{sys.executable} Arobot.py"],
        cwd=WORK_DIR,
        shell=True
    )
elif user_input == "q":
    print("Thoát")
else:
    print("[innit] Bạn không nhấn 'i', Arobot sẽ không được khởi động.")
