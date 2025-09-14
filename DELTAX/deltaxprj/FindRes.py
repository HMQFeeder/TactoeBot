import cv2

# Danh sách độ phân giải phổ biến
common_res = [
    (320, 240),
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 1024),
    (1600, 1200),
    (1920, 1080),
    (2048, 1536),
    (2560, 1440),
    (2592, 1944),
    (3840, 2160)  # 4K
]

# Các backend phổ biến trên Windows
backends = [
    (cv2.CAP_DSHOW, "DirectShow")
]

# Các codec phổ biến
codecs = [
    ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
    ('YUYV', cv2.VideoWriter_fourcc(*'YUYV')),
    ('H264', cv2.VideoWriter_fourcc(*'H264')),
    ('RAW ', cv2.VideoWriter_fourcc(*'RAW '))
]

for backend_id, backend_name in backends:
    print(f"\n=== Backend: {backend_name} ===")
    cap = cv2.VideoCapture(0, backend_id)

    if not cap.isOpened():
        print("Không mở được camera!")
        continue

    for codec_name, codec in codecs:
        print(f"\n--- Codec: {codec_name} ---")
        for w, h in common_res:
            cap.set(cv2.CAP_PROP_FOURCC, codec)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

            ret, frame = cap.read()
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            status = "✅" if ret else "❌"
            print(f"{status} Requested: {w}x{h} | Actual: {actual_w}x{actual_h}")

    cap.release()
