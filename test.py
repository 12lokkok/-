import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import time
import numpy as np
from ultralytics import YOLO

# ================== 全局配置 ==================
CONF_THRES = 0.25
MODEL_PATH = r"E:\apple_dataset\train0\weights\best.pt"
model = YOLO(MODEL_PATH)

# ================== 成熟度 → 颜色映射 ==================
COLOR_MAP = {
    "immature": (255, 0, 0),     # 蓝色
    "mature": (0, 255, 0),       # 绿色
    "overmature": (0, 0, 255)    # 红色
}

# ================== 成熟度融合判定 ==================
def maturity_fusion(cls_name, s_mean):
    final_class = cls_name
    if cls_name != "immature":
        if s_mean < 80:
            final_class = "immature"
    return final_class

# # ================== 绘制颜色图例（Legend）==================
# def draw_legend(img):
#     """
#     绘制成熟度颜色图例，提高系统可视化与可解释性
#     """
#     x, y = 10, 120
#     box_size = 15
#     gap = 25
#
#     for cls in ["immature", "mature", "overmature"]:
#         color = COLOR_MAP[cls]
#
#         cv2.rectangle(
#             img,
#             (x, y),
#             (x + box_size, y + box_size),
#             color,
#             -1
#         )
#
#         cv2.putText(
#             img,
#             cls,
#             (x + box_size + 10, y + box_size - 2),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (0, 0, 0),
#             1
#         )
#
#         y += gap

# ================== 图片检测 ==================
def detect_image():
    file_path = filedialog.askopenfilename(
        title="选择图片",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    img = cv2.imread(file_path)

    start_time = time.time()
    results = model(img)[0]
    fps = 1 / (time.time() - start_time)

    total_count = 0
    count_dict = {"immature": 0, "mature": 0, "overmature": 0}

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        s_mean = hsv[:, :, 1].mean()

        final_class = maturity_fusion(cls_name, s_mean)
        color = COLOR_MAP[final_class]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{final_class} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        total_count += 1
        count_dict[final_class] += 1

    # 统计文字
    y_offset = 30
    for cls in ["immature", "mature", "overmature"]:
        cv2.putText(
            img,
            f"{cls}: {count_dict[cls]}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_MAP[cls],
            2
        )
        y_offset += 25

    cv2.putText(
        img,
        f"Total: {total_count}  FPS: {fps:.1f}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

    # # 绘制图例
    # draw_legend(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((520, 420))
    img_tk = ImageTk.PhotoImage(img)

    label_img.config(image=img_tk)
    label_img.image = img_tk

def detect_video():
    """
    视频文件检测功能
    支持本地视频文件输入，实现逐帧检测与可视化
    """
    video_path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )

    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        results = model(frame)[0]
        fps = 1 / (time.time() - start_time)

        total_count = 0
        count_dict = {"immature": 0, "mature": 0, "overmature": 0}

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            s_mean = hsv[:, :, 1].mean()

            final_class = maturity_fusion(cls_name, s_mean)
            color = COLOR_MAP[final_class]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, final_class,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            total_count += 1
            count_dict[final_class] += 1

        # ===== 统计显示 =====
        y_offset = 30
        for cls in ["immature", "mature", "overmature"]:
            cv2.putText(
                frame,
                f"{cls}: {count_dict[cls]}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_MAP[cls],
                2
            )
            y_offset += 30

        cv2.putText(
            frame,
            f"Total: {total_count}  FPS: {fps:.1f}",
            (10, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        # draw_legend(frame)

        cv2.imshow("Video Detection (Press Q to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================== 摄像头实时检测 ==================
def detect_camera():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        results = model(frame)[0]
        fps = 1 / (time.time() - start_time)

        total_count = 0
        count_dict = {"immature": 0, "mature": 0, "overmature": 0}

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            s_mean = hsv[:, :, 1].mean()

            final_class = maturity_fusion(cls_name, s_mean)
            color = COLOR_MAP[final_class]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, final_class,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            total_count += 1
            count_dict[final_class] += 1

        y_offset = 30
        for cls in ["immature", "mature", "overmature"]:
            cv2.putText(
                frame,
                f"{cls}: {count_dict[cls]}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_MAP[cls],
                2
            )
            y_offset += 30

        cv2.putText(
            frame,
            f"Total: {total_count}  FPS: {fps:.1f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        # # 绘制图例
        # draw_legend(frame)

        cv2.imshow("Apple Detection System (Press Q to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================== GUI ==================
root = tk.Tk()
root.title("苹果实时检测与成熟度分级系统")
root.geometry("720x650")

btn_img = tk.Button(root, text="图片检测", command=detect_image)
btn_img.pack(pady=10)

btn_video = tk.Button(root, text="视频文件检测", command=detect_video)
btn_video.pack(pady=10)

btn_cam = tk.Button(root, text="摄像头实时检测", command=detect_camera)
btn_cam.pack(pady=10)

label_img = tk.Label(root)
label_img.pack()

root.mainloop()


