# ================== 导入必要的库 ==================
import sys
import os
import torch
import torchvision   # 用于跨类别 NMS
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import time
import numpy as np
from ultralytics import YOLO

# ================== 路径处理（支持 PyInstaller 打包）==================
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# ================== 全局配置 ==================
CONF_THRES = 0.25          # 置信度阈值，低于此值的检测框被丢弃
IOU_NMS = 0.2              # 跨类别 NMS 的 IoU 阈值，重叠超过此值的框会被合并
SAVE_DIR = r"./detection_results"   # 检测结果保存目录
MODEL_PATH = os.path.join(base_path, 'best.pt')
model = YOLO(MODEL_PATH)   # 加载 YOLOv8 模型

os.makedirs(SAVE_DIR, exist_ok=True)  # 创建保存目录（如果不存在）

# 成熟度到边框颜色的映射（BGR 格式）
COLOR_MAP = {
    "immature": (255, 0, 0),   # 未成熟 → 蓝色
    "mature": (0, 255, 0),     # 成熟 → 绿色
    "overmature": (0, 0, 255)  # 过熟 → 红色
}

# ================== 成熟度修正（基于色调和饱和度的联合规则）==================
def maturity_fusion_v2(cls_name, h_mean, s_mean, l_mean):
    """
    根据 HSL 色调、饱和度、亮度联合修正成熟度。
    规则：
        1. 若 YOLO 判为成熟/过熟，且亮度 > 50（避免阴暗误判），
           色调在青绿范围(35~85)且饱和度<90 → 改为未成熟
        2. 若 YOLO 判为未成熟，色调在红色范围(0~10 或 156~180)且饱和度>100 → 改为成熟
        3. 其他情况保持原类别。
    """

    # 成熟/过熟修正为未成熟：仅在亮度足够时进行（避免阴暗环境下误判）
    if cls_name != "immature" and l_mean > 50:
        if (35 <= h_mean <= 85) and s_mean < 90:
            return "immature"
    # 未成熟修正为成熟
    if cls_name == "immature":
        if (h_mean <= 10 or h_mean >= 156) and s_mean > 100:
            return "mature"
    return cls_name

# ================== 遮挡补偿：基于颜色分割估算苹果个数 ==================
def estimate_occluded_by_color(roi, cls_name, normal_area):
    """
    对面积过大的检测框（怀疑有重叠），利用 HSL 颜色分割估算内部苹果个数。
    步骤：
        1. 将 ROI 转为 HLS 颜色空间。
        2. 根据成熟度选择颜色阈值（青绿或红色），生成二值掩码。
        3. 形态学操作去噪，查找轮廓。
        4. 过滤小轮廓，统计有效轮廓数。
        5. 若轮廓数在 1~10 之间则返回，否则回退到面积比例估算。
    """
    if roi.size == 0:
        return 1
    hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    if cls_name == "immature":
        lower = np.array([35, 0, 50])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hls, lower, upper)
    else:
        lower1 = np.array([0, 0, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([156, 0, 50])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hls, lower1, upper1)
        mask2 = cv2.inRange(hls, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # 闭运算填充孔洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # 开运算去除噪点
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = normal_area / 10   # 过滤极小噪声轮廓
    valid = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    cnt_count = len(valid)
    if 1 <= cnt_count <= 10:
        return cnt_count
    else:
        area = roi.shape[0] * roi.shape[1]
        return max(1, round(area / normal_area))

# ================== 遮挡判定（边缘密度 + 颜色占比）==================
def is_occluded(roi, cls_name, conf):
    """
    判断检测框是否被严重遮挡（枝叶遮挡）。
    特征：
        1. 边缘密度（Canny 边缘占比）高。
        2. 绿色像素占比高。
        3. 红色占比低（成熟特征不明显）。
    返回 True 表示需要二次检测修正成熟度。
    """
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_ratio = np.sum(green_mask > 0) / (roi.shape[0] * roi.shape[1])

    red_mask1 = cv2.inRange(hsv, (0, 40, 40), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (156, 40, 40), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_ratio = np.sum(red_mask > 0) / (roi.shape[0] * roi.shape[1])

    # 如果红色占比明显高且绿色不多，认为不是遮挡（成熟特征显著）
    if red_ratio > 0.2 and green_ratio < 0.4:
        return False
    # 边缘密度高且绿色多，或者置信度低且绿色多 → 遮挡
    return (edge_ratio > 0.15 and green_ratio > 0.3) or (conf < 0.5 and green_ratio > 0.3)

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

    # 1. 获取原始检测结果（不进行内置 NMS，设置 iou=1.0）
    raw_results = model(img, iou=1.0, conf=CONF_THRES)[0]

    if raw_results.boxes is None or len(raw_results.boxes) == 0:
        # 没有检测到苹果：显示原图，清空统计
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((520, 420))
        img_tk = ImageTk.PhotoImage(img_pil)
        label_img.config(image=img_tk)
        label_img.image = img_tk
        stat_mature.set("mature: 0")
        stat_immature.set("immature: 0")
        stat_overmature.set("overmature: 0")
        stat_total.set("总数: 0")
        stat_fps.set("FPS: --")
        return

    # 2. 提取所有检测框的坐标、置信度、类别
    boxes = raw_results.boxes.xyxy.cpu()      # (N, 4) 每个框 [x1,y1,x2,y2]
    scores = raw_results.boxes.conf.cpu()     # (N,) 置信度
    classes = raw_results.boxes.cls.cpu()     # (N,) 类别索引

    # 3. 跨类别 NMS：将不同类别的框视为同一类，仅根据位置和置信度合并
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_NMS)
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    fps = 1 / (time.time() - start_time)

    # 4. 构建 boxes_data 列表供后续处理
    boxes_data = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i].tolist())
        area = (x2 - x1) * (y2 - y1)
        boxes_data.append({
            'box': (x1, y1, x2, y2),
            'conf': float(scores[i]),
            'cls_name': model.names[int(classes[i])],
            'area': area
        })

    # 5. 计算正常单果面积（中位数，用于遮挡补偿）
    all_areas = [d['area'] for d in boxes_data]
    normal_area = np.median(all_areas)

    count_dict = {"immature": 0, "mature": 0, "overmature": 0}
    total_estimated = 0

    for data in boxes_data:
        x1, y1, x2, y2 = data['box']
        conf = data['conf']
        cls_name = data['cls_name']
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # 判断是否被严重遮挡
        occluded = is_occluded(roi, cls_name, conf)

        if occluded:
            # 遮挡框：二次检测修正成熟度，并估算内部苹果个数
            sub_results = model(roi, iou=0.3)[0]
            if sub_results.boxes is not None and len(sub_results.boxes) > 0:
                best = sub_results.boxes[0]   # 取置信度最高的框
                new_conf = float(best.conf[0])
                new_cls_name = model.names[int(best.cls[0])]
                if new_conf > conf:
                    cls_name = new_cls_name
                    conf = new_conf
            estimated = estimate_occluded_by_color(roi, cls_name, normal_area)
        else:
            # 非遮挡框：单个苹果，用颜色规则微调成熟度
            estimated = 1
            hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
            h_mean = hls[:, :, 0].mean()   # 色调均值
            s_mean = hls[:, :, 2].mean()   # 饱和度均值
            l_mean = hls[:, :, 1].mean()   # 饱和度均值
            cls_name = maturity_fusion_v2(cls_name, h_mean, s_mean,l_mean)

        total_estimated += estimated
        color = COLOR_MAP[cls_name]
        # 绘制矩形框和文本
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{cls_name} {conf:.2f}",
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        count_dict[cls_name] += estimated

    # 更新 GUI 底部统计栏
    stat_mature.set(f"mature: {count_dict['mature']}")
    stat_immature.set(f"immature: {count_dict['immature']}")
    stat_overmature.set(f"overmature: {count_dict['overmature']}")
    stat_total.set(f"total: {total_estimated}")
    stat_fps.set(f"FPS: {fps:.1f}")

    # 保存检测结果图片
    base_name = os.path.basename(file_path)
    save_path = os.path.join(SAVE_DIR, f"detected_{base_name}")
    cv2.imwrite(save_path, img)
    print(f"检测结果已保存至: {save_path}")

    # 在 GUI 中显示结果图片
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((520, 420))
    img_tk = ImageTk.PhotoImage(img_pil)
    label_img.config(image=img_tk)
    label_img.image = img_tk

# ================== 视频文件检测 ==================
def detect_video():
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

    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    base_name = os.path.basename(video_path)
    save_video_path = os.path.join(SAVE_DIR, f"detected_{base_name}")
    out = cv2.VideoWriter(save_video_path, fourcc, fps_orig, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        raw_results = model(frame, iou=1.0, conf=CONF_THRES)[0]

        if raw_results.boxes is None or len(raw_results.boxes) == 0:
            cv2.putText(frame, "No apple detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(frame)
            cv2.imshow("Video Detection (Press Q to exit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        boxes = raw_results.boxes.xyxy.cpu()
        scores = raw_results.boxes.conf.cpu()
        classes = raw_results.boxes.cls.cpu()
        keep = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_NMS)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        fps = 1 / (time.time() - start_time)

        boxes_data = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i].tolist())
            area = (x2 - x1) * (y2 - y1)
            boxes_data.append({
                'box': (x1, y1, x2, y2),
                'conf': float(scores[i]),
                'cls_name': model.names[int(classes[i])],
                'area': area
            })

        all_areas = [d['area'] for d in boxes_data]
        normal_area = np.median(all_areas)

        count_dict = {"immature": 0, "mature": 0, "overmature": 0}
        total_estimated = 0

        for data in boxes_data:
            x1, y1, x2, y2 = data['box']
            conf = data['conf']
            cls_name = data['cls_name']
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            occluded = is_occluded(roi, cls_name, conf)

            if occluded:
                sub_results = model(roi, iou=0.3)[0]
                if sub_results.boxes is not None and len(sub_results.boxes) > 0:
                    best = sub_results.boxes[0]
                    new_conf = float(best.conf[0])
                    new_cls_name = model.names[int(best.cls[0])]
                    if new_conf > conf:
                        cls_name = new_cls_name
                        conf = new_conf
                estimated = estimate_occluded_by_color(roi, cls_name, normal_area)
            else:
                estimated = 1
                hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
                h_mean = hls[:, :, 0].mean()  # 色调均值
                s_mean = hls[:, :, 2].mean()  # 饱和度均值
                l_mean = hls[:, :, 1].mean()  # 饱和度均值
                cls_name = maturity_fusion_v2(cls_name, h_mean, s_mean, l_mean)

            total_estimated += estimated
            color = COLOR_MAP[cls_name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}",
                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            count_dict[cls_name] += estimated

        # 在视频帧左上角绘制统计信息
        y_offset = 30
        for cls in ["immature", "mature", "overmature"]:
            cv2.putText(frame, f"{cls}: {count_dict[cls]}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MAP[cls], 2)
            y_offset += 30
        cv2.putText(frame, f"Total: {total_estimated}  FPS: {fps:.1f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        out.write(frame)
        cv2.imshow("Video Detection (Press Q to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"检测视频已保存至: {save_video_path}")

# ================== 摄像头实时检测 ==================
def detect_camera():
    cap = cv2.VideoCapture(0)
    save_video = messagebox.askyesno("保存选项", "是否保存摄像头检测过程为视频文件？")
    out = None
    if save_video:
        fps_cam = cap.get(cv2.CAP_PROP_FPS)
        if fps_cam <= 0:
            fps_cam = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_path = os.path.join(SAVE_DIR, "camera_detection.mp4")
        out = cv2.VideoWriter(save_path, fourcc, fps_cam, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        raw_results = model(frame, iou=1.0, conf=CONF_THRES)[0]

        if raw_results.boxes is None or len(raw_results.boxes) == 0:
            cv2.putText(frame, "No apple detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if out is not None:
                out.write(frame)
            cv2.imshow("Apple Detection System (Press Q to exit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        boxes = raw_results.boxes.xyxy.cpu()
        scores = raw_results.boxes.conf.cpu()
        classes = raw_results.boxes.cls.cpu()
        keep = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_NMS)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        fps = 1 / (time.time() - start_time)

        boxes_data = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i].tolist())
            area = (x2 - x1) * (y2 - y1)
            boxes_data.append({
                'box': (x1, y1, x2, y2),
                'conf': float(scores[i]),
                'cls_name': model.names[int(classes[i])],
                'area': area
            })

        all_areas = [d['area'] for d in boxes_data]
        normal_area = np.median(all_areas)

        count_dict = {"immature": 0, "mature": 0, "overmature": 0}
        total_estimated = 0

        for data in boxes_data:
            x1, y1, x2, y2 = data['box']
            conf = data['conf']
            cls_name = data['cls_name']
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            occluded = is_occluded(roi, cls_name, conf)

            if occluded:
                sub_results = model(roi, iou=0.3)[0]
                if sub_results.boxes is not None and len(sub_results.boxes) > 0:
                    best = sub_results.boxes[0]
                    new_conf = float(best.conf[0])
                    new_cls_name = model.names[int(best.cls[0])]
                    if new_conf > conf:
                        cls_name = new_cls_name
                        conf = new_conf
                estimated = estimate_occluded_by_color(roi, cls_name, normal_area)
            else:
                estimated = 1
                hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
                h_mean = hls[:, :, 0].mean()  # 色调均值
                s_mean = hls[:, :, 2].mean()  # 饱和度均值
                l_mean = hls[:, :, 1].mean()  # 饱和度均值
                cls_name = maturity_fusion_v2(cls_name, h_mean, s_mean, l_mean)

            total_estimated += estimated
            color = COLOR_MAP[cls_name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}",
                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            count_dict[cls_name] += estimated

        # 绘制统计信息
        y_offset = 30
        for cls in ["immature", "mature", "overmature"]:
            cv2.putText(frame, f"{cls}: {count_dict[cls]}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MAP[cls], 2)
            y_offset += 30
        cv2.putText(frame, f"Total: {total_estimated}  FPS: {fps:.1f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if out is not None:
            out.write(frame)

        cv2.imshow("Apple Detection System (Press Q to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out is not None:
        out.release()
        print(f"摄像头检测视频已保存至: {save_path}")
    cv2.destroyAllWindows()

# ================== GUI 界面构建 ==================
root = tk.Tk()
root.title("苹果实时检测与成熟度分级系统")
root.geometry("800x750")
root.configure(bg="#f0f2f5")

# 设置样式
style = ttk.Style()
style.theme_use("clam")
style.configure("Custom.TButton",
                font=("微软雅黑", 12, "bold"),
                foreground="#2c3e50",
                background="#3498db",
                borderwidth=0,
                focuscolor="none",
                relief="flat")
style.map("Custom.TButton",
          background=[("active", "#2980b9")],
          foreground=[("active", "white")])
style.configure("Title.TLabel",
                font=("微软雅黑", 14, "bold"),
                foreground="#2c3e50",
                background="#f0f2f5")
style.configure("Subtitle.TLabel",
                font=("微软雅黑", 10),
                foreground="#7f8c8d",
                background="#f0f2f5")

# 主框架
main_frame = tk.Frame(root, bg="#f0f2f5")
main_frame.pack(expand=True, fill="both", padx=20, pady=20)

# 标题区域
title_label = ttk.Label(main_frame, text="🍎 苹果检测与成熟度分级系统", style="Title.TLabel")
title_label.pack(pady=(0, 10))
subtitle_label = ttk.Label(main_frame, text="基于 YOLOv8 的实时检测 | 遮挡补偿 | 成熟度分级", style="Subtitle.TLabel")
subtitle_label.pack(pady=(0, 20))

# 按钮区域
button_frame = tk.Frame(main_frame, bg="#f0f2f5")
button_frame.pack(pady=10)

btn_img = ttk.Button(button_frame, text="📷 图片检测", command=detect_image, style="Custom.TButton", width=15)
btn_img.pack(side="left", padx=10, ipady=5)

btn_video = ttk.Button(button_frame, text="🎥 视频文件检测", command=detect_video, style="Custom.TButton", width=15)
btn_video.pack(side="left", padx=10, ipady=5)

btn_cam = ttk.Button(button_frame, text="📹 摄像头实时检测", command=detect_camera, style="Custom.TButton", width=15)
btn_cam.pack(side="left", padx=10, ipady=5)

# 图片显示区域
img_frame = tk.Frame(main_frame, bg="#ffffff", relief="solid", bd=1, padx=2, pady=2)
img_frame.pack(pady=10, fill="both", expand=True)
label_img = tk.Label(img_frame, bg="#ffffff", text="等待选择图片或视频...", font=("微软雅黑", 12), fg="#95a5a6")
label_img.pack(expand=True, fill="both", padx=5, pady=5)

# 底部统计栏
stat_frame = tk.Frame(main_frame, bg="#f0f2f5")
stat_frame.pack(side="bottom", fill="x", pady=10)
stat_immature = tk.StringVar(value="immature: 0")
stat_mature = tk.StringVar(value="mature: 0")
stat_overmature = tk.StringVar(value="overmature: 0")
stat_total = tk.StringVar(value="总数: 0")
stat_fps = tk.StringVar(value="FPS: --")

tk.Label(stat_frame, textvariable=stat_immature, font=("微软雅黑", 11), fg="#3498db", bg="#f0f2f5").grid(row=0, column=0, padx=15)
tk.Label(stat_frame, textvariable=stat_mature, font=("微软雅黑", 11), fg="#2ecc71", bg="#f0f2f5").grid(row=0, column=1, padx=15)
tk.Label(stat_frame, textvariable=stat_overmature, font=("微软雅黑", 11), fg="#e67e22", bg="#f0f2f5").grid(row=0, column=2, padx=15)
tk.Label(stat_frame, textvariable=stat_total, font=("微软雅黑", 11, "bold"), fg="#2c3e50", bg="#f0f2f5").grid(row=0, column=3, padx=15)
tk.Label(stat_frame, textvariable=stat_fps, font=("微软雅黑", 11), fg="#7f8c8d", bg="#f0f2f5").grid(row=0, column=4, padx=15)

# 底部提示信息
info_label = ttk.Label(main_frame, text="检测结果自动保存至 ./detection_results 文件夹", style="Subtitle.TLabel")
info_label.pack(side="bottom", pady=5)

# 窗口居中显示
root.update_idletasks()
width = root.winfo_width()
height = root.winfo_height()
x = (root.winfo_screenwidth() // 2) - (width // 2)
y = (root.winfo_screenheight() // 2) - (height // 2)
root.geometry(f"{width}x{height}+{x}+{y}")

root.mainloop()

"""
1 跨类别 NMS：使用 torchvision.ops.nms 对所有检测框（不分成熟度类别）进行合并，彻底解决同一苹果出现多个不同类别框的问题。阈值 IOU_NMS=0.2 可根据实际效果调整。

2 遮挡判定：综合边缘密度、绿色占比、红色占比和置信度，判断是否为枝叶遮挡。红色占比高的成熟苹果不会被误判为遮挡。

3 二次检测：对遮挡框裁剪后再次送入 YOLO 模型，用局部特写重新识别成熟度，修正结果。

4 颜色修正：非遮挡框使用色调+饱和度联合规则微调成熟度，进一步提高准确率。

5 计数补偿：对面积过大的框采用 HSL 颜色分割估算内部苹果个数，减少漏计。

6 GUI 支持：图片、视频、摄像头三种模式均已实现，底部统计栏实时更新。
"""