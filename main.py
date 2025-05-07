import sys
import os
import cv2
import numpy as np
import onnxruntime as ort
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                           QComboBox, QSlider, QGroupBox, QRadioButton, QButtonGroup)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

class ObjectDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        # 使用固定的输入尺寸，YOLOv8默认为640x640
        self.img_size = (640, 640)
        
    def preprocess(self, img):
        # 调整图像大小
        img_resized = cv2.resize(img, self.img_size)
        # 转换为RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # 归一化
        img_norm = img_rgb.astype(np.float32) / 255.0
        # 调整维度顺序并添加批次维度
        img_input = np.transpose(img_norm, (2, 0, 1))
        img_input = np.expand_dims(img_input, 0)
        return img_input
    
    def detect(self, img, conf_threshold=0.25, iou_threshold=0.45):
        # 预处理图像
        input_tensor = self.preprocess(img)
        
        # 执行推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 处理输出 (假设输出格式为YOLO格式)
        boxes = self.process_output(outputs[0], img.shape, conf_threshold, iou_threshold)
        return boxes
    
    def process_output(self, output, orig_shape, conf_threshold, iou_threshold):
        # 处理YOLO输出
        # YOLOv8 ONNX输出格式: [batch, 84, 8400] 其中84 = 4(box) + 80(class scores)
        # 需要先转置为 [batch, 8400, 84] 以便处理每个检测结果
        output = np.transpose(output, (0, 2, 1))
        
        boxes = []
        scores = []
        class_ids = []
        
        # 获取原始图像尺寸和输入尺寸
        orig_h, orig_w = orig_shape[0:2]
        input_h, input_w = self.img_size
        
        # 计算缩放比例
        ratio_h = orig_h / input_h
        ratio_w = orig_w / input_w
        
        # 获取置信度大于阈值的检测结果
        for detection in output[0]:
            # 获取类别得分
            class_scores = detection[4:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            
            # 如果类别得分大于阈值，保存检测结果
            if class_score >= conf_threshold:
                # 获取边界框坐标 (x, y, w, h 格式)
                x, y, w, h = detection[0:4]
                
                # 转换为左上角和右下角坐标，并缩放到原始图像尺寸
                x1 = max(0, int((x - w/2) * ratio_w))
                y1 = max(0, int((y - h/2) * ratio_h))
                x2 = min(orig_w, int((x + w/2) * ratio_w))
                y2 = min(orig_h, int((y + h/2) * ratio_h))
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(class_score))
                class_ids.append(class_id)
        
        # 应用非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        
        result = []
        for i in indices:
            # OpenCV 4.x 和 OpenCV 3.x 返回的索引格式不同
            if isinstance(i, (list, tuple)):
                i = i[0]
                
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            
            # 不限制特定类别ID，接受所有检测结果
            result.append({
                'box': box,
                'score': score,
                'class_id': class_id
            })
        
        return result

class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("小米SU7检测系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化变量
        self.image = None
        self.video_path = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 模型路径
        self.model_paths = {
            "YOLOv8n": os.path.join("weight", "v8n.onnx"),
            "YOLOv8s": os.path.join("weight", "v8s.onnx")
        }
        
        # 默认加载YOLOv8n模型
        self.current_model = "YOLOv8n"
        self.detector = ObjectDetector(self.model_paths[self.current_model])
        
        # 设置界面
        self.setup_ui()
    
    def setup_ui(self):
        # 主布局
        main_layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv8n", "YOLOv8s"])
        self.model_combo.currentTextChanged.connect(self.change_model)
        model_layout.addWidget(self.model_combo)
        control_layout.addWidget(model_group)
        
        # 输入选择
        input_group = QGroupBox("输入选择")
        input_layout = QVBoxLayout(input_group)
        self.image_btn = QPushButton("选择图片")
        self.image_btn.clicked.connect(self.load_image)
        self.video_btn = QPushButton("选择视频")
        self.video_btn.clicked.connect(self.load_video)
        self.camera_btn = QPushButton("打开摄像头")
        self.camera_btn.clicked.connect(self.open_camera)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_detection)
        input_layout.addWidget(self.image_btn)
        input_layout.addWidget(self.video_btn)
        input_layout.addWidget(self.camera_btn)
        input_layout.addWidget(self.stop_btn)
        control_layout.addWidget(input_group)
        
        # 检测参数
        param_group = QGroupBox("检测参数")
        param_layout = QVBoxLayout(param_group)
        
        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_label = QLabel("置信度阈值:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(99)
        self.conf_slider.setValue(25)  # 默认0.25
        self.conf_value = QLabel("0.25")
        self.conf_slider.valueChanged.connect(self.update_conf_value)
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value)
        param_layout.addLayout(conf_layout)
        
        # IOU阈值
        iou_layout = QHBoxLayout()
        iou_label = QLabel("IOU阈值:")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(99)
        self.iou_slider.setValue(45)  # 默认0.45
        self.iou_value = QLabel("0.45")
        self.iou_slider.valueChanged.connect(self.update_iou_value)
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_value)
        param_layout.addLayout(iou_layout)
        
        control_layout.addWidget(param_group)
        
        # 添加空白区域
        control_layout.addStretch()
        
        # 右侧显示区域
        display_panel = QWidget()
        display_layout = QVBoxLayout(display_panel)
        
        # 图像显示
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 1px solid black;")
        display_layout.addWidget(self.image_label)
        
        # 状态栏
        status_layout = QHBoxLayout()
        self.status_label = QLabel("就绪")
        status_layout.addWidget(self.status_label)
        display_layout.addLayout(status_layout)
        
        # 添加到主布局
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(display_panel, 4)
        
        # 设置中央窗口
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def update_conf_value(self):
        value = self.conf_slider.value() / 100
        self.conf_value.setText(f"{value:.2f}")
    
    def update_iou_value(self):
        value = self.iou_slider.value() / 100
        self.iou_value.setText(f"{value:.2f}")
    
    def change_model(self, model_name):
        self.current_model = model_name
        self.detector = ObjectDetector(self.model_paths[model_name])
        self.status_label.setText(f"已加载模型: {model_name}")
    
    def load_image(self):
        # 停止任何正在进行的视频检测
        self.stop_detection()
        
        # 打开文件对话框选择图片
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        
        if file_path:
            # 读取图片
            self.image = cv2.imread(file_path)
            if self.image is None:
                self.status_label.setText("无法读取图片")
                return
            
            # 进行检测
            self.detect_image()
    
    def detect_image(self):
        if self.image is None:
            return
        
        # 获取当前参数
        conf_threshold = self.conf_slider.value() / 100
        iou_threshold = self.iou_slider.value() / 100
        
        # 复制图像以便绘制
        display_image = self.image.copy()
        
        # 执行检测
        detections = self.detector.detect(display_image, conf_threshold, iou_threshold)
        
        # 在图像上绘制检测结果
        for det in detections:
            box = det['box']
            score = det['score']
            
            # 绘制边界框
            cv2.rectangle(display_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"Object-{det['class_id']}: {score:.2f}"
            cv2.putText(display_image, label, (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 更新状态
        self.status_label.setText(f"检测到 {len(detections)} 个小米SU7")
        
        # 显示图像
        self.display_image(display_image)
    
    def load_video(self):
        # 停止任何正在进行的视频检测
        self.stop_detection()
        
        # 打开文件对话框选择视频
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov)")
        
        if file_path:
            # 打开视频
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                self.status_label.setText("无法打开视频")
                return
            
            # 开始定时器
            self.timer.start(30)  # 约30fps
            self.status_label.setText("正在播放视频...")
    
    def open_camera(self):
        # 停止任何正在进行的视频检测
        self.stop_detection()
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("无法打开摄像头")
            return
        
        # 开始定时器
        self.timer.start(30)  # 约30fps
        self.status_label.setText("摄像头已打开")
    
    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.stop_detection()
            return
        
        # 读取一帧
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            self.status_label.setText("视频播放结束")
            return
        
        # 获取当前参数
        conf_threshold = self.conf_slider.value() / 100
        iou_threshold = self.iou_slider.value() / 100
        
        # 执行检测
        detections = self.detector.detect(frame, conf_threshold, iou_threshold)
        
        # 在图像上绘制检测结果
        for det in detections:
            box = det['box']
            score = det['score']
            
            # 绘制边界框
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"Object-{det['class_id']}: {score:.2f}"
            cv2.putText(frame, label, (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 更新状态
        self.status_label.setText(f"检测到 {len(detections)} 个小米SU7")
        
        # 显示图像
        self.display_image(frame)
    
    def stop_detection(self):
        # 停止定时器
        self.timer.stop()
        
        # 释放视频资源
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        
        self.status_label.setText("就绪")
    
    def display_image(self, img):
        # 转换为RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建QImage
        h, w, c = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, w * c, QImage.Format_RGB888)
        
        # 创建QPixmap并显示
        pixmap = QPixmap.fromImage(q_img)
        
        # 缩放以适应标签大小
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())