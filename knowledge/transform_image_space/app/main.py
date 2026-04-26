import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel,
    QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QSlider, QPushButton
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from transforms import apply_transform


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV2 Transform Demo")
        self.setMinimumSize(900, 600)

        # --- camera ---
        self.cap = cv2.VideoCapture(0)
        self.current_transform = "none"
        self.param = 50

        # --- layout ---
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        root.addWidget(self.video_label, stretch=3)

        # control panel
        ctrl = QVBoxLayout()
        root.addLayout(ctrl, stretch=1)

        # dropdown chọn transform
        self.combo = QComboBox()
        self.combo.addItems([
            "none", "grayscale", "blur",
            "canny", "rotate", "flip",
            "threshold", "perspective",
            "resize"
        ])
        self.combo.currentTextChanged.connect(self.on_transform_change)
        ctrl.addWidget(self.combo)

        # slider param
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.on_param_change)
        ctrl.addWidget(self.slider)

        self.param_label = QLabel("param: 50")
        ctrl.addWidget(self.param_label)

        # snapshot button
        btn = QPushButton("Chụp ảnh")
        btn.clicked.connect(self.snapshot)
        ctrl.addWidget(btn)

        ctrl.addStretch()

        # --- timer cập nhật frame ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 fps

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = apply_transform(frame, self.current_transform, self.param)

        # convert BGR → RGB để hiển thị
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
        )

    def on_transform_change(self, text):
        self.current_transform = text

    def on_param_change(self, val):
        self.param = val
        self.param_label.setText(f"param: {val}")

    def snapshot(self):
        ret, frame = self.cap.read()
        if ret:
            frame = apply_transform(frame, self.current_transform, self.param)
            cv2.imwrite("snapshot.jpg", frame)

    def closeEvent(self, e):
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())