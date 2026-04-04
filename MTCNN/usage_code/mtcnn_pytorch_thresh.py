from facenet_pytorch import MTCNN
import torch
import cv2
import threading
from myFunctions.notworle_mtcnn_utils import mtcnn_torch_draw
import copy

frame_to_detect = None
result_frame = None
lock = threading.Lock()

def detection_worker():
    global frame_to_detect, result_frame
    while True:
        with lock:
            if frame_to_detect is None:
                continue
            frame = frame_to_detect.copy()  # copy để tránh race condition

        faces, probs = mtcnn_detector.detect(frame)
        if faces is not None:
            mtcnn_torch_draw(frame, faces, probs)

        with lock:
            result_frame = frame  # lưu frame đã vẽ xong

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn_detector = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)

thread_detection = threading.Thread(target=detection_worker, daemon=True)
thread_detection.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    is_cap, frame = cap.read()
    if not is_cap:
        print("Can't capture")
        break
    frame = cv2.flip(frame, 1)

    with lock:
        frame_to_detect = frame  # cập nhật frame mới cho thread
        display = result_frame if result_frame is not None else frame  # hiển thị frame đã detect

    cv2.imshow('Face detection', display)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()