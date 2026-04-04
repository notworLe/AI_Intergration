import cv2
import threading

import torch
from mtcnn import MTCNN


def mtcnn_draw_faces(result_detection, image, *args, **kwargs):
    count_face = 0
    for face in result_detection:
        x, y, w, h = face['box']
        count_face +=1

        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(173, 250, 93), *args, **kwargs)

    cv2.putText(
        image,
        f"People: {str(count_face)}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

detector = MTCNN()

latest_frame = None
latest_result = []

# Avoid race condition
lock = threading.Lock()

def detection_worker():
    global latest_result
    while True:
        with lock:
            frame = latest_frame
        if frame is None:
            continue

        result= detector.detect_faces(frame)

        with lock:
            latest_result = result

thread = threading.Thread(target=detection_worker, daemon=True)
thread.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)

        with lock:
            latest_frame = frame.copy()
            result = latest_result
        mtcnn_draw_faces(result, frame)

    cv2.imshow("MTCNN", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()