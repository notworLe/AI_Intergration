from facenet_pytorch import MTCNN
import torch
import cv2
from myFunctions.notworle_mtcnn_utils import mtcnn_torch_draw



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Threshold for P, R, O layers
mtcnn_detector = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    is_cap, frame = cap.read()
    if not is_cap:
        print("Can't capture")
        break
    frame = cv2.flip(frame, 1)

    faces, probs = mtcnn_detector.detect(frame)
    if faces is not None:
        mtcnn_torch_draw(frame, faces, probs)

    cv2.imshow('Face detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()