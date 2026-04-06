from face_recognition.my_mtcnn import detector_mtcnn, device
import cv2

image = cv2.imread(r'D:\gitvanhub\Computer_vision_intergration\project\dataset\do_mi_xi\raw\download (4).jpg', cv2.COLOR_BGR2RGB)
faces, ss = detector_mtcnn.detect(image)
print(len(faces))