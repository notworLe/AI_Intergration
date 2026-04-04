from mtcnn import MTCNN
from mtcnn.utils.images import load_image
import cv2
from myFunctions.notworlecv2 import read_display_img

# create detector instance
detector = MTCNN()

# Load image
image_path = '../dataset/do_mi_xi/do2.jpg'
do_mi_xi = cv2.imread('../dataset/do_mi_xi/do2.jpg')

# Read for detector
# image = load_image(image_path)

# Detect faces in image
result = detector.detect_faces(do_mi_xi.copy())
print(result)
for face in result:
    x, y, width, height = face['box']
    cv2.rectangle(do_mi_xi, pt1=(x, y), pt2=(x + width, y + height), color=(173, 250, 93), thickness=2)


read_display_img(do_mi_xi)