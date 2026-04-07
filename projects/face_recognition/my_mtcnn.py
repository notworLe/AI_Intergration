from facenet_pytorch import MTCNN
import torch
import cv2
from pathlib import Path
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detector_mtcnn = MTCNN(
    image_size=160,
    margin=0,
    keep_all=True,
    device=device,
).to(device)



# Lỗi thời
def detect_processed(image_path:Path):
    if not isinstance(image_path, Path):
        image_path = Path(image_path)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Can't read image, {image_path}")

    faces, probs = detector_mtcnn.detect(image)
    if faces is not None:
        for face, prob in zip(faces, probs):
            x1, y1, x2, y2 = [int(x) for x in face]
            face_crop = image[y1:y2, x1:x2]

            new_path = image_path.parent.parent / "processed" / image_path.name
            cv2.imwrite(new_path, face_crop)
    else:
        print(f"No face in {image_path}")

# Lỗi thời
def detect_post_processing(image:Path):
    """
    Detect and convert to 160 x 160 with tensor
    :param image_path:
    :return: batch of tensor
    """
    if not isinstance(image, np.ndarray):
        image = cv2.imread(image)

    if image is None:
        raise ValueError("Can't read image")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_images = detector_mtcnn(image)
    if face_images is None:
        raise ValueError("Can't detect faces")
    return face_images.to(device)







if __name__ == '__main__':
    detect = detect_post_processing(r"D:\gitvanhub\Computer_vision_intergration\dataset\many_faces.jpg")
    print(type(detect))
    print(detect.shape)
    print(detect[0].shape)
