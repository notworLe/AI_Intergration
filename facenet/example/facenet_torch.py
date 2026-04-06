from facenet_pytorch import InceptionResnetV1, MTCNN
from torch.utils.data import DataLoader
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from notworLeUtils.notworle_utils.dataset import FolderImage
from notworLeUtils.notworle_utils.mtcnn import mtcnn_torch_get_face
import pandas as pd

# Set GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# MTCNN
detector_mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# facenet
resnet  = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# Load image
def collate_fn(x):
    """
    Để convert bình thường thay vì tuple khi dùng dataloader
    :param x:
    :return:
    """
    return x[0]

print("Load image")
# Dataset path
dataset_path = Path.cwd() / ".." / ".." / "dataset" / "do_mi_xi" / "raw"
# Read images
do_mi_xi_dataset = FolderImage(dataset_path)
# Load dataset, add collate_fn for solve output
do_mi_xi_dataloader = DataLoader(dataset=do_mi_xi_dataset, collate_fn=collate_fn)

# Save face in "do_mi_xi / processed
path_write = dataset_path / ".." / "processed"

# Face detection
face_list = []
for image, name, image_path in do_mi_xi_dataloader:
    # Detection
    faces, probs = detector_mtcnn.detect(image)
    if faces is not None:
        face_images = mtcnn_torch_get_face(image, faces, probs)
        face_list.extend(face_images)

        # Write image
        # for i, face_image in enumerate(face_images):
        #     # Tên ảnh và đuôi
        #     new_name = f"{image_path.stem}_{i + 1}_{image_path.suffix}"
        #
        #     image_path = path_write / new_name
        #     print(f"Get face: {image_path}")
        #     cv2.imwrite(image_path, face_image)

face_list = torch.stack(face_list).to(device)

embeddings = resnet(face_list).detach().cpu()
cos_sim = [[ F.cosine_similarity(face1, face2, dim=0).item() for face2 in embeddings] for face1 in embeddings]
print(pd.DataFrame(cos_sim))