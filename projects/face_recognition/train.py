from my_mtcnn import detector_mtcnn, device
from my_facenet import make_embedding
from pathlib import Path
from notworle_utils.dataset import FolderImage
import numpy as np
from database.db import update_embedding1
import cv2
import torch

class UserFaceRecognition:
    def __init__(self, user_path:Path, dataset=False):
        self.user_path = user_path
        self.embedding = None
        if dataset:
            self.dataset_raw = FolderImage(user_path / "raw")
        self.name = user_path.name

    def __str__(self):
        return f"{self.name}"

# Dataset path
dataset_path = (Path.cwd() / "dataset").resolve()

# Path of all users
user_paths = list(dataset_path.iterdir())

# Create user object
users_list = [UserFaceRecognition(user_path, dataset=True) for user_path in user_paths]

# Loop each user
for i in range(len(users_list)):
    # Loop each dataset of each user
    temp_embed = []
    for image, name, path in users_list[i].dataset_raw:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # It can return one or more faces
        faces = detector_mtcnn(image)
        if faces is not None:
            faces = faces.to(device)
            # Make sure only have 1 face
            n_faces = len(faces)
            if len(faces) <= 1:
                # Resnet expected batch
                # face_batch = torch.stack([faces]).to(device)

                # Embedding face
                embed = make_embedding(faces)

                temp_embed.append(embed)
            else:
                print(f"Can't solve more one face: {path} ")
        else:
            print(f"No face: {path}")

    # End loop each dataset of each user
    users_list[i].embedding = torch.stack(temp_embed).mean(dim=0).cpu().numpy()

# Update to db
for i in range(len(users_list)):
    update_embedding1(users_list[i].name, users_list[i].embedding)

