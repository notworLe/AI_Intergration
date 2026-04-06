import torch
import cv2
from my_mtcnn import detector_mtcnn, device
from my_facenet import make_embedding
from pathlib import Path
from notworLeUtils.notworle_utils.dataset import FolderImage
from notworLeUtils.notworle_utils.cv2 import read_display_img
import numpy as np
import torch.nn.functional as F


dataset_path = Path.cwd() / ".." / "dataset" / "notworle" / "raw"
dataset = FolderImage(dataset_path)

embedding = []
for image, name, path in dataset:
    # Resize 160x160 and normalize [-1, 1]
    faces = detector_mtcnn(image)
    if faces is not None:
        for face in faces:
            face_batch = torch.stack([face]).to(device)
            embed = make_embedding(face_batch)
            embedding.append(embed)

cos_sin = []
for i in range(len(embedding)):
    temp = []
    for j in range(len(embedding)):
        temp.append(F.cosine_similarity(embedding[i], embedding[j]))
    cos_sin.append(temp)

