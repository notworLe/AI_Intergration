import torch
from facenet_pytorch import InceptionResnetV1

device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
resnet = InceptionResnetV1(
    pretrained='vggface2',  # ← phải có cái này, nếu None thì weights random
    classify=False
).eval().to(device)


def make_embedding(face_images_batch):
    # Resnet expected tuple
    embedding_face = resnet(face_images_batch).detach()
    return embedding_face