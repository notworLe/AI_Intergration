import cv2
import torch
from my_facenet import make_embedding, device
from my_mtcnn import detector_mtcnn
from database.db import get_embedding
import torch.nn.functional as F


if __name__ == '__main__':
    embedding_db = [
        (row[0], torch.tensor(row[1], dtype=torch.float32).squeeze().to(device))
        for row in get_embedding()
    ]

    cap = cv2.VideoCapture(0)
    THRESHOLD = 0.7

    while True:
        is_cap, frame = cap.read()
        if is_cap is None:
            print("Capture failure")
            break
        # Avoid reverse camera
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Return faces with boxes
        boxes, probs = detector_mtcnn.detect(frame_rgb)

        # Face has post processing 160x160 [-1,1]
        faces = detector_mtcnn(frame_rgb)

        if faces is not None and len(faces) == len(boxes):
            faces = faces.to(device)
            embedding = make_embedding(faces)

            for embedd, box in zip(embedding, boxes):
                embedd = embedd.squeeze(0)
                cos_sim = [
                    F.cosine_similarity(embedd, emb, dim=0).item()
                    for _, emb in embedding_db
                ]
                best_idx = int(torch.argmax(torch.tensor(cos_sim)))
                best_score = cos_sim[best_idx]

                if best_score > THRESHOLD:
                    label = f"{embedding_db[best_idx][0]} {best_score:.2f}"
                    color = (0, 255, 0)
                else:
                    label = f"Unknown {best_score:.2f}"
                    color = (0, 0, 255)

                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            print("No face detection")

        cv2.imshow("Face recognition", frame)
        if cv2.waitKey(1) == ord('q'):
            break