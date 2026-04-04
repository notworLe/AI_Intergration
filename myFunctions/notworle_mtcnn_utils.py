import cv2

def mtcnn_torch_draw(image, face, probs=None, *args, **kwargs):
    count_face = 0
    for box, prob in zip(face, probs):
        # by default x1, y1, x2, y2 is 'object' type
        x1, y1, x2, y2 = [int(x) for x in box]
        cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=(173, 250, 93), *args, **kwargs)
        count_face += 1

    cv2.putText(
        img=image,
        text=f"People: {count_face}",
        org=(50, 50),
        fontFace=cv2.FONT_ITALIC,
        fontScale=1,
        color=(173, 250, 93),
        thickness=2
    )

