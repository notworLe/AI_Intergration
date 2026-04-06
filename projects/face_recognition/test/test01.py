import torch
import cv2
from notworLeUtils.notworle_utils.cv2 import read_display_img

x = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])

print(x.mean(dim=0))  # theo cột
print(x.mean(dim=1))  # theo hàng

img = cv2.imread(r'D:\gitvanhub\Computer_vision_intergration\project\dataset\notworle\raw\WIN_20260406_15_49_58_Pro.jpg', cv2.IMREAD_COLOR_RGB)
read_display_img(img)