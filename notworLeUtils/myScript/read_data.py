import torchvision
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import random

def is_path_exist(path: str):
    current_path = os.getcwd()
    path = os.path.join(current_path, path)
    if not os.path.exists(path):
        raise ValueError(f'{path} isn\'t existed')
        
class Object18Dataset(Dataset):
    def __init__(self, root, n_cate=None, size=None):
        # Kiểm tra root có tồn tại không
        root = os.path.join(os.getcwd(), root)
        is_path_exist(root)

        self.categorical = os.listdir(root)
        self.images = []
        self.labels = []


        for label, cate in enumerate(self.categorical):
            if n_cate and label == n_cate:
                break

            # path của categorical folder
            cate_path = os.path.join(root, cate)

            # trong từng categorical folder có những ảnh nào
            image_name_list = os.listdir(cate_path)


            for index, name in enumerate(image_name_list.copy()):
                if size == index:
                    break
                if size:
                    # chose random image index
                    image_index = random.randrange(len(image_name_list))

                    # pop theo index
                    image_name = image_name_list.pop(image_index)

                    # attach with parent path
                    image_path = os.path.join(cate_path, image_name)
                else:
                    # Path của image
                    image_path = os.path.join(cate_path, name)

                self.labels.append(label)
                self.images.append(image_path)

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR_BGR)
        image = MyImage(image)
        label = self.labels[index]
        return image, label


class MyImage:
    def __init__(self, image, path=False):
        if path:
            self.image = cv2.imread(image)
        else:
            self.image = image if isinstance(image, np.ndarray) else np.array(image)

    def __str__(self):
        return f'{self.image}'

    def __getitem__(self):
        return self.image

    def show(self, size=None):
        if size is not None:
            self.image = cv2.resize(self.image, size)

        cv2.imshow('image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def resize(self, *args, **kwargs):
    #     self.image = cv2.resize(self.image, *args, **kwargs)


class SiameseDataset(Dataset):
    def __init__(self, root_path, size_0, size_1):
        # kiểm tra root_path có tồn tại hay không
        root_path = os.path.join(os.getcwd(), root_path)
        is_path_exist(root_path)






# utils function
def make_siamese_data(dataset, n_cate, n_image, size_true=50, size_false=50):
    siamese = []
    i_true = 0
    i_false = 0
    while i_true + i_false < size_true + size_false:
        cate1, cate2 = np.random.randint(n_cate, size=2)
        img1 = np.random.randint(cate1 * n_image, cate1 * n_image + n_image)
        img2 = np.random.randint(cate2 * n_image, cate2 * n_image + n_image)

        if cate1 == cate2 and i_true < size_true:
            siamese.append([dataset.images[img1], dataset.images[img2], 1])
            i_true += 1
        elif cate1 != cate2 and i_false < size_false:
            siamese.append([dataset.images[img1], dataset.images[img2], 0])
            i_false += 1

    return np.array(siamese, dtype='object')

if __name__ == '__main__':
    # data = Object18Dataset(root='../../data/SiameseData/object_18', n_cate=3)
    # print(len(data))
    #
    # # Chọn ảnh thứ 123
    # n = 123
    # image ,label = data[n]
    # print(f'Ảnh thuộc class: {label} {data.categorical[label]}')
    # image.show()

    n_categorical = 3
    n_images = 10
    data = Object18Dataset(root='../../data/SiameseData/object_18', n_cate=n_categorical, size=n_images)

    pos_siamese = make_siamese_data(data, n_cate=n_categorical, n_image=n_images)

    for img1, img2, label in pos_siamese:
        print(img1, img2, label)


