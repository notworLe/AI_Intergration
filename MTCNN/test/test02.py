from torchvision.transforms import ToTensor
import numpy as np

a = np.random.randint(0,10, (10,2))
to_tensor = ToTensor()
print(a)
print(to_tensor(a))
