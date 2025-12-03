# utils/randaugment.py

import random
from torchvision import transforms

class RandAugment:
    """
    Minimal RandAugment implementation compatible with:
       RandAugment(N, M)
    N = number of augmentation ops
    M = magnitude (1–10)
    """

    def __init__(self, N=2, M=9):
        self.N = N
        self.M = min(10, max(1, M))  # force 1–10

        # Define available augmentation operations
        self.transforms = [
            transforms.ColorJitter(brightness=0.5 * M / 10),
            transforms.ColorJitter(contrast=0.5 * M / 10),
            transforms.ColorJitter(saturation=0.5 * M / 10),
            transforms.ColorJitter(hue=0.1 * M / 10),
            transforms.RandomRotation(degrees=15 * M / 10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0 * M / 10)),
        ]

    def __call__(self, img):
        ops = random.sample(self.transforms, self.N)
        for op in ops:
            img = op(img)
        return img
