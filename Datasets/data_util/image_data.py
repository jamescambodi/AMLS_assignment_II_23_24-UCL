from torch.utils.data import Dataset
from PIL import Image
from torchvision.io import read_image
import os


def img_collate_fn(batch):
    return list(batch)


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # image = read_image(self.image_paths[index])
        image = Image.open(self.image_paths[index])
        return image
