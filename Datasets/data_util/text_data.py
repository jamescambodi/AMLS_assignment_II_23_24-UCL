from torch.utils.data import Dataset
from PIL import Image
from torchvision.io import read_image
import os


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]
