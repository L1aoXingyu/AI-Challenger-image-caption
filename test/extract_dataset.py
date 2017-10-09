__author__ = "sherlock"
import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


class MyDataset(data.Dataset):
    def __init__(self, img_path, transform):
        self.img_path = img_path
        self.img_list = os.listdir(img_path)
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.img_path, img_name)
        img = Image.open(img_path)
        img = self.transform(img)

        return img, img_name

    def __len__(self):
        return len(self.img_list)


def img_transform(x):
    x = x.resize((224, 224))
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(x)
    return x


def collate_fn(batch):
    img, img_name = zip(*batch)
    img = torch.stack(img, 0)
    return img, img_name


def get_loader(img_transform=img_transform,
               batch_size=32,
               shuffle=False,
               collate_fn=collate_fn):
    dset = MyDataset(
        img_path="/home/node/dhn/Image_caption/image-caption/data/ai_challenger_caption_test1_20170923/caption_test1_images_20170923",
        transform=img_transform)

    return data.DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
