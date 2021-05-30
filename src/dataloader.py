
import os

import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import config


def load_csv(path, root="/kaggle/input/unicode", encoding='utf-8'):
    df = pd.read_csv(path, encoding=encoding)
    # print("total length: ", len(df))
    # print(df.head())
    # print("---------")
    return df


def get_image_transforms(level=0):
    img_size = config.img_size
    level=1
    if level == 0:
    # Data Transforms
        tf_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    else:
        level = min(5, level)
        p = [0.2, 0.3, 0.4, 0.5][level-1]
        distortion_scale = [0.2, 0.3, 0.4, 0.5][level-1]
        scale = [(0.1, 0.2), (0.15, 0.25), (0.15, 0.3), (0.2, 0.35)][level-1]
        tf_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # transforms.RandomApply([
            #     transforms.RandomChoice([
            #         transforms.Pad(round(img_size * 0.06)),
            #         transforms.RandomCrop(round(img_size * 0.94)),
            #     ]),
            #     transforms.Resize((img_size, img_size))
            # ], p=0.4),
            transforms.ToTensor(),
        ])
    tf_valid = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return tf_train, tf_valid


class YuShanDataset(Dataset):
    def __init__(self, csv_file, img_path, root="/kaggle/input/unicode", transform=None):
        csv_file = os.path.join(root, csv_file)
        self.annotations = load_csv(csv_file)
        self.root = os.path.join(root, img_path)
        self.img_path = img_path
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        img_name = os.path.join(self.root, img_name)
        label = torch.tensor(int(self.annotations.iloc[index, 1]))
        raw_img = Image.open(img_name)
        raw_img = self.transform(raw_img)

        if config.multi_channel:
            root = f"{config.cut_root}/{self.img_path}"
            cut_name = img_name.replace(self.root, root).replace('.jpg', '.png').replace('.jepg', '.png')
            cut_img = Image.open(cut_name)
            cut_img = self.transform(cut_img)
            img = torch.cat((raw_img, cut_img), 0)
            return (img, label)
        else:
            return (raw_img, label)
        