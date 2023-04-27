import torch
from torch.utils.data.dataset import Dataset
from glob import glob
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image
from torch import nn

class NYUUWDataset(Dataset):
    def __init__(self, data_path, label_path, img_format='png', size=30000, mode='train', train_start=0, val_start=30000, test_start=33000):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.size = size
        self.train_start = train_start
        self.test_start = test_start
        self.val_start = val_start

        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))

        if self.mode == 'train':
            self.uw_images = self.uw_images[self.train_start:self.train_start+self.size]
        elif self.mode == 'test':
            self.uw_images = self.uw_images[self.test_start:self.test_start+self.size]
        elif self.mode == 'val':
            self.uw_images = self.uw_images[self.val_start:self.val_start+self.size]

        self.cl_images = []

        for img in self.uw_images:
             self.cl_images.append(os.path.join(self.label_path, os.path.basename(img)))

        for uw_img, cl_img in zip(self.uw_images, self.cl_images):
            assert os.path.basename(uw_img).split('.')[0] == os.path.basename(cl_img).split('.')[0], ("Files not in sync.")

        self.transform = transforms.Compose([
            transforms.Resize((270, 360)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        uw = Image.open(self.uw_images[index])
        w, h = uw.size[-2], uw.size[-1]
        uw_img = self.transform(Image.open(self.uw_images[index]))
        cl_img = self.transform(Image.open(self.cl_images[index]))
        # water_type = int(os.path.basename(self.uw_images[index])[-5])
        name = os.path.basename(self.uw_images[index])[:-4]

        return uw_img, cl_img, -1, name, (w, h)

    def __len__(self):
        return self.size

class UIEBDataset(Dataset):
    def __init__(self, data_path, label_path, img_format='jpg', size=30000, mode='train', train_start=0, val_start=30000, test_start=33000):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.size = size
        self.train_start = train_start
        self.test_start = test_start
        self.val_start = val_start

        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))
        self.cl_images = glob(os.path.join(self.label_path, '*.' + img_format))

        if self.mode == 'train':
            self.uw_images = self.uw_images[self.train_start:self.train_start+self.size]
        elif self.mode == 'test':
            self.uw_images = self.uw_images[self.test_start:self.test_start+self.size]
        elif self.mode == 'val':
            self.uw_images = self.uw_images[self.val_start:self.val_start+self.size]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
            ])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        uw_img = self.transform(Image.open(self.uw_images[index]))
        uw_img2 = self.transform2(Image.open(self.uw_images[index]))
        cl_img = self.transform2(Image.open(self.cl_images[index]))

        return uw_img, uw_img2, cl_img, -1, os.path.basename(self.uw_images[index])

    def __len__(self):
        return self.size


class Re_size(nn.Module):
    def __init__(self):
        super(Re_size, self).__init__()
        # self.transform = transforms.Compose([
        #     transforms.Resize((270, 360)),
        # ])
    def forward(self, x, size):
        return transforms.Compose([
            transforms.Resize((size[1], size[0])),
        ])(x)
