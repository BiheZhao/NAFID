import os
import scipy.misc as misc
from PIL import Image
from PIL import ImageFile
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from config import opt
import cv2


real_map = {'stylegan':'FFHQ',
    'stargan':'CELEBA',
    'deeper':'Deeper_Real',
    'deepfake':'Original',
    'attgan':'CELEBA_New',
    'pggan':'Pggan_Real',
    'stylegan2':'FFHQ_New'}
fake_map = {'stylegan':'100K_Faces',
    'stargan':'Stargan_V2',
    'deeper':'Deeper_Fake',
    'deepfake':'Deepfake',
    'attgan':'Attgan',
    'pggan':'Pggan_Fake',
    'stylegan2':'Stylegan2'}

class CommonDataloader(data.Dataset):

    def __init__(self, root, dataset, noise=None, denoised_data=None,
                 transforms=None, train=True, test=False):
        self.train   = train
        self.test    = test
        self.dataset = dataset
        if dataset == 'combined':
            self.real_datasets = ["FFHQ","CELEBA"]
            self.fake_datasets = ["100K_Faces","Stargan_V2"]
            real_dirs = []
            fake_dirs = []
            imgs = []
            if self.train:
                for d in self.real_datasets:
                    real_dirs.append(root+"source/"+d+"/"+d+"_Train/")
                for d in self.fake_datasets:
                    fake_dirs.append(root+"source/"+d+"/"+d+"_Train/")
            elif self.test:
                for d in self.real_datasets:
                    real_dirs.append(root+"source/"+d+"/"+d+"_Test/")
                for d in self.fake_datasets:
                    fake_dirs.append(root+"source/"+d+"/"+d+"_Test/")
            else:
                for d in self.real_datasets:
                    real_dirs.append(root+"source/"+d+"/"+d+"_Validate/")
                for d in self.fake_datasets:
                    fake_dirs.append(root+"source/"+d+"/"+d+"_Validate/")

            for real_dir in real_dirs:
                imgs += [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
            for fake_dir in fake_dirs:
                imgs += [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        
        else:
            self.real_name = real_map[dataset]
            self.fake_name = fake_map[dataset]
            if self.train:
                real_dir = root+"source/%s/%s_Train"%(self.real_name,self.real_name)
                fake_dir = root+"source/%s/%s_Train"%(self.fake_name,self.fake_name)
            elif self.test:
                real_dir = root+"source/%s/%s_Test"%(self.real_name,self.real_name)
                fake_dir = root+"source/%s/%s_Test"%(self.fake_name,self.fake_name)
            else:
                real_dir = root+"source/%s/%s_Validate"%(self.real_name,self.real_name)
                fake_dir = root+"source/%s/%s_Validate"%(self.fake_name,self.fake_name)
            if noise:
                real_dir += '_%s'%noise
                fake_dir += '_%s'%noise
            # get img list
            imgs =  [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
            imgs += [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        
        self.imgs = imgs
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        resize = int(opt.img_size/7*8)
        if self.train:
            self.transforms = T.Compose(
                [T.RandomHorizontalFlip(),
                T.Resize(resize),
                T.RandomCrop(opt.img_size, padding=4),
                T.ToTensor(),
                T.Normalize(mean, std)])
        else:
            self.transforms = T.Compose(
                [T.Resize(resize),
                T.RandomCrop(opt.img_size, padding=4),
                T.ToTensor(),
                T.Normalize(mean, std)])
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __getitem__(self, index):
        # no label for testset, return filename
        img_path = self.imgs[index]
        if self.dataset == 'combined':
            if self.test:
                label = int(self.imgs[index].split('.')[-2].split('/'[-1]))
            else:
                label = 0
                for real_dataset in self.real_datasets:
                    if real_dataset in img_path.split('.')[-2]:
                        label = 1
        else:
            if self.test:
                label = int(self.imgs[index].split('.')[-2].split('/'[-1]))
            elif self.real_name in img_path.split('.')[-2]:
                label = 1
            else:
                label = 0

        data = Image.open(img_path)
        data_ = np.array(data).copy()
        hr = data_.copy()
        # add noise
        noises = np.random.normal(scale=opt.noise_scale, size=data_.shape)
        noises = noises.round()
        x_noise = data_.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        lr = x_noise
        hr = Image.fromarray(hr).convert('RGB')
        lr = Image.fromarray(lr).convert('RGB')
        hr = self.transforms(hr)
        lr = self.transforms(lr)

        return lr, hr, label

    def __len__(self):
        return len(self.imgs)