#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask=None, body=None, detail=None):
        image = (image - self.mean)/self.std
        if mask is None:
            return image
        return image, mask/255, body/255, detail/255

class RandomCrop(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if mask is None:
            return image[p0:p1,p2:p3, :]
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3], body[p0:p1,p2:p3], detail[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        if np.random.randint(2)==0:
            if mask is None:
                return image[:,::-1,:].copy()
            return image[:,::-1,:].copy(), mask[:, ::-1].copy(), body[:, ::-1].copy(), detail[:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask, body, detail

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, body=None, detail=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        body  = cv2.resize( body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        detail= cv2.resize( detail, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, body, detail

class ToTensor(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)
        body  = torch.from_numpy(body)
        detail= torch.from_numpy(detail)
        return image, mask, body, detail


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(352, 352)
        self.totensor   = ToTensor()

        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name  = self.samples[idx]
        image = cv2.imread(self.cfg.datapath+'/image/'+name+'.jpg')[:,:,::-1].astype(np.float32)

        if self.cfg.mode=='train':
            mask  = cv2.imread(self.cfg.datapath+'/mask/' +name+'.png', 0).astype(np.float32)
            body  = cv2.imread(self.cfg.datapath+'/body/' +name+'.png', 0).astype(np.float32)
            detail= cv2.imread(self.cfg.datapath+'/detail/' +name+'.png', 0).astype(np.float32)
            image, mask, body, detail = self.normalize(image, mask, body, detail)
            image, mask, body, detail = self.randomcrop(image, mask, body, detail)
            image, mask, body, detail = self.randomflip(image, mask, body, detail)
            return image, mask, body, detail
        else:
            shape = image.shape[:2]
            image = self.normalize(image)
            image = self.resize(image)
            image = self.totensor(image)
            return image, shape, name

    def __len__(self):
        return len(self.samples)
    
    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask, body, detail = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            body[i]  = cv2.resize(body[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            detail[i]= cv2.resize(detail[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image  = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        mask   = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        body   = torch.from_numpy(np.stack(body, axis=0)).unsqueeze(1)
        detail = torch.from_numpy(np.stack(detail, axis=0)).unsqueeze(1)
        return image, mask, body, detail
