#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import dataset
from torch.utils.data import DataLoader
from net import LDF

class Test(object):
    def __init__(self, Dataset, Network, Path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, snapshot='./out/model-40', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                outb1, outd1, out1, outb2, outd2, out2 = self.net(image, shape)
                out  = out2
                pred = torch.sigmoid(out[0,0]).cpu().numpy()*255
                head = '../eval/maps/LDF/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))


if __name__=='__main__':
    for path in ['../data/ECSSD', '../data/PASCAL-S', '../data/DUTS', '../data/HKU-IS', '../data/DUT-OMRON', '../data/THUR15K']:
        t = Test(dataset, LDF, path)
        t.save()
