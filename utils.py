#coding=utf-8
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_map(datapath):
    print(datapath)
    for name in os.listdir(datapath+'/mask'):
        mask = cv2.imread(datapath+'/mask/'+name,0)
        body = cv2.blur(mask, ksize=(5,5))
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        body = body**0.5

        tmp  = body[np.where(body>0)]
        if len(tmp)!=0:
            body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)

        if not os.path.exists(datapath+'/body-origin/'):
            os.makedirs(datapath+'/body-origin/')
        cv2.imwrite(datapath+'/body-origin/'+name, body)

        if not os.path.exists(datapath+'/detail-origin/'):
            os.makedirs(datapath+'/detail-origin/')
        cv2.imwrite(datapath+'/detail-origin/'+name, mask-body)


if __name__=='__main__':
    split_map('./data/DUTS')

