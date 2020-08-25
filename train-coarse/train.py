#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from net  import LDF

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='../data/DUTS', savepath='./out', mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=48)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=8)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask, body, detail) in enumerate(loader):
            image, mask, body, detail = image.cuda(), mask.cuda(), body.cuda(), detail.cuda()
            outb1, outd1, out1, outb2, outd2, out2 = net(image)
            
            lossb1 = F.binary_cross_entropy_with_logits(outb1, body)
            lossd1 = F.binary_cross_entropy_with_logits(outd1, detail)
            loss1  = F.binary_cross_entropy_with_logits(out1, mask) + iou_loss(out1, mask)

            lossb2 = F.binary_cross_entropy_with_logits(outb2, body)
            lossd2 = F.binary_cross_entropy_with_logits(outd2, detail)
            loss2  = F.binary_cross_entropy_with_logits(out2, mask) + iou_loss(out2, mask)
            loss   = (lossb1 + lossd1 + loss1 + lossb2 + lossd2 + loss2)/2

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'lossb1':lossb1.item(), 'lossd1':lossd1.item(), 'loss1':loss1.item(), 'lossb2':lossb2.item(), 'lossd2':lossd2.item(), 'loss2':loss2.item()}, global_step=global_step)
            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | lossb1=%.6f | lossd1=%.6f | loss1=%.6f | lossb2=%.6f | lossd2=%.6f | loss2=%.6f'
                    %(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], lossb1.item(), lossd1.item(), loss1.item(), lossb2.item(), lossd2.item(), loss2.item()))

        if epoch > cfg.epoch*3/4:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))


if __name__=='__main__':
    train(dataset, LDF)