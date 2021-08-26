import torch
import numpy as np
def encoder(boxes, labels):
        """
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 14x14x30
        """
        grid_num = 14
        target = torch.zeros((grid_num, grid_num, 30))
        print('target.shape = ', target.shape)
        cell_size = 1. / grid_num  # 归一化后每个网格的大小
        wh = boxes[:, 2:] - boxes[:, :2]  # 这张图片，有n=8个box，每个box的wh torch.Size([8, 2])，读出每个bounding box的宽高
        print('wh=', wh)
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2  # n个box的中心xy  torch.Size([8, 2])， 读出每个bounding box的中心点
        print('cxcy = ', cxcy)
        for i in range(cxcy.size()[0]):  # 有几个obj就是[n, n]
                cxcy_sample = cxcy[i]  # 第 i 个obj的bounding box 的归一化中心点
                ij = (cxcy_sample / cell_size).ceil() - 1  # 将cxcy_sample定位到cell（网格）中，看属于那个网格
                print('ij = ', ij)
                target[int(ij[1]), int(ij[0]), 4] = 1  # 将target中box设为1,找到某个块，将该块的第一个c改为1，也就是存在obj
                print('target1.shape = ', target.shape)
                target[int(ij[1]), int(ij[0]), 9] = 1  # 将target中box设为1，找到某个块，将该块的第二个c改为1，也就是存在obj
                print('target2.shape = ', target.shape)
                target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1  # 将target中class设为1  ，第15类，
                print('target3.shape = ', target.shape)
                xy = ij * cell_size  # 匹配到网格的左上角相对坐标
                print('xy = ', xy)
                delta_xy = (cxcy_sample - xy) / cell_size
                print('delta_xy = ', delta_xy)
                target[int(ij[1]), int(ij[0]), 2:4] = wh[i]  # 每个bounding box的归一化宽高
                print('target4.shape = ', target.shape)
                target[int(ij[1]), int(ij[0]), :2] = delta_xy  # 相对中心点的偏移量
                print('target5.shape = ', target.shape)
                target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
                print('target6.shape = ', target.shape)
                target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
                print('target7.shape = ', target.shape)
        return target

boxes = torch.tensor([[0.2620, 0.0057, 0.9980, 0.9971],
        [0.6880, 0.2829, 0.7320, 0.3400],
        [0.4180, 0.2543, 0.5160, 0.6600]])
labels = torch.tensor([19, 15, 15])

target = encoder(boxes, labels)
print(target)


