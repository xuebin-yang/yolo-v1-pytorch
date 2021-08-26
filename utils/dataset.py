# -*- coding: utf-8 -*-
"""
@Time          : 2020/08/07 15:07
@Author        : FelixFu
@File          : train.py
@Noice         :
@Modificattion : txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
    @Author    :
    @Time      :
    @Detail    :
"""

import os
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt


class yoloDataset(data.Dataset):
    image_size = 448

    def __init__(self, root, list_file, train, transform):
        self.root = root    # 数据集根目录 datasests
        self.train = train  # 是否为训练
        self.transform = transform  # 转换
        self.fnames = []    # 文件名s [001.jpg, 002.jpg]
        self.boxes = []     # boxes  [ [box], [[x1,y1,x2,y2], ...], ... ]
        self.labels = []    # labels [ [1], [2], ... ]
        self.mean = (123, 117, 104)  # RGB
        self.num_samples = 0  # 样本总数

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = os.path.join(root, 'images.txt')
            list_file = [os.path.join(root, list_file[0]), os.path.join(root, list_file[1])]
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file
        else:
            list_file = os.path.join(root, list_file)

        # 处理标签
        with open(list_file) as f:
            lines = f.readlines()   # 直接读取所有行，直到碰到EOF结束
        for line in lines:
            splited = line.strip().split()  # ['005246.jpg', '84', '48', '493', '387', '2']
            self.fnames.append(splited[0])   # 用来存放所有图像的文件名
            num_boxes = (len(splited) - 1) // 5               #说明这个图片里面有多少个物体
            box = []
            label = []
            for i in range(num_boxes):
                x = float(splited[1+5*i])     # 标注框左上角x坐标（左上角为坐标原点，往右为x轴）
                y = float(splited[2+5*i])     # 标注框左上角y坐标（左上角往下为y轴）
                x2 = float(splited[3+5*i])    # 标注框右下角x坐标
                y2 = float(splited[4+5*i])    # 标注框左下角y坐标
                c = splited[5+5*i]            # 代表类别
                box.append([x, y, x2, y2])    # 该图片的标注框坐标放入box中保存
                label.append(int(c)+1)        # 该图片的obj所属的类别放入label中
            self.boxes.append(torch.Tensor(box))   # 存入一个大的boxes中，这个boxes负责存所有图像的obj的标注框和类别， 一张图片一个tensor
            self.labels.append(torch.LongTensor(label)) # 一个大的labels负责存所有的标签， 一张图片一个tensor
        self.num_samples = len(self.boxes)   # 训练集样本数量

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, "images", fname))
        boxes = self.boxes[idx].clone()  # 返回一个张量的副本，其与原张量的尺寸和数据类型相同。与copy_()不同，这个函数记录在计算图中。传递到克隆张量的梯度将传播到原始张量。
        labels = self.labels[idx].clone()  # 这里的 boxes 是个二维的

        # 数据增强
        # if self.train:
        #     img = self.random_bright(img)
        #     img, boxes = self.random_flip(img, boxes)
        #     img, boxes = self.randomScale(img, boxes)
        #     img = self.randomBlur(img)
        #     img = self.RandomBrightness(img)
        #     img = self.RandomHue(img)
        #     img = self.RandomSaturation(img)
        #     img, boxes, labels = self.randomShift(img, boxes, labels)
        #     img, boxes, labels = self.randomCrop(img, boxes, labels)

        # # debug
        # box_show = boxes.numpy().reshape(-1)
        # print(box_show)
        # img_show = self.BGR2RGB(img)
        # pt1 = (int(box_show[0]), int(box_show[1]))
        # pt2 = (int(box_show[2]), int(box_show[3]))
        # cv2.rectangle(img_show, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=1)
        # print(type(img_show))
        # plt.figure()
        # plt.imshow(img_show)
        # plt.show()
        # plt.savefig("a.png")
        # #debug
        h, w, _ = img.shape
        print("img.shape", img.shape)
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)   # boxes的归一化
        img = self.BGR2RGB(img)  # because pytorch pretrained model use RGB，but cv2 readed image is BGR, this operation not change h,w,c
        print("img.shape", img.shape)
        img = self.subMean(img, self.mean)  # 图像只是减去了均值减去均值
        img = cv2.resize(img, (self.image_size, self.image_size))  # 调整图片尺寸，此时图片没有归一化
        target = self.encoder(boxes, labels)  # 7x7x30
        for t in self.transform:
            img = t(img)

        return img, target

    def __len__(self):
        return self.num_samples

    # yolo 输出适配
    def encoder(self, boxes, labels):
        """
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 14x14x30
        """
        grid_num = 14
        target = torch.zeros((grid_num, grid_num, 30)) # [14, 14, 30],可以看作一张图片的每个patch都被编码成一个30维的向量
        cell_size = 1./grid_num    # 归一化后图像的长宽都为1，所以这里表示归一化后每个网格的大小
        wh = boxes[:, 2:] - boxes[:, :2]    # 读出每个bounding box的（归一化）宽高
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2    # 读出bounding box的（归一化）后的中心坐标，如果有两个bbox，cxcy为[2, 2]
        for i in range(cxcy.size()[0]):  # 读出obj的数量，也就是框的数量
            cxcy_sample = cxcy[i]   # 第 i 个obj的bounding box 的归一化中心点坐标
            ij = (cxcy_sample/cell_size).ceil()-1  # 将cxcy_sample定位到cell（网格）中，看属于那个网格
            target[int(ij[1]), int(ij[0]), 4] = 1   # 将第[i,j]个patch的30维的向量中的target位置设为1
            target[int(ij[1]), int(ij[0]), 9] = 1   # 将第[i,j]个patch的30维的向量中的target位置设为1，因为标签标注框是唯一的，所以一个patch预测的两个bounding box是相同的
            target[int(ij[1]), int(ij[0]), int(labels[i])+9] = 1    # 将第[i,j]个patch的30维的向量中的class位置设为1
            xy = ij*cell_size  # 归一化后的patch[i,j]的左上角坐标
            # 下面这句代码在干嘛？？？？
            # 物体中心相对左上的坐标 ---> 坐标x,y代表了预测的bounding box的中心与栅格边界的相对值，即中心坐标偏移
            delta_xy = (cxcy_sample - xy)/cell_size  # bbox中心占该patch的比例。第 i 个obj的标注框中心（不是patch的中心）（落在[i,j]patch内）与[i,j]patch的左上角的相对距离

            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]    # 将第[i,j]个patch的30维的向量中的存放bbox宽高的位置置为bounding box的归一化宽高
            target[int(ij[1]), int(ij[0]), :2] = delta_xy  # 将第[i,j]个patch的30维的向量中一定的位置置为相对标注框中心点的偏移量
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]    # 由于一个patch预测两个bbox，所以第[i,j]个patch的30维的向量的存放bbox的位置相同
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy # 由于一个patch预测两个bbox，所以第[i,j]个patch的30维的向量的存放bbox的位置相同
        return target

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        # 平移变换
        center = (boxes[:, 2:]+boxes[:, :2])/2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width*0.2, width*0.2)
            shift_y = random.uniform(-height*0.2, height*0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height-int(shift_y), :width-int(shift_x), :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height+int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width-int(shift_x), :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width+int(shift_x), :] = bgr[:height-int(shift_y), -int(shift_x):, :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height+int(shift_y), :width+int(shift_x), :] = bgr[-int(shift_y):, -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[: ,1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y),
                                            int(shift_x), int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width*scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:]+boxes[:, :2])/2
            height, width, c = bgr.shape
            h = random.uniform(0.6*height, height)
            w = random.uniform(0.6*width, width)
            x = random.uniform(0, width-w)
            y = random.uniform(0, height-h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if(len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h, x:x+w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    file_root = "../datasets"
    # train_dataset = yoloDataset(root=file_root, list_file=['voc2012.txt', 'voc2007.txt'],
    #                             train=True, transform=[transforms.ToTensor()])
    train_dataset = yoloDataset(root=file_root, list_file='images.txt',
                                train=True, transform=[transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    train_iter = iter(train_loader)
    for i in range(1):
        img, target = next(train_iter)
        print(img.shape, target.shape)
    print(train_dataset.num_samples)


