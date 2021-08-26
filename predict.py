# -*- coding: utf-8 -*-
"""
@Time          : 2020/08/10 15:07
@Author        : FelixFu
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
"""
import torch

from models.resnet_yolo import resnet50
import torchvision.transforms as transforms
import cv2
import numpy as np

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

Color = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128]
]


def decoder(pred):
    """
    pred (tensor)  torch.Size([1, 14, 14, 30])
    return (tensor) box[[x1,y1,x2,y2]] label[...]     # 这是对一张图片的结果
    """
    grid_num = 14
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1./grid_num
    pred = pred.data  # torch.Size([1, 14, 14, 30])
    pred = pred.squeeze(0)  # torch.Size([14, 14, 30])
    contain1 = pred[:, :, 4].unsqueeze(2)  # torch.Size([14, 14, 1]) 找出pred中即所有patch的bounding box的置信度
    contain2 = pred[:, :, 9].unsqueeze(2)  # torch.Size([14, 14, 1]) 找出pred中的30维的第9个位置的值，即bounding box的置信度
    contain = torch.cat((contain1, contain2), 2)    # torch.Size([14, 14, 2])

    mask1 = contain > 0.6  # 所有patch的bounding box的置信度大于0.6的置为True，其他的为False torch.Size([14, 14, 2])。首先初步筛选一次
    print(contain.max())    # 预测的 contain 的最大 c
    print(contain.shape)   # [14, 14, 2]
    mask2 = (contain == contain.max()) # 所有patch的bounding box的置信度等于所有置信度的最大值的patch置为True，其他的为False， 挑出所有置信度最高的 bounding box
    mask = (mask1+mask2).gt(0)  #
    print('mask1+mask2', (mask1+mask2).shape)
    print('mask1.shape', mask1.shape)  # [14, 14, 2]
    print('mask2.shape', mask2.shape)  # [14, 14, 2]

    # min_score,min_index = torch.min(contain,2) # 每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i, j, b] == 1:   # 说明第 i j 的索引patch上存在obj
                    box = pred[i, j, b*5:b*5+4]  # 找出该 patch 预测的第一个bbox的位置
                    contain_prob = torch.FloatTensor([pred[i, j, b*5+4]])  # 取出该 bbox 的 c
                    xy = torch.FloatTensor([j, i])*cell_size  # cell左上角坐标，为什么是[j, i]
                    box[:2] = box[:2]*cell_size + xy  # box[:2]是偏移量delta_xy，delta_xy = (cxcy_sample - xy)/cell_size, 恢复原来的bbox的归一化中心点
                    box_xy = torch.FloatTensor(box.size())  # 转换成xy形式 convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]      # 转换成xy形式 convert[cx,cy,w,h] to [x1,xy1,x2,y2]，为什么都是负数
                    box_xy[2:] = box[:2] + 0.5*box[2:]      # 转换成xy形式 convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)  # 取出这个patch预测出来的最高类别和最高类别概率
                    if float((contain_prob*max_prob)[0]) > 0.6:   # 如果该 bbox有obj的概率（c） * max_prob（有某一类obj的最大概率）> 0.6， 在筛选一次
                        boxes.append(box_xy.view(1, 4))   # 则将该 bbox 考虑到最后的 nms 中
                        cls_indexs.append(cls_index.item())
                        probs.append(contain_prob*max_prob)   # 该 bbox有obj的概率（c） * max_prob（有某一类obj的最大概率）
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0)  # (n,4)
        probs = torch.cat(probs, 0)  # (n,)
        cls_indexs = torch.IntTensor(cls_indexs)  # (n,)
    keep = nms(boxes, probs)

    a = boxes[keep]  # keep 中保留的是最后的boxes的索引
    b = cls_indexs[keep]
    c = probs[keep]
    return a, b, c


def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    这里的bboxes都是归一化后的bbox（除了图像的宽高），左上角坐标和右下角坐标
    '''
    x1 = bboxes[:, 0]  # 把满足条件的预测的bbox的第一位置取出来
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2-x1) * (y2-y1)   # 求出满足条件的预测的bbox的面积
    # print(scores)   # tensor([0.1006, 0.2381, 0.1185, 0.5342, 0.2892, 0.3521, 0.6027])
    _, order = scores.sort(0, descending=True)    # 分数从大到小排序，order 代表索引
    keep = []
    # print("order:", order)    # order: tensor([6, 3, 5, 4, 1, 2, 0])
    # print("order.numel:", order.numel())  # 7
    while order.numel() > 0:    #
        if order.numel() == 1:   # 返回数组中元素的个数
            # print("end1")
            # print(type(order))
            # print(order)
            i = order
            keep.append(i)
            break
        # print("len:", order.size())
        # print(keep)
        i = order[0]   # 找到最大的 prob_scores 的索引，保留剩余box中得分最高的一个
        keep.append(i)
        # 得到相交区域,左上及右下
        xx1 = x1[order[1:]].clamp(min=x1[i])   # 除最大prob的bbox外，其他的按prob从大到小的顺序order[1:0]数组中元素的内容来定位x1中的内容。x1[order[1:]]这是典型的找位置的方法
        yy1 = y1[order[1:]].clamp(min=y1[i])   # clamp表示钳位，小于最大prob的bbox的右下角坐标就钳位    y1[i]的值全部钳位为y1[i]
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # 求交并比
        ids = (ovr <= threshold).nonzero().squeeze()   # 找出小于等于阈值的bbox的索引
        if ids.numel() == 0:
            break
        order = order[ids+1]   # 把小于等于阈值的bbox的bbox的索引取出来，注意这里要加 1
    return torch.LongTensor(keep)


# start predict one image
def predict_gpu(model, image_name, root_path=''):
    result = []
    image = cv2.imread(root_path+image_name)
    # print(root_path , image_name)
    h, w, _ = image.shape
    img = cv2.resize(image, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123, 117, 104)  # RGB
    img = img - np.array(mean, dtype=np.float32)   # 减去均值

    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img)  # torch.Size([3, 448, 448])
    img = img[None, :, :, :]  # 升 1 维， img: torch.Size([1, 3, 448, 448])
    img = img.cuda()

    pred = model(img)  # 1x7x7x30
    pred = pred.cpu()
    print('pred.shape', pred.shape)   # [1, 14, 14, 30]
    boxes, cls_indexs, probs = decoder(pred)

    for i, box in enumerate(boxes):
        x1 = int(box[0]*w)   # 去归一化
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], image_name, prob])
    return result


if __name__ == '__main__':
    model = resnet50()
    print('load model...')
    #model.load_state_dict(torch.load('checkpoints/best.pth'))
    model.eval()
    model.cuda()
    image_name = 'imgs/person.jpg'
    image = cv2.imread(image_name)
    print('predicting...')
    result = predict_gpu(model, image_name)

    for left_up, right_bottom, class_name, _, prob in result:
        color = Color[VOC_CLASSES.index(class_name)]
        cv2.rectangle(image, left_up, right_bottom, color, 2)
        label = class_name+str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    cv2.imwrite('imgs/person_result.jpg', image)




