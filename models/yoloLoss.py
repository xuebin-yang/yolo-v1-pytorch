# -*- coding: utf-8 -*-
"""
@Time          : 2020/08/07 15:07
@Author        : FelixFu
@File          : yoloLoss.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class yoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(yoloLoss, self).__init__()
        self.S = S    # S 代表 split 为 7 * 7
        self.B = B    # B 代表 bounding box，每个窗口预测两个 bounding box
        self.l_coord = l_coord  # l_coord = 5
        self.l_noobj = l_noobj  # = 0.5

    def compute_iou(self, box1, box2):
        """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        """

        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(   # 本来是求最小值的，因为左上角的坐标小于 0， 所以应该是求左上角的最大值，求出相交区域的左上角坐标
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(  # 求相交区域的右下角坐标
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # 相交区域的宽高如果小于 0 的位置就置 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]， box1 和 box2 的交集

        area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
        area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        """
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        torch.Size([24, 14, 14, 30])

        target_tensor: (tensor) size(batchsize,S,S,30)
        torch.Size([24, 14, 14, 30])
        """
        N = pred_tensor.size()[0]     # batch-size

        # 筛选有object和无object的bbox + confidence + classes——生成掩码，cx cy w h confidence
        # 筛选过程中是从目标矩阵中生成的
        coo_mask = target_tensor[:, :, :, 4] > 0       # 应该要有object的patch置为True， torch.Size([B, 14, 14])
        noo_mask = target_tensor[:, :, :, 4] == 0    # 应该没有object的patch， torch.Size([B, 14, 14])
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)  # torch.Size([B, 14, 14, 30]), 应该要有object的位置置为True
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)  # torch.Size([B, 14, 14, 30])
        print('coo_mask.shape', coo_mask.shape)
        print('noo_mask.shape', noo_mask.shape)

        # 筛选有object和无object的bbox + confidence + classes——筛选pred中有object
        # 挑出有object的split，后面都是在有obj的patch中操作
        coo_pred = pred_tensor[coo_mask].view(-1, 30)   #  [4, 30]（根据真实目标）把coo_mask为true的位置的pred_tensor挑出来，表示把预测的向量中应该有obj的patch挑出来
        print('coo_pred.shape', coo_pred.shape)
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # 把预测的向量中应该有obj的patch的前10维表示预测框的坐标提取出来
        class_pred = coo_pred[:, 10:]  # torch.Size([4, 20]) # 把预测的向量中应该有obj的patch的后20维表示类别的坐标提取出来。(一个patch的30维向量中的后20维只会预测1个类别)

        # 筛选有object和无object的bbox + confidence + classes——筛选target中有object
        coo_target = target_tensor[coo_mask].view(-1, 30)    # [4, 30] 把真实的标签中有obj的patch挑出来，也就是把obj的中心落在的那个patch取出来
        box_target = coo_target[:, :10].contiguous().view(-1, 5) # 把真实的向量中应该有obj的patch的前10维表示预测框的坐标提取出来
        class_target = coo_target[:, 10:] # 把真实的向量中有obj的patch的后20维表示类别的坐标提取出来 [4, 20]
        print('coo_target.shape', coo_target.shape)
        print('class_target.shape', class_target.shape)

        # compute not contain obj loss
        # 两张图片为什么把所有的patch统一？
        noo_pred = pred_tensor[noo_mask].view(-1, 30)   # （根据真实目标）把noo_mask为true的位置的pred_tensor挑出来，表示把预测的向量中不应该有obj的patch挑出来
        noo_target = target_tensor[noo_mask].view(-1, 30)   # 把noo_mask为true的位置的target_tensor挑出来，表示把真实的向量中没有obj的patch挑出来
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()).bool()   # 构建一个和 noo_pred 相同大小的 bool 类型的张量, 张量元素都为 false
        noo_pred_mask.zero_()   # noo_pred_mask 中的元素全都变成 False， 和 noo_pred 同维度
        noo_pred_mask[:, 4] = 1  # noo_pred_mask 中的元素的置信度位置上的元素置为 true
        noo_pred_mask[:, 9] = 1  # noo_pred_mask 中的元素的置信度位置上的元素置为 true
        noo_pred_c = noo_pred[noo_pred_mask]  # 把不应该有obj的patch的置信度提取出来
        noo_target_c = noo_target[noo_pred_mask]    # 把真实标签中没有obj的patch1z置信度提取出来，全0
        # 论文中第4个误差，边框内无对象的误差，只对错误的patch进行计算
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)   # 本应该没有 obj 的 patch 但是却被预测有了 obj，置信度损失

        # compute contain obj loss
        # 这一部分损失包括三个部分，中心点损失 + 宽高损失 + 有目标的置信度回归损失
        coo_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()  # [8, 5] 构建一个和 box_target 相同尺寸的 bool 类型张量，
        coo_response_mask.zero_()  # 张量元素都置为 false
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()   # 构建一个和 box_target 相同尺寸的 bool 类型张量
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()  # 构建一个和 box_target 相同尺寸的张量， torch.Size([86, 5])
        for i in range(0, box_target.size()[0], 2):  # 每次读两个框，也就是一个 patch 预测的两个框。 choose the best iou box  [0, 86, 2]
            box1 = box_pred[i:i+2]  # 获取和真实标签的patch一样的patch预测的 2 个box
            box1_xyxy = torch.FloatTensor(box1.size())  # 生成一个和 box1 相同尺寸的向量
            box1_xyxy[:, :2] = box1[:, :2]/14. - 0.5 * box1[:, 2:4]    # delta_xy = (cxcy_sample - xy)/cell_size，由于最后预测的是delta_xy，所以除14就相当于求得了cxcy_sample - xy，最后计算的是cxcy_sample - xy相对格点的偏移量
            box1_xyxy[:, 2:4] = box1[:, :2]/14. + 0.5 * box1[:, 2:4]   # 得到归一化的bounding box的右下角坐标，现在不代表归一化宽高了，现在是代表右下角的坐标了
            box2 = box_target[i].view(-1, 5)   # 获取真实目标的 bounding box
            box2_xyxy = torch.FloatTensor(box2.size())  # 生成一个和 box2 相同尺寸的向量
            box2_xyxy[:, :2] = box2[:, :2]/14. - 0.5*box2[:, 2:4]    # 恢复到标准化之前， delta_xy = (cxcy_sample - xy)/cell_size
            box2_xyxy[:, 2:4] = box2[:, :2]/14. + 0.5*box2[:, 2:4]   #
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # 计算应该有 obj 的 patch 预测出的 2 个 bounding box 的置信度
            max_iou, max_index = iou.max(0)  # 选择置信度高的那个，并记下 index
            max_index = max_index.data.cuda()
            
            coo_response_mask[i+max_index] = 1    # 将应该有 obj 的 patch 预测出的置信度高的 bbox 位置置为 True
            coo_not_response_mask[i+1-max_index] = 1 # 将应该有 obj 的 patch 预测出的置信度低的 bbox 位置置为 True

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()   # 将应该有 obj 的 patch 预测出的置信度高的 bbox 位置的置信度赋值为较大的 iou 值
        box_target_iou = box_target_iou.cuda()  # torch.Size([92, 5])

        # 1.response loss，应该有obj的patch的置信度损失
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)  # 将应该有 obj 的 patch 的置信度高的 bounding box 找出来
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)  # 将应该有 obj 的 patch 的置信度高的 bounding box 的 iou
        box_target_response = box_target[coo_response_mask].view(-1, 5)  # 挑出真实的负责预测的bbox
        # 论文第3个误差
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)   # 应该有obj的 iou 损失
        # box_pred_response[:, 4] 是预测出来的置信度，box_target_response_iou[:, 4]是根据 box_pred 和 box_target 计算出来的IoU
        # 论文第1，2个误差的和
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)  # 前一部分是中心点损失（其实是偏移量损失），后一部分是宽高损失。也叫定位损失
        # 这里 box_pred_response[:, :2]是中心点坐标，他没有经过坐标恢复
        # 2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        
        # I believe this bug is simply a typo
        # 第3个
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], size_average=False)

        # 3.class loss
        # 第5个
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)

        return (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N   # 最后要求一个平均




