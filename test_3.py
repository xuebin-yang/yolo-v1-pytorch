import torch


contain = torch.tensor([[0, 1], [0, 1]])

mask1 = contain > 0.1
# print(mask1)
# print(contain.max())
#
# mask2 = (contain == contain.max())
# print(mask2)
#
# x1 = torch.arange(20).reshape(4, 5)
# print(x1)
# order = torch.tensor([1, 2, 3, 3, 3])
# xx1 = x1[order[1:]].clamp(min=x1[0])
# print('xx1 = ', xx1)

"""
//测试tensor[bool_tensor]的返回值
pred = torch.tensor([[[[0.2640, 0.4015],
          [0.2747, 0.5303],
          [0.2886, 0.6246],
          [0.3807, 0.5587],
          [0.3946, 0.1991],
          [0.2544, 0.3650]],

         [[0.5372, 0.2157],
          [0.4061, 0.7594],
          [0.5727, 0.4114],
          [0.2916, 0.5266],
          [0.1520, 0.6539],
          [0.1279, 0.2084]],

         [[0.5800, 0.3092],
          [0.4140, 0.4651],
          [0.3696, 0.4537],
          [0.2148, 0.2110],
          [0.5782, 0.2978],
          [0.3711, 0.1879]]],

        [[[0.2321, 0.4340],
          [0.3379, 0.5766],
          [0.3026, 0.2850],
          [0.2474, 0.5060],
          [0.5776, 0.5219],
          [0.7219, 0.3672]],

         [[0.5937, 0.6841],
          [0.5484, 0.5678],
          [0.5758, 0.8691],
          [0.5701, 0.6360],
          [0.6727, 0.7815],
          [0.7371, 0.5170]],

         [[0.5890, 0.4122],
          [0.5718, 0.7518],
          [0.1275, 0.7522],
          [0.5493, 0.5719],
          [0.6908, 0.3216],
          [0.5794, 0.7438]]]])
coo = torch.tensor([[[[True, True],
          [False, False],
          [False, False],
          [False, False],
          [False, False],
          [False, False]],

         [[False, False],
          [False, False],
          [False, False],
          [False, False],
          [False, False],
          [False, False]],

         [[False, False],
          [False, False],
          [False, False],
          [False, False],
          [False, False],
          [False, False]]],

        [[[False, False],
          [False, False],
          [False, False],
          [False, False],
          [False, False],
          [False, False]],

         [[False, False],
          [False, False],
          [False, False],
          [False, False],
          [False, False],
          [False, False]],

         [[False, False],
          [False, False],
          [False, False],
          [False, False],
          [False, False],
          [False, False]]]])

coo_pred = pred[coo]
print(coo_pred)
"""
#
# noo_pred_mask = torch.ByteTensor(contain.size()).bool()
# noo_pred_mask.zero_()
# print(noo_pred_mask)

# a = torch.tensor([[-0.2095, -0.0171,  0.2547,  0.1164],
#                  [-0.2739, -0.2781,  0.3397,  0.3583]])
#
# a = a[:, :2].unsqueeze(1).expand(2, 1, 2)
# print('a', a)
#
# b = torch.tensor([[-0.4034, -0.3760,  0.4146,  0.5000]])
# b = b[:, :2].unsqueeze(0).expand(2, 1, 2)
# print('b', b)
#
# lt = torch.max(a, b)
# rt = torch.min(a, b)
# print(lt)
# print(rt)
# wh = rt - lt
# print(wh)
# wh[wh < 0] = 0
# print(wh)



def compute_iou(box1, box2):
        """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        """

        N = box1.size(0)  # 2
        M = box2.size(0)  # 1

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]， box1 和 box2 的交集

        area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
        area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

box1 = torch.tensor([[-0.2095, -0.0171,  0.2547,  0.1164],
                    [-0.2739, -0.2781,  0.3397,  0.3583]])
box2 = torch.tensor([[-0.4034, -0.3760,  0.4146,  0.5000]])   # 四个元素分别是 ， ， 归一化宽， 归一化高

iou = compute_iou(box1, box2)
print(iou)