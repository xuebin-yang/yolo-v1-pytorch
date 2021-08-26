import torch
import numpy as np

target_tensor = torch.randn(2, 5)
print('target_tensor', target_tensor)
coo_mask = target_tensor[:, 4] > 0
print('coo_mask', coo_mask)
pred_tensor = torch.randn(2, 5)

coo_pred = pred_tensor[coo_mask] #.view(-1, 30)
print('coo_pred', coo_pred)

noo_pred_mask = torch.ByteTensor(target_tensor.size()).bool()
noo_pred_mask.zero_()
noo_pred_mask[:, 4] = 1
print(noo_pred_mask)

for i in range(0, 86, 2):
    print(i)