import torch
import torch.nn.functional as F

import pdb

def anneal_weight(data_seq, min_weight=0.05, temp=1):
    # data_seq is a list of 2D tensors of shape (m,n)
    # this function return a convex combination of these tensors
    # the larger the original tensor is, the large weight it has in the final tensor
    # weight is a 3D tensor (m,n,len), where len = len(data_seq)
    # and weight.sum(dim=2) = 1
    data = torch.stack(data_seq, dim=2)
    with torch.no_grad():
        weight = F.softmax(data/temp, dim=2)
        weight.clamp_(max=1-min_weight, min=min_weight)
        if len(data_seq)>2:
            weight_sum = weight.sum(dim=2, keepdim=True)
            weight = weight / weight_sum
    weight = weight.detach()
    bound = (data*weight).sum(dim=2)
    return bound, weight

if __name__ == '__main__':
    batch = 2
    num_class = 10
    r = 10
    x = torch.rand(batch, num_class) * r - r/2
    y = torch.rand(batch, num_class) * r - r/2
    z = torch.rand(batch, num_class) * r - r/2

    # data = torch.stack([x,y], dim=2)
    # weight = F.softmax(data, dim=2)
    # weight.clamp_(max = 0.9, min=0.1)
    # torch.nn.functional.softmax(input, dim=None)
    bound, weight = anneal_weight([x,y,z], min_weight=0.05, temp=1)
    pdb.set_trace()