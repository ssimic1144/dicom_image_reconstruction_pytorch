import torch
import numpy as np

def batch_tensor_convert(batch_tensor, target_min, target_max):
    new_batch_list = []
    for tensor in batch_tensor:
        new_tensor = tensor_convert(tensor[0], target_min, target_max)
        while new_tensor.min().item() < target_min or new_tensor.max().item() > target_max:
            new_tensor = tensor_convert(new_tensor, target_min, target_max)
        new_tensor = new_tensor[None, None,...]
        new_batch_list.append(new_tensor)
    new_batch = torch.cat(new_batch_list,dim = 0)
    return new_batch

def numpy_convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def tensor_convert(tensor, target_min, target_max):
    tmin = tensor.min().item()
    tmax = tensor.max().item()
    a = (target_max  - target_min) / (tmax - tmin)
    b = target_max - a * tmax
    new_tensor = (a * tensor + b)
    return new_tensor