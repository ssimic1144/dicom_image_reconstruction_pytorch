import torch
import numpy as np 

def baseline_tensor(previous_image_tensor, next_image_tensor):
    previous_np_array = previous_image_tensor.cpu().detach().numpy()[0]
    next_np_array = next_image_tensor.cpu().detach().numpy()[0]

    output_np_array = (previous_np_array+next_np_array)/2

    output_tensor_image = torch.from_numpy(output_np_array)
    output_tensor_image = output_tensor_image[None, ...]

    return output_tensor_image
    
