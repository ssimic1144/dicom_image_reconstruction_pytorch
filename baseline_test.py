import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from dicom_dataset import DicomDataset
from ssim_loss import SSIM
from baseline import baseline_tensor

transformations = transforms.Compose([transforms.ToTensor()])
load_dataset = DicomDataset("../slike/", transform=transformations)
dataset = DataLoader(dataset=load_dataset,batch_size=1,shuffle=True)

criterion = SSIM()
for _ in range(1):
    all_ssim_values = []
    for _, (prev_img, next_img, expcted_img) in enumerate(dataset):
        ssim_value = criterion(baseline_tensor(prev_img,next_img),expcted_img)
        all_ssim_values.append(ssim_value.item())
    avg_ssim_value = np.array(all_ssim_values).mean()
    min_ssim_value = np.array(all_ssim_values).min()
    max_ssim_value = np.array(all_ssim_values).max()
    print("Avg. SSIM value for expected and output tensor : {:.4f}\nMin : {:.4f}\nMax : {:.4f}".format(avg_ssim_value, min_ssim_value, max_ssim_value))