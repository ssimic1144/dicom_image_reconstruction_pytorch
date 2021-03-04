import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from dicom_dataset import DicomDataset
from ssim_loss import SSIM
from baseline import baseline_tensor

from piq import SSIMLoss

transformations = transforms.Compose([transforms.ToTensor()])
load_dataset = DicomDataset("../slike/", transform=transformations)
dataset = DataLoader(dataset=load_dataset,batch_size=1,shuffle=True)

#criterion = SSIM()
criterion = SSIMLoss()
for _ in range(1):
    all_values = []
    for _, (prev_img, next_img, expcted_img) in enumerate(dataset):
        value = criterion(baseline_tensor(prev_img,next_img),expcted_img)
        all_values.append(value.item())
    avg_value = np.array(all_values).mean()
    min_value = np.array(all_values).min()
    max_value = np.array(all_values).max()
    print("Avg. SSIM value for expected and output tensor : {:.4f}\nMin : {:.4f}\nMax : {:.4f}".format(avg_value, min_value, max_value))