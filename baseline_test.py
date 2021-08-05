import torch
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from models.piq_nc_net import Net
from dicom_dataset import DicomDataset
from baseline import baseline_tensor

from piq import SSIMLoss, VIFLoss, HaarPSILoss

transformations = transforms.Compose([transforms.ToTensor()])

load_dataset = DicomDataset("../test_slika/", transform=transformations)
dataset = DataLoader(dataset=load_dataset,batch_size=1,shuffle=True)

transformations_for_model = transforms.Compose([
    transforms.Normalize((0.5,),(0.5,))
])

model = Net()
model.load_state_dict(torch.load("piq_model.pt"))
model.eval()

criterion_ssim = SSIMLoss()
criterion_vif = VIFLoss()
criterion_haarPSI = HaarPSILoss()
for _ in range(1):
    all_baseline_ssim_values = []
    all_model_ssim_values = []
    all_baseline_vif_values = []
    all_model_vif_values = []
    all_baseline_haarPSI_values = []
    all_model_haarPSI_values = []

    for _, (prev_img, next_img, expcted_img) in enumerate(dataset):
        prev_img_for_model = transformations_for_model(prev_img)
        next_img_for_model = transformations_for_model(next_img)

        baseline_ssim_value = criterion_ssim(baseline_tensor(prev_img,next_img),expcted_img)
        all_baseline_ssim_values.append(baseline_ssim_value.item())
        model_ssim_value = criterion_ssim(model.forward(prev_img_for_model,next_img_for_model),expcted_img)
        all_model_ssim_values.append(model_ssim_value.item())

        baseline_vif_value = criterion_vif(baseline_tensor(prev_img,next_img),expcted_img)
        all_baseline_vif_values.append(baseline_vif_value.item())
        model_vif_value = criterion_vif(model.forward(prev_img_for_model,next_img_for_model),expcted_img)
        all_model_vif_values.append(model_vif_value.item())
        
        """
        baseline_haarPSI_value = criterion_haarPSI(baseline_tensor(prev_img,next_img),expcted_img)
        all_baseline_haarPSI_values.append(baseline_haarPSI_value.item())
        model_haarPSI_value = criterion_haarPSI(model.forward(prev_img_for_model,next_img_for_model),expcted_img)
        all_model_haarPSI_values.append(model_haarPSI_value.item())
        """

    avg_baseline_value = np.array(all_baseline_ssim_values).mean()
    min_baseline_value = np.array(all_baseline_ssim_values).min()
    max_baseline_value = np.array(all_baseline_ssim_values).max()
    avg_model_value = np.array(all_model_ssim_values).mean()
    min_model_value = np.array(all_model_ssim_values).min()
    max_model_value = np.array(all_model_ssim_values).max()
    print("---------SSIM---------")
    print("BASELINE avg. value for expected and output tensor : {:.4f}\nBaseline Min : {:.4f}\nBaseline Max : {:.4f}".format(avg_baseline_value, min_baseline_value, max_baseline_value))
    print("MODEL avg. value for expected and output tensor : {:.4f}\nModel Min : {:.4f}\nModel Max : {:.4f}".format(avg_model_value, min_model_value, max_model_value))

    avg_baseline_value = np.array(all_baseline_vif_values).mean()
    min_baseline_value = np.array(all_baseline_vif_values).min()
    max_baseline_value = np.array(all_baseline_vif_values).max()
    avg_model_value = np.array(all_model_vif_values).mean()
    min_model_value = np.array(all_model_vif_values).min()
    max_model_value = np.array(all_model_vif_values).max()
    print("---------VIF---------")
    print("BASELINE avg. value for expected and output tensor : {:.4f}\nBaseline Min : {:.4f}\nBaseline Max : {:.4f}".format(avg_baseline_value, min_baseline_value, max_baseline_value))
    print("MODEL avg. value for expected and output tensor : {:.4f}\nModel Min : {:.4f}\nModel Max : {:.4f}".format(avg_model_value, min_model_value, max_model_value))
    
    """
    avg_baseline_value = np.array(all_baseline_haarPSI_values).mean()
    min_baseline_value = np.array(all_baseline_haarPSI_values).min()
    max_baseline_value = np.array(all_baseline_haarPSI_values).max()
    avg_model_value = np.array(all_model_haarPSI_values).mean()
    min_model_value = np.array(all_model_haarPSI_values).min()
    max_model_value = np.array(all_model_haarPSI_values).max()
    print("---------HaarPSI---------")
    print("BASELINE avg. value for expected and output tensor : {:.4f}\nBaseline Min : {:.4f}\nBaseline Max : {:.4f}".format(avg_baseline_value, min_baseline_value, max_baseline_value))
    print("MODEL avg. value for expected and output tensor : {:.4f}\nModel Min : {:.4f}\nModel Max : {:.4f}".format(avg_model_value, min_model_value, max_model_value))
    """