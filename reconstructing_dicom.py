import torch
from torchvision.transforms import transforms
import pydicom
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image

from models.no_changes_net import Net
#from models.basic_net import Net
#from models.simple_nn_model import Net
from ssim_loss import SSIM

model = Net()
model.load_state_dict(torch.load("model.pt"))
model.eval()

dicom_path = "../test_slika/jazack1.IMA"

dicom_file = pydicom.dcmread(dicom_path)

array_dicom = dicom_file.pixel_array
array_dicom = array_dicom.astype("uint8")
print(array_dicom.shape)

first_img = array_dicom[1,:,:]
second_img = array_dicom[3,:,:]
expected_img = array_dicom[2,:,:]


transformations_for_model = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(64),
    transforms.Normalize((0.5,),(0.5,))
])

transformations_from_model = transforms.Compose([
    transforms.Resize(128)

])

first_img = transformations_for_model(first_img)
#print(first_img.shape)
second_img = transformations_for_model(second_img)

first_img = first_img[None,...]
second_img = second_img[None,...]

output = model.forward(first_img,second_img)

output = transformations_from_model(output)

output_numpy = output.cpu().detach().numpy().transpose(0,2,3,1)[0]
#Convert to 0-255 range 
output_numpy = (255*(output_numpy - np.min(output_numpy))/np.ptp(output_numpy)).astype(int) 

print(output_numpy)
print(torch.max(output))
print(torch.min(output))

#plt.gray()
plt.imshow(output_numpy)
plt.show()
plt.imshow(expected_img)
plt.show()