import torch
from torchvision.transforms import transforms
import pydicom
import matplotlib.pyplot as plt 

from nn_model import Net

model = Net()
model.load_state_dict(torch.load("model.pt"))
model.eval()

dicom_path = "../slike/jazack1.IMA"

dicom_file = pydicom.dcmread(dicom_path)

array_dicom = dicom_file.pixel_array
array_dicom = array_dicom.astype("uint8")
print(array_dicom.shape)

first_img = array_dicom[0,:,:]
second_img = array_dicom[2,:,:]
expected_img = array_dicom[1,:,:]

transformations_for_model = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(64),
    transforms.Normalize((0.5), (0.5))
])

transformations_from_model = transforms.Compose([
    transforms.Resize(128),
    transforms.Normalize((-0.5/0.5),(1.0/0.5))
])

first_img = transformations_for_model(first_img)
print(first_img.shape)
second_img = transformations_for_model(second_img)

first_img = first_img[None,...]
second_img = second_img[None,...]
print(first_img.shape)
output = model.forward(first_img,second_img)

output = transformations_from_model(output)

output_numpy = output.cpu().detach().numpy()

output_numpy = output_numpy[0,:,:,:]
print(output_numpy.shape)
plt.imshow(output_numpy[0])
plt.show()
plt.imshow(expected_img)
plt.show()