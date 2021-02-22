import pydicom
import numpy as np
import torch
from torchvision.transforms import transforms

from models.no_changes_net import Net

def test_dicom_reconstruction(array_dicom, model):
    new_array_dicom_list = []
    
    len_of_array_dicom = array_dicom.shape[0]

    transformations_for_model = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize((0.5,),(0.5,))
    ])
    
    transformations_from_model = transforms.Compose([
        transforms.Resize(128)
    ])
    
    model.eval()

    for position in range(0, len_of_array_dicom, 2):
        previous_projection = array_dicom[position,:,:]
        previous_projection = previous_projection[..., None]
        next_projection = array_dicom[(position+2)%len_of_array_dicom,:,:]
        next_projection = next_projection[..., None]

        previous_projection = transformations_for_model(previous_projection)
        next_projection = transformations_for_model(next_projection)
        previous_projection = previous_projection[None, ...]
        next_projection = next_projection[None, ...]

        output = model.forward(previous_projection,next_projection)

        previous_projection = transformations_from_model(previous_projection)
        output = transformations_from_model(output)
        
        previous_projection = previous_projection.cpu().detach().numpy()[0]
        previous_projection = (255*(previous_projection - np.min(previous_projection))/np.ptp(previous_projection)).astype(int) 
        output = output.cpu().detach().numpy()[0]
        output = (255*(output - np.min(output))/np.ptp(output)).astype(int) 
        
        new_array_dicom_list.append(previous_projection)
        new_array_dicom_list.append(output)


    new_array_dicom  = np.vstack(new_array_dicom_list)
    new_array_dicom = new_array_dicom.astype("uint16")
    return new_array_dicom

def production_dicom_reconstruction(array_dicom, model):
    new_array_dicom_list = []

    len_of_array_dicom = array_dicom.shape[0]   
    
    transformations_for_model = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize((0.5,),(0.5,))
    ])

    transformations_from_model = transforms.Compose([
        transforms.Resize(128)
    ])

    model.eval()

    for position in range(len_of_array_dicom):
        previous_projection = array_dicom[position,:,:]
        previous_projection = previous_projection[..., None]
        next_projection = array_dicom[(position+1)%len_of_array_dicom,:,:]
        next_projection = next_projection[..., None]

        previous_projection = transformations_for_model(previous_projection)
        next_projection = transformations_for_model(next_projection)
        previous_projection = previous_projection[None, ...]
        next_projection = next_projection[None, ...]

        output = model.forward(previous_projection,next_projection)

        previous_projection = transformations_from_model(previous_projection)
        output = transformations_from_model(output)
        
        previous_projection = previous_projection.cpu().detach().numpy()[0]
        previous_projection = (255*(previous_projection - np.min(previous_projection))/np.ptp(previous_projection)).astype(int)
        output = output.cpu().detach().numpy()[0]
        output = (255*(output - np.min(output))/np.ptp(output)).astype(int) 
        
        new_array_dicom_list.append(previous_projection)
        new_array_dicom_list.append(output)

    new_array_dicom  = np.vstack(new_array_dicom_list)
    new_array_dicom = new_array_dicom.astype("uint16")
    return new_array_dicom

def convert(img, target_type_min, target_type_max, target_type):
        imin = img.min()
        imax = img.max()
        a = (target_type_max - target_type_min) / (imax - imin)
        b = target_type_max - a * imax
        new_img = (a * img + b).astype(target_type)
        return new_img


dicom_path = "../test_slika/jazack1.IMA"

dicom_file = pydicom.dcmread(dicom_path)

original_array_dicom = dicom_file.pixel_array
original_array_dicom = convert(original_array_dicom, 0, 255, np.uint8)

model = Net()
model.load_state_dict(torch.load("model.pt"))

generated_pixel_array = test_dicom_reconstruction(original_array_dicom, model)
#You should use this one for reconstruction and comment out test
#generated_pixel_array = production_dicom_reconstruction(original_array_dicom, model) 

dicom_file.PixelData = generated_pixel_array.tobytes()
dicom_file.Rows, dicom_file.Columns = generated_pixel_array.shape[1], generated_pixel_array.shape[2]

dicom_file.save_as("generated_dicom.IMA")