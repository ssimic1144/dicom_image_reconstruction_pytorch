import pydicom
import numpy as np
import torch
from torchvision.transforms import transforms

from models.no_changes_net import Net
from dicom_dataset import convert


def get_numpy_array_from_tensor(tensor, min_value_for_conversion, max_value_for_conversion, numpy_type):
    numpy_array = tensor.cpu().detach().numpy()[0]
    numpy_array = convert(numpy_array, min_value_for_conversion, max_value_for_conversion, numpy_type)
    return numpy_array

def test_dicom_reconstruction(dicom_path, model, transformations_for_model, transformations_from_model):
    dicom_file = pydicom.dcmread(dicom_path)

    original_array_dicom = dicom_file.pixel_array
    original_type = original_array_dicom.dtype
    original_min = original_array_dicom.min()
    original_max = original_array_dicom.max()
    original_array_dicom = convert(original_array_dicom, 0, 255, np.uint8)
    
    new_array_dicom_list = []
    
    len_of_array_dicom = original_array_dicom.shape[0]
    
    model.eval()

    for position in range(0, len_of_array_dicom, 2):
        previous_projection = original_array_dicom[position,:,:]
        previous_projection = previous_projection[..., None]
        next_projection = original_array_dicom[(position+2)%len_of_array_dicom,:,:]
        next_projection = next_projection[..., None]

        previous_projection = transformations_for_model(previous_projection)
        next_projection = transformations_for_model(next_projection)
        previous_projection = previous_projection[None, ...]
        next_projection = next_projection[None, ...]

        output = model.forward(previous_projection,next_projection)

        previous_projection = transformations_from_model(previous_projection)
        output = transformations_from_model(output)

        previous_projection = get_numpy_array_from_tensor(previous_projection, original_min, original_max, original_type)
        output = get_numpy_array_from_tensor(output, original_min, original_max, original_type)
        
        new_array_dicom_list.append(previous_projection)
        new_array_dicom_list.append(output)

    new_array_dicom  = np.vstack(new_array_dicom_list)
    
    dicom_file.PixelData = new_array_dicom.tobytes()
    dicom_file.Rows, dicom_file.Columns = new_array_dicom.shape[1], new_array_dicom.shape[2]
    
    return dicom_file

def production_dicom_reconstruction(dicom_path, model, transformations_for_model, transformations_from_model):
    dicom_file = pydicom.dcmread(dicom_path)

    original_array_dicom = dicom_file.pixel_array
    original_type = original_array_dicom.dtype
    original_min = original_array_dicom.min()
    original_max = original_array_dicom.max()
    original_array_dicom = convert(original_array_dicom, 0, 255, np.uint8)

    new_array_dicom_list = []

    len_of_array_dicom = original_array_dicom.shape[0]

    model.eval()

    for position in range(len_of_array_dicom):
        previous_projection = original_array_dicom[position,:,:]
        previous_projection = previous_projection[..., None]
        next_projection = original_array_dicom[(position+1)%len_of_array_dicom,:,:]
        next_projection = next_projection[..., None]

        previous_projection = transformations_for_model(previous_projection)
        next_projection = transformations_for_model(next_projection)
        previous_projection = previous_projection[None, ...]
        next_projection = next_projection[None, ...]

        output = model.forward(previous_projection,next_projection)

        previous_projection = transformations_from_model(previous_projection)
        output = transformations_from_model(output)
        
        previous_projection = get_numpy_array_from_tensor(previous_projection, original_min, original_max, original_type)
        output = get_numpy_array_from_tensor(output, original_min, original_max, original_type)
        
        new_array_dicom_list.append(previous_projection)
        new_array_dicom_list.append(output)

    new_array_dicom  = np.vstack(new_array_dicom_list)
    
    dicom_file.PixelData = new_array_dicom.tobytes()
    dicom_file.Rows, dicom_file.Columns = new_array_dicom.shape[1], new_array_dicom.shape[2]
    
    return dicom_file



if __name__=="__main__":
    dicom_path = "../test_slika/jazack1.IMA"

    model = Net()
    model.load_state_dict(torch.load("model.pt"))
    
    transformations_for_model = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize((0.5,),(0.5,))
    ])
    
    transformations_from_model = transforms.Compose([
        transforms.Resize(128),
        transforms.Normalize(mean=(-0.5/0.5), std=(1/0.5))
    ])
    

    dicom_file = test_dicom_reconstruction(dicom_path, model, transformations_for_model, transformations_from_model)
    #Production function has not been tested yet
    #dicom_file = production_dicom_reconstruction(dicom_path, model, transformations_for_model, transformations_from_model) 

    dicom_file.save_as("generated_dicom.IMA")
    print("Dicom file generated.")