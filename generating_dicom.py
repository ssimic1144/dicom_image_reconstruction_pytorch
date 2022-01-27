import pydicom
import numpy as np
import torch
from torchvision.transforms import transforms

from models.piq_nc_net import Net
from utils.common import numpy_convert
from baseline import baseline_tensor


def get_numpy_array_from_tensor(tensor, min_value_for_conversion, max_value_for_conversion, numpy_type):
    numpy_array = tensor.cpu().detach().numpy()[0]
    numpy_array = numpy_convert(numpy_array, min_value_for_conversion, max_value_for_conversion, numpy_type)
    return numpy_array

def test_dicom_reconstruction(dicom_path, model, transformations_for_model, transformations_from_model):
    dicom_file = pydicom.dcmread(dicom_path)

    original_array_dicom = dicom_file.pixel_array
    original_type = original_array_dicom.dtype
    original_min = original_array_dicom.min()
    original_max = original_array_dicom.max()
    original_array_dicom = numpy_convert(original_array_dicom, 0, 255, np.uint8)
    
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
        #output = baseline_tensor(previous_projection, next_projection)

        previous_projection = get_numpy_array_from_tensor(previous_projection, original_min, original_max, original_type)
        output = get_numpy_array_from_tensor(output, original_min, original_max, original_type)
        
        new_array_dicom_list.append(previous_projection)
        new_array_dicom_list.append(output)

    new_array_dicom  = np.vstack(new_array_dicom_list)
    
    dicom_file.PixelData = new_array_dicom.tobytes()
    dicom_file.Rows, dicom_file.Columns = new_array_dicom.shape[1], new_array_dicom.shape[2]
    dicom_file.NumberOfFrames = new_array_dicom.shape[0]

    return dicom_file

def production_dicom_reconstruction(dicom_path, model, transformations_for_model, transformations_from_model):
    dicom_file = pydicom.dcmread(dicom_path)

    original_array_dicom = dicom_file.pixel_array
    original_type = original_array_dicom.dtype
    original_min = original_array_dicom.min()
    original_max = original_array_dicom.max()
    original_array_dicom = numpy_convert(original_array_dicom, 0, 255, np.uint8)

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
    dicom_file.NumberOfFrames = new_array_dicom.shape[0]
    return dicom_file

def four_input_dicom_reconstruction(dicom_path, model, transformations_for_model, transformations_from_model):
    dicom_file = pydicom.dcmread(dicom_path)

    original_array_dicom = dicom_file.pixel_array
    original_type = original_array_dicom.dtype
    original_min = original_array_dicom.min()
    original_max = original_array_dicom.max()
    original_array_dicom = numpy_convert(original_array_dicom, 0, 255, np.uint8)

    new_array_dicom_list = []

    len_of_array_dicom = original_array_dicom.shape[0]

    model.eval()
    for position in range(1, len_of_array_dicom, 2):
        one_projection = original_array_dicom[(position-3)%len_of_array_dicom,:,:]
        three_projection = original_array_dicom[(position-1)%len_of_array_dicom,:,:]
        five_projection = original_array_dicom[(position+1)%len_of_array_dicom,:,:]
        seven_projection = original_array_dicom[(position+3)%len_of_array_dicom,:,:]

        one_projection, three_projection, five_projection, seven_projection = one_projection[... , None], three_projection[... , None], five_projection[... , None], seven_projection[... , None]

        one_projection = transformations_for_model(one_projection)
        three_projection = transformations_for_model(three_projection)
        five_projection = transformations_for_model(five_projection)
        seven_projection = transformations_for_model(seven_projection)

        one_projection, three_projection, five_projection, seven_projection = one_projection[None,...], three_projection[None,...], five_projection[None,...], seven_projection[None,...]
        
        output = model.forward(one_projection,three_projection,five_projection,seven_projection)

        three_projection = transformations_from_model(three_projection)
        output = transformations_from_model(output)

        three_projection = get_numpy_array_from_tensor(three_projection, original_min, original_max, original_type)
        output = get_numpy_array_from_tensor(output, original_min, original_max, original_type)
        
        new_array_dicom_list.append(three_projection)
        new_array_dicom_list.append(output)

    new_array_dicom  = np.vstack(new_array_dicom_list)
    dicom_file.PixelData = new_array_dicom.tobytes()
    dicom_file.Rows, dicom_file.Columns = new_array_dicom.shape[1], new_array_dicom.shape[2]
    dicom_file.NumberOfFrames = new_array_dicom.shape[0]

    return dicom_file

if __name__=="__main__":
    dicom_path = "data/jaszczak_rare_acq_test.dcm"

    model = Net()
    model.load_state_dict(torch.load("improved_3_loss_300_epochs_piq_model.pt"))
    
    
    transformations_for_model = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    
    transformations_from_model = transforms.Compose([
        transforms.Normalize(mean=(-0.5/0.5), std=(1/0.5))
    ])
    

    dicom_file = test_dicom_reconstruction(dicom_path, model, transformations_for_model, transformations_from_model)
    #Production function has not been tested yet
    #dicom_file = production_dicom_reconstruction(dicom_path, model, transformations_for_model, transformations_from_model) 


    dicom_file.save_as("NN_jaszczak_rare_acq_test_hf_generated.IMA")
    print("Dicom file generated.")
