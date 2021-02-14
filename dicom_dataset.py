from torch.utils.data import Dataset
import pydicom
import os
import numpy as np
from PIL import Image

class DicomDataset(Dataset):
    def __init__(self, dicom_dir, transform = None):
        self.dicom_dir = dicom_dir
        self.transform = transform
        self.dicom_files = os.listdir(self.dicom_dir)
        self.all_np_dicoms = self.get_all_dicoms_in_one_array()
        self.list_of_shapes = self.get_list_of_shapes()

    def __len__(self):
        return self.all_np_dicoms.shape[0] - 1
    
    def __getitem__(self, index):
        expected_image_index = None
        previous_image_index = None
        next_image_index = None
        for position, x in enumerate(self.list_of_shapes):
            if index == 0:
                current_shape = self.list_of_shapes[position+1]
                expected_image_index = index
                previous_image_index = index + current_shape - 1
                next_image_index = index + 1
                break
            #prvi
            if index == x:
                current_shape = x - self.list_of_shapes[position-1]
                expected_image_index = index
                previous_image_index = index + current_shape - 1 
                next_image_index = index + 1 
                break
            #zadnji
            if index == x-1:
                current_shape = x - self.list_of_shapes[position-1]
                expected_image_index = index
                previous_image_index = index -1 
                next_image_index = index - current_shape + 1
                break
        
        if expected_image_index is None:
            expected_image_index = index
            previous_image_index = index -1 
            next_image_index = index + 1

        previous_image = self.all_np_dicoms[previous_image_index,:,:]
        next_image = self.all_np_dicoms[next_image_index,:,:]
        expected_image = self.all_np_dicoms[expected_image_index,:,:]

        if self.transform:
            previous_image = self.transform(previous_image)
            next_image = self.transform(next_image)
            expected_image = self.transform(expected_image)

        return previous_image, next_image, expected_image
    
    def get_all_dicoms_in_one_array(self):
        list_of_all_dicoms = []
        for dicom in self.dicom_files:
            array_dicom = self.get_dicom_pixel_array(self.dicom_dir, dicom)
            list_of_all_dicoms.append(array_dicom)
        np_dicoms = np.vstack(list_of_all_dicoms)
        np_dicoms = np_dicoms.astype('uint8')
        return np_dicoms

    def get_list_of_shapes(self):
        current_number = 0
        dicom_shape_list = []
        dicom_shape_list.append(current_number)
        for dicom in self.dicom_files:
            array_dicom = self.get_dicom_pixel_array(self.dicom_dir, dicom)
            current_number += array_dicom.shape[0]
            dicom_shape_list.append(current_number)
        return dicom_shape_list

    def get_dicom_pixel_array(self, dicom_dir, dicom):
        dicom_file_path = os.path.join(dicom_dir, dicom)
        dicom_file = pydicom.dcmread(dicom_file_path)
        array_dicom = dicom_file.pixel_array
        return array_dicom
    

if __name__=="__main__":
    new_dataset = DicomDataset("slike/")
    #print(new_dataset.dicom_files)
    print(new_dataset.__len__())
    #print(new_dataset.all_in_one())
    print(new_dataset.list_of_shapes)
    print(new_dataset.__getitem__(257))
    print(new_dataset.__getitem__(0))
    print(new_dataset.__getitem__(5))
    print(new_dataset.__getitem__(512))
    print(new_dataset.__getitem__(2975))
    print(new_dataset.__getitem__(1024))
    print(new_dataset.__getitem__(1025))
    #print(new_dataset.number_of_projection_per_dicom())
