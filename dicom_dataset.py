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
            if index == x:
                current_shape = x - self.list_of_shapes[position-1]
                expected_image_index = index
                previous_image_index = index + current_shape - 1 
                next_image_index = index + 1 
                break
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
            array_dicom = convert(array_dicom,0,255,np.uint8)
            list_of_all_dicoms.append(array_dicom)
        np_dicoms = np.vstack(list_of_all_dicoms)
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
    
def convert(img, target_type_min, target_type_max, target_type):
        imin = img.min()
        imax = img.max()
        a = (target_type_max - target_type_min) / (imax - imin)
        b = target_type_max - a * imax
        new_img = (a * img + b).astype(target_type)
        return new_img

class DicomDataset1357(DicomDataset):
    def __init__(self, dicom_dir, transform=None):
        super(DicomDataset1357, self).__init__(dicom_dir, transform)
    
    def __getitem__(self, index):
        expected_index = index
        one_index, three_index, five_index, seven_index = self.get_index(expected_index)
        
        #print(f"Positions : {one_index} {three_index} {five_index} {seven_index}")
        
        one_image = self.all_np_dicoms[one_index,:,:]
        three_image = self.all_np_dicoms[three_index,:,:]
        five_image = self.all_np_dicoms[five_index,:,:]
        seven_image = self.all_np_dicoms[seven_index,:,:]
        expected_image = self.all_np_dicoms[expected_index,:,:]

        if self.transform:
            one_image = self.transform(one_image)
            three_image = self.transform(three_image)
            five_image = self.transform(five_image)
            seven_image = self.transform(seven_image)
            expected_image = self.transform(expected_image)

        return one_image, three_image, five_image, seven_image, expected_image

    def get_index(self, expected_index):
        for position, x in enumerate(self.list_of_shapes):
            if expected_index == x:
                ending_index = self.list_of_shapes[position+1]
                
                one_index = ending_index - 3 
                three_index = ending_index - 1 
                five_index = expected_index + 1
                seven_index = expected_index + 3

                return one_index, three_index, five_index, seven_index

            if (expected_index < x) & (expected_index > self.list_of_shapes[position-1]):
                starting_index = self.list_of_shapes[position-1]
                current_shape = x - starting_index
                scaled_expected_index = expected_index - starting_index

                one_index = ((scaled_expected_index-3)%current_shape) + starting_index
                three_index = ((scaled_expected_index-1)%current_shape) + starting_index
                five_index = ((scaled_expected_index+1)%current_shape) + starting_index
                seven_index = ((scaled_expected_index+3)%current_shape) + starting_index

                return one_index, three_index, five_index, seven_index

if __name__=="__main__":
    new_dataset = DicomDataset1357("../slike/")