import numpy as np
import os
from rt_utils import RTStructBuilder
import nibabel as nib
import subprocess
from mayavi import mlab


class ConstructModel:
    def __init__(self, input_path, model_name, save, plot=False):
        self.input_path = input_path
        self.plot = plot
        self.save = save
        self.model_name = model_name
        
    def read_dicom(self):
        dicom_series_path = os.path.join(self.input_path, "original_data")
        rt_struct_path = os.path.join(self.input_path, 'rt_struct/1-1.dcm')
        # Load existing RT Struct. Requires the series path and existing RT Struct path
        rtstruct = RTStructBuilder.create_from(
        dicom_series_path=dicom_series_path,
        rt_struct_path=rt_struct_path
        )
        tumour_mask = rtstruct.get_roi_mask_by_name('GTV-1')
        tumour_mask = tumour_mask.astype(np.uint8) # + mask_3d_l.astype(np.uint8)
        tumour_mask = np.flip(tumour_mask, axis=2)
        tumour_mask = np.rot90(tumour_mask, k=1, axes=(0, 1))
        return tumour_mask
    
    def read_airway_segmented(self):
        subprocess.run("gzip -dc original_data_airway.nii.gz > original_data_airway.nii", shell=True, cwd=self.input_path)
        nii_file = os.path.join(self.input_path, 'original_data_airway.nii')
        nii_img = nib.load(nii_file)
        nii_array = np.where(nii_img.get_fdata() > 0, 1, 0).astype(np.uint8)
        return nii_array
    
    def create_model(self):
        nii_array = self.read_airway_segmented()
        tumour_mask = self.read_dicom()
        nii_array[tumour_mask==1] = 2
        if self.save:
            nii_array.tofile(os.path.join(self.input_path, f'{model_name}.bin'))
        if self.plot:
            mlab.figure()
            mlab.contour3d(nii_array, contours=5, opacity=0.1)
            mlab.show()

if __name__ == '__main__':
    plot = False
    save = True
    root_path = '/mnt/e/dataset'
    model_name = 'LUNG1-289'
    input_path = os.path.join(root_path, model_name)
    make_model = ConstructModel(input_path, model_name, save, plot)
    make_model.create_model()
