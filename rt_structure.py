import pydicom
import numpy as np
from skimage.draw import polygon
import os
import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import nibabel as nib


paticipent_id = '089'
original_data = '1.000000-NA-00092'

# Load the RTSTRUCT file and the corresponding image DICOM file(s)
dataset_path = f'/home/mbpl/morizane/analysis_sensitivity/dataset/LUNG1-{paticipent_id}'
dicom_series_path = os.path.join(dataset_path, 'original_data')
rt_struct_path = os.path.join(dataset_path, 'rt_struct/1-1.dcm')


# Load existing RT Struct. Requires the series path and existing RT Struct path
rtstruct = RTStructBuilder.create_from(
  dicom_series_path=dicom_series_path,
  rt_struct_path=rt_struct_path
)

# View all of the ROI names from within the image
print(rtstruct.get_roi_names())

# Loading the 3D Mask from within the RT Struct
mask_3d = rtstruct.get_roi_mask_by_name('Lung-Right')
mask_3d_l = rtstruct.get_roi_mask_by_name('Lung-Left')

tumour_mask = mask_3d.astype(np.uint8) + mask_3d_l.astype(np.uint8)
tumour_mask = np.flip(tumour_mask, axis=2)
tumour_mask = np.rot90(tumour_mask, k=1, axes=(0, 1))

print(mask_3d.shape)

nii_file = os.path.join(dataset_path, 'original_data_airway.nii')
nii_img = nib.load(nii_file)
nii_array = nii_img.get_fdata().astype(np.uint8)
nii_array[tumour_mask == 1] = 2
print(nii_array.shape)

nii_file = os.path.join(dataset_path, 'original_data_all_segmentations.nii/original_data_all_segmentations.nii')
nii_img = nib.load(nii_file)
aa = nii_img.get_fdata().astype(np.uint8)
aa = np.flip(aa, axis=2)
aa = np.rot90(aa, k=1, axes=(0, 1))

from matplotlib import pyplot as plt
import matplotlib

plt.imshow(aa[:, 200, :])
plt.show()

# Correct usage of cv2.fillPoly
# Example usage:
# mask = np.zeros((height, width), dtype=np.uint8)
# points = np.array([[x1, y1], [x2, y2], [x3, y3]])
# cv2.fillPoly(mask, [points], 1)

