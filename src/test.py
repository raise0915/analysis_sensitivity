import json
import numpy as np
import zlib
import base64
from matplotlib import pyplot as plt

def open_jnii_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Example usage
if __name__ == "__main__":
    with open('/home/mbpl/morizane/analysis_sensitivity/test_res.mc2') as f:
        data = np.fromfile(f, dtype=np.float32).reshape((600,600,600), order='F')
    plt.imshow(data[246,:,:])
    plt.show()

    import matplotlib.pyplot as plt

    file_path = '/home/mbpl/ToMorizane/results/air.jnii'
    data = open_jnii_file(file_path)
    # Extract the compressed NIFTI data
    compressed_data = data['NIFTIData']
    
    compressed_data = zlib.decompress(base64.b64decode(compressed_data['_ArrayZipData_']))
    
    # Convert the decompressed data to a numpy array and reshape it
    nifti_array = np.frombuffer(compressed_data, dtype=np.float32).reshape((600, 600, 600), order='F')
    
    # Display a slice of the 3D array using plt.imshow
    plt.imshow(nifti_array[:, :, 300], origin='lower')
    plt.show()
