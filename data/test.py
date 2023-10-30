import time
import nibabel as nib
import numpy as np

# Step 1: Read the .nii.gz file
nii_path = './case_00000.nii.gz'
start = time.time()
nii_data = nib.load(nii_path)
data_array = np.array(nii_data.dataobj)
print('Time for reading .nii.gz: ', time.time() - start)
# Step 2: Save data as .npy and .npz
npy_path = './case_00000.npy'
npz_path = './case_00000.npz'
data_array = data_array.astype(np.int16)
np.save(npy_path, data_array)
np.savez_compressed(npz_path, data=data_array)

# Step 3: Load and test opening time for .npy and .npz
start_time_npy = time.time()
loaded_npy = np.load(npy_path)
loading_time_npy = time.time() - start_time_npy

start_time_npz = time.time()
loaded_npz = np.load(npz_path)['data']
loading_time_npz = time.time() - start_time_npz

print('Loading time for .npy: ', loading_time_npy)
print('Loading time for .npz: ', loading_time_npz)