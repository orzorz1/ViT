import os
import nibabel as nib
import numpy as np


def convert_nii_to_npz(src_dir, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # List all files in the source directory
    file_list = os.listdir(src_dir)

    for file_name in file_list:
        # Check if the file has the .nii.gz extension
        if file_name.endswith('.nii.gz'):
            print('Converting: ' + file_name)
            # Load the .nii.gz file
            nii_data = nib.load(os.path.join(src_dir, file_name))
            # Get the data as an array and convert to int16
            array_data = nii_data.get_fdata().astype(np.int16)

            # Create the .npz file name
            npz_file_name = file_name.replace('.nii.gz', '.npy')
            npz_file_path = os.path.join(dest_dir, npz_file_name)

            # Save the data as .npz
            np.save(npz_file_path, array_data)


# Paths
train_image_path = '../../dataset/Task001_LiTS17/imagesTr'
train_label_path = '../../dataset/Task001_LiTS17/labelsTr'
test_image_path = '../../dataset/Task001_LiTS17/imagesTs'
test_label_path = '../../dataset/Task001_LiTS17/labelsTs'

# Destination directories
dest_train_image_path = '../../dataset/LiTS_npy/imagesTr'
dest_train_label_path = '../../dataset/LiTS_npy/labelsTr'
dest_test_image_path = '../../dataset/LiTS_npy/imagesTs'
dest_test_label_path = '../../dataset/LiTS_npy/labelsTs'

# Convert the files
convert_nii_to_npz(train_image_path, dest_train_image_path)
convert_nii_to_npz(train_label_path, dest_train_label_path)
convert_nii_to_npz(test_image_path, dest_test_image_path)
convert_nii_to_npz(test_label_path, dest_test_label_path)
