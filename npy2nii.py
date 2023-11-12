import numpy as np
import nibabel as nib


def combine_npy_with_nii(npy_file_path, nii_template_path, output_nii_path):
    # 读取.npy文件，这会得到一个numpy数组
    npy_data = np.load(npy_file_path)

    # 读取.nii文件，这会得到一个包含图像数据和头部信息的Nifti1Image对象
    nii_template = nib.load(nii_template_path)

    # 创建一个新的Nifti1Image对象，使用.npy的数据和.nii的头部信息
    new_nii_image = nib.Nifti1Image(npy_data, nii_template.affine, nii_template.header)

    # 保存新的.nii文件
    nib.save(new_nii_image, output_nii_path)
    print(f'New NIfTI file saved: {output_nii_path}')

# 调用函数
# combine_npy_with_nii('path/to/your/data.npy', 'path/to/your/template.nii', 'path/to/your/output.nii')
for i in range(31):
    npy_file_path = './UNETRpp_LiTS17_pre_{index}.npy'.format(index=i)
    nii_template_path = '../dataset/LiTS17/imagesTs/case_00{index}.nii.gz'.format(index=i+100)
    output_nii_path = './save/LiTS17/UNETRpp/norm40x6/pre_2/UNETRpp_LiTS17_pre_{index}.nii.gz'.format(index=i)
    combine_npy_with_nii(npy_file_path, nii_template_path, output_nii_path)