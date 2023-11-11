import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import nibabel as nib
import torch
# from driver.config import patch_size, batch_size, batch_size_val, epochs, train_step, test_target_image_list,\
#     saveModel_name, saveImage_name, save_log, trainOrPredict, model, pre_model_path, train_model_path,\
#     train_target_image_list, train_target_label_list, train_source_image_list, test_source_image_list, openAMP, predict_list


def set_label(x):
    if x != 2:
        x = 0
    else:
        x = 2
    return x

def read_nii(path):
    img = nib.load(path)
    img_arr = np.array(img.dataobj)
    triangle_ufunc1 = np.frompyfunc(set_label, 1, 1)
    out = triangle_ufunc1(img_arr)
    out = out.astype(np.float)
    # out = img_arr
    return out

def dice_coef(y_pred, y_true):
    im1 = np.asarray(y_true).astype(np.bool)
    im2 = np.asarray(y_pred).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def get_accuracy(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()
    corr = torch.sum(SR == GT)
    tensor_size = torch.prod(torch.tensor(SR.size()))
    acc = float(corr) / float(tensor_size)
    return acc

def jaccard(y_pred, y_true):
    intersect = np.sum(y_true * y_pred)  # Intersection points
    union = np.sum(y_true) + np.sum(y_pred)  # Union points
    return (float(intersect)) / (union - intersect + 1e-7)

def compute_jaccard(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_jaccard = 0.
    thresholded_jaccard = 0.

    for im_index in range(y_pred.shape[0]):
        current_jaccard = jaccard(y_true=y_true[im_index], y_pred=y_pred[im_index])

        mean_jaccard += current_jaccard
        thresholded_jaccard += 0 if current_jaccard < 0.65 else current_jaccard

    mean_jaccard = mean_jaccard / y_pred.shape[0]
    thresholded_jaccard = thresholded_jaccard / y_pred.shape[0]

    return mean_jaccard, thresholded_jaccard

def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TP : True Positive
    # FN : False Negative
    TP = (((SR == 1).int() + (GT == 1).int()).int() == 2).int()
    FN = (((SR == 0).int() + (GT == 1).int()).int() == 2).int()

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE

def get_specificity(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TN : True Negative
    # FP : False Positive
    TN = (((SR == 0).int() + (GT == 0).int()).int() == 2).int()
    FP = (((SR == 1).int() + (GT == 0).int()).int() == 2).int()

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP

def get_precision(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TP : True Positive
    # FP : False Positive
    TP = (((SR == 1).int() + (GT == 1).int()).int() == 2).int()
    FP = (((SR == 1).int() + (GT == 0).int()).int() == 2).int()

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC

def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    Inter = torch.sum((SR + GT) == 2).int()
    Union = torch.sum((SR + GT) >= 1).int()

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    Inter = torch.sum((SR + GT) == 2).int()
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC

# import surface_distance as surfdist
#
# def get_ASSD(SR, GT):
#     mask_gt = np.asarray(GT).astype(np.bool)
#     mask_pred = np.asarray(SR).astype(np.bool)
#     surface_distances = surfdist.compute_surface_distances(
#         mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
#     avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
#
#     return avg_surf_dist
# (average_distance_gt_to_pred, average_distance_pred_to_gt)


metric = {"Dice": dice_coef,
          "Accuracy": get_accuracy,
          # "Jaccard": compute_jaccard,x
          "Sensitivity": get_sensitivity,
          "Specificity": get_specificity,
          "Precision": get_precision,
          # "F1": get_F1,x
          "Jaccard": get_JS,
          # "DC": get_DC,x
          # "ASSD":get_ASSDx
}

# for key, value in metric.items():
#     print(key)
#     for index in range(0,3):
#         path_x = predict_list[index]
#         x = read_nii(path_x)
#         x = torch.tensor(x)
#         path_y = train_target_label_list[index+17]
#         y = read_nii(path_y)
#         y = torch.tensor(y)
#         print(value(x, y))

# for key, value in metric.items():
#     print(key)
#     for n in range(0,4):
#         path_x = "../save/cycleGAN_3D/test/MR2CT_cycleGAN_RA_SE_128_pre_"+str(n)+".nii.gz"
#         x = read_nii(path_x)
#         x = torch.tensor(x)
#         path_y = "../save/cycleGAN_3D/test/labels/"+str(n)+".nii"
#         y = read_nii(path_y)
#         y = torch.tensor(y)
#         print(value(x, y))

for key, value in metric.items():
    print(key)
    for n in range(0,31):
        path_x = "../save/LiTS17/UNETRpp/40x6/UNETRpp_LiTS17_pre_"+str(n)+".nii.gz"
        x = read_nii(path_x)
        x = torch.tensor(x)
        path_y = "../../../../dataset/nnUNet_raw/Task001_LiTS17/labelsTs/case_00"+str(n+100)+".nii.gz"
        # path_y = "G:/file/Project/Deng/dataset/nnUNet_raw/Task20_CHAOSct/labelsTs/case_000"+str(n+15)+".nii.gz"
        y = read_nii(path_y)
        y = torch.tensor(y)
        print(value(x, y))