from commons.tool import listdir
from models.UNETR_pp import UNETR_PP

patch_size = (128, 128, 32)

ViT_patch_size = 16
num_classes = 3
channel_in = 1

model_lr = 0.0001
batch_size = 30
batch_size_val = 4
epochs = 100
train_step = 3
model = UNETR_PP(in_channels=channel_in, out_channels=num_classes, patch_size=patch_size, dims=(32, 64, 128, 256),do_ds=False)

train_model_path = ""  # 从0开始训练填""
pre_model_path = "/root/autodl-tmp/ViT/UNETRpp_LiTS17_3.pth"
trainOrPredict = "train"  # "train" or "predict"
openAMP = True  # 是否开启自动混合精度
save_log = True  # 是否记录训练日志
saveModel_name = "UNETRpp_LiTS17"
saveImage_name = "UNETRpp_LiTS17_pre"

train_image_path = '../dataset/Task001_LiTS17/imagesTr'
train_label_path = '../dataset/Task001_LiTS17/labelsTr'
test_image_path = '../dataset/Task001_LiTS17/imagesTs'
test_label_path = '../dataset/Task001_LiTS17/labelsTs'

train_image_list = listdir(train_image_path)
train_label_list = listdir(train_label_path)
# test_image_list = listdir(test_image_path)
# test_label_list = listdir(test_image_path)
test_image_list = listdir(test_image_path)
test_label_list = listdir(test_label_path)