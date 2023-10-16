from commons.tool import listdir
from monai.networks.nets.swin_unetr import SwinUNETR

patch_size = [128, 128, 32]

ViT_patch_size = 16
num_classes = 1
channel_in = 1

model_lr = 0.0001
batch_size = 5
batch_size_val = 1
epochs = 100
train_step = 3
model = SwinUNETR(in_channels=channel_in, out_channels=num_classes, img_size=patch_size)

train_model_path = ""  # 从0开始训练填""
pre_model_path = "./SwinUNETR_CHAOSct_3.pth"
trainOrPredict = "predict"  # "train" or "predict"
openAMP = True  # 是否开启自动混合精度
save_log = False  # 是否记录训练日志
saveModel_name = "SwinUNETR_CHAOSct"
saveImage_name = "SwinUNETR_CHAOSct_pre"

train_image_path = '../dataset/Task20_CHAOSct/imagesTr'
train_label_path = '../dataset/Task20_CHAOSct/labelsTr'
test_image_path = '../dataset/Task20_CHAOSct/imagesTs'
test_label_path = '../dataset/Task20_CHAOSct/labelsTs'

train_image_list = listdir(train_image_path)
train_label_list = listdir(train_label_path)
# test_image_list = listdir(test_image_path)
# test_label_list = listdir(test_image_path)
test_image_list = listdir(test_image_path)
test_label_list = listdir(test_label_path)