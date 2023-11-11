from commons.tool import listdir
from models.ResNet18 import ResNet18

patch_size = [128, 128, 64]

ViT_patch_size = 16
num_classes = 3
channel_in = 1

model_lr = 0.0001
batch_size = 5
batch_size_val = 1
epochs = 40
train_step = 6
model = ResNet18(in_channels=channel_in, num_classes=num_classes)

train_model_path = ""
pre_model_path = "/root/autodl-tmp/ViT/ResNet18_LiTS17_6.pth"
trainOrPredict = "train"  # "train" or "predict"
openAMP = True  # 是否开启自动混合精度
save_log = False  # 是否记录训练日志
saveModel_name = "ResNet18_LiTS17"
saveImage_name = "ResNet18_LiTS17_pre"


train_image_path = '../dataset/LiTS17_npy/imagesTr'
train_label_path = '../dataset/LiTS17_npy/labelsTr'
test_image_path = '../dataset/LiTS17_npy/imagesTs'
test_label_path = '../dataset/LiTS17_npy/labelsTs'

train_image_list= listdir(train_image_path)
train_label_list = listdir(train_label_path)
test_image_list = listdir(test_image_path)
test_label_list = listdir(test_label_path)
