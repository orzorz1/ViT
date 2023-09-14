from commons.tool import listdir
from models.ViT import ViTSeg

patch_size = [128, 128, 32]

ViT_patch_size = 16
num_classes = 2   #这个好像还不能用
channel_in = 1

model_lr = 0.0001
batch_size = 10
batch_size_val = 2
epochs = 100
train_step = 3
model = ViTSeg(image_size=patch_size, patch_size=ViT_patch_size, num_classes=num_classes,
               dim= 2048, depth=24, heads=48, mlp_dim=4096, channels=channel_in, learned_pos=False, use_token=True)

train_model_path = ""  # 从0开始训练填""
pre_model_path = "./save/ViT/3/model/ViTseg_3.pth"
trainOrPredict = "predict"  # "train" or "predict"
openAMP = True  # 是否开启自动混合精度
save_log = False  # 是否记录训练日志
saveModel_name = "ViTseg"
saveImage_name = "ViT_CHAOSct_pre"

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

