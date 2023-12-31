from commons.tool import listdir
from models.KiUNet import kiunet3d

patch_size = [64, 64, 32]

num_classes = 1
channel_in = 1

model_lr = 0.000001
batch_size = 5
batch_size_val = 1
epochs = 100
train_step = 1
model = kiunet3d(c=channel_in, n=1, num_classes=num_classes)


train_model_path = ""
pre_model_path = ""
trainOrPredict = "train"
openAMP = True  # 是否开启自动混合精度
save_log = False  # 是否记录训练日志
saveModel_name = "kiunet"
saveImage_name = "kiunet_CHAOSct_pre"

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
