import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
from modules.functions import dice_loss, ce_loss,bce_loss, adjust_learning_rate, sigmoid
from commons.plot import save_nii, draw, draw1, save_nii_
from data.LoadData import *
from torch.utils.data import DataLoader
import torch
import math
from commons.plot import save_nii
import torchsummary
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
from config.config_ViT import *
from commons.log import make_print_to_file
from torch.cuda.amp import autocast as autocast, GradScaler


class BaseTrainHelper(object):
    def __init__(self):
        self.model = model

    def train(self, model_load = ""):
        loss_train = []
        loss_val = []
        model = self.model
        scaler = GradScaler()  #自动混合精度运算
        try:
            model.load_state_dict(torch.load(model_load, map_location='cpu'))
        except FileNotFoundError:
            print("模型不存在")
        else:
            print("加载模型成功")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_lr)
        print(torch.cuda.memory_summary())
        torchsummary.summary(model, (1,128,128,32), batch_size=batch_size, device="cuda")
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [epochs//3, epochs//3*2], 0.1)
        for i in range(1, train_step+1):
            print("训练进度：{index}/{train_step}".format(index=i,train_step=train_step))
            dataset = load_dataset(train_image_list, train_label_list, 0, 14, i, patch_size)
            val_data = load_dataset_one(train_image_list, train_label_list, 15, patch_size)
            train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(dataset=val_data, batch_size=batch_size_val)
            for epoch in range(epochs):
                # training-----------------------------------
                model.train()
                train_loss = 0
                adjust_learning_rate(optimizer, epoch, epochs, 1, model_lr)
                for batch, (batch_x, batch_y, position) in enumerate(train_loader):
                    batch_x, batch_y = torch.autograd.Variable(batch_x.to(device)), torch.autograd.Variable(batch_y.to(device))
                    optimizer.zero_grad()
                    if openAMP == False:
                        out = model(batch_x)['out']
                        loss = bce_loss(out, batch_y)
                        loss.backward()
                        optimizer.step()
                        # print(torch.cuda.memory_summary())
                    else:
                        with autocast():
                            out = model(batch_x)['out']
                            loss = bce_loss(out, batch_y)
                        scaler.scale(loss).backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2) #梯度裁剪,防止梯度爆炸
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()

                    train_loss += loss.item()
                    print('epoch: %2d/%d batch %3d/%d  Train Loss: %.6f'
                          % (epoch + 1, epochs, batch + 1, math.ceil(len(dataset) / batch_size),loss.item(),))
                # scheduler.step()  # 更新learning rate
                print('Train Loss: %.6f' % (train_loss / (math.ceil(len(dataset) / batch_size))))
                loss_train.append(train_loss / (math.ceil(len(dataset) / batch_size)))

                #evaluation---------------------
                model.eval()
                eval_loss = 0
                for batch, (batch_x, batch_y, position) in enumerate(val_loader):
                    batch_x, batch_y = torch.autograd.Variable(batch_x.to(device)), torch.autograd.Variable(batch_y.to(device))
                    if openAMP == False:
                        out = model(batch_x)['out']
                        # loss, l, n = dice_loss(out, batch_y)
                        loss = bce_loss(out, batch_y)
                    else:
                        out = model(batch_x)['out']
                        loss = bce_loss(out, batch_y)
                        # with autocast():
                        #     out = model(batch_x)
                        #     loss = ce_loss(out, batch_y)
                        # scaler.update()
                    eval_loss += loss.item()
                    if (batch == 3 or batch == 6 or batch == 9) and (epoch == 99 or epoch == 79 or epoch == 49 or epoch == 29 or epoch == 0):
                        print(1)
                        save_nii(batch_x.cpu().numpy().astype(np.int16)[0][0],'{name}-{e}-{batch}X'.format(name=i, e=epoch+1, batch=batch))
                        save_nii(batch_y.cpu().numpy().astype(np.int16)[0][0],'{name}-{e}-{batch}Y'.format(name=i, e=epoch+1, batch=batch))
                        out = np.around(sigmoid(out.cpu().detach().numpy()[0]))
                        save_nii(out[0], '{name}-{e}-{batch}Out0'.format(name=i, e=epoch+1, batch=batch))

                print('Val Loss: %.6f' % (eval_loss / (math.ceil(len(val_data) / batch_size_val))))
                loss_val.append((eval_loss / (math.ceil(len(val_data) / batch_size))))
            torch.save(model.state_dict(), saveModel_name+"_"+str(i)+".pth")
            draw1(loss_train, "{i}-train".format(i=i))
            draw1(loss_val, "{i}-val".format(i=i))
            print(loss_train)
            print(loss_val)



    def predct(self, begin, end, model_load):
        model = self.model
        model.load_state_dict(torch.load(model_load, map_location='cpu'))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for index in range(begin, end + 1):
            print("predicting {i}".format(i=index))
            val_data = load_dataset_test(test_image_list, test_label_list, index, patch_size)
            val_loader = DataLoader(dataset=val_data, batch_size=batch_size)
            model.eval()
            path_x = test_image_list[index]
            print(path_x)
            x1 = read_dataset(path_x)
            print(x1.shape)
            x = np.zeros(( x1.shape[0], x1.shape[1], x1.shape[2]))
            predict = np.zeros_like(x)
            count = np.zeros_like(x)
            for batch, (batch_x, batch_y, p) in enumerate(val_loader):
                print(p)
                batch_x, batch_y = torch.autograd.Variable(batch_x.to(device)), torch.autograd.Variable(
                    batch_y.to(device))

                out = model(batch_x)['out'][0]
                    # with autocast():
                    #     out = model(batch_x)
                out = out.cpu().detach().numpy()
                for i in range(out.shape[0]):
                    position = [0, 0, 0]
                    o = out[i]
                    for j in range(len(p)):
                        position[j] = p[j].cpu().numpy().tolist()[i]
                    predict[position[0]:position[0] + patch_size[0], position[1]:position[1] + patch_size[1],
                    position[2]:position[2] + patch_size[2]] += o
                    count[position[0]:position[0] + patch_size[0], position[1]:position[1] + patch_size[1],
                    position[2]:position[2] + patch_size[2]] += np.ones_like(o)

            pre = predict / count
            pre[np.isnan(pre)] = 0.0001
            pre = np.around(sigmoid(pre))
            print(pre.shape)
            save_nii_(pre.astype(np.int16), saveImage_name+"_"+str(index), path_x)

if __name__ == '__main__':
    if save_log:
        make_print_to_file("./")
    torch.cuda.empty_cache()
    NetWork = BaseTrainHelper()
    if trainOrPredict == "train":
        NetWork.train(train_model_path)
    else:
        print(pre_model_path)
        NetWork.predct(0, 4, pre_model_path)

    os.system("shutdown")