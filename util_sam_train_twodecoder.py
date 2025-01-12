import sys
sys.path.append('..')
import numpy as np
import os
from datetime import datetime
import torch
import torch.nn.functional as F
from torch import nn

import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


BCE_loss = nn.BCELoss(size_average=True, reduce=False)
num_binary = 3



class DiceLoss_deep(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):

        # print("pred.shape, target.shape:", pred.shape, target.shape, uout_l.shape, liver.shape)
        # 首先将金标准拆开
        # 通过3个通道的 target 生成 organ_target，使用 argmax 会得到0、1、2（表示类别索引）
        organ_target = torch.argmax(target, dim=1)  # shape: [B, 256, 512]
        # 创建一个背景掩码，判断哪些像素点没有被前3类标注 (即前3个通道都是0)
        background_mask = (target.sum(dim=1) == 0)  # shape: [B, 256, 512]
        # 将 organ_target 中的背景像素标记为类别3
        organ_target[background_mask] = 3     # 0\1\2\3
        organ_target = organ_target + 1     # 1\2\3\4
        organ_target[organ_target==4] = 0     # 0\1\2\3
        # 将 organ_target 转换为 one-hot 编码，生成 [B, 256, 512, 4] 张量
        organ_target_one_hot = F.one_hot(organ_target, num_binary+1)
        organ_target_one_hot = organ_target_one_hot.permute(0, 3, 1, 2).float()

        dice = 0.0
        for organ_index in range(num_binary + 1):
            dice += 2 * (pred[:,organ_index,:,:] * organ_target_one_hot[:,organ_index,:,:]).sum(dim=1).\
                sum(dim=1) / (pred[:,organ_index,:,:].pow(2).sum(dim=1).sum(dim=1) +organ_target_one_hot[:,organ_index,:,:].pow(2).sum(dim=1).sum(dim=1) + 1e-5)

        



        BCE_loss = F.binary_cross_entropy(pred, organ_target_one_hot, reduce=True)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** 2 * BCE_loss




        dice_loss = 1 - dice / (num_binary + 1) + torch.mean(F_loss)
        # 返回的是dice距离
        return dice_loss

criterion_deep = DiceLoss_deep()


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, uout_l, liver):

        # print("pred.shape, target.shape:", pred.shape, target.shape, uout_l.shape, liver.shape)
        # 首先将金标准拆开
        # 通过3个通道的 target 生成 organ_target，使用 argmax 会得到0、1、2（表示类别索引）
        organ_target = torch.argmax(target, dim=1)  # shape: [B, 256, 512]
        # 创建一个背景掩码，判断哪些像素点没有被前3类标注 (即前3个通道都是0)
        background_mask = (target.sum(dim=1) == 0)  # shape: [B, 256, 512]
        # 将 organ_target 中的背景像素标记为类别3
        organ_target[background_mask] = 3     # 0\1\2\3
        organ_target = organ_target + 1     # 1\2\3\4
        organ_target[organ_target==4] = 0     # 0\1\2\3
        # 将 organ_target 转换为 one-hot 编码，生成 [B, 256, 512, 4] 张量
        organ_target_one_hot = F.one_hot(organ_target, num_binary+1)
        organ_target_one_hot = organ_target_one_hot.permute(0, 3, 1, 2).float()

        dice = 0.0
        for organ_index in range(num_binary + 1):
            dice += 2 * (pred[:,organ_index,:,:] * organ_target_one_hot[:,organ_index,:,:]).sum(dim=1).\
                sum(dim=1) / (pred[:,organ_index,:,:].pow(2).sum(dim=1).sum(dim=1) +organ_target_one_hot[:,organ_index,:,:].pow(2).sum(dim=1).sum(dim=1) + 1e-5)





        """区域一致性的Dice约束"""
        # 检查 liver 张量的类型
        # print("The dtype of liver tensor is: {liver.dtype}", liver.dtype, liver.shape, target.dtype, organ_target.dtype)


        liver_target_one_hot = F.one_hot(liver.long(), 2)
        liver_target_one_hot = liver_target_one_hot.permute(0, 3, 1, 2).float()
        """肝脏分支的Dice约束"""
        dice_liver = 0.0
        for organ_index in range(2):
            dice_liver += 2 * (uout_l[:, organ_index, :, :] * liver_target_one_hot[:, organ_index, :, :]).sum(dim=1). \
                sum(dim=1) / (uout_l[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) + liver_target_one_hot[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) + 1e-5)



        # 获取肝脏的注释（内部和外部）
        liver_target_inside = liver_target_one_hot[:, 1, :, :].unsqueeze(1)  # B*1*H*W
        liver_target_outside = liver_target_one_hot[:, 0, :, :].unsqueeze(1)  # B*1*H*W
        # 获取肝脏的预测（内部和外部）
        liver_pred_inside = uout_l[:, 1, :, :].unsqueeze(1)  # B*1*H*W
        liver_pred_outside = uout_l[:, 0, :, :].unsqueeze(1)  # B*1*H*W
        # ------------------------------
        # 将解剖学曲线的注释与肝脏的注释相乘
        # ------------------------------
        # 肝脏内部区域与解剖学曲线的注释相乘
        target_inside_liver = organ_target_one_hot * liver_target_inside  # B*4*H*W
        # 肝脏外部区域与解剖学曲线的注释相乘
        target_outside_liver = organ_target_one_hot * liver_target_outside  # B*4*H*W

        # ------------------------------
        # 将解剖学曲线的预测与肝脏的预测相乘
        # ------------------------------
        # 肝脏内部区域与解剖学曲线的预测相乘
        pred_inside_liver = pred * liver_pred_inside  # B*4*H*W
        # 肝脏外部区域与解剖学曲线的预测相乘
        pred_outside_liver = pred * liver_pred_outside  # B*4*H*W



        dice_inside = 0.0
        for organ_index in range(num_binary + 1):
            dice_inside += 2 * (pred_inside_liver[:, organ_index, :, :] * target_inside_liver[:, organ_index, :, :]).sum(dim=1). \
                sum(dim=1) / (pred_inside_liver[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) + target_inside_liver[:,organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) + 1e-5)

        dice_outside = 0.0
        for organ_index in range(num_binary + 1):
            dice_outside += 2 * (pred_outside_liver[:, organ_index, :, :] * target_outside_liver[:, organ_index, :, :]).sum(dim=1). \
                sum(dim=1) / (pred_outside_liver[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) + target_outside_liver[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) + 1e-5)



        BCE_loss = F.binary_cross_entropy(pred, organ_target_one_hot, reduce=True)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** 2 * BCE_loss




        dice_loss = 1 - dice / (num_binary + 1) + torch.mean(F_loss) + 1 -dice_liver/2 + 1 - dice_inside / (num_binary + 1) + 1 - dice_outside / (num_binary + 1)
        # 返回的是dice距离
        return dice_loss







def get_batch_acc(uout, label):
#def get_acc(uout, uout_1, label, label_1):
    """soft dice score"""
    eps = 1e-7
    uout = torch.Tensor(uout)
    label = torch.Tensor(label)


    #print("type(uout), uout.shape, type(label), label.shape:", type(uout), uout.shape, type(label), label.shape)
    iflat = uout.view(-1) .float()
    tflat = label.view(-1).float()
    intersection = (iflat * tflat).sum()
    dice_0 = 2 * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)
    # print("dice_0",dice_0)
    return dice_0


def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
        print("使用了 cuda")
    else:
        print("没使用cuda")
    prev_time = datetime.now()
    # 超算上用于保存模型的路径
    save = './save_model'     # /home/zxk/Code/self-sub/code/save
    save_results_data = "./save_result"
    # 定义初始化正确率为 0
    best_acc_1 = 0.0
    best_acc_2 = 0.0
    best_acc_3 = 0.0
    best_acc = 0.0





    for epoch in range(num_epochs):
#         if epoch % 20 == 0:
#             for p in optimizer.param_groups:
#                 p['lr'] *= 0.9

#         print("当前学习率为{:.6f}".format(p['lr']))

        train_loss = 0
        dice_1_acc = 0
        dice_2_acc = 0
        dice_3_acc = 0
        number_tra, number_val = 0, 0
        train_case_n, val_case_n = 0, 0
        net = net.train()
        print("len(train_data):", len(train_data))
        for batch in train_data:
            img_sequence, label_sequence, name, size, liver_sequence = [], [], [], [], []
            for i in range(len(batch)):
                """image enhance"""
                # print("batch[i][0].shape:", batch[i][0].shape, batch[i][0].max(), batch[i][0].min(), batch[i][1].shape, batch[i][2], batch[i][3])
                # img_tem = batch[i][0]

                img_sequence.extend([batch[i][0]])
                label_sequence.extend([batch[i][1]])
                name.extend([batch[i][2]])
                size.extend([batch[i][3]])
                liver_sequence.extend([batch[i][4]])


            img_sequence, label_sequence, liver_sequence = np.array(torch.stack(img_sequence)), np.array(torch.stack(label_sequence)), np.array(torch.stack(liver_sequence))
            #label_sequence[label_sequence > 1] = 1
            # img_sequence = (img_sequence) / 400.0
            img_sequence[img_sequence > 1], img_sequence[img_sequence < 0] = 1, 0
            # print("img_sequence.shape, label_sequence.shape:", img_sequence.shape, label_sequence.shape)


            # print("img_sequence, label_sequence, name, size:", len(img_sequence), img_sequence.shape, name, size)
            img_sequence = torch.Tensor(img_sequence)
            label_sequence = torch.Tensor(label_sequence)
            liver_sequence = torch.Tensor(liver_sequence)
            """得到tensor形式的数据"""
            im_1, label, name, size, liver = img_sequence, label_sequence, name, size, liver_sequence[:, 0, :, :]
            # print("liver.shape:", liver.shape)
            number_tra = number_tra + 1



            # saveimg, savelabel = np.array(im_1), np.array(label)
            # saveimg = saveimg[0,0,:,:]
            # #print("savelabel.shape:", savelabel.shape)
            # savelabel = savelabel.reshape((1, 256, 512))
            # saveimg = sitk.GetImageFromArray(saveimg)
            # sitk.WriteImage(saveimg, './demo/saveimg_train.nii.gz')
            # savelabel = sitk.GetImageFromArray(savelabel)
            # sitk.WriteImage(savelabel, './demo/savelabel_train.nii.gz')



            im_1, label, liver = im_1.cuda(), label.cuda(), liver.cuda()#

            # print("im_1.shape, label.shape:", im_1.shape, label.shape, liver.shape)
            uout, uout_l, landmark_5, landmark_4, landmark_3 = net(im_1)    # 因为术中所有的轮廓都是label，所以先使用全1的label进行训练；

            

            


            alpha = 1.0
            loss_1 = criterion(uout, label, uout_l, liver) + criterion_deep(landmark_5, label) + criterion_deep(landmark_4, label) + criterion_deep(landmark_3, label)



            new_uout = np.zeros((label.shape[0], 1, 1024, 1024))
            uout = uout.cpu()


            new_uout[uout[:,1:2,:,:]>0.5] = 1
            new_uout[uout[:,2:3,:,:]>0.5] = 2
            new_uout[uout[:,3:4,:,:]>0.5] = 3



            optimizer.zero_grad()
            loss_1.backward()
            optimizer.step()

            label = label.cpu().numpy()
            new_uout_1 = np.zeros(new_uout.shape)
            new_uout_1[new_uout == 1] =1
            label_1 = label[:, 0, :, :]

            new_uout_2 = np.zeros(new_uout.shape)
            new_uout_2[new_uout == 2] =1
            label_2 = label[:, 1, :, :]

            new_uout_3 = np.zeros(new_uout.shape)
            new_uout_3[new_uout == 3] =1
            label_3 = label[:, 2, :, :]



            dice_1 = get_batch_acc(new_uout_1, label_1)
            dice_2 = get_batch_acc(new_uout_2, label_2)
            dice_3 = get_batch_acc(new_uout_3, label_3)




            # print("savelabel.shape:", savelabel.shape)

            dice_1_acc = dice_1_acc + dice_1
            dice_2_acc = dice_2_acc + dice_2
            dice_3_acc = dice_3_acc + dice_3
            # if epoch % 10 == 0:
            #     print("traing---当前%s 的dice1值为:%f,dice2值为:%f,dice3值为:%f"%(name[0],dice_1,dice_2,dice_3))
            train_case_n = train_case_n + 1



            new_uout = new_uout[:, 0, :, :]
            # image_nii = sitk.GetImageFromArray(im.cpu())
            # train_img_path = os.path.join(save_results_data,"train/img")
            # sitk.WriteImage(image_nii, os.path.join(train_img_path, "UNet_" + str(name[0]) + '.nii.gz'))
            #
            # label_nii = sitk.GetImageFromArray(label)
            # train_label_path = os.path.join(save_results_data, "train/label")
            # sitk.WriteImage(label_nii, os.path.join(train_label_path, "UNet_" + str(name[0]) + '.nii.gz'))


            mask_array = new_uout.astype(np.float32)
            # print("mask_array.dtype:", mask_array.dtype)
            # mask_nii = sitk.GetImageFromArray(mask_array)
            # train_mask_path = os.path.join(save_results_data,"train/mask")
            # sitk.WriteImage(mask_nii, os.path.join(train_mask_path, "UNet_" + str(name[0]) + '.nii.gz'))


            train_loss += loss_1.item()


        print('index in train-data, and the length of train-data:', number_tra)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        number = 0
        # im_list, uout_list, map_list, label_list = [], [], [], []
        # name_zero = ["0"]
        if valid_data is not None:
            valid_loss = 0
            val_acc_1 = 0
            val_acc_2 = 0
            val_acc_3 = 0

            with torch.no_grad():
                net = net.eval()
                for batch in valid_data:
                    img_sequence_1, label_sequence, name, size, liver_sequence = [], [], [], [], []
                    for i in range(len(batch)):
                        img_sequence_1.extend([batch[i][0]])

                        label_sequence.extend([batch[i][1]])
                        name.extend([batch[i][2]])
                        size.extend([batch[i][3]])
                        liver_sequence.extend([batch[i][4]])

                    img_sequence_1, label_sequence, liver_sequence = np.array(torch.stack(img_sequence_1)), np.array(torch.stack(label_sequence)), np.array(torch.stack(liver_sequence))
                    # print("max(img_sequence_1):", img_sequence_1.shape, img_sequence_1.max(), img_sequence_1.min())
                    #label_sequence[label_sequence > 1] = 1
                    img_sequence_1[img_sequence_1 > 1], img_sequence_1[img_sequence_1 < 0] = 1, 0


                    img_sequence_1 = torch.Tensor(img_sequence_1)
                    label_sequence = torch.Tensor(label_sequence)
                    liver_sequence = torch.Tensor(liver_sequence)
                    """得到tensor形式的数据"""
                    im_1, label, name, size, liver = img_sequence_1, label_sequence, name, size, liver_sequence[:, 0, :, :]
                    number = number + 1
                    im_1, label, liver = im_1.cuda(), label.cuda(), liver.cuda()
                    im = im_1


                    uout, uout_l, landmark_5, landmark_4, landmark_3 = net(im_1)




                    new_uout = np.zeros((label.shape[0], 1, 1024, 1024))
                    label = label.cpu().numpy()
                    uout = uout.cpu()
                    new_uout[uout[:, 1:2, :, :] > 0.5] = 1
                    new_uout[uout[:, 2:3, :, :] > 0.5] = 2
                    new_uout[uout[:, 3:4, :, :] > 0.5] = 3

                    new_uout_1 = np.zeros(new_uout.shape)
                    new_uout_1[new_uout == 1] = 1
                    label_1 = label[:, 0, :, :]

                    new_uout_2 = np.zeros(new_uout.shape)
                    new_uout_2[new_uout == 2] = 1
                    label_2 = label[:, 1, :, :]

                    new_uout_3 = np.zeros(new_uout.shape)
                    new_uout_3[new_uout == 3] = 1
                    label_3 = label[:, 2, :, :]

                    val_dice_1 = get_batch_acc(new_uout_1, label_1)
                    val_dice_2 = get_batch_acc(new_uout_2, label_2)
                    val_dice_3 = get_batch_acc(new_uout_3, label_3)


                    val_acc_1 = val_acc_1 + val_dice_1
                    val_acc_2 = val_acc_2 + val_dice_2
                    val_acc_3 = val_acc_3 + val_dice_3

                    #print("val---当前%s 的dice1值为:%f,dice2值为:%f,dice3值为:%f"%(name[0],val_dice_1,val_dice_2,val_dice_3))
                    val_case_n = val_case_n + 1

                    im = im[:, 0, :, :]
                    label = label[:, 0, :, :]
                    new_uout = new_uout[:, 0, :, :]

            val_acc = (val_acc_1+val_acc_2+val_acc_3)/3
            epoch_str = (
                        "Epoch %d. Train Loss: %f, Train dice 1: %f, Valid dice 1: %f, Train dice 1: %f, Valid dice 2: %f, Train dice 3: %f, Valid dice 3: %f, Valid avg dice:%f,len(valid_data): %d"
                        % (epoch, train_loss / len(train_data),
                    dice_1_acc / train_case_n,
                    val_acc_1 / val_case_n,
                    dice_2_acc / train_case_n,
                    val_acc_2 / val_case_n,
                    dice_3_acc / train_case_n,
                    val_acc_3 / val_case_n,
                    val_acc / val_case_n,
                    val_case_n))

            # print('dice list, and conf list:', dice_list, conf_list)
            sys.stdout.flush()
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          dice_1_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        '''
        保存最终的模型：
            torch.save(net.state_dict(), os.path.join(save, 'model_half.dat'))
        '''
        # Determine if model is the best
        if (val_acc / val_case_n) > best_acc:
            best_acc = (val_acc / val_case_n)
            save_path = os.path.join(save,"best_acc/")
            torch.save(net.state_dict(), os.path.join(save_path,'SAM_twodecoder_resaall.dat'))
            torch.save(net, os.path.join(save_path, 'SAM_twodecoder_resaall.pth'))

        best_str = (
                " Best dice 1: %f, Best dice 2: %f, Best dice 3: %f, Best dice: %f"
                % (best_acc_1,best_acc_2,best_acc_3,best_acc))
        print(best_str)

        if epoch > 100 and epoch % 100 == 0:
            now_acc = (val_acc / val_case_n)
            save_path_epoch = os.path.join(save, "epoch/")
            torch.save(net.state_dict(), os.path.join(save_path_epoch, str(now_acc.item())[0:6] + '_' + str(epoch) + '_SAM_twodecoder_resaall.dat'))
            torch.save(net, os.path.join(save_path_epoch, str(now_acc.item())[0:6] + '_' + str(epoch) + '_SAM_twodecoder_resaall.pth'))







