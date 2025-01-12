import sys
sys.path.append('..')
import time
import numpy as np
import os
from datetime import datetime
import torch
from torch import nn
import cv2
import SimpleITK as sitk
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



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

    eps = 1e-7
    uout = torch.Tensor(uout)
    label = torch.Tensor(label)
    iflat = uout.view(-1) .float()
    tflat = label.view(-1).float()
    intersection = (iflat * tflat).sum()
    dice_0 = 2 * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)
    return dice_0

def get_batch_precision(pred, target):
    eps = 1e-7
    uout = torch.Tensor(pred)
    label = torch.Tensor(target)
    iflat = uout.view(-1).float()
    tflat = label.view(-1).float()
    intersection = (iflat * tflat).sum()
    precision_1 = intersection / (iflat.sum() + eps)
    return precision_1

def get_batch_recall(pred, target):
    eps = 1e-7
    uout = torch.Tensor(pred)
    label = torch.Tensor(target)
    iflat = uout.view(-1).float()
    tflat = label.view(-1).float()
    intersection = (iflat * tflat).sum()
    recall_1 = intersection / (tflat.sum() + eps)
    return recall_1

def F1_score(precision_1, precision_2, precision_3, recall_1, recall_2, recall_3):
    eps = 1e-7
    f1_score_1 = (2*precision_1*recall_1) / (precision_1+recall_1+eps)
    f1_score_2 = (2*precision_2*recall_2) / (precision_2+recall_2+eps)
    f1_score_3 = (2*precision_3*recall_3) / (precision_3+recall_3+eps)
    return f1_score_1, f1_score_2, f1_score_3




def test(net, test_data):
    if torch.cuda.is_available():
        net = net.cuda()
        print("使用了 cuda")
    else:
        print("没使用cuda")
    prev_time = datetime.now()
    execution_time = 0 
    save_results_data = "./save_result/sam_twodecoder_p2ilf"
    if 1:

        dice_avg_acc = 0
        dice_1_acc = 0
        dice_2_acc = 0
        dice_3_acc = 0

        dice_liver = 0

        pre_1, pre_2, pre_3, pre_a = 0, 0, 0, 0
        rec_1, rec_2, rec_3, rec_a = 0, 0, 0, 0
        f1_1, f1_2, f1_3, f1_a = 0, 0, 0, 0
        number_tra, number_val = 0, 0
        train_case_n, val_case_n = 0, 0
        with torch.no_grad():
            net = net.eval()

            for batch in test_data:
                img_sequence, label_sequence, name, size, liver_sequence = [], [], [], [], []
                for i in range(len(batch)):

                    img_sequence.extend([batch[i][0]])
                    label_sequence.extend([batch[i][1]])
                    name.extend([batch[i][2]])
                    size.extend([batch[i][3]])
                    liver_sequence.extend([batch[i][4]])

                img_sequence, label_sequence, liver_sequence = np.array(torch.stack(img_sequence)), np.array(torch.stack(label_sequence)), np.array(torch.stack(liver_sequence))
                img_sequence[img_sequence > 1], img_sequence[img_sequence < 0] = 1, 0

                img_sequence = torch.Tensor(img_sequence)
                label_sequence = torch.Tensor(label_sequence)
                liver_sequence = torch.Tensor(liver_sequence)

                """得到tensor形式的数据"""
                im_1, label, name, size, liver = img_sequence, label_sequence, name, size, liver_sequence[:, 0, :, :]
                number_tra = number_tra + 1



                im_1, label = im_1.cuda(), label.cuda()#

                # 测量函数执行时间
                start_time = time.time()
                uout, uout_l, landmark_5, landmark_4, landmark_3 = net(im_1)    # 因为术中所有的轮廓都是label，所以先使用全1的label进行训练；
                end_time = time.time()

                execution_time = execution_time + end_time - start_time

                print("uout.shape, uout_l.shape:", uout.shape, uout_l.shape)
                save_uout = uout[0, :, :, :].cpu().numpy()
                save_uout_l = uout_l[0, :, :, :].cpu().numpy()
                save_name = name[0]
                land_name, liver_name = save_name+"_land", save_name+"_liver"
                print("save_name:", save_name)
                save_path_landandliver = "/home/zxk/code/D2GPLand-2/test_results"
                # 转换为 SimpleITK 图像并保存为 nii.gz 文件
                nii_img = sitk.GetImageFromArray(save_uout)
                output_file = os.path.join(save_path_landandliver, f"{land_name}.nii.gz")
                sitk.WriteImage(nii_img, output_file)
                
                nii_img = sitk.GetImageFromArray(save_uout_l)
                output_file = os.path.join(save_path_landandliver, f"{liver_name}.nii.gz")
                sitk.WriteImage(nii_img, output_file)
                
                
                new_uout = np.zeros((label.shape[0], 1, 1024, 1024))
                uout, uout_l = uout.cpu(), uout_l.cpu()


                uout_liver = torch.argmax(uout_l, dim=1).numpy()
                liver = liver.cpu().numpy()
                # print("uout_liver.shape, liver.shape:", uout_liver.shape, liver.shape)
                dice_l = get_batch_acc(uout_liver, liver)

                new_uout[uout[:,1:2,:,:]>0.5] = 1
                new_uout[uout[:,2:3,:,:]>0.5] = 2
                new_uout[uout[:,3:4,:,:]>0.5] = 3

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


                # print("new_uout_1.shape, label_1.shape:", new_uout_1.shape, label_1.shape)
                dice_1 = get_batch_acc(new_uout_1, label_1)
                dice_2 = get_batch_acc(new_uout_2, label_2)
                dice_3 = get_batch_acc(new_uout_3, label_3)
                precision_1 = get_batch_precision(new_uout_1, label_1)
                precision_2 = get_batch_precision(new_uout_2, label_2)
                precision_3 = get_batch_precision(new_uout_3, label_3)
                recall_1 = get_batch_recall(new_uout_1, label_1)
                recall_2 = get_batch_recall(new_uout_2, label_2)
                recall_3 = get_batch_recall(new_uout_3, label_3)
                F1_score_1, F1_score_2, F1_score_3 = F1_score(precision_1, precision_2, precision_3, recall_1, recall_2, recall_3)

                pre_1, pre_2, pre_3 = pre_1 + precision_1, pre_2 + precision_2, pre_3 + precision_3
                pre_a = pre_a + (precision_1+precision_2+precision_3)/3

                rec_1, rec_2, rec_3 = rec_1 + recall_1, rec_2 + recall_2, rec_3 + recall_3
                rec_a = rec_a + (recall_1+recall_2+recall_3) / 3

                f1_1, f1_2, f1_3 = f1_1 + F1_score_1, f1_2 + F1_score_2, f1_3 + F1_score_3
                f1_a = f1_a + (F1_score_1+F1_score_2+F1_score_3) / 3

                dice_1_acc = dice_1_acc + dice_1
                dice_2_acc = dice_2_acc + dice_2
                dice_3_acc = dice_3_acc + dice_3
                dice_avg_acc = dice_avg_acc + (dice_1 + dice_2 + dice_3)/3

                dice_liver = dice_liver + dice_l





                train_case_n = train_case_n + 1

                # print("name:", name)
                # print("size:", size)
                # print("im.shape, label.shape, new_uout.shape:", im.shape, label.shape, new_uout.shape)

                new_uout = new_uout[:, 0, :, :]

                for na in range(len(name)):
                    mask_name = name[na]
                    mask_size = size[na]
                    mask_array = new_uout[na].astype(np.float32)
                    # print("mask_name, mask_size, mask_array.shape:", mask_name, mask_size, mask_array.shape)

                    # 创建一个空的RGB通道图像，大小与mask_array一致
                    mask_rgb = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)

                    # 将值1映射为红色通道
                    mask_rgb[mask_array == 1] = [255, 0, 0]  # 红色
                    # 将值2映射为绿色通道
                    mask_rgb[mask_array == 2] = [0, 255, 0]  # 绿色
                    # 将值3映射为蓝色通道
                    mask_rgb[mask_array == 3] = [0, 0, 255]  # 蓝色

                    # 调整mask大小到原始图像大小，使用cv2.resize
                    resized_mask_rgb = cv2.resize(mask_rgb, (mask_size[1], mask_size[0]), interpolation=cv2.INTER_NEAREST)

                    # 保存为PNG图片
                    save_path = save_results_data + "/" + "label" + "/" + mask_name + ".png"
                    cv2.imwrite(save_path, resized_mask_rgb)


                    """保存肝脏的分割预测"""
                    mask_array = uout_liver[na].astype(np.float32)
                    # print("mask_name, mask_size, mask_array.shape:", mask_name, mask_size, mask_array.shape)

                    # 创建一个空的RGB通道图像，大小与mask_array一致
                    mask_rgb = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)

                    # 将肝脏的值映射为白色的前景
                    mask_rgb[mask_array == 1] = [255, 255, 255]

                    # 调整mask大小到原始图像大小，使用cv2.resize
                    resized_mask_rgb = cv2.resize(mask_rgb, (mask_size[1], mask_size[0]),
                                                  interpolation=cv2.INTER_NEAREST)

                    # 保存为PNG图片
                    save_path = save_results_data + "/" + "liver" + "/" + mask_name + ".png"
                    cv2.imwrite(save_path, resized_mask_rgb)


            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            epoch_str = ("dice 1: %f, dice 2: %f, dice 3: %f, avg dice:%f, liver dice:%f, len(data): %d"
                            % (dice_1_acc / train_case_n,
                        dice_2_acc / train_case_n,
                        dice_3_acc / train_case_n,
                        dice_avg_acc / train_case_n,
                        dice_liver / train_case_n,
                        train_case_n))

            prev_time = cur_time
            print(epoch_str + time_str)

            print("pre_1, pre_2, pre_3, pre_a:", pre_1/ train_case_n, pre_2/ train_case_n, pre_3/ train_case_n, pre_a/ train_case_n)
            print("rec_1, rec_2, rec_3, rec_a:", rec_1/ train_case_n, rec_2/ train_case_n, rec_3/ train_case_n, rec_a/ train_case_n)
            print("f1_1, f1_2, f1_3, f1_a:", f1_1/ train_case_n, f1_2/ train_case_n, f1_3/ train_case_n, f1_a/ train_case_n)
            print("execution_time/109:", execution_time/109)
