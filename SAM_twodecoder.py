import math
import warnings
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.resnet import ResNet18, ResNet34, ResNet50
from models.context_modules import get_context_module
from models.model_utils import ConvBNAct, Swish, Hswish, SqueezeAndExcitation
from models.decoder import Decoder
from segment_anything import sam_model_registry


'''
    Basic Block     
'''
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x

    
    
    

# class BasicBlock_land(nn.Module):
#     def __init__(self, in_channels, out_channels, n_groups=8):
#         super(BasicBlock_land, self).__init__()
#
#         # 水平卷积分支
#         self.h_conv = nn.Sequential(
#             nn.GroupNorm(n_groups, in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
#             nn.GroupNorm(n_groups, out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
#             # nn.GroupNorm(n_groups, out_channels),
#             # nn.ReLU(inplace=True),
#             # nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
#         )
#
#         # 垂直卷积分支
#         self.v_conv = nn.Sequential(
#             nn.GroupNorm(n_groups, in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
#             nn.GroupNorm(n_groups, out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
#             # nn.GroupNorm(n_groups, out_channels),
#             # nn.ReLU(inplace=True),
#             # nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
#         )
#
#         # 标准 3×3 卷积分支
#         self.std_conv = nn.Sequential(
#             nn.GroupNorm(n_groups, in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.GroupNorm(n_groups, out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             # nn.GroupNorm(n_groups, out_channels),
#             # nn.ReLU(inplace=True),
#             # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         )
#
#         # 自适应融合模块（轻量级注意力机制）
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # 全局平均池化
#             nn.Conv2d(out_channels * 3, out_channels // 4, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels // 4, out_channels * 3, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#         # 最终的卷积层（用于降维）
#         self.final_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         residul = x
#
#         # 分别通过水平、垂直和标准卷积分支
#         h_feat = self.h_conv(x)
#         v_feat = self.v_conv(x)
#         std_feat = self.std_conv(x)
#
#         # 将特征在通道维度上拼接
#         combined_feat = torch.cat([h_feat, v_feat, std_feat], dim=1)
#
#         # 计算自适应权重
#         attention_weights = self.attention(combined_feat)
#         # 对特征进行加权
#         weighted_feat = combined_feat * attention_weights
#
#         # 通过最终的卷积层（降维）
#         out = self.final_conv(weighted_feat)
#
#         # 残差连接
#         out = out + residul
#         return out



class SAMTDecoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, init_channels=8, p=0.2):

        super(SAMTDecoder, self).__init__()

        sam = sam_model_registry["vit_b"](
            checkpoint="sam_path/sam_vit_b_01ec64.pth")
        self.sam_encoder = sam.image_encoder
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_decoder()
        
    def make_decoder(self):
        init_channels = self.init_channels
        self.up6conva = nn.Conv2d(init_channels * 32, init_channels * 16, 1)
        self.up6 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up6convb = BasicBlock(init_channels * 16, init_channels * 16)

        self.up5conva = nn.Conv2d(init_channels * 16, init_channels * 8, 1)
        self.up5 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up5convb = BasicBlock(init_channels * 8, init_channels * 8)

        self.up4conva = nn.Conv2d(init_channels * 8, init_channels * 4, 1)
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv2d(init_channels * 4, init_channels * 2, 1)
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv2d(init_channels * 2, init_channels, 1)
        # self.up2 = nn.Upsample(scale_factor=2)
        # self.up2convb = BasicBlock(init_channels, init_channels)
        self.up1conv = nn.Conv2d(init_channels, self.out_channels, 1)
        # 用于生成地标 CAM 的卷积层
        self.landmark_cam_5 = nn.Conv2d(init_channels * 8, 4, 1)

        self.landmark_cam_4 = nn.Conv2d(init_channels * 4, 4, 1)
        self.landmark_cam_3 = nn.Conv2d(init_channels * 2, 4, 1)
        
        
        
        
        
        """
        肝脏分支解码器
        """
        self.up6conva_l = nn.Conv2d(init_channels * 32, init_channels * 16, 1)
        self.up6_l = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up6convb_l = BasicBlock(init_channels * 16, init_channels * 16)

        self.up5conva_l = nn.Conv2d(init_channels * 16, init_channels * 8, 1)
        self.up5_l = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up5convb_l = BasicBlock(init_channels * 8, init_channels * 8)

        self.up4conva_l = nn.Conv2d(init_channels * 8, init_channels * 4, 1)
        self.up4_l = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb_l = BasicBlock(init_channels * 4, init_channels * 4)

        
        self.up3conva_l = nn.Conv2d(init_channels * 4, init_channels * 2, 1)
        self.up3_l = nn.Upsample(scale_factor=2)
        self.up3convb_l = BasicBlock(init_channels * 2, init_channels * 2)

        
        
        self.up2conva_l = nn.Conv2d(init_channels * 2, init_channels, 1)
        # self.up2_l = nn.Upsample(scale_factor=2)
        # self.up2convb_l = BasicBlock(init_channels, init_channels)

        self.up1conv_l = nn.Conv2d(init_channels, 2, 1)
        # 用于生成肝脏 CAM 的卷积层
        # self.liver_cam_conv = nn.Conv2d(init_channels * 2, 1, kernel_size=1)

    def forward(self, image):
        image = self.sam_encoder(image)
        

        
        u6 = self.up6conva(image)
        # print("u6.shape:", u6.shape)
        u6 = self.up6(u6)

        # print("u6.shape:", u6.shape)
        u6 = self.up6convb(u6)


        u5 = self.up5conva(u6)
        landmark_5 = self.landmark_cam_5(u5)  
        landmark_5 = torch.softmax(landmark_5,dim=1)

        landmark_5 = F.interpolate(landmark_5, size=image.shape[2]*16, mode='bilinear', align_corners=False)

        u5 = self.up5(u5)

        u5 = self.up5convb(u5)

        u4 = self.up4conva(u5)



        landmark_4 = self.landmark_cam_4(u4)  

        landmark_4 = torch.softmax(landmark_4,dim=1)

        landmark_4 = F.interpolate(landmark_4, size=image.shape[2]*16, mode='bilinear', align_corners=False)
        u4 = self.up4(u4)

        u4 = self.up4convb(u4)
        u3 = self.up3conva(u4)


        landmark_3 = self.landmark_cam_3(u3)  
        landmark_3 = torch.softmax(landmark_3,dim=1)
        landmark_3 = F.interpolate(landmark_3, size=image.shape[2]*16, mode='bilinear', align_corners=False)
        
        u3 = self.up3(u3)
        
        
        # 提取地标 CAM
        # landmark_cam = self.landmark_cam_conv(u3)  # u3 的尺寸较大，适合生成 CAM
        # # print("landmark_cam.shape:", landmark_cam.shape)
        # # # 上采样到输入图像尺寸
        # # landmark_cam = F.interpolate(landmark_cam, size=image.shape[2]*16, mode='bilinear', align_corners=False)
        # landmark_cam = torch.sigmoid(landmark_cam)  # 将值映射到 [0,1]

        u3 = self.up3convb(u3)
        u2 = self.up2conva(u3)
        uout = self.up1conv(u2)
        uout = torch.softmax(uout,dim=1)
        
        
        
        """
        肝脏分支
        """
        # print("c6d.shape:", c6d.shape)
        u6_l = self.up6conva_l(image)
        # print("u6.shape:", u6.shape)
        u6_l = self.up6_l(u6_l)

        # print("u6.shape:", u6.shape)
        u6_l = self.up6convb_l(u6_l)

        u5_l = self.up5conva_l(u6_l)
        u5_l = self.up5_l(u5_l)

        u5_l = self.up5convb_l(u5_l)
        u4_l = self.up4conva_l(u5_l)
        u4_l = self.up4_l(u4_l)

        u4_l = self.up4convb_l(u4_l)
        u3_l = self.up3conva_l(u4_l)
        u3_l = self.up3_l(u3_l)

        u3_l = self.up3convb_l(u3_l)
        
        

        # # 提取肝脏 CAM
        # liver_cam = self.liver_cam_conv(u3_l)  # u3_l 的尺寸较大，适合生成 CAM
        # # 上采样到输入图像尺寸
        # # liver_cam = F.interpolate(liver_cam, size=image.shape[2]*16, mode='bilinear', align_corners=False)
        # liver_cam = torch.sigmoid(liver_cam)  # 将值映射到 [0,1]
        
        u2_l = self.up2conva_l(u3_l)

        uout_l = self.up1conv_l(u2_l)
        uout_l = torch.softmax(uout_l, dim=1)
        # print("landmark_cam.shape, liver_cam.shape:", landmark_cam.shape, liver_cam.shape)
        return uout, uout_l, landmark_5, landmark_4, landmark_3
