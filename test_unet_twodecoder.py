import sys
sys.path.append('..')

import numpy as np
import torch
import seg_transforms_twodecoder
from skimage.measure import label as la
from data_load_twodecoder import custom_dataset
from torch.utils.data import DataLoader
# from Unet2D import UNet2D
# from models.D2GPLand_twodecoder import D2GPLand
from SAM_twodecoder import SAMTDecoder
from torchvision import transforms as tfs
from util_unet_test_twodecoder import test



def collate_fn(batch):

    return batch


img_height, img_width = 1024, 1024
data_transforms = {
        'train': seg_transforms_twodecoder.Compose([seg_transforms_twodecoder.ConvertImgFloat(),
                                         seg_transforms_twodecoder.PhotometricDistort(),
                                         seg_transforms_twodecoder.Expand(),
                                         # seg_transforms_twodecoder.RandomSampleCrop(),
                                         # seg_transforms_twodecoder.RandomMirror_w(),
                                         # seg_transforms_twodecoder.RandomMirror_h(),
                                         seg_transforms_twodecoder.Resize(img_height, img_width),
                                         seg_transforms_twodecoder.ToTensor()]),

        'val': seg_transforms_twodecoder.Compose([seg_transforms_twodecoder.ConvertImgFloat(),
                                       seg_transforms_twodecoder.Resize(img_height, img_width),
                                       seg_transforms_twodecoder.ToTensor()])
        # 'val': seg_transforms.Compose([seg_transforms.ToTensor_newdata()])
    }




test_dataset = custom_dataset('./D2PGL_DATA/Test_P2ILF', transform=data_transforms['val']) # 读入 .pkl 文件
test_data = DataLoader(test_dataset, 1, shuffle=False, collate_fn=collate_fn) # batch size 设置为 8


net = torch.load('./save_model/best_acc/SAM_twodecoder_60deep.pth')

# net = UNet2D()

# optimize = torch.optim.SGD(net.parameters(), lr=0.01)
# optimizer_1 = torch.optim.Adam(net_1.parameters(),lr=0.001,betas=(0.9,0.999))
# optimizer_2 = torch.optim.Adam(net_2.parameters(),lr=0.001,betas=(0.9,0.999))
# optimizer_1 = torch.optim.SGD(net_1.parameters(), lr=0.001)
# optimizer_2 = torch.optim.SGD(net_2.parameters(), lr=0.001)


print("---- 开始训练：")
test(net, test_data)
