import sys
sys.path.append('..')
from numpy import random
import numpy as np
import torch
import seg_transforms_twodecoder
import os
from data_load_twodecoder import custom_dataset
from torch.utils.data import DataLoader
# from Unet2D import UNet2D
from SAM_twodecoder import SAMDecoder
from torchvision import transforms as tfs
from util_unet_train_twodecoder import train,DiceLoss
# import segmentation_models_pytorch as smp
# import pretrainedmodels
def seed_torch(seed=3047):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
seed_torch()








def collate_fn(batch):

    return batch

img_height, img_width = 1024, 1024
data_transforms = {
        'train': seg_transforms_twodecoder.Compose([seg_transforms_twodecoder.ConvertImgFloat(),
                                         seg_transforms_twodecoder.PhotometricDistort(),
                                         seg_transforms_twodecoder.Expand(),
                                         # seg_transforms_twodecoder.RandomSampleCrop(),
                                         seg_transforms_twodecoder.RandomMirror_w(),
                                         seg_transforms_twodecoder.RandomMirror_h(),
                                         seg_transforms_twodecoder.Resize(img_height, img_width),
                                         seg_transforms_twodecoder.ToTensor()]),

        'val': seg_transforms_twodecoder.Compose([seg_transforms_twodecoder.ConvertImgFloat(),
                                       seg_transforms_twodecoder.Resize(img_height, img_width),
                                       seg_transforms_twodecoder.ToTensor()])
    }







test_dataset = custom_dataset('/home/zxk/code/D2GPLand-2/D2PGL_DATA/Val', transform=data_transforms['val'])
test_data = DataLoader(test_dataset, 1, shuffle=False, collate_fn=collate_fn)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
train_dataset_100 = custom_dataset('/home/zxk/code/D2GPLand-2/D2PGL_DATA/Train', transform=data_transforms['train'])
train_data_100 = DataLoader(train_dataset_100, 1, shuffle=True, collate_fn=collate_fn)
net = D2GPLand()
optimize = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = DiceLoss()

print("---- 开始训练：")
train(net, train_data_100, test_data, 1000, optimize, criterion)