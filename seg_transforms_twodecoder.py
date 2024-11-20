import numpy as np
import random
import cv2
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels=None, livers=None):
        for t in self.transforms:
            # print("-----img.shape, labels.shape:", img.shape, labels.shape, t)
            img, labels, livers = t(img, labels, livers)
        return img, labels, livers



class ConvertImgFloat(object):
    def __call__(self, img, labels=None, livers = None):
        return img.astype(np.float32), labels, livers

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, labels=None, livers=None):
        if random.randint(0, 2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img, labels, livers


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img, labels=None, livers=None):
        if random.randint(0, 2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img, labels, livers

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, img, labels=None, livers=None):
        if random.randint(0, 2):
            swap = self.perms[random.randint(0, len(self.perms)-1)]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img, labels, livers


class PhotometricDistort(object):
    def __init__(self):
        self.pd = RandomContrast()
        self.rb = RandomBrightness()
        self.rln = RandomLightingNoise()

    def __call__(self, img, labels=None, livers=None):
        img, labels, livers = self.rb(img, labels, livers)

        distort = self.pd
        img, labels, livers = distort(img, labels, livers)
        img, labels, livers = self.rln(img, labels, livers)
        return img, labels, livers


class Expand(object):
    def __init__(self, max_scale = 1.5, mean = (0.485, 0.456, 0.406)):
        self.mean = mean
        self.max_scale = max_scale

    def __call__(self, img, labels=None, livers=None):
        # print("img.shape, labels.shape:", img.shape, labels.shape)
        if random.randint(0, 2):
            return img, labels, livers
        h,w,c = img.shape
        ratio = random.uniform(1,self.max_scale)
        y1 = random.uniform(0, h*ratio-h)
        x1 = random.uniform(0, w*ratio-w)
        expand_img = np.zeros(shape=(int(h*ratio), int(w*ratio),c),dtype=img.dtype)
        expand_img[:,:,:] = self.mean
        expand_img[int(y1):int(y1+h), int(x1):int(x1+w)] = img
        img = expand_img

        num_obj,h,w = labels.shape
        expand_mask = np.zeros(shape=(num_obj,int(h*ratio), int(w*ratio)),dtype=labels.dtype)
        expand_mask[:,:,:] = 0.
        expand_mask[:, int(y1):int(y1+h), int(x1):int(x1+w)] = labels
        masks = expand_mask

        num_obj, h, w = livers.shape
        expand_livers = np.zeros(shape=(num_obj, int(h * ratio), int(w * ratio)), dtype=livers.dtype)
        expand_livers[:, :, :] = 0.
        expand_livers[:, int(y1):int(y1 + h), int(x1):int(x1 + w)] = livers
        livermask = expand_livers

        return img, masks, livermask



# class RandomSampleCrop(object):
#     def __init__(self, ratio=(0.75, 1.25), min_win = 0.9):
#         self.sample_options = (
#             # using entire original input image
#             None,
#             # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
#             # (0.1, None),
#             # (0.3, None),
#             (0.7, None),
#             (0.9, None),
#             # randomly sample a patch
#             (None, None),
#         )
#         self.ratio = ratio
#         self.min_win = min_win
#
#     def __call__(self, img, labels=None):
#         height, width ,_ = img.shape
#         while True:
#             mode = random.choice(self.sample_options)
#             if mode is None:
#                 return img, labels
#
#             for _ in range(50):
#                 current_img = img
#                 current_labels = labels
#                 w = random.uniform(self.min_win*width, width)
#                 h = random.uniform(self.min_win*height, height)
#                 if h/w<self.ratio[0] or h/w>self.ratio[1]:
#                     continue
#                 y1 = random.uniform(0, height-h)
#                 x1 = random.uniform(0, width-w)
#                 rect = np.array([int(y1), int(x1), int(y1+h), int(x1+w)])
#
#                 current_img = current_img[rect[0]:rect[2], rect[1]:rect[3], :]
#                 current_labels = current_labels[:, rect[0]:rect[2], rect[1]:rect[3]]
#
#                 return current_img, current_labels

class RandomMirror_w(object):
    def __call__(self, img, labels, livers):
        _,w,_ = img.shape
        if random.randint(0, 2):
            img = img[:,::-1,:]
            labels = labels[:,:,::-1]
            livers = livers[:, :, ::-1]
        return img, labels, livers


class RandomMirror_h(object):
    def __call__(self, img, labels, livers):
        h,_,_ = img.shape
        if random.randint(0, 2):
            img = img[::-1,:,:]
            labels = labels[:,::-1,:]
            livers = livers[:, ::-1, :]
        return img, labels, livers


class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, labels, livers):
        h,w,c = img.shape

        img = cv2.resize(img, dsize=(self.width, self.height))

        num_obj, h, w = labels.shape
        output_labels = np.zeros((num_obj,self.height, self.width),dtype=labels.dtype)
        for i in range(num_obj):
            output_labels[i] = cv2.resize(labels[i,:,:],
                                         dsize=(self.width, self.height),
                                         interpolation=cv2.INTER_NEAREST)

        num_obj, h, w = livers.shape
        output_livers = np.zeros((num_obj, self.height, self.width), dtype=livers.dtype)
        for i in range(num_obj):
            output_livers[i] = cv2.resize(livers[i, :, :],
                                          dsize=(self.width, self.height),
                                          interpolation=cv2.INTER_NEAREST)

        return img, output_labels, output_livers


class ToTensor(object):
    def __call__(self, img, labels, livers):
        if isinstance(img, np.ndarray):
            img = torch.Tensor(img.copy().transpose((2,0,1)))

        if isinstance(labels, np.ndarray):
            labels = torch.Tensor(labels.copy())

        if isinstance(livers, np.ndarray):
            livers = torch.Tensor(livers.copy())
        return img, labels, livers


