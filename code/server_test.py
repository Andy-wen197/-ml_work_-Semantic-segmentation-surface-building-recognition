#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# 将图片编码成rle格式
from torchvision.transforms import transforms


def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return ''  ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return ''  ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# 将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')  # Needed to align to RLE direction



import pandas as pd
import cv2

# mask为rle编码
train_rle = pd.read_csv('../dataset/train_mask.csv/train_mask.csv',
                        sep='\t', names=['name', 'mask'])
# 将rle转为mask mask为numpy.ndarray mode为L模式
for i in range(10):
    try:
        print('正在处理第{}个'.format(i))
        train_mask = rle_decode(train_rle['mask'].iloc[i])
        print(train_mask.shape)
        print(type(train_mask))
        # print(type(train_mask))  # 矩阵形式
        train_mask = train_mask * 255
        train_mask = train_mask.astype(np.uint8)
        if os.path.exists('../dataset/train/build_image/' + train_rle['name'].iloc[i]):
            cv2.imwrite('../dataset/train/build_label/' + train_rle['name'].iloc[i], train_mask)
    except:
        # 剔除异常数据
        print("存在异常文件")
        if os.path.exists('../dataset/train/build_image/' + train_rle['name'].iloc[i]):
            os.remove('../dataset/train/build_image/' + train_rle['name'].iloc[i])  # 剔除异常标签对应的图像
            print("异常rle对应的图像已剔除")
        else:
            print("异常rle对应的图像已剔除")