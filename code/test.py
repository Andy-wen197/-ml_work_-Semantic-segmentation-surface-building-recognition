# 用于验证训练集精度
import csv
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import transforms

from data_aug import test_dataset
from data_prepro import test_dataloader


# 将图片编码成rle格式
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


def input_img(i):
    test_list = pd.read_csv('D:\\00Com_TianChi\\dataset\\test_a_samplesubmit.csv',
                            sep='\t', names=['name', 'mask'])
    img = Image.open("D:\\00Com_TianChi\\dataset\\test\\img\\" + test_list['name'].iloc[i])
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = transform(img)
    return img, test_list


def text_model(model):
 # 训练好的模型
    with torch.no_grad():
        with open("output_03.csv", "w", newline='') as csvfile:
            filenames = ['label', 'mask']
            writer = csv.DictWriter(csvfile, fieldnames=filenames)
            writer.writeheader()
            # for index, imgs in enumerate(test_dataloader):
            for index in range(2500):
                print("正在计算第{}/2500个测试集图像".format(index))
                imgs, test_list = input_img(index)
                imgs = imgs.reshape(1, 3, 512, 512)
                # 加载模型
                model = torch.load("FCN_model.pth", map_location='cpu')
                output = model(imgs.to(torch.float32))  # 输出值
                output = torch.sigmoid(output)  # 归一化后的概率

                output_np = output.cpu().detach().numpy().copy()
                output_np = np.argmax(output_np, axis=1)
                output_np = output_np.astype(int)
                output_np = output_np.astype(np.uint8)
                output_np = output_np.reshape(512, 512)
                print(type(output_np))
                print(output_np.shape)
                #cv2.imwrite('D:\\00Com_TianChi\\dataset\\test\\output\\' + test_dataset.image_list[index], output_np)
                rle = rle_encode(output_np)
                writer.writerow({'label': test_list['name'].iloc[index], 'mask': rle})


if __name__ == "__main__":
    test_model = torch.load("FCN_model.pth", map_location='cpu')
    text_model(test_model)




