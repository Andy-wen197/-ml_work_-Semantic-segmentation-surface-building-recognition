# this test use for data amplification dataset and load
import cv2
import pandas as pd
import torch
from cv2 import imread
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import albumentations as aug   # 数据扩增
import scipy.misc


# ---------------数据集的定义----------------
class MyData(Dataset):
    def __init__(self, root_dir, image_dir, label_dir, transform):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        img = Image.open(img_item_path)
        img = self.transform(img)
        if len(self.label_list) > 0:
            label_name = self.label_list[idx]
            label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
            label = Image.open(label_item_path)
            label = np.asarray(label)
            label = torch.tensor(label)
            label = label / 255
            return img, label
        else:
            return img
        # label = self.label_dir
        # trans_tensor = transforms.ToTensor()
        # img = trans_tensor(img)  # 将图片变为tensor格式
        # label = trans_tensor(label)
        # img = np.asarray(img)
        # img = np.transpose(img, (2, 0, 1))
        # img = torch.tensor(img)

        # with open(label_item_path, 'r') as f:
        #     label = f.readline()
        #
        # # img = np.array(img)
        # img = self.transform(img)
        # sample = {'img': img, 'label': label}
        # return sample

    def __len__(self):
        #assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)


# if __name__ == '__main__':
# 定义训练集
transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_root_dir = "D:/00Com_TianChi/dataset/train/"
train_image_build = "build_image"
train_label_build = "build_label"
build_dataset = MyData(train_root_dir, train_image_build, train_label_build, transform=transform)
# 定义验证集
ver_root_dir = "D:/00Com_TianChi/dataset/ver/"
ver_image_build = "build_img"
ver_label_build = "build_lab"
ver_dataset = MyData(ver_root_dir, ver_image_build, ver_label_build, transform=transform)

# 定义测试集
test_dir = "D:/00Com_TianChi/dataset/test/"
image_build_test = "img"
label_build_test = "label"
test_dataset = MyData(test_dir, image_build_test, label_build_test, transform=transform)

#
# if __name__ == '__main__':
#     # ---------------数据扩增部分---------------
#     aug_data = 'D:\\00Com_TianChi\\dataset\\train_aug\\'
#     image_build_aug = "build_image_aug"
#     label_build_aug = "build_label_aug"
#     # 扩增img和扩增label的路径
#     image_build_aug_path = os.path.join(aug_data, image_build_aug)
#     label_build_aug_path = os.path.join(aug_data, label_build_aug)
#
#         # 原始图像的名称 build_dataset.image_list[0] build_dataset.label_list[0]
#
#     # 路径测试
#     # print(os.path.join(root_dir, image_build,  build_dataset.image_list[0]))
#
#     # print( os.path.join(image_build_aug_path,  'scale' + build_dataset.image_list[0]))
#
#     for i in range(0, 5):
#         print(i)
#         # 将 原始图像和原始标签路径 放入函数 得到路径
#         img_path = os.path.join(root_dir, image_build,  build_dataset.image_list[i])
#         label_path = os.path.join(root_dir, label_build, build_dataset.label_list[i])
#         # 根据路径加载图片 转为np类
#         trans_img = np.asarray(Image.open(img_path))
#         trans_label = np.asarray(Image.open(label_path))
#
#
#         # 水平翻转操作
#         augments = aug.HorizontalFlip(p=1)(image=trans_img, mask=trans_label)
#         img_aug_hor, mask_aug_hor = augments['image'], augments['mask']
#         # 随即裁剪操作
#         augments = aug.RandomCrop(p=1, height=256, width=256)(image=trans_img, mask=trans_label)
#         img_aug_ran, mask_aug_ran = augments['image'], augments['mask']
#         # 旋转操作
#         augments = aug.ShiftScaleRotate(p=1)(image=trans_img, mask=trans_label)
#         img_aug_rot, mask_aug_rot = augments['image'], augments['mask']
#         # 复合操作
#         trfm = aug.Compose([
#             aug.Resize(256, 256),
#             aug.HorizontalFlip(p=0.5),
#             aug.VerticalFlip(p=0.5),
#             aug.RandomRotate90(),
#         ])
#         augments = trfm(image=trans_img, mask=trans_label)
#         img_aug_mix, mask_aug_mix = augments['image'], augments['mask']
#
#
#         # 保存路径 变换后的文件名
#         # 水平翻转
#         save_hor_path_img = os.path.join(image_build_aug_path,  'hor' + build_dataset.image_list[i])
#         save_hor_path_label = os.path.join(label_build_aug_path, 'hor' + build_dataset.label_list[i])
#         cv2.imwrite(save_hor_path_img, img_aug_hor)
#         cv2.imwrite(save_hor_path_label, mask_aug_hor)
#         # 随即裁剪
#         save_ran_path_img = os.path.join(image_build_aug_path, 'ran' + build_dataset.image_list[i])
#         save_ran_path_label = os.path.join(label_build_aug_path, 'ran' + build_dataset.label_list[i])
#         cv2.imwrite(save_ran_path_img, img_aug_ran)
#         cv2.imwrite(save_ran_path_label, mask_aug_ran)
#         # 旋转操作
#         save_rot_path_img = os.path.join(image_build_aug_path, 'rot' + build_dataset.image_list[i])
#         save_rot_path_label = os.path.join(label_build_aug_path, 'rot' + build_dataset.label_list[i])
#         cv2.imwrite(save_rot_path_img, img_aug_rot)
#         cv2.imwrite(save_rot_path_label, mask_aug_rot)
#         # 复合操作
#         save_mix_path_img = os.path.join(image_build_aug_path, 'rot' + build_dataset.image_list[i])
#         save_mix_path_label = os.path.join(label_build_aug_path, 'rot' + build_dataset.label_list[i])
#         cv2.imwrite(save_mix_path_img, img_aug_mix)
#         cv2.imwrite(save_mix_path_label, mask_aug_mix)
# #
#
#
#
#
#
#
#
#
#
#
#
