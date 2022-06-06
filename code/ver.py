# 测试集
# 从训练集提取部分图片做测试
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
from data_aug import ver_dataset
from data_prepro import ver_dataloader


# 获取验证集数据
def get_data():
    rle = pd.read_csv('D:\\00Com_TianChi\\dataset\\train_mask.csv\\train_mask.csv',
                        sep='\t', names=['name', 'mask'])
    before_img_path = '../dataset/train/build_image/'
    before_lab_path = '../dataset/train/build_label_2/'
    after_img_path = '../dataset/ver/build_img/'
    after_lab_path = '../dataset/ver/build_lab/'
    for index in range(0, 6000, 2):  # 取前6000张图片 间隔1个
        try:
            # 复制Img
            shutil.copy(before_img_path + rle['name'].iloc[index],
                 after_img_path + rle['name'].iloc[index])
            # 复制lab
            shutil.copy(before_lab_path + rle['name'].iloc[index],
                 after_lab_path + rle['name'].iloc[index])
            print("正在复制第{}个图像".format(index))
        except:
            pass
    return 0


# 评价指标
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def dice_coeff(self, pred, target):
        smooth = 1.
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()
        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    pa_sum = 0
    cpa_sum = 0
    mpa_sum = 0
    mIoU_sum = 0
    dice_sum = 0
    with torch.no_grad():
        for index, (imgs, labels) in enumerate(ver_dataloader):
            print("正在计算第{}/{}个验证集准确率".format(index, len(ver_dataloader)))
            # 加载模型
            model = torch.load("FCN_model.pth", map_location='cpu')
            output = model(imgs)  # 输出值
            output = torch.sigmoid(output)  # 归一化后的概率

            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmax(output_np, axis=1)
            labels_np = labels.cpu().detach().numpy().copy().astype(int)
            output_np = output_np.astype(int)

            output_tensor = torch.tensor(output_np)

            metric = SegmentationMetric(2)  # 类的实例化 输入分类的类别数
            metric.addBatch(output_np, labels_np)  # 输出 与 真值放入评价类

            pa = metric.pixelAccuracy()
            cpa = metric.classPixelAccuracy()
            mpa = metric.meanPixelAccuracy()
            mIoU = metric.meanIntersectionOverUnion()
            dice = metric.dice_coeff(output_tensor, labels)

            print("pa:{}".format(pa))
            print("cpa:{}".format(cpa))
            print("mpa:{}".format(mpa))
            print("mIoU:{}".format(mIoU))
            print("dice:{}".format(dice))

            # 计算总数
            pa_sum = pa_sum + pa
            cpa_sum = cpa_sum + cpa
            mpa_sum = mpa_sum + mpa
            mIoU_sum = mIoU_sum + mIoU
            dice_sum = dice_sum + dice

    # 平均值
    mpa = pa_sum / 313
    mcpa = cpa_sum / 313
    mmpa = mpa_sum / 313
    mmIoU = mIoU_sum / 313
    mdice = dice_sum / 313

    print('mpa is : %f' % mpa)
    print('mcpa is :{}'.format(mcpa))  # 列表
    print('mmpa is : %f' % mmpa)
    print('mmIoU is : %f' % mmIoU)
    print('mdice is : %f' % mdice)

