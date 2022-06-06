# this test use for pre-processing data
import numpy as np

from data_aug import build_dataset, test_dataset, ver_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler


sampler = SequentialSampler(test_dataset)
# 利用DataLoader生成一个分batch获取数据的可迭代对象
train_dataloader = DataLoader(build_dataset, batch_size=8, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, drop_last=True, sampler=sampler)
ver_dataloader = DataLoader(ver_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)


