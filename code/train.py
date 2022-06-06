from model import*
from data_prepro import*
import torch

# 在训练网络前定义函数用于计算Acc 和 mIou


#
# # 根据混淆矩阵计算Acc和mIou
# def label_accuracy_score(label_trues, label_preds, n_class):
#     """Returns accuracy score evaluation result.
#       - overall accuracy
#       - mean accuracy
#       - mean IU
#     """
#     hist = np.zeros((n_class, n_class))
#     for lt, lp in zip(label_trues, label_preds):
#         hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
#     acc = np.diag(hist).sum() / hist.sum()
#     with np.errstate(divide='ignore', invalid='ignore'):
#         acc_cls = np.diag(hist) / hist.sum(axis=1)
#     acc_cls = np.nanmean(acc_cls)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         iu = np.diag(hist) / (
#             hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
#         )
#     mean_iu = np.nanmean(iu)
#     freq = hist.sum(axis=1) / hist.sum()
#     return acc, acc_cls, mean_iu


from datetime import datetime

import torch.optim as optim
import matplotlib.pyplot as plt


def train(epo_num, show_vgg_params=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建VGG模型 根据VGG模型构建FCN8s模型
    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=2)
    fcn_model = fcn_model.to(device)
    # 这里只有两类，采用二分类常用的损失函数BCE
    criterion = nn.CrossEntropyLoss().to(device)
    # 随机梯度下降优化，学习率0.001，惯性分数0.7
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.7)

    # 记录训练过程相关指标
    all_train_iter_loss = []
    all_test_iter_loss = []
    test_Acc = []
    test_mIou = []
    # start timing
    prev_time = datetime.now()

    for epo in range(epo_num):

        # 训练
        train_loss = 0
        fcn_model.train()
        for index, (imgs, labels) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = fcn_model(imgs.to(torch.float32))
            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            print(output.size())
            print(labels.size())
            labels = labels.view(8, 512, 512)  # labels降低1维
            labels = torch.LongTensor(labels.numpy())  # 标签改为长整型
            loss = criterion(output, labels)
            loss.backward()  # 需要计算导数，则调用backward
            iter_loss = loss.item()  # .item()返回一个具体的值，一般用于loss和acc
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmin(output_np, axis=1)
            label_np = labels.cpu().detach().numpy().copy()
            blabel_np = np.argmin(label_np, axis=1)

            # 每15个bacth，输出一次训练过程的数据
            if np.mod(index, 15) == 0:
                print('epoch：{}, 当前批次/总批次：{}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))

    return fcn_model


if __name__ == '__main__':
    model = train(epo_num=7, show_vgg_params=False)
    torch.save(model, "FCN_model.pth")



