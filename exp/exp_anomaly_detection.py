from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

# 设置多进程共享策略为 file_system
# 避免某些平台下 DataLoader 多进程读取数据时出现共享内存问题
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

# 忽略警告信息
warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    """
    异常检测实验类

    继承自 Exp_Basic，主要实现：
    1. 模型构建
    2. 数据加载
    3. 训练过程
    4. 验证过程
    5. 测试与异常检测评估
    """
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        """
        构建模型

        根据 args.model 从 model_dict 中选择对应模型并实例化。
        如果开启多 GPU，则使用 DataParallel 并行训练。
        """
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        获取数据集和数据加载器

        参数:
            flag: str
                数据集标识，一般为 'train' / 'val' / 'test'

        返回:
            data_set: 数据集对象
            data_loader: 数据加载器
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        选择优化器

        这里使用 Adam 优化器。
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        选择损失函数

        异常检测任务中，这里使用均方误差 MSELoss，
        因为模型本质上是通过重构误差来判断异常。
        """
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """
        验证函数

        在验证集上计算平均损失。

        参数:
            vali_data: 验证集
            vali_loader: 验证集加载器
            criterion: 损失函数

        返回:
            total_loss: float
                验证集平均损失
        """
        total_loss = []

        # 切换到评估模式
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                # 输入数据转为 float 并放到设备上
                batch_x = batch_x.float().to(self.device)

                # 前向传播，得到重构结果
                outputs = self.model(batch_x, None, None, None)

                # 根据特征模式选择输出维度
                # MS: 多变量到单变量，取最后一个维度
                # 否则从第 0 维开始取
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # 预测值和真实值
                pred = outputs.detach()
                true = batch_x.detach()

                # 计算重构误差
                loss = criterion(pred, true)
                total_loss.append(loss.item())

        # 计算平均损失
        total_loss = np.average(total_loss)

        # 恢复训练模式
        self.model.train()
        return total_loss

    def train(self, setting):
        """
        模型训练函数

        训练流程：
        1. 加载训练/验证/测试集
        2. 创建保存路径
        3. 迭代训练模型
        4. 在验证集上做 early stopping
        5. 保存并加载最佳模型
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 模型检查点保存路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        # 每个 epoch 的迭代次数
        train_steps = len(train_loader)

        # 提前停止机制
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 优化器与损失函数
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 开始训练多个 epoch
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # 训练模式
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # 输入数据转到设备
                batch_x = batch_x.float().to(self.device)

                # 前向传播，输出重构结果
                outputs = self.model(batch_x, None, None, None)

                # 根据任务特征类型选择维度
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # 计算重构损失
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                # 每 100 次迭代打印一次训练信息
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

                    # 计算平均每步耗时
                    speed = (time.time() - time_now) / iter_count

                    # 估计剩余时间
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

                    iter_count = 0
                    time_now = time.time()

                # 反向传播和参数更新
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # 计算平均训练损失
            train_loss = np.average(train_loss)

            # 在验证集和测试集上评估
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # 提前停止机制：监控验证损失
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 加载验证集上表现最好的模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """
        模型测试函数

        异常检测流程：
        1. 在训练集上统计正常样本的重构误差分布
        2. 在测试集上计算重构误差
        3. 根据设定异常比例确定阈值
        4. 根据阈值得到预测标签
        5. 对检测结果进行调整
        6. 计算并输出评价指标
        """
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        # 如果 test=1，则从保存的检查点加载模型
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []

        # 测试结果保存目录
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 评估模式
        self.model.eval()

        # 不做 reduction，保留每个样本、每个时间步的误差信息
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) 在训练集上统计“正常”重构误差
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)

                # 重构输出
                outputs = self.model(batch_x, None, None, None)

                # 计算每个时间步的重构误差，最后在特征维求平均
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)

                # 转为 numpy 并保存
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        # 拼接训练集所有误差分数
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) 在测试集上计算误差，并记录真实标签
        attens_energy = []
        test_labels = []

        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)

            # 重构输出
            outputs = self.model(batch_x, None, None, None)

            # 计算重构误差
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()

            attens_energy.append(score)
            test_labels.append(batch_y)

        # 拼接测试集误差
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        # 合并训练和测试误差，用于确定异常阈值
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        # 根据 anomaly_ratio 设定阈值
        # 例如 anomaly_ratio=1 表示取 99% 分位点作为阈值
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) 在测试集上根据阈值判断异常
        pred = (test_energy > threshold).astype(int)

        # 获取真实标签
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) 对检测结果做调整
        # 一般用于时间序列异常段修正：
        # 如果某个异常区间被部分命中，则将整个区间视为检测到
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)

        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        # (5) 计算分类指标
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average='binary'
        )

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))

        # (6) 将结果追加写入文件
        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()

        return