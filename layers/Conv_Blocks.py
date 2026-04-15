import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    """
    Inception 卷积块 V1

    使用多个不同大小的方形卷积核并行提取特征，
    再对各分支输出取平均，融合多尺度信息。
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()

        # 输入通道数
        self.in_channels = in_channels
        # 输出通道数
        self.out_channels = out_channels
        # 卷积分支数量
        self.num_kernels = num_kernels

        # 用于存放多个不同卷积核大小的卷积层
        kernels = []

        # 构造多个卷积分支，卷积核大小依次为 1, 3, 5, 7, ...
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * i + 1,  # 卷积核尺寸：1,3,5,...
                    padding=i               # 为保持输出尺寸一致，padding 与 i 对应
                )
            )

        # 将所有卷积分支注册为模块列表
        self.kernels = nn.ModuleList(kernels)

        # 是否进行权重初始化
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化卷积层权重

        使用 Kaiming 初始化，适合 ReLU 类激活函数。
        偏置初始化为 0。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状一般为 [B, C, H, W]

        返回:
            res: 多分支卷积结果取平均后的输出
        """
        res_list = []

        # 每个卷积分支分别提取特征
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))

        # 将多个分支结果在新维度堆叠
        # 形状变为 [B, out_channels, H, W, num_kernels]
        res = torch.stack(res_list, dim=-1).mean(-1)

        # 对最后一个维度取平均，完成多尺度特征融合
        return res


class Inception_Block_V2(nn.Module):
    """
    Inception 卷积块 V2

    与 V1 不同，V2 使用非对称卷积核：
    - [1, k] 横向卷积
    - [k, 1] 纵向卷积
    再额外加一个 1x1 卷积分支，用于融合不同方向与尺度的信息。
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()

        # 输入通道数
        self.in_channels = in_channels
        # 输出通道数
        self.out_channels = out_channels
        # 卷积分支参数
        self.num_kernels = num_kernels

        # 存放所有卷积分支
        kernels = []

        # 构造非对称卷积分支
        # 每轮加入两个分支：
        # 1) [1, 2*i+3] 横向卷积
        # 2) [2*i+3, 1] 纵向卷积
        for i in range(self.num_kernels // 2):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[1, 2 * i + 3],
                    padding=[0, i + 1]
                )
            )
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[2 * i + 3, 1],
                    padding=[i + 1, 0]
                )
            )

        # 再增加一个 1x1 卷积分支
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        # 注册为模块列表
        self.kernels = nn.ModuleList(kernels)

        # 是否初始化权重
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化卷积层权重

        使用 Kaiming 初始化，偏置置零。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状一般为 [B, C, H, W]

        返回:
            res: 所有卷积分支输出的平均结果
        """
        res_list = []

        # 分支总数 = num_kernels // 2 * 2 + 1
        # 前一部分是横向和纵向卷积分支，最后一个是 1x1 卷积
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))

        # 堆叠后对所有分支结果求平均
        res = torch.stack(res_list, dim=-1).mean(-1)

        return res