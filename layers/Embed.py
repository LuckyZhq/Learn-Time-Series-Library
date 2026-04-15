import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """
    位置编码模块
    使用 Transformer 中经典的正弦/余弦位置编码，为序列中的每个时间步注入位置信息。
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # 初始化位置编码矩阵 pe，形状为 [max_len, d_model]
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False  # 位置编码不参与训练

        # position: [max_len, 1]，表示每个位置的下标
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # div_term: 控制不同维度上的正弦/余弦波长
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # 偶数维使用 sin，奇数维使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度，变为 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为 buffer，模型保存时会一起保存，但不会被优化器更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 根据输入序列长度截取对应的位置编码
        # x.size(1) 表示序列长度 T
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    数值特征嵌入模块
    使用 1D 卷积将输入特征映射到 d_model 维空间。
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()

        # 不同版本的 PyTorch 对 circular padding 支持略有差异
        padding = 1 if torch.__version__ >= '1.5.0' else 2

        # 一维卷积：输入通道 c_in，输出通道 d_model
        # kernel_size=3 表示局部时间窗口建模
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',
            bias=False
        )

        # 对卷积层进行 Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # 输入 x: [B, T, C]
        # Conv1d 需要输入为 [B, C, T]，因此先交换维度
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        # 输出恢复为 [B, T, d_model]
        return x


class FixedEmbedding(nn.Module):
    """
    固定嵌入模块
    使用不可训练的正弦/余弦方式构造 embedding，常用于离散时间特征嵌入。
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        # 构造固定嵌入表 [c_in, d_model]
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False  # 固定，不参与训练

        # position 表示离散索引位置
        position = torch.arange(0, c_in).float().unsqueeze(1)

        # 与位置编码相同的频率缩放项
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # 偶数维 sin，奇数维 cos
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # 创建嵌入层，并将权重设置为固定值
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        # 返回固定 embedding，并显式 detach，防止梯度传播
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    时间特征嵌入模块
    对月、日、星期、小时、分钟等离散时间特征分别嵌入后相加。
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        # 各类时间特征的取值范围
        minute_size = 4      # 一般用于 15 分钟粒度，取值 0~3
        hour_size = 24       # 小时 0~23
        weekday_size = 7     # 星期 0~6
        day_size = 32        # 日期 1~31，预留到 32
        month_size = 13      # 月份 1~12，预留到 13

        # 根据 embed_type 选择固定嵌入或可训练嵌入
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        # 当频率为分钟级别时，额外引入 minute embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)

        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        # 输入 x 通常为时间标记，形状 [B, T, C_time]
        # 转成 long 类型以便做 embedding 查表
        x = x.long()

        # x[:, :, 4] -> minute
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.

        # x 的最后一个维度通常依次表示 [month, day, weekday, hour, minute]
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # 将各类时间嵌入相加，得到综合时间表示
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    连续时间特征嵌入模块
    当 embed_type='timeF' 时使用线性层对连续时间特征进行映射。
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # 不同时间频率下输入时间特征维度不同
        freq_map = {
            'h': 4,  # hour
            't': 5,  # minute
            's': 6,  # second
            'm': 1,  # month
            'a': 1,  # annual
            'w': 2,  # week
            'd': 3,  # day
            'b': 3   # business day
        }
        d_inp = freq_map[freq]

        # 线性映射到 d_model 维
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        # 输入 x: [B, T, d_inp]
        # 输出: [B, T, d_model]
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    完整数据嵌入模块
    由三部分组成：
    1. value embedding（数值特征嵌入）
    2. temporal embedding（时间特征嵌入）
    3. positional embedding（位置编码）
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        # 数值特征嵌入
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        # 位置编码
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # 时间特征嵌入
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        )

        # dropout 防止过拟合
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 若没有时间标记，仅使用数值嵌入 + 位置编码
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # 否则叠加数值嵌入、时间嵌入和位置编码
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)

        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """
    倒置式嵌入模块
    将输入从 [B, T, C] 变为 [B, C, T] 后，再用线性层对每个变量进行映射。
    常用于以“变量”为建模对象的结构。
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()

        # 线性层：将长度为 c_in 的向量映射到 d_model
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 输入 x: [B, T, C]
        # 转换为 [B, C, T]
        x = x.permute(0, 2, 1)

        # x: [Batch, Variate, Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # 若有时间标记，则在变量维拼接时间信息
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))

        # 输出: [Batch, Variate, d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """
    不含位置编码的数据嵌入模块
    仅使用数值嵌入和时间嵌入，不加位置编码。
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)  # 虽定义，但 forward 中不使用
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 若没有时间标记，仅进行数值嵌入
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # 数值嵌入 + 时间嵌入
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Patch 嵌入模块
    将时间序列按滑动窗口切分成多个 patch，再映射到 d_model 维空间。
    常用于 PatchTST 等模型。
    """
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()

        # patch 的长度和步长
        self.patch_len = patch_len
        self.stride = stride

        # 在序列右侧进行复制填充，保证最后一个 patch 也能完整取到
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # 将每个 patch（长度 patch_len）映射到 d_model 维
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # patch 级位置编码
        self.position_embedding = PositionalEmbedding(d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入 x 一般形状为 [B, N, T]
        # B: batch_size, N: 变量数, T: 时间长度

        # 记录变量个数
        n_vars = x.shape[1]

        # 对时间维右侧填充
        x = self.padding_patch_layer(x)

        # 按时间维切分 patch
        # unfold 后形状近似为 [B, N, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # 将 batch 维和变量维合并
        # 变为 [B*N, patch_num, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # 对每个 patch 做线性映射并加上位置编码
        x = self.value_embedding(x) + self.position_embedding(x)

        # 返回嵌入结果和变量个数
        return self.dropout(x), n_vars