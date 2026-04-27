import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    '''卷积层，用于编码器中的蒸馏操作，对序列长度进行下采样'''

    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        # 1D卷积：降采样序列长度，保留通道数不变
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')  # 循环填充，保持边界信息
        self.norm = nn.BatchNorm1d(c_in)          # 批归一化
        self.activation = nn.ELU()                 # ELU 激活函数
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 最大池化，序列长度减半

    def forward(self, x):
        # 输入 x: [B, L, D]，Permute 为 [B, D, L] 适配 Conv1d
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)  # 恢复为 [B, L', D]
        return x


class EncoderLayer(nn.Module):
    '''编码器层：自注意力 + 前馈网络（两层 1x1 卷积）'''

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention                    # 自注意力模块
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)   # 升维
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)   # 降维
        self.norm1 = nn.LayerNorm(d_model)            # 第一层归一化
        self.norm2 = nn.LayerNorm(d_model)            # 第二层归一化
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # 自注意力子层（残差连接）
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        # 前馈网络子层（残差连接）
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # 升维+激活
        y = self.dropout(self.conv2(y).transpose(-1, 1))                   # 降维回 d_model

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    '''编码器：由多个 EncoderLayer 堆叠，层间可选插入 ConvLayer 进行蒸馏'''

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)          # 注意力层列表
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None  # 卷积层列表
        self.norm = norm_layer                                 # 最终归一化层

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []  # 收集各层的注意力权重
        if self.conv_layers is not None:
            # 带蒸馏的编码器：每一对注意力层后接一个卷积层
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None  # 仅第一层传入 delta
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)                 # 卷积层缩短序列长度
                attns.append(attn)
            # 最后一层注意力层不接卷积
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            # 无蒸馏：依次通过所有注意力层
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    '''解码器层：掩码自注意力 + 交叉注意力 + 前馈网络'''

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention            # 自注意力（带掩码，防止看到未来信息）
        self.cross_attention = cross_attention          # 交叉注意力（Q来自解码器，K/V来自编码器）
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)              # 自注意力后的归一化
        self.norm2 = nn.LayerNorm(d_model)              # 交叉注意力后的归一化
        self.norm3 = nn.LayerNorm(d_model)              # 前馈网络后的归一化
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # 带掩码的自注意力（残差连接）
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # 交叉注意力（残差连接）：解码器查询与编码器输出交互
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        # 前馈网络（残差连接）
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    '''解码器：由多个 DecoderLayer 堆叠，最后可选归一化和线性投影'''

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)      # 解码器层列表
        self.norm = norm_layer                   # 最终归一化
        self.projection = projection             # 最终线性投影到输出维度

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # 依次通过所有解码器层
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x