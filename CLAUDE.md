# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Time Series Library (TSLib) 是清华大学开源的深度时间序列分析库，覆盖 6 大任务：长期预测、短期预测、缺失值填补、异常检测、分类、零样本预测。包含 40+ 模型实现，统一入口 `run.py`。

## 核心命令

```bash
# 训练（示例：TimesNet 长期预测 ETTh1）
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model TimesNet \
  --model_id ETTh1_96_96 \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --e_layers 2 --d_layers 1 --d_model 16 --d_ff 32 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --top_k 5 --itr 1

# 测试（is_training=0，自动加载 checkpoints 中的最佳模型）
python -u run.py --task_name long_term_forecast --is_training 0 ...

# 批量运行脚本
bash scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
```

## 架构

### 入口与任务分发

`run.py` → 根据 `--task_name` 分发到对应实验类：

```
task_name → Exp 类（位于 exp/）
───────────────────────────────────────────
long_term_forecast    → exp/exp_long_term_forecasting.py
short_term_forecast   → exp/exp_short_term_forecasting.py
imputation            → exp/exp_imputation.py
anomaly_detection     → exp/exp_anomaly_detection.py
classification        → exp/exp_classification.py
zero_shot_forecast    → exp/exp_zero_shot_forecasting.py
```

所有 Exp 类继承 `exp/exp_basic.py → Exp_Basic`。

### 模型注册机制

**约定大于配置**：将 `.py` 文件放入 `models/` 目录即可自动注册。

- `Exp_Basic.__init__()` 调用 `_scan_models_directory()` 扫描 `models/` 下所有 `.py` 文件
- 建立 `{模型名: "models.模型名"}` 映射存入 `LazyModelDict`
- 访问 `model_dict['TimesNet']` 时才动态 `importlib.import_module`，首次加载后缓存
- 模型模块必须导出 `class Model(nn.Module)` 或与文件名同名的类

### 数据加载

`data_provider/data_factory.py → data_provider(args, flag)`：
- 维护 `data_dict` 映射：数据集名 → Dataset 类
- 根据 `args.task_name` 分支处理（anomaly_detection / classification / 其他）
- 返回 `(data_set, data_loader)` 元组

### 模型通用接口

所有模型的 `Model.forward()` 统一签名：

```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
```

根据 `self.task_name` 内部分发到 `forecast()` / `imputation()` / `anomaly_detection()` / `classification()`。输入输出约定：
- **x_enc**: `[B, seq_len, enc_in]`，编码器输入
- **x_mark_enc**: `[B, seq_len, time_features]`，编码器时间特征
- **x_dec**: `[B, label_len + pred_len, dec_in]`，解码器输入（forecast 任务）
- **x_mark_dec**: `[B, label_len + pred_len, time_features]`，解码器时间特征
- **mask**: `[B, seq_len, 1]`，imputation 任务的观测掩码

### 目录结构

```
run.py                  ← 统一入口
exp/                    ← 实验流水线（继承 Exp_Basic）
data_provider/          ← 数据集加载器和工厂
models/                 ← 模型实现（每个文件一个模型，自动扫描注册）
layers/                 ← 可复用网络层（注意力、编码器-解码器、嵌入、卷积）
utils/                  ← 工具：metrics.py（评估指标）、tools.py（早停/训练辅助）、masking.py
scripts/                ← Shell 脚本，按 任务类型/数据集/模型 组织
dataset/                ← 数据文件存放目录
checkpoints/            ← 训练检查点
docs/models/            ← 算法结构图（Mermaid）
```

### 关键层文件（layers/）

| 文件 | 职责 | 包含类/函数 |
|------|------|-------------|
| `Embed.py` | 数据嵌入全家桶 | `TokenEmbedding`（Conv1d 数值嵌入）、`PositionalEmbedding`（sin/cos 位置编码）、`FixedEmbedding`（固定正弦嵌入）、`TemporalEmbedding`（月/日/周/时/分特征嵌入）、`TimeFeatureEmbedding`（连续时间特征线性映射）、`DataEmbedding`（Token+Temporal+Position 三路叠加）、`DataEmbedding_wo_pos`（无位置编码，Autoformer 用）、`DataEmbedding_inverted`（倒置嵌入，iTransformer 用）、`PatchEmbedding`（滑动窗口分段，PatchTST 用） |
| `SelfAttention_Family.py` | 注意力机制全家桶 | `FullAttention`（标准点积 O(L²)）、`ProbAttention`（ProbSparse O(L log L)，Informer 用）、`DSAttention`（De-stationary 注意力，Nonstationary Transformer 用）、`AttentionLayer`（通用 Q/K/V 投影+多头包装）、`ReformerLayer`（LSH 注意力包装）、`TwoStageAttentionLayer`（段级+跨维度两阶段注意力，Crossformer 用） |
| `Transformer_EncDec.py` | 标准 Transformer 编解码器 | `ConvLayer`（蒸馏下采样：Conv1d+BN+ELU+MaxPool，Informer 用）、`EncoderLayer`（自注意力+FFN+LayerNorm）、`Encoder`（多层堆叠+可选蒸馏）、`DecoderLayer`（掩码自注意力+交叉注意力+FFN）、`Decoder`（多层堆叠+投影） |
| `Autoformer_EncDec.py` | Autoformer 渐进式分解编解码器 | `moving_avg`（AvgPool1d 移动平均）、`series_decomp`（趋势-季节分解）、`series_decomp_multi`（多尺度分解）、`my_Layernorm`（季节性专用归一化：LayerNorm 减时间维均值）、`EncoderLayer`（注意力+两次 decomp+FFN，舍弃 trend）、`DecoderLayer`（自注意力+交叉注意力+三次 decomp，累积 trend）、`Encoder`/`Decoder` |
| `AutoCorrelation.py` | Auto-Correlation 时延聚合机制 | `AutoCorrelation`（频域互相关找周期 → top-k 时延聚合，O(L log L)；训练用 `torch.roll`，推理用 `torch.gather`）、`AutoCorrelationLayer`（Q/K/V 投影+多头包装） |
| `FourierCorrelation.py` | 频域注意力（FEDformer Fourier 版） | `get_frequency_modes`（频率选择：random/low）、`FourierBlock`（rfft→选 modes→复数乘法→irfft，自注意力）、`FourierCrossAttention`（频域交叉注意力，Q/K 频率索引可不同） |
| `MultiWaveletCorrelation.py` | 小波域注意力（FEDformer Wavelets 版） | `get_filter`（构造 Legendre/Chebyshev 小波滤波器 H0/H1/G0/G1）、`MultiWaveletTransform`（MWT_CZ1d 小波分解-FFT 变换-重构，自注意力）、`MultiWaveletCross`（小波域交叉注意力：4 个 FourierCrossAttentionW 分别处理 detail/smooth 系数）、`sparseKernelFT1d`（FFT 稀疏核线性变换）、`MWT_CZ1d`（单个小波变换块：分解→A/B/C 稀疏核→重构） |
| `Conv_Blocks.py` | 多尺度卷积块 | `Inception_Block_V1`（多分支方形卷积核 1/3/5/7/... 并行+均值融合，TimesNet 用）、`Inception_Block_V2`（非对称卷积核 [1,k]+[k,1]+1×1 并行） |
| `Crossformer_EncDec.py` | Crossformer 金字塔编解码器 | `SegMerging`（相邻段合并降采样，Linear+LayerNorm）、`scale_block`（金字塔编码块：可选 SegMerging + 多层 TwoStageAttentionLayer）、`Encoder`（多层 scale_block，保存各层输出用于跳跃连接）、`DecoderLayer`/`Decoder`（金字塔解码：跨尺度交叉注意力） |
| `ETSformer_EncDec.py` | ETSformer 指数平滑编解码器 | `Transform`（jitter+shift+scale 数据增强，sigma=0.2）、`conv1d_fft`（FFT 快速卷积）、`ExponentialSmoothing`（可学习指数平滑权重，FFT 实现）、`GrowthLayer`（多头增长趋势建模）、`SeasonalLayer`（多头季节性建模，含傅里叶模式选择）、`ETSformerEncoderLayer`/`ETSformerEncoder`/`ETSformerDecoderLayer`/`ETSformerDecoder` |
| `Pyraformer_EncDec.py` | Pyraformer 金字塔注意力编解码器 | `get_mask`（金字塔注意力掩码：intra-scale 窗口 + inter-scale 跨层）、`refer_points`（金字塔到原始分辨率的索引映射）、`RegularMask`（掩码包装）、`EncoderLayer`/`Encoder`（金字塔编码，window_size 控制逐层下采样倍率）、`Decoder`（Conv1d 线性解码） |
| `StandardNorm.py` | 可逆实例归一化（RevIN） | `Normalize`（支持 norm/denorm 双向模式；可选 affine 可学习参数、subtract_last 模式、non_norm 跳过归一化；PatchTST/iTransformer 等模型使用） |
| `DWT_Decomposition.py` | 离散小波包分解（WPMixer 用） | `DWT1DForward`（前向 DWT，基于 PyWavelets）、`DWT1DInverse`（逆 DWT 重构）、`Decomposition`（完整的 DWT 分解+重构流水线，支持多级分解） |
| `MambaBlock.py` | Mamba 选择性状态空间模块 | `Mamba_TimeVariant`（支持 timevariant dt/B/C 的选择性 SSM，依赖 mamba_ssm 库；in_proj→conv1d→SSM→out_proj；MambaSingleLayer 用） |
| `MSGBlock.py` | MSGNet 多尺度图卷积模块 | `GCN`（图卷积：adj 归一化 + einsum 邻域聚合）、`mask_topk_moe`（MoE 风格 mask 过滤：S/T/ST 三种拓扑 mask + 阈值过滤）、`mask_topk_area`（面积比例过滤）、`Predict`（逐变量独立预测或共享 Linear）、`Attention_Block`（标准自注意力+FFN）、`GraphBlock`（GCN + TopK 激活）、`ScaleGraphBlock`（GraphBlock + Attention_Block + 残差） |
| `TimeFilter_layers.py` | TimeFilter 频域过滤模块 | `GCN`（图卷积）、`mask_topk_moe`/`mask_topk_area`（S/T/ST 三类拓扑 mask 过滤）、`Predict`（seq_len→pred_len 线性投影）、`Attention_Block`（自注意力+FFN）、`GraphBlock`（图卷积+TopK）、`ScaleGraphBlock`（图卷积+注意力+残差聚合）、`TimeFilter_Backbone`（PatchEmbed → 多层 ScaleGraphBlock，alpha 控制过滤强度） |

## 添加新模型

1. 在 `models/` 下创建 `YourModel.py`
2. 定义 `class Model(nn.Module)`，实现 `__init__(self, configs)` 和 `forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)`
3. 在 `__init__` 中根据 `configs.task_name` 分支构建不同输出头
4. 在 `forward` 中根据 `self.task_name` 分发到对应处理函数
5. 添加 `scripts/` 下的运行脚本即可训练，无需修改任何注册代码

## 数据集约定

- `--root_path`: 数据文件根目录（如 `./dataset/ETT-small/`）
- `--data_path`: CSV 文件名（如 `ETTh1.csv`）
- `--features`: `M`（多变量→多变量）、`S`（单变量→单变量）、`MS`（多变量→单变量）
- `--enc_in` / `--dec_in` / `--c_out`: 输入/解码器输入/输出变量数（必须与数据集列数匹配）

## 全部模型一览（41 个）

### Transformer 家族

| 模型 | 文件 | 核心技术 | 支持任务 |
|------|------|----------|----------|
| **Transformer** | `Transformer.py` | 标准 FullAttention，O(L²) 基准实现 | 全部 5 任务 |
| **Informer** | `Informer.py` | ProbSparse 注意力 + ConvLayer 蒸馏，O(L log L) | 全部 5 任务 |
| **Autoformer** | `Autoformer.py` | Auto-Correlation 时延聚合 + series_decomp 趋势-季节分解 | 全部 5 任务 |
| **FEDformer** | `FEDformer.py` | 频域注意力（Fourier/Wavelet 双版本），O(N) | 全部 5 任务 |
| **Nonstationary_Transformer** | `Nonstationary_Transformer.py` | De-stationary 注意力：Projector 学习 tau/delta | 全部 5 任务 |
| **Pyraformer** | `Pyraformer.py` | 金字塔多尺度窗口注意力，O(L) | 全部 5 任务 |
| **Reformer** | `Reformer.py` | LSH 局部敏感哈希注意力，O(L log L) | 全部 5 任务 |
| **Crossformer** | `Crossformer.py` | 两阶段注意力（段级 + 跨维度），TwoStageAttentionLayer | 全部 5 任务 |
| **ETSformer** | `ETSformer.py` | 指数平滑注意力，分别建模 level/growth/season | 全部 5 任务 |
| **iTransformer** | `iTransformer.py` | 反转 Transformer：变量作为 token，时间步作为特征 | 全部 5 任务 |
| **PatchTST** | `PatchTST.py` | Patch 分段 + Channel-Independent Transformer | 全部 5 任务 |
| **TimeXer** | `TimeXer.py` | Patch 编码器 + 全局 token 跨注意力融合外生变量 | long/short forecast |

### MLP / 线性模型

| 模型 | 文件 | 核心技术 | 支持任务 |
|------|------|----------|----------|
| **DLinear** | `DLinear.py` | series_decomp + 两个 Linear，极简无注意力 | 全部 5 任务 |
| **LightTS** | `LightTS.py` | IEBlock（输入/通道/输出投影）+ chunk 处理，极低参数量 | 全部 5 任务 |
| **TiDE** | `TiDE.py` | ResBlock 堆叠的纯 MLP 架构，逐变量独立预测 | long/short forecast, imputation |
| **TSMixer** | `TSMixer.py` | MLP-Mixer：交替时间混合 + 通道混合 ResBlock | long/short forecast |

### 频域 / 小波模型

| 模型 | 文件 | 核心技术 | 支持任务 |
|------|------|----------|----------|
| **FiLM** | `FiLM.py` | HiPPO-LegT + 频谱卷积多尺度频率建模 | 全部 5 任务 |
| **FreTS** | `FreTS.py` | 频域 MLP：FFT 后用复数权重矩阵学习时间/通道维度 | 全部 5 任务 |
| **WPMixer** | `WPMixer.py` | 小波包分解（DWT）+ Token/Embedding 双通道 MLP-Mixer | long/short forecast |
| **TimeFilter** | `TimeFilter.py` | Patch 嵌入 + 频域感知过滤（S/T/ST 三种 mask） | 全部 5 任务 |

### 状态空间模型

| 模型 | 文件 | 核心技术 | 支持任务 |
|------|------|----------|----------|
| **Mamba** | `Mamba.py` | 选择性 SSM，O(L) 线性复杂度，依赖 mamba_ssm 库 | long/short forecast |
| **MambaSimple** | `MambaSimple.py` | Mamba 纯 PyTorch 重实现，无外部依赖 | long/short forecast |
| **MambaSingleLayer** | `MambaSingleLayer.py` | 单层 Mamba + TimeVariant 参数，专注分类 | classification, long/short, imputation |

### RNN / 序列模型

| 模型 | 文件 | 核心技术 | 支持任务 |
|------|------|----------|----------|
| **SegRNN** | `SegRNN.py` | GRU 编码分段序列 + 位置/通道嵌入循环预测 | 全部 5 任务 |

### 卷积 / 多尺度模型

| 模型 | 文件 | 核心技术 | 支持任务 |
|------|------|----------|----------|
| **TimesNet** | `TimesNet.py` | FFT 找周期 + 2D Inception 卷积捕获时序变化 | 全部 5 任务 |
| **MICN** | `MICN.py` | 多尺度等距卷积 + series_decomp | 全部 5 任务 |
| **SCINet** | `SCINet.py` | 奇偶分裂 + 交互调制 + 递归树结构 | 全部 5 任务 |
| **MultiPatchFormer** | `MultiPatchFormer.py` | 4 种 patch 大小 + 共享 MHA + 通道注意力 | long/short forecast |

### 图 / 混合架构

| 模型 | 文件 | 核心技术 | 支持任务 |
|------|------|----------|----------|
| **MSGNet** | `MSGNet.py` | FFT 找周期 + 多尺度图卷积 + 注意力聚合 | 全部 5 任务 |
| **TimeMixer** | `TimeMixer.py` | 多尺度趋势-季节分解 + 底部向上/顶部向下混合 | 全部 5 任务 |
| **Koopa** | `Koopa.py` | Koopman 算子：FourierFilter 分离时变/时不变 + DMD | long forecast |

### 其他专用模型

| 模型 | 文件 | 核心技术 | 支持任务 |
|------|------|----------|----------|
| **PAttn** | `PAttn.py` | Patch 嵌入 + FullAttention，62 行极简实现 | long/short forecast |
| **KANAD** | `KANAD.py` | 周期余弦特征 + 卷积网络异常检测 | anomaly_detection, classification |
| **TemporalFusionTransformer** | `TemporalFusionTransformer.py` | 变量选择 + 门控残差 + 静态/观测/已知输入分类处理 | long/short forecast |

### 预训练基础模型（仅 zero_shot_forecast）

| 模型 | 文件 | 来源 | 特点 |
|------|------|------|------|
| **Chronos** | `Chronos.py` | Amazon chronos-bolt-base | 按通道独立预测再 stack |
| **Chronos2** | `Chronos2.py` | Amazon Chronos-2 | 支持 0.1/0.5/0.9 分位数预测 |
| **Moirai** | `Moirai.py` | Salesforce Moirai-2.0-R-small | 按变量独立 predict 再 stack |
| **Sundial** | `Sundial.py` | thuml/sundial-base-128m | 因果语言模型时序生成，多采样取均值 |
| **TimeMoE** | `TimeMoE.py` | Maple728/TimeMoE-50M | MoE Transformer 自回归生成 |
| **TimesFM** | `TimesFM.py` | Google TimesFM 2.5 | 连续分位数头，逐变量 numpy 推理 |
| **TiRex** | `TiRex.py` | NX-AI/TiRex | 分位数预测，逐变量独立预测 |

### 任务覆盖统计

| 支持任务范围 | 模型 |
|-------------|------|
| 全部 5 任务（long/short/imputation/anomaly/classification） | Autoformer, Crossformer, DLinear, ETSformer, FEDformer, FiLM, FreTS, Informer, iTransformer, LightTS, MICN, MSGNet, Nonstationary_Transformer, PatchTST, Pyraformer, Reformer, SCINet, SegRNN, TimeFilter, TimeMixer, TimesNet, Transformer |
| 仅预测任务 | TimeXer, MultiPatchFormer, PAttn, TSMixer, WPMixer, Koopa, TiDE |
| 仅分类/异常检测 | KANAD |
| 仅零样本预测 | Chronos, Chronos2, Moirai, Sundial, TimeMoE, TimesFM, TiRex |

## 技术栈

Python 3.10+, PyTorch 2.5+, NumPy, Pandas, Scikit-learn, SciPy, sktime, einops, reformer-pytorch, PyWavelets, sympy
