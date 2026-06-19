# TimesNet 算法结构图

> **论文**: [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://openreview.net/pdf?id=ju_Uqw384Oq)
>
> **核心思想**: 将一维时间序列通过 FFT 检测周期，reshape 为二维张量，用 2D 卷积同时捕获周期内局部模式与跨周期全局模式。

---

## 1. 总体架构总览

```mermaid
flowchart TD
    A["输入 x_enc\n[B, T, C]"] --> B["DataEmbedding\nToken+Pos+Temporal"]
    B --> C["predict_linear\n(仅 forecast)"]
    C --> D["TimesBlock × L\ne_layers 层堆叠"]
    D --> E["LayerNorm"]
    E --> F{"task_name?"}
    F -->|forecast| G["Linear → d_model→c_out"]
    F -->|imputation| H["Linear → d_model→c_out"]
    F -->|anomaly_det| I["Linear → d_model→c_out"]
    F -->|classification| J["GELU+Dropout\n→ Flatten → Linear"]
    G --> K["反标准化\n×stdev + means"]
    H --> K
    I --> K
    K --> L["输出 dec_out\n[B, pred_len, C]"]
    J --> M["输出 logits\n[B, num_class]"]

    style A fill:#4A90D9,color:#fff
    style L fill:#52C41A,color:#fff
    style M fill:#52C41A,color:#fff
    style D fill:#FA8C16,color:#fff
    style F fill:#722ED1,color:#fff
```

**说明**: 模型入口在 `Model.__init__()`，根据 `task_name` 走不同的输出头。Forecast / Imputation / Anomaly Detection 三个任务共享「标准化 → 编码 → TimesBlock × L → 线性投影 → 反标准化」的主路径，Classification 则在编码后走 flatten + 线性分类头，不做标准化/反标准化。

---

## 2. TimesBlock 核心算法（单层展开）

```mermaid
flowchart TD
    subgraph TimesBlock["TimesBlock 单层处理流程"]
        X["输入 x\n[B, T, d_model]"] --> FFT["FFT_for_Period\n找 top-k 周期"]
        FFT --> PERIOD["period_list\n周期长度数组"]
        FFT --> WEIGHT["period_weight\n频率幅值权重"]

        PERIOD --> LOOP["对每个周期 i=0..k-1"]
        LOOP --> PAD["padding\n补齐到 period 整数倍"]
        PAD --> RESHAPE["reshape 2D\n[B, N, L//P, P]"]
        RESHAPE --> CONV["Inception Conv2D ×2\n多尺度卷积+GELU"]
        CONV --> BACK["reshape 回 1D\n[B, T, d_model]"]
        BACK --> STACK["堆叠 k 个结果\n[B, T, d_model, k]"]

        STACK --> AGG["自适应加权聚合\nsoftmax(weight) × res"]
        AGG --> RES["残差连接\nres + x"]
        RES --> OUT["输出 [B, T, d_model]"]
    end

    style X fill:#4A90D9,color:#fff
    style FFT fill:#722ED1,color:#fff
    style CONV fill:#FA8C16,color:#fff
    style AGG fill:#EB2F96,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: TimesBlock 是 TimesNet 的核心计算单元。对每个检测到的周期 P，将序列 reshape 成二维 `[B, C, num_periods, P]`（行=周期序号，列=周期内位置），使 2D Inception 卷积能同时建模周期内局部变化和跨周期全局趋势。k 个周期的结果通过 softmax 加权聚合，再加残差连接。

---

## 3. FFT 周期检测算法

```mermaid
flowchart LR
    A["输入 x\n[B, T, C]"] --> B["rfft\n沿时间维"]
    B --> C["abs 取幅值\nmean(C)→频率能量"]
    C --> D["置零 DC\nfrequency[0]=0"]
    D --> E["topk\n取前 k 个"]
    E --> F["period = T // freq_idx"]
    E --> G["weight = 幅值均值\n[B, k]"]

    style A fill:#4A90D9,color:#fff
    style B fill:#722ED1,color:#fff
    style F fill:#FA8C16,color:#fff
    style G fill:#EB2F96,color:#fff
```

**说明**: `torch.fft.rfft` 对时间维做实数 FFT，取各频率幅值的跨变量均值作为能量指标，排除直流分量（index=0）后取 top-k 频率索引，对应周期 `T // freq_idx`。`period_weight` 为各 batch 在 top-k 频率上的幅值均值，送入 TimesBlock 中的 softmax 层进行自适应加权。

---

## 4. Inception 多尺度卷积块

```mermaid
flowchart LR
    A["输入 [B,C,H,W]"] --> K1["Conv2d k=1 p=0"]
    A --> K2["Conv2d k=3 p=1"]
    A --> K3["Conv2d k=5 p=2"]
    A --> K4["Conv2d k=11 p=5"]
    K1 --> S["stack+mean\n多尺度融合"]
    K2 --> S
    K3 --> S
    K4 --> S
    S --> O["输出 [B,C',H,W]"]

    style A fill:#4A90D9,color:#fff
    style S fill:#EB2F96,color:#fff
    style O fill:#52C41A,color:#fff
```

**说明**: `Inception_Block_V1`（位于 `layers/Conv_Blocks.py`）构造 `num_kernels`（默认 6）个不同尺寸的方形卷积核（`kernel_size = 2*i+1`，即 1, 3, 5, 7, 9, 11），每个分支使用对应 padding 保证输出尺寸一致，最后 stack 取均值。各分支用 Kaiming Normal 初始化。TimesBlock 中连续使用两个 Inception_Block_V1，中间夹 GELU 激活。

---

## 5. DataEmbedding 嵌入结构

```mermaid
flowchart LR
    A["x_enc [B,T,C]"] --> V["TokenEmbedding\nConv1d k=3\nc_in→d_model"]
    MARK["x_mark [B,T,5]"] --> T["TemporalEmbedding\nmonth+day+weekday\n+hour+minute"]
    A --> P["PositionalEmbedding\nsin/cos 固定编码"]
    V --> ADD["三者相加"]
    T --> ADD
    P --> ADD
    ADD --> DROP["Dropout"]
    DROP --> OUT["[B, T, d_model]"]

    style A fill:#4A90D9,color:#fff
    style MARK fill:#8B9DAF,color:#fff
    style ADD fill:#EB2F96,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: 三个嵌入组件均位于 `layers/Embed.py`。

- **TokenEmbedding**: 一维卷积（k=3, circular padding），将原始变量投影到 `d_model` 维，用 Kaiming 初始化。
- **PositionalEmbedding**: 经典 Transformer 正弦/余弦位置编码（`max_len=5000`），注册为 buffer 不参与训练。
- **TemporalEmbedding**: 对 month(13)、day(32)、weekday(7)、hour(24)、minute(4) 分别查表嵌入后相加。当 `x_mark=None`（如 anomaly_detection）时跳过时间嵌入，仅用 Token + Position 两路。

---

## 6. 四种任务的 forward 分支

```mermaid
flowchart TD
    F["forward(x_enc,\nx_mark_enc,\nx_dec,\nx_mark_dec,\nmask)"] --> BR{"task_name"}

    BR -->|forecast| F1["forecast()\n标准化 → Embed\n→ predict_linear\n→ TimesBlock × L\n→ Linear → 反标准化"]
    BR -->|imputation| F2["imputation()\nmask-aware标准化\n→ Embed → TimesBlock×L\n→ Linear → 反标准化"]
    BR -->|anomaly_det| F3["anomaly_detection()\n标准化 → Embed\n→ TimesBlock × L\n→ Linear → 反标准化"]
    BR -->|classification| F4["classification()\nEmbed → TimesBlock×L\n→ GELU → Dropout\n→ mask zero-out\n→ Flatten → Linear"]

    F1 --> R1["返回 [:, -pred_len:, :]"]
    F2 --> R2["返回完整序列"]
    F3 --> R3["返回完整序列"]
    F4 --> R4["返回 [B, num_class]"]

    style F fill:#4A90D9,color:#fff
    style BR fill:#722ED1,color:#fff
    style R1 fill:#52C41A,color:#fff
    style R2 fill:#52C41A,color:#fff
    style R3 fill:#52C41A,color:#fff
    style R4 fill:#52C41A,color:#fff
```

**说明**:

- **forecast**: `predict_linear` 将 `seq_len` 维线性映射为 `seq_len + pred_len`，再送入 TimesBlock，最终只取后 `pred_len` 个时间步（`[:, -pred_len:, :]`）。
- **imputation**: 与 forecast 相同主路径，但输入带 mask，标准化时仅对观测到的值计算均值/标准差，输出取完整序列。
- **anomaly_detection**: 简化版 forecast，`x_mark=None`，输入只有 `x_enc`，无 predict_linear，输出完整序列用于重构误差计算。
- **classification**: 唯一不走标准化/反标准化的任务；`x_mark_enc` 在此任务中作为 mask（标记有效位置），通过 `output * x_mark_enc.unsqueeze(-1)` 将 padding 位置置零，再 flatten 投影到 `num_class` 维。

---

## 7. 标准化与反标准化（RevIN 模式）

```mermaid
flowchart LR
    A["x_enc"] --> M["mean(dim=1)\n[B,1,C]"]
    A --> X1["x - mean"]
    X1 --> S["sqrt(var+ε)\n[B,1,C]"]
    S --> X2["x / stdev"]
    X2 --> E["TimesBlock×L"]
    E --> R1["× stdev"]
    R1 --> R2["+ mean"]
    R2 --> O["dec_out"]

    style A fill:#4A90D9,color:#fff
    style O fill:#52C41A,color:#fff
    style E fill:#FA8C16,color:#fff
```

**说明**: 可逆实例归一化（Reversible Instance Normalization, RevIN）：先按时间维减均值除标准差做标准化，模型处理后再乘标准差加均值还原。这让模型在统一尺度上学习，对多变量时间序列尤为关键。

`imputation` 的标准化特殊处理了 mask——仅用 `torch.sum(mask == 1, dim=1)` 作为有效观测数计算均值和标准差，避免缺失值（填充的 0）污染统计量。

---

## 模块依赖关系

```mermaid
flowchart TD
    MT["models/TimesNet.py"] --> EMB["layers/Embed.py\nDataEmbedding"]
    MT --> CB["layers/Conv_Blocks.py\nInception_Block_V1"]
    EMB --> TE["TokenEmbedding\nConv1d"]
    EMB --> PE["PositionalEmbedding\nsin/cos"]
    EMB --> TMPE["TemporalEmbedding\n或 TimeFeatureEmbedding"]

    style MT fill:#4A90D9,color:#fff
    style EMB fill:#722ED1,color:#fff
    style CB fill:#FA8C16,color:#fff
```

---

## 关键超参数说明

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `top_k` | FFT 检测保留的周期数量 | 2 ~ 5 |
| `num_kernels` | Inception 卷积核数量 | 3 ~ 6 |
| `e_layers` | TimesBlock 堆叠层数 | 2 ~ 4 |
| `d_model` | 隐层维度 | 64 ~ 512 |
| `d_ff` | Inception 中间层维度 | 128 ~ 1024 |
| `seq_len` | 输入序列长度 | 96 ~ 720 |
| `pred_len` | 预测序列长度 | 96 ~ 720 |
