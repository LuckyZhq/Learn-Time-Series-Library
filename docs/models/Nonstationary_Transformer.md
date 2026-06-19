# Nonstationary Transformer 算法结构图

> **论文**: [Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting](https://openreview.net/pdf?id=ucNDIDRNjjv)
>
> **核心思想**: 标准 Transformer 的注意力机制假设时间序列平稳，但真实序列存在分布漂移。Nonstationary Transformer 通过学习 De-stationary 因子（tau 和 delta），自适应调整注意力的温度和偏置，使模型适应非平稳特性。

---

## 1. 总体架构总览

```mermaid
flowchart TD
    A["x_enc [B,T,C]"] --> RAW["x_raw\n(clone.detach)"]
    RAW --> TAU_L["tau_learner\n→ tau [B,1]"]
    RAW --> DEL_L["delta_learner\n→ delta [B,T]"]
    A --> NORM["RevIN 标准化\nmean/std"]
    NORM --> EMB["enc_embedding\nDataEmbedding"]

    EMB --> ENC["Encoder\ne_layers × EncoderLayer\n(DSAttention)"]
    ENC --> DEC["Decoder\nd_layers × DecoderLayer\n(DSAttention)"]

    TAU_L --> ENC
    TAU_L --> DEC
    DEL_L --> ENC
    DEL_L --> DEC

    DEC --> DENORM["反标准化\n×std + mean"]
    DENORM --> OUT["dec_out [:,-pred_len:]"]

    style A fill:#4A90D9,color:#fff
    style TAU_L fill:#EB2F96,color:#fff
    style DEL_L fill:#EB2F96,color:#fff
    style ENC fill:#FA8C16,color:#fff
    style DEC fill:#722ED1,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: 与标准 Transformer 的核心区别在于两个额外组件——`tau_learner` 和 `delta_learner`。它们从原始（未标准化）序列中学习 De-stationary 因子，注入 Encoder 和 Decoder 的每一层注意力中。标准化后的序列送入标准 Transformer 流程，最终反标准化输出。

---

## 2. Projector — De-stationary 因子学习器

```mermaid
flowchart TD
    X["x_raw [B,S,E]"] --> CONV["Conv1d k=3 circular\nS→1 压缩时间维"]
    CONV --> CAT["cat [conv_out, stats]\n→ [B, 2, E]"]
    STATS["stats [B,1,E]\n(std 或 mean)"] --> CAT
    CAT --> FLAT["view [B, 2E]"]
    FLAT --> MLP["MLP backbone\nLinear→ReLU→...→Linear"]
    MLP --> OUT_tau["output_dim=1\n→ tau [B,1]"]
    MLP --> OUT_delta["output_dim=S\n→ delta [B,S]"]

    style X fill:#4A90D9,color:#fff
    style STATS fill:#8B9DAF,color:#fff
    style CONV fill:#EB2F96,color:#fff
    style MLP fill:#FA8C16,color:#fff
    style OUT_tau fill:#52C41A,color:#fff
    style OUT_delta fill:#52C41A,color:#fff
```

**说明**: `Projector` 是学习 De-stationary 因子的 MLP。两个实例分别负责：
- **tau_learner**（`output_dim=1`）：接收 `(x_raw, std_enc)` → 输出温度缩放因子 tau（标量）
- **delta_learner**（`output_dim=seq_len`）：接收 `(x_raw, mean_enc)` → 输出注意力偏置 delta（长度为 seq_len 的向量）

Conv1d 用 circular padding 将整个时间序列压缩为 1 个向量（通道维保留），与统计量拼接后送入 MLP。`hidden_dims` 和 `hidden_layers` 由 `configs.p_hidden_dims` 和 `configs.p_hidden_layers` 控制。

---

## 3. DSAttention — De-stationary 注意力机制

```mermaid
flowchart TD
    Q["Q [B,L,H,E]"] --> SCORES["scores = einsum(Q,K)"]
    K["K [B,S,H,E]"] --> SCORES

    TAU["tau [B,1]\n→ [B,1,1,1]"] --> MUL["scores × tau"]
    DELTA["delta [B,S]\n→ [B,1,1,S]"] --> ADD["scores + delta"]

    SCORES --> MUL
    MUL --> ADD
    ADD --> MASK["causal mask\n(若 mask_flag=True)"]
    MASK --> SOFTMAX["softmax(scale × scores)"]
    SOFTMAX --> AGG["einsum(attn, V)\n→ [B,L,H,D]"]
    V["V [B,S,H,D]"] --> AGG
    AGG --> OUT["输出"]

    style Q fill:#4A90D9,color:#fff
    style K fill:#4A90D9,color:#fff
    style V fill:#4A90D9,color:#fff
    style TAU fill:#EB2F96,color:#fff
    style DELTA fill:#EB2F96,color:#fff
    style SCORES fill:#FA8C16,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: DSAttention 的核心公式为：

```
scores = (Q · K^T) × tau + delta
attn   = softmax(scale × scores)
output = attn · V
```

- **tau**（温度缩放）：正值标量，通过 `exp()` 保证正性。tau > 1 使注意力分布更尖锐（聚焦少数关键位置），tau < 1 使分布更平滑。
- **delta**（注意力偏置）：长度为 S 的向量，为每个 key 位置添加可学习偏置，捕获时间趋势或周期性偏移。
- **scale**：默认 `1/sqrt(E)`，标准 Transformer 缩放。

与标准注意力 `softmax(Q·K^T / sqrt(d))` 相比，DSAttention 的 tau 和 delta 使注意力分布能够自适应于序列的非平稳统计特性。

---

## 4. Forecast 完整数据流

```mermaid
flowchart TD
    X["x_enc [B,seq_len,C]"] --> RAW["x_raw = x.clone().detach()"]
    RAW --> MEAN["mean_enc = mean(dim=1)\n[B,1,C]"]
    RAW --> STD["std_enc = sqrt(var)\n[B,1,C]"]
    X --> SUB["x_enc - mean_enc"]
    SUB --> DIV["x_enc / std_enc"]
    DIV --> XN["x_enc_normalized\n[B,seq_len,C]"]

    RAW --> TAU_L["tau_learner(x_raw, std_enc)"]
    STD --> TAU_L
    TAU_L --> CLAMP["clamp(max=80)\n→ exp()"]
    CLAMP --> TAU["tau [B,1]"]

    RAW --> DEL_L["delta_learner(x_raw, mean_enc)"]
    MEAN --> DEL_L
    DEL_L --> DELTA["delta [B,seq_len]"]

    XN --> ENC["Encoder(enc_out, tau, delta)"]
    TAU --> ENC
    DELTA --> ENC

    XN --> CAT["cat [x_enc[:,-label_len:],\nzeros[:,-pred_len:]]"]
    CAT --> DEC["Decoder(dec_out, enc_out,\ntau, delta)"]
    ENC --> DEC
    TAU --> DEC
    DELTA --> DEC

    DEC --> DENORM["dec_out × std + mean"]
    STD --> DENORM
    MEAN --> DENORM
    DENORM --> SLICE["[:, -pred_len:, :]"]

    style X fill:#4A90D9,color:#fff
    style TAU fill:#EB2F96,color:#fff
    style DELTA fill:#EB2F96,color:#fff
    style ENC fill:#FA8C16,color:#fff
    style DEC fill:#722ED1,color:#fff
    style SLICE fill:#52C41A,color:#fff
```

**说明**:
- `x_raw = x.clone().detach()` 截断梯度——Projector 从原始统计量学习因子，不参与主路径梯度传播。
- `tau` 经过 `clamp(max=80)` 再 `exp()`，防止数值溢出（`exp(80) ≈ 5.5×10³⁴`，足够大但不会 `inf`）。
- tau 的输入是 `(x_raw, std_enc)`，delta 的输入是 `(x_raw, mean_enc)`——tau 关注波动幅度（std），delta 关注趋势水平（mean）。
- Decoder 输入构造方式与 TimesNet 类似：取已知序列后 `label_len` 步，拼接零张量占位。

---

## 5. EncoderLayer 结构（标准 Transformer + tau/delta 透传）

```mermaid
flowchart TD
    X["输入 x"] --> ATT["DSAttention 自注意力\n(tau, delta 注入)"]
    ATT --> ADD1["x + dropout(attn)"]
    ADD1 --> NORM1["LayerNorm₁"]
    NORM1 --> FFN["Conv1d→d_ff\n+ activation\n+ Conv1d→d_model"]
    FFN --> ADD2["norm₁_out + ffn"]
    ADD2 --> NORM2["LayerNorm₂"]
    NORM2 --> OUT["输出, attn"]

    style X fill:#4A90D9,color:#fff
    style ATT fill:#FA8C16,color:#fff
    style NORM1 fill:#EB2F96,color:#fff
    style NORM2 fill:#EB2F96,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: EncoderLayer 结构为标准 Transformer Encoder（自注意力 + FFN + 残差 + LayerNorm），但注意力机制使用 `DSAttention`，每一层接收 tau 和 delta 并注入注意力分数计算。与 Autoformer/FEDformer 的 EncoderLayer 不同，这里**没有 series_decomp 分解**，只有标准的 LayerNorm。

---

## 6. DecoderLayer 结构（标准 Transformer + tau/delta 透传）

```mermaid
flowchart TD
    X["输入 x"] --> SELF["DSAttention 掩码自注意力\n(tau, delta=None)"]
    SELF --> ADD1["x + dropout"]
    ADD1 --> NORM1["LayerNorm₁"]
    NORM1 --> CROSS["DSAttention 交叉注意力\n(tau, delta 注入)"]
    CROSS --> ADD2["norm₁ + dropout(cross)"]
    ADD2 --> NORM2["LayerNorm₂"]
    NORM2 --> FFN["Conv1d FFN"]
    FFN --> ADD3["norm₂ + ffn"]
    ADD3 --> NORM3["LayerNorm₃"]
    NORM3 --> OUT["输出"]

    style X fill:#4A90D9,color:#fff
    style SELF fill:#FA8C16,color:#fff
    style CROSS fill:#FA8C16,color:#fff
    style NORM1 fill:#EB2F96,color:#fff
    style NORM2 fill:#EB2F96,color:#fff
    style NORM3 fill:#EB2F96,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: 关键细节——**自注意力传 `delta=None`，交叉注意力传 `delta=delta`**。原因是 delta 编码了原始序列的位置级偏置（趋势信息），在自注意力中（decoder 对自身）不需要；在交叉注意力中（decoder query 与 encoder key 交互）注入 delta 可以让 decoder 感知 encoder 端的非平稳趋势。tau 在两者中都注入，控制注意力温度。

---

## 7. DSAttention 与标准注意力对比

```mermaid
flowchart LR
    subgraph Standard["标准 FullAttention"]
        S1["scores = Q·K^T"]
        S2["softmax(scores/√d)"]
        S1 --> S2
        S2 --> S3["output = attn·V"]
    end

    subgraph DS["De-stationary Attention"]
        D1["scores = Q·K^T"]
        D2["scores × tau\n温度缩放"]
        D3["scores + delta\n位置偏置"]
        D4["softmax(scores/√d)"]
        D1 --> D2 --> D3 --> D4
        D4 --> D5["output = attn·V"]
    end

    style Standard fill:#8B9DAF,color:#fff
    style DS fill:#4A90D9,color:#fff
```

| 对比维度 | 标准注意力 | De-stationary 注意力 |
|---------|-----------|---------------------|
| 注意力公式 | `softmax(QK^T / √d)` | `softmax((QK^T × τ + δ) / √d)` |
| 温度控制 | 固定 `1/√d` | 可学习 τ（自适应温度） |
| 位置偏置 | 无 | 可学习 δ（每位置偏置） |
| 统计假设 | 平稳序列 | 非平稳序列 |
| 额外参数 | 无 | tau_learner + delta_learner（两个 Projector） |

---

## 8. tau 和 delta 的语义解释

```mermaid
flowchart TD
    subgraph Tau["tau 温度缩放"]
        T_HIGH["tau >> 1\n→ 分布尖锐\n聚焦少数关键位置"]
        T_LOW["tau << 1\n→ 分布平滑\n均匀关注所有位置"]
    end

    subgraph Delta["delta 偏置"]
        D_POS["delta[i] > 0\n→ 增加位置 i 的注意力权重\n(趋势上升区)"]
        D_NEG["delta[i] < 0\n→ 降低位置 i 的注意力权重\n(趋势下降区)"]
    end

    STD["std_enc\n波动幅度"] --> TAU_L["tau_learner"]
    MEAN["mean_enc\n趋势水平"] --> DEL_L["delta_learner"]
    TAU_L --> T_HIGH
    TAU_L --> T_LOW
    DEL_L --> D_POS
    DEL_L --> D_NEG

    style TAU_L fill:#EB2F96,color:#fff
    style DEL_L fill:#EB2F96,color:#fff
    style T_HIGH fill:#FA8C16,color:#fff
    style D_POS fill:#722ED1,color:#fff
```

**说明**:
- **tau** 接收 std_enc 作为输入——std 越大（序列波动越剧烈），tau 趋向更大值，注意力越聚焦。这对应论文的直觉：非平稳序列的分布漂移使某些时间步更重要，需要更尖锐的注意力。
- **delta** 接收 mean_enc 作为输入——mean 编码了趋势水平，delta 学习出一个随时间变化的偏置向量，让注意力自然地偏向趋势上升区或下降区。
- 两者共同作用：tau 控制注意力"锐度"，delta 控制注意力"重心"。

---

## 9. 四种任务的 forward 分支

```mermaid
flowchart TD
    F["forward(...)"] --> BR{"task_name"}

    BR -->|forecast| F1["forecast()\nRevIN → enc+dec\n→ 反标准化"]
    BR -->|imputation| F2["imputation()\nmask-aware RevIN\n→ enc → projection"]
    BR -->|anomaly_det| F3["anomaly_detection()\nRevIN → enc\n→ projection"]
    BR -->|classification| F4["classification()\n(no RevIN)\n→ enc → GELU → mask"]

    F1 --> R1["[:, -pred_len:]"]
    F2 --> R2["完整序列"]
    F3 --> R3["完整序列"]
    F4 --> R4["[B, num_class]"]

    style F fill:#4A90D9,color:#fff
    style BR fill:#722ED1,color:#fff
    style R1 fill:#52C41A,color:#fff
    style R2 fill:#52C41A,color:#fff
    style R3 fill:#52C41A,color:#fff
    style R4 fill:#52C41A,color:#fff
```

**说明**: 所有任务都使用 tau_learner 和 delta_learner 学习 De-stationary 因子。Forecast/Imputation/Anomaly Detection 三个任务走 RevIN（标准化→处理→反标准化），Classification 不走标准化（但仍然计算 tau 和 delta）。注意 classification 不走标准化但仍计算 tau 和 delta，且 `x_mark_enc` 在此任务中作为 mask 使用。

---

## 10. 模块依赖关系

```mermaid
flowchart TD
    NST["models/Nonstationary_Transformer.py\nModel + Projector"] --> EMB["layers/Embed.py\nDataEmbedding"]
    NST --> TE["layers/Transformer_EncDec.py\nEncoder/Decoder\nEncoderLayer/DecoderLayer"]
    NST --> SA["layers/SelfAttention_Family.py\nDSAttention\nAttentionLayer"]

    TE --> SA

    style NST fill:#4A90D9,color:#fff
    style TE fill:#FA8C16,color:#fff
    style SA fill:#722ED1,color:#fff
    style EMB fill:#EB2F96,color:#fff
```

**说明**: Nonstationary Transformer 使用标准 Transformer 的 Encoder/Decoder 结构（来自 `Transformer_EncDec.py`），但将注意力机制替换为 `DSAttention`（来自 `SelfAttention_Family.py`）。与 Autoformer/FEDformer 不同，它不使用 `Autoformer_EncDec.py` 中的分解架构（无 `series_decomp`、`my_Layernorm`），而是使用标准的 LayerNorm。`Projector` 定义在模型文件内部，是该模型独有的组件。

---

## 关键超参数说明

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `p_hidden_dims` | Projector MLP 各层维度列表 | `[64, 64]` 或 `[128]` |
| `p_hidden_layers` | Projector MLP 层数 | 2 |
| `e_layers` | Encoder 层数 | 2 ~ 4 |
| `d_layers` | Decoder 层数 | 1 ~ 2 |
| `n_heads` | 注意力头数 | 8 |
| `d_model` | 隐层维度 | 512 |
| `d_ff` | FFN 中间维度 | 2048 |
| `factor` | 注意力缩放因子（未实际用于 DSAttention） | 5 |
| `pred_len` | 预测序列长度 | 96 ~ 720 |
| `label_len` | Decoder 输入的已知序列长度 | 48 ~ 96 |
