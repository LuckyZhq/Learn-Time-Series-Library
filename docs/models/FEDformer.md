# FEDformer 算法结构图

> **论文**: [FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting](https://proceedings.mlr.press/v162/zhou22g.html)
>
> **核心思想**: 在频域（Fourier / Wavelet）执行注意力机制，实现 O(N) 复杂度；结合 Autoformer 的渐进式分解架构，将序列显式拆分为趋势项（trend）和季节项（seasonal）分别建模。

---

## 1. 总体架构总览

```mermaid
flowchart TD
    A["x_enc [B,T,C]"] --> DECOMP["series_decomp\nmoving_avg 分解"]
    A --> E_EMB["enc_embedding\nDataEmbedding"]
    DECOMP --> TREND_INIT["trend_init\n[-label_len:]\n+ mean"]
    DECOMP --> SEA_INIT["seasonal_init\npad(pred_len)"]

    E_EMB --> ENC["Encoder\ne_layers × EncoderLayer"]
    SEA_INIT --> D_EMB["dec_embedding\nDataEmbedding"]
    D_EMB --> DEC["Decoder\nd_layers × DecoderLayer"]
    TREND_INIT --> DEC
    ENC --> DEC

    DEC --> COMBINE["trend_part\n+ seasonal_part"]
    COMBINE --> OUT["dec_out [:,-pred_len:]"]

    style A fill:#4A90D9,color:#fff
    style ENC fill:#FA8C16,color:#fff
    style DEC fill:#722ED1,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: FEDformer 采用 Encoder-Decoder 结构，但在 Decoder 输入端先用 `series_decomp`（移动平均）将序列分解为 trend 和 seasonal 两路。Encoder 只处理 seasonal 路径；Decoder 同时接收 Encoder 输出和 trend 初始值，每层 DecoderLayer 会进一步分解并累积 trend，最终 trend + seasonal 合并输出。

---

## 2. Forecast 完整数据流（最复杂路径）

```mermaid
flowchart TD
    X["x_enc [B,seq_len,C]"] --> MEAN["mean(dim=1)\n[B,1,C] repeat→pred_len"]
    X --> DECOMP["series_decomp(x_enc)\nseasonal_init, trend_init"]
    MEAN --> T_CAT["trend_init[:, -label_len:]\ncat mean → [B,label_len+pred_len,C]"]
    DECOMP --> T_CAT
    DECOMP --> S_PAD["seasonal_init[:, -label_len:]\nF.pad → [B,label_len+pred_len,C]"]

    X --> E_EMB["enc_embedding(x_enc, x_mark_enc)"]
    E_EMB --> ENC["Encoder → enc_out"]
    S_PAD --> D_EMB["dec_embedding(seasonal, x_mark_dec)"]
    D_EMB --> DEC
    T_CAT --> DEC["Decoder(dec_out, enc_out,\ntrend=trend_init)"]
    ENC --> DEC

    DEC --> S_OUT["seasonal_part"]
    DEC --> T_OUT["trend_part"]
    S_OUT --> ADD["dec_out = trend + seasonal"]
    T_OUT --> ADD
    ADD --> SLICE["[:, -pred_len:, :]"]

    style X fill:#4A90D9,color:#fff
    style ENC fill:#FA8C16,color:#fff
    style DEC fill:#722ED1,color:#fff
    style SLICE fill:#52C41A,color:#fff
```

**说明**:
- `trend_init` 由原始序列的移动平均拼接全局均值构成，作为 Decoder 中 trend 累积的起点。
- `seasonal_init` 取原始 seasonal 分解的后 `label_len` 步，右侧 zero-pad `pred_len` 位，送入 Decoder 嵌入层。
- Decoder 每层输出的 `residual_trend` 会逐步累加到 `trend_init` 上，实现渐进式趋势建模。

---

## 3. EncoderLayer 渐进式分解结构

```mermaid
flowchart TD
    X["输入 x [B,T,d_model]"] --> ATT["频率域自注意力\nFourierBlock 或\nMultiWaveletTransform"]
    ATT --> ADD1["x + dropout(attn_out)"]
    ADD1 --> DECOMP1["series_decomp₁\n→ seasonal₁, _ (舍弃trend)"]
    DECOMP1 --> FFN["Conv1d→d_ff\n+ activation\n+ Conv1d→d_model"]
    FFN --> ADD2["seasonal₁ + ffn_out"]
    ADD2 --> DECOMP2["series_decomp₂\n→ seasonal₂, _ (舍弃trend)"]
    DECOMP2 --> OUT["返回 seasonal₂, attn"]

    style X fill:#4A90D9,color:#fff
    style ATT fill:#FA8C16,color:#fff
    style DECOMP1 fill:#EB2F96,color:#fff
    style DECOMP2 fill:#EB2F96,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: EncoderLayer 采用两次 `series_decomp`（移动平均分解）。第一次在注意力之后去除残余趋势，第二次在 FFN 之后再次去趋势。两次分解均**舍弃 trend 分支**，只保留 seasonal 传给下一层——Encoder 专注于季节性特征提取。FFN 使用两层 1×1 Conv1d 替代 Transformer 的全连接层。

---

## 4. DecoderLayer 三层分解与 trend 累积

```mermaid
flowchart TD
    X["输入 x"] --> SELF_ATT["频率域自注意力"]
    SELF_ATT --> ADD1["x + dropout(attn)"]
    ADD1 --> DECOMP1["decomp₁ → sea₁, trend₁"]

    sea1["sea₁"] --> CROSS_ATT["频率域交叉注意力\n(Q=sea₁, KV=enc_out)"]
    CROSS_ATT --> ADD2["sea₁ + dropout(cross)"]
    ADD2 --> DECOMP2["decomp₂ → sea₂, trend₂"]

    sea2["sea₂"] --> FFN["Conv1d FFN"]
    FFN --> ADD3["sea₂ + ffn"]
    ADD3 --> DECOMP3["decomp₃ → sea₃, trend₃"]

    trend1["trend₁"] --> T_ADD["trend₁ + trend₂ + trend₃"]
    trend2["trend₂"] --> T_ADD
    trend3["trend₃"] --> T_ADD
    T_ADD --> PROJ["Conv1d k=3 projection\ntrend → c_out 维"]
    PROJ --> T_OUT["residual_trend"]

    sea3["sea₃"] --> S_OUT["seasonal 输出"]

    style X fill:#4A90D9,color:#fff
    style SELF_ATT fill:#FA8C16,color:#fff
    style CROSS_ATT fill:#FA8C16,color:#fff
    style DECOMP1 fill:#EB2F96,color:#fff
    style DECOMP2 fill:#EB2F96,color:#fff
    style DECOMP3 fill:#EB2F96,color:#fff
    style T_OUT fill:#722ED1,color:#fff
    style S_OUT fill:#52C41A,color:#fff
```

**说明**: DecoderLayer 有 3 次分解（比 Encoder 多 1 次），分别发生在自注意力后、交叉注意力后、FFN 后。**三次分解的 trend 分支被累加**，经过 Conv1d（k=3, circular padding）投影到 `c_out` 维后，作为 `residual_trend` 返回。Decoder 外层将各层 `residual_trend` 逐步累加到初始 `trend_init` 上，形成渐进式趋势建模。

---

## 5. FourierBlock — 频域自注意力（Fourier 版本）

```mermaid
flowchart LR
    Q["q [B,L,H,E]"] --> PERM["permute→[B,H,E,L]"]
    PERM --> FFT["rfft(dim=-1)\n频域表示"]
    FFT --> SEL["选取 top-modes\n频率分量"]
    SEL --> MUL["complex 乘法\n× learnable weights\n[bhi,hio→bho]"]
    MUL --> IFFT["irfft\n→ 回到时域"]
    IFFT --> OUT["(x, None)"]

    style Q fill:#4A90D9,color:#fff
    style FFT fill:#722ED1,color:#fff
    style MUL fill:#FA8C16,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: `FourierBlock`（位于 `layers/FourierCorrelation.py`）在频域执行线性变换替代传统注意力。输入做 `rfft` 后，只保留 `modes` 个频率分量（`mode_select='random'` 随机选取，`'low'` 选低频），在频域与可学习权重做复数乘法，再 `irfft` 回时域。复杂度 O(N·modes)，远低于标准注意力的 O(N²)。两个权重矩阵 `weights1/weights2`（实部/虚部）各维度为 `[n_heads, E//H, E//H, modes]`。

---

## 6. FourierCrossAttention — 频域交叉注意力（Fourier 版本）

```mermaid
flowchart TD
    Q["q [B,Lq,H,E]"] --> FFT_Q["rfft → 选取\nindex_q 频率"]
    K["k [B,Lkv,H,E]"] --> FFT_K["rfft → 选取\nindex_kv 频率"]
    V["v"] --> FFT_V["rfft → 选取\nindex_kv 频率"]

    FFT_Q --> QK["complex einsum\nQ·K† → attention"]
    FFT_K --> QK
    QK --> ACT["tanh 激活\n(实部虚部分别)"]
    ACT --> QKV["complex einsum\nattn·V → context"]
    FFT_V --> QKV
    QKV --> MW["× learnable weights\n频域线性变换"]
    MW --> SCATTER["scatter 回\n原始频率位置"]
    SCATTER --> IFFT["irfft → 时域"]
    IFFT --> OUT["(out, None)"]

    style Q fill:#4A90D9,color:#fff
    style K fill:#4A90D9,color:#fff
    style V fill:#4A90D9,color:#fff
    style QK fill:#EB2F96,color:#fff
    style ACT fill:#FA8C16,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: `FourierCrossAttention` 在频域实现交叉注意力。Q 和 K/V 分别做 rfft 后选取各自的频率子集（`index_q`, `index_kv`），在频域计算 `Q·K†` 得到注意力矩阵，用 `tanh` 激活（而非 softmax，保证频域稳定性），再乘 V 得到上下文表示。最后经过可学习权重线性变换后 scatter 回原始频率位置，irfft 回时域。关键区别：Q 和 K 的频率索引可以不同（`seq_len_q ≠ seq_len_kv`），适应 Encoder-Decoder 不同序列长度。

---

## 7. MultiWaveletTransform — 小波域自注意力（Wavelets 版本）

```mermaid
flowchart TD
    V["values [B,L,H,E]"] --> PROJ["Lk0 线性映射\n→ [B,L,c,k]"]
    PROJ --> MWT["MWT_CZ1d × nCZ\n多小波变换块"]
    MWT --> OUT_PROJ["Lk1 线性映射\n→ [B,L,H,E]"]
    OUT_PROJ --> OUT["(V, None)"]

    subgraph MWT_CZ["MWT_CZ1d 单块"]
        IN["输入 [B,N,c,k]"] --> PAD["pad 到 2^k 长度"]
        PAD --> DECOMP["小波分解 ×log₂N 层\n每层: odd/even拆分\n→ detail d + smooth s"]
        DECOMP --> PROC["每层: A(d)+B(s)\nC(d) (FFT 稀疏核)"]
        PROC --> RECON["小波重构\n从最粗尺度逐层上采样"]
        RECON --> T0["T0 线性层\n粗尺度变换"]
    end

    style V fill:#4A90D9,color:#fff
    style MWT fill:#722ED1,color:#fff
    style DECOMP fill:#EB2F96,color:#fff
    style RECON fill:#FA8C16,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: `MultiWaveletTransform`（位于 `layers/MultiWaveletCorrelation.py`）使用多小波变换（MWT）替代 FFT 进行频域表示学习。基函数默认为 Legendre 多项式。`MWT_CZ1d` 块执行：(1) 将序列 pad 到 2 的幂次长度；(2) 逐层小波分解，每层将信号拆为 detail（高频）和 smooth（低频）；(3) 对各层系数用 FFT 稀疏核（`sparseKernelFT1d`）做频域线性变换；(4) 从最粗尺度逐层小波重构回原始长度。`sparseKernelFT1d` 内部对各层系数做 rfft → 选 top-modes → complex 乘法 → irfft，实现 O(N log N) 复杂度。

---

## 8. MultiWaveletCross — 小波域交叉注意力（Wavelets 版本）

```mermaid
flowchart TD
    Q["q"] --> LQ["Lq 线性→[B,N,c,k]"]
    K["k"] --> LK["Lk 线性→[B,S,c,k]"]
    V["v"] --> LV["Lv 线性→[B,S,c,k]"]

    LQ --> DECOMP_Q["小波分解 Q"]
    LK --> DECOMP_K["小波分解 K"]
    LV --> DECOMP_V["小波分解 V"]

    DECOMP_Q --> ATT_D["attn1+attn2\ndetail 交叉注意力"]
    DECOMP_K --> ATT_D
    DECOMP_V --> ATT_D

    DECOMP_Q --> ATT_S["attn3\nsmooth 交叉注意力"]
    DECOMP_K --> ATT_S
    DECOMP_V --> ATT_S

    LQ --> ATT_C["attn4\n粗尺度交叉注意力"]
    LK --> ATT_C
    LV --> ATT_C

    ATT_D --> RECON["逐层小波重构\n+ evenOdd 上采样"]
    ATT_S --> RECON
    ATT_C --> RECON
    RECON --> OUT_PROJ["out 线性→[B,N,H,E]"]
    OUT_PROJ --> OUT["(v, None)"]

    style Q fill:#4A90D9,color:#fff
    style K fill:#4A90D9,color:#fff
    style V fill:#4A90D9,color:#fff
    style ATT_D fill:#EB2F96,color:#fff
    style ATT_S fill:#FA8C16,color:#fff
    style ATT_C fill:#722ED1,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: `MultiWaveletCross` 在小波域实现交叉注意力。对 Q/K/V 分别做小波分解后，在**每一层**分别对 detail 系数（attn1+attn2）和 smooth 系数（attn3）执行 `FourierCrossAttentionW`（无权重的简化版频域注意力），最粗尺度还有单独的 attn4。重构时从最粗尺度逐层上采样（`evenOdd`），每层加上 smooth 注意力结果和 detail 注意力结果。这实现了多尺度的交叉注意力——粗尺度捕获全局趋势对应，细尺度捕获局部细节对应。

---

## 9. series_decomp — 时间序列分解

```mermaid
flowchart LR
    X["输入 x [B,T,C]"] --> MA["moving_avg\nAvgPool1d\nk=kernel_size"]
    MA --> TREND["trend\nmoving_mean"]
    X --> SUB["x - moving_mean"]
    SUB --> SEASONAL["seasonal\n残差"]

    style X fill:#4A90D9,color:#fff
    style MA fill:#EB2F96,color:#fff
    style TREND fill:#722ED1,color:#fff
    style SEASONAL color:#fff,fill:#52C41A
```

**说明**: `series_decomp`（位于 `layers/Autoformer_EncDec.py`）用移动平均提取趋势项。`moving_avg` 对序列两端做镜像填充（首尾各 `(kernel_size-1)//2` 个点），再用 `AvgPool1d(kernel_size, stride=1)` 平滑，保证输出长度与输入一致。`seasonal = x - trend` 为季节性残差。`kernel_size` 对应 `configs.moving_avg`，典型值 25。

---

## 10. my_Layernorm — 季节性专用归一化

```mermaid
flowchart LR
    X["x [B,T,d]"] --> LN["LayerNorm(x)"]
    LN --> BIAS["mean(dim=1)\n→ bias"]
    BIAS --> SUB["x_hat - bias"]
    SUB --> OUT["输出 [B,T,d]"]

    style X fill:#4A90D9,color:#fff
    style LN fill:#EB2F96,color:#fff
    style OUT fill:#52C41A,color:#fff
```

**说明**: `my_Layernorm` 在标准 LayerNorm 基础上减去时间维均值，使得归一化后的季节性分量在时间维上零均值，符合"季节性应围绕零波动"的先验。这是 Autoformer/FEDformer 专为 trend-seasonal 分解设计的归一化策略。

---

## 11. 两种版本注意力机制对比

```mermaid
flowchart LR
    subgraph Fourier["version='Fourier'"]
        F1["rfft"] --> F2["选 modes 个频率"]
        F2 --> F3["频域复数乘法\n+ learnable weights"]
        F3 --> F4["irfft"]
    end

    subgraph Wavelets["version='Wavelets'"]
        W1["小波分解\n(log₂N 层)"] --> W2["每层: FFT 稀疏核"]
        W2 --> W3["小波重构"]
        W3 --> W4["evenOdd 上采样"]
    end

    style Fourier fill:#4A90D9,color:#fff
    style Wavelets fill:#722ED1,color:#fff
```

| 对比维度 | Fourier 版 | Wavelets 版 |
|---------|-----------|------------|
| 变换方式 | 全局 FFT | 多尺度小波分解+重构 |
| 注意力位置 | 频域线性变换 / 频域 QKV 注意力 | 每层小波系数上做频域注意力 |
| 频率选择 | 随机 / 低频 | 自适应（小波多分辨率） |
| 自注意力 | `FourierBlock` | `MultiWaveletTransform`（MWT_CZ1d） |
| 交叉注意力 | `FourierCrossAttention` | `MultiWaveletCross`（含 4 个 FourierCrossAttentionW） |
| 复杂度 | O(N·modes) | O(N log N) |
| 适合场景 | 周期性明显的信号 | 多尺度非平稳信号 |

---

## 12. 模块依赖关系

```mermaid
flowchart TD
    FED["models/FEDformer.py\nModel"] --> EMB["layers/Embed.py\nDataEmbedding"]
    FED --> AE["layers/Autoformer_EncDec.py\nEncoder/Decoder/series_decomp"]
    FED --> AC["layers/AutoCorrelation.py\nAutoCorrelationLayer"]
    FED --> FC["layers/FourierCorrelation.py\nFourierBlock\nFourierCrossAttention"]
    FED --> MC["layers/MultiWaveletCorrelation.py\nMultiWaveletTransform\nMultiWaveletCross"]

    AE --> AC
    FC --> AC
    MC --> AC

    style FED fill:#4A90D9,color:#fff
    style AE fill:#FA8C16,color:#fff
    style AC fill:#EB2F96,color:#fff
    style FC fill:#722ED1,color:#fff
    style MC fill:#52C41A,color:#fff
```

**说明**: `AutoCorrelationLayer` 是通用包装层，提供 Q/K/V 线性投影和多头拆分，内部 `inner_correlation` 可以是 `FourierBlock`、`FourierCrossAttention`、`MultiWaveletTransform` 或 `MultiWaveletCross` 中的任意一个。Encoder/Decoder 层通过 `AutoCorrelationLayer` 透明地切换频域注意力后端。

---

## 关键超参数说明

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `version` | 注意力后端：`'Fourier'` 或 `'Wavelets'` | `'fourier'` |
| `mode_select` | 频率选择策略：`'random'` / `'low'` | `'random'` |
| `modes` | 保留的频率模态数 | 32 |
| `moving_avg` | 移动平均窗口大小（分解用） | 25 |
| `e_layers` | Encoder 层数 | 2 ~ 4 |
| `d_layers` | Decoder 层数 | 1 ~ 2 |
| `n_heads` | 注意力头数 | 8 |
| `d_model` | 隐层维度 | 512 |
| `d_ff` | FFN 中间维度 | 2048 |
| `label_len` | Decoder 输入的已知序列长度 | 48 ~ 96 |
| `pred_len` | 预测序列长度 | 96 ~ 720 |
