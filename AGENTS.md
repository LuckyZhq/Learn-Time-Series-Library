# AGENTS.md - 时间序列库 (TSLib) 项目指南

## 语言要求
- 与用户交流时使用中文回复
- 编写文档和代码注释时使用中文

## 项目概述
时间序列库 (TSLib) 是一个用于深度时间序列分析的开源库，涵盖五个主流任务：
- 长期预测：预测未来较长时段的值
- 短期预测：预测近期的值
- 填补：填补时间序列中的缺失值
- 异常检测：识别时间序列中的异常模式
- 分类：对时间序列数据进行分类

## 项目结构

主要目录：
- run.py：所有实验的主入口
- requirements.txt：Python 依赖
- dataset/：数据文件目录
  - ETT-small/：ETT 数据集 (ETTh1, ETTh2, ETTm1, ETTm2)
  - SWaT/：SWaT 异常检测数据集
  - PSM/：PSM 异常检测数据集
  - MSL/：MSL 异常检测数据集
  - SMAP/：SMAP 异常检测数据集
  - SMD/：SMD 异常检测数据集
- data_provider/：数据加载与处理
  - data_loader.py：各类任务的数据集类
  - data_factory.py：DataLoader 工厂
  - m4.py：M4 数据集工具
  - uea.py：UEA 分类数据集工具
- models/：模型实现（40+ 模型）
  - TimesNet.py：TimesNet（各任务表现最佳）
  - iTransformer.py：iTransformer
  - PatchTST.py：PatchTST
  - TimeXer.py：TimeXer（长期预测最佳）
  - TimeMixer.py：TimeMixer
  - Autoformer.py：Autoformer
  - Informer.py：Informer
  - Transformer.py：标准 Transformer
  - DLinear.py：DLinear
  - Mamba.py：Mamba（需要 mamba_ssm）
  - Chronos.py：Chronos（基础模型）
- layers/：神经网络层
  - Embed.py：嵌入层
  - SelfAttention_Family.py：注意力机制
  - Transformer_EncDec.py：Transformer 编解码器
- exp/：实验运行器
  - exp_basic.py：基础实验类
  - exp_long_term_forecasting.py：长期预测
  - exp_short_term_forecasting.py：短期预测
  - exp_anomaly_detection.py：异常检测
  - exp_classification.py：分类
  - exp_imputation.py：填补
  - exp_zero_shot_forecasting.py：零样本预测
- utils/：工具函数
  - metrics.py：评估指标
  - losses.py：损失函数
  - timefeatures.py：时间特征编码
  - augmentation.py：数据增强
  - dtw.py：动态时间规整
  - tools.py：早停、可视化工具
- scripts/：各任务的示例脚本
  - long_term_forecast/
  - short_term_forecast/
  - anomaly_detection/
  - classification/
  - imputation/
- checkpoints/：模型检查点
- results/：实验结果

## 数据集格式

### 1. 预测任务 (ETT, Weather, ECL, Traffic, Exchange, ILI)
格式：带表头的 CSV 文件
- 第一列：时间戳 (date)
- 中间列：特征变量
- 最后一列：目标变量（默认 OT）
- 划分比例：70%% 训练集，10%% 验证集，20%% 测试集（ETT 使用固定日期边界）
- 加载器类：Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

### 2. 异常检测任务 (SWaT, PSM, MSL, SMAP, SMD)
格式：CSV 或 NPY 文件

CSV 格式 (PSM, SWaT)：
- 无时间戳列
- 最后一列：标签 (0=正常, 1=异常)
- 训练集：仅正常数据；测试集：正常 + 异常数据
- SWaT 有 51 个特征

NPY 格式 (MSL, SMAP, SMD)：
- name_train.npy：训练数据（仅正常）
- name_test.npy：测试数据（正常 + 异常）
- name_test_label.npy：测试标签

加载器类：PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader

### 3. 分类任务 (UEA)
格式：TS 文件（sktime 格式）
- 通过 sktime.datasets.load_from_tsfile_to_dataframe 加载
- 加载器类：UEAloader

### 4. 短期预测 (M4)
- 使用 M4 竞赛数据集
- 加载器类：Dataset_M4

## 关键配置参数

### 任务与模型
--task_name：任务类型 (long_term_forecast, anomaly_detection, classification, imputation)
--is_training：训练模式 (1=训练, 0=测试)
--model_id：实验 ID (如 ETTh1_96_96)
--model：模型名称 (TimesNet, iTransformer, PatchTST, TimeXer)
--data：数据集键名 (ETTh1, SWAT, PSM, custom)

### 数据路径
--root_path：数据根目录 (如 ./dataset/ETT-small/)
--data_path：数据文件名 (如 ETTh1.csv)

### 特征配置
--features：特征类型 (M=多变量->多变量, S=单变量->单变量, MS=多变量->单变量)
--target：目标列名（默认 OT）
--freq：时间编码频率 (s/t/h/d/b/w/m)

### 序列长度
--seq_len：输入（回看）长度（默认 96）
--label_len：解码器起始 token 长度（默认 48）
--pred_len：预测长度（默认 96）

### 模型架构
--enc_in：编码器输入维度（特征数量）（默认 7）
--dec_in：解码器输入维度（默认 7）
--c_out：输出维度（默认 7）
--d_model：模型隐藏维度（默认 512）
--d_ff：前馈网络维度（默认 2048）
--n_heads：注意力头数（默认 8）
--e_layers：编码器层数（默认 2）
--d_layers：解码器层数（默认 1）
--dropout：Dropout 率（默认 0.1）
--top_k：TimesNet top-k 周期数（默认 5）
--factor：注意力因子（默认 1）

### 异常检测专用
--anomaly_ratio：先验异常比例 %%（默认 0.25）

### 训练配置
--batch_size：批大小（默认 32）
--learning_rate：学习率（默认 0.0001）
--train_epochs：最大训练轮数（默认 10）
--patience：早停耐心值（默认 3）
--num_workers：数据加载器工作进程数（默认 10）
--loss：损失函数（默认 MSE）
--lradj：学习率调整策略（默认 type1）
--use_amp：混合精度训练（默认 False）

## 数据加载流程

1. 选择：data_factory.py 根据 args.data 映射到对应的 Dataset 类
2. 读取：Dataset 类从 root_path + data_path 读取 CSV/NPY 文件
   - 如果本地文件不存在，自动从 HuggingFace 下载 (thuml/Time-Series-Library)
3. 归一化：在训练集上拟合 StandardScaler，应用到所有划分
4. 时间特征：从时间戳提取（月、日、星期、小时）或使用 time_features() 编码
5. 划分：固定边界 (ETT) 或按比例划分（自定义数据集 70/10/20）
6. 批处理：返回 seq_x, seq_y, seq_x_mark, seq_y_mark

### 批数据形状（预测任务）
- seq_x: (batch, seq_len, n_features) - 编码器输入
- seq_y: (batch, label_len + pred_len, n_features) - 解码器输入
- seq_x_mark: (batch, seq_len, time_features) - 编码器时间特征
- seq_y_mark: (batch, label_len + pred_len, time_features) - 解码器时间特征

### 批数据形状（异常检测任务）
- 返回：(batch, win_size, n_features) 数据，(batch, win_size) 标签

## 添加新模型

1. 创建 models/YourModel.py，定义 class Model(nn.Module)
2. 实现接口：
   - __init__(self, configs)：初始化模型层
   - forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec)：返回 (batch, pred_len, c_out)
   - anomaly_detection(self, x_enc)：返回 (batch, seq_len, c_out)
   - classification(self, x_enc, x_mark_enc)：返回 (batch, num_classes)
   - imputation(self, x_enc, x_mark_enc, mask)：返回 (batch, seq_len, c_out)
   - forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)：路由到对应方法
3. 在 exp/exp_basic.py 中注册：from models.YourModel import Model
4. 在 scripts/<task>/<dataset>/YourModel.sh 中创建脚本

## 添加新数据集

1. 将数据文件放入 dataset/<dataset_name>/
2. 在 data_provider/data_loader.py 中创建 Dataset 类
3. 在 data_provider/data_factory.py 的 data_dict 中注册
4. 在 scripts/<task>/<dataset_name>/ 中创建脚本

## 顶级模型排行（2025 年）

长期预测（回看长度 96）：1.TimeXer, 2.iTransformer, 3.TimeMixer
长期预测（回看长度搜索）：1.TimeMixer, 2.PatchTST, 3.DLinear
短期预测：1.TimesNet, 2.Non-stationary Transformer, 3.FEDformer
填补：1.TimesNet, 2.Non-stationary Transformer, 3.Autoformer
分类：1.TimesNet, 2.Non-stationary Transformer, 3.Informer
异常检测：1.TimesNet, 2.FEDformer, 3.Autoformer

## 大型时序模型（零样本预测）

以下基础模型支持零样本预测：
- Chronos / Chronos2 (Amazon)
- TimesFM (Google)
- Moirai (Salesforce)
- TiRex, Sundial, Time-MoE, Toto

## 评估指标

预测：MSE, MAE
异常检测：Precision, Recall, F1
分类：Accuracy
填补：MSE, MAE

## 注意事项

### 运行环境
- 本项目运行在 WSL2 的 Ubuntu 环境中
- 运行时使用 conda 的 `py312_torch` 虚拟环境

### 其他说明
- 所有脚本使用 export HF_ENDPOINT=https://hf-mirror.com 配置国内 HuggingFace 镜像
- 本地不存在数据时会自动从 HuggingFace 下载
- 模型检查点保存到 ./checkpoints/
- 结果保存到 ./results/
--des 参数是实验描述标签
- 异常检测任务需要设置 --pred_len 0