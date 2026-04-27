def print_args(args):
    # =========================================================
    # Basic Config: 基本配置
    # =========================================================
    print("\033[1m" + "Basic Config" + "\033[0m")
    # 任务名称 & 是否为训练模式
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    # 模型ID & 模型类型
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    # =========================================================
    # Data Loader: 数据加载相关参数
    # =========================================================
    print("\033[1m" + "Data Loader" + "\033[0m")
    # 数据集名称 & 根路径
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    # 数据文件路径 & 使用的特征
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    # 目标列 & 数据频率（如小时、天、月）
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    # 检查点保存路径
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    # =========================================================
    # Forecasting Task: 预测任务相关参数
    # =========================================================
    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        # 输入序列长度 & 标签长度
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        # 预测长度 & 是否考虑季节性模式
        print(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        # 是否对数据进行逆变换（如标准化后的逆操作）
        print(f'  {"Inverse:":<20}{args.inverse:<20}')
        print()

    # =========================================================
    # Imputation Task: 缺失值填充任务相关参数
    # =========================================================
    if args.task_name == 'imputation':
        print("\033[1m" + "Imputation Task" + "\033[0m")
        # 缺失值比例（Mask Rate）
        print(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
        print()

    # =========================================================
    # Anomaly Detection Task: 异常检测任务相关参数
    # =========================================================
    if args.task_name == 'anomaly_detection':
        print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        # 异常点比例
        print(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
        print()

    # =========================================================
    # Model Parameters: 模型参数
    # =========================================================
    print("\033[1m" + "Model Parameters" + "\033[0m")
    # Top-k注意力 & 卷积核数量
    print(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    # 编码器输入维度 & 解码器输入维度
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    # 输出通道数 & 模型维度
    print(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    # 注意力头数 & 编码器层数
    print(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    # 解码器层数 & 前馈网络维度
    print(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    # 移动平均窗口大小 & 因子（用于季节性分解等）
    print(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    # 是否使用蒸馏 & dropout概率
    print(f'  {"Distil:":<20}{args.distil:<20}{"Dropout:":<20}{args.dropout:<20}')
    # 嵌入类型 & 激活函数
    print(f'  {"Embed:":<20}{args.embed:<20}{"Activation:":<20}{args.activation:<20}')
    print()

    # =========================================================
    # Run Parameters: 训练/运行相关参数
    # =========================================================
    print("\033[1m" + "Run Parameters" + "\033[0m")
    # 数据加载器使用的线程数 & 迭代次数
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    # 训练轮数 & 批量大小
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    # 早停耐心 & 学习率
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    # 任务描述 & 损失函数类型
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    # 学习率调度策略 & 是否使用混合精度训练
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print()

    # =========================================================
    # GPU Parameters: GPU相关设置
    # =========================================================
    print("\033[1m" + "GPU" + "\033[0m")
    # 是否使用GPU & GPU编号
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    # 是否使用多GPU & GPU设备列表
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

    # =========================================================
    # De-stationary Projector Params: 非平稳投影器参数
    # =========================================================
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    # 投影器隐藏层维度 & 隐藏层数
    print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}')
    print()