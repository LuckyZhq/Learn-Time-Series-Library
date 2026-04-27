# 导入必要的库
import argparse
import os
import torch
import torch.backends
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    # 设置随机种子以确保实验的可重复性
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='TimesNet')

    # ==================== 基础配置 ====================
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='任务名称，可选: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='状态标识：1表示训练，0表示测试')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='模型ID标识')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='模型名称，可选: [Autoformer, Transformer, TimesNet]')

    # ==================== 数据加载器配置 ====================
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='数据集类型')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='数据文件根路径')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件名')
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务类型，可选: [M, S, MS]; M:多变量预测多变量, S:单变量预测单变量, MS:多变量预测单变量')
    parser.add_argument('--target', type=str, default='OT', help='S或MS任务中的目标特征')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间特征编码的频率，可选: [s:秒级, t:分钟级, h:小时级, d:天级, b:工作日, w:周级, m:月级]，也可以使用更详细的频率如15min或3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点保存位置')

    # ==================== 预测任务配置 ====================
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=48, help='起始token长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='M4数据集的子集')
    parser.add_argument('--inverse', action='store_true', help='是否对输出数据进行反归一化', default=False)

    # ==================== 填补任务配置 ====================
    parser.add_argument('--mask_rate', type=float, default=0.25, help='掩码比例')

    # ==================== 异常检测任务配置 ====================
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='先验异常比例(%)')

    # ==================== 模型定义配置 ====================
    parser.add_argument('--expand', type=int, default=2, help='Mamba模型的扩展因子')
    parser.add_argument('--d_conv', type=int, default=4, help='Mamba模型的卷积核大小')
    parser.add_argument('--tv_dt', type=int, default=0, help='MambaSL是否使用时间变化的dt')
    parser.add_argument('--tv_B', type=int, default=0, help='MambaSL是否使用时间变化的B')
    parser.add_argument('--tv_C', type=int, default=0, help='MambaSL是否使用时间变化的C')
    parser.add_argument('--use_D', type=int, default=0, help='MambaSL是否使用D')
    parser.add_argument('--top_k', type=int, default=5, help='TimesBlock使用的top-k值')
    parser.add_argument('--num_kernels', type=int, default=6, help='Inception模块的卷积核数量')
    parser.add_argument('--enc_in', type=int, default=7, help='编码器输入维度')
    parser.add_argument('--dec_in', type=int, default=7, help='解码器输入维度')
    parser.add_argument('--c_out', type=int, default=7, help='输出维度')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--moving_avg', type=int, default=25, help='移动平均窗口大小')
    parser.add_argument('--factor', type=int, default=1, help='注意力因子')
    parser.add_argument('--distil', action='store_false',
                        help='是否在编码器中使用蒸馏，使用此参数表示不使用蒸馏',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout概率')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码方式，可选: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='FreTS模型的通道独立性：0表示通道依赖，1表示通道独立')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='序列分解方法，仅支持moving_avg或dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='是否使用归一化；1表示是，0表示否')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='下采样层数')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='下采样窗口大小')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='下采样方法，仅支持avg、max、conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='SegRNN分段迭代的长度')

    # ==================== 优化配置 ====================
    parser.add_argument('--num_workers', type=int, default=10, help='数据加载器的工作进程数')
    parser.add_argument('--itr', type=int, default=1, help='实验重复次数')
    parser.add_argument('--train_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='训练数据的批次大小')
    parser.add_argument('--patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='优化器学习率')
    parser.add_argument('--des', type=str, default='test', help='实验描述')
    parser.add_argument('--loss', type=str, default='MSE', help='损失函数')
    parser.add_argument('--lradj', type=str, default='type1', help='学习率调整策略')
    parser.add_argument('--use_amp', action='store_true', help='是否使用自动混合精度训练', default=False)

    # ==================== GPU配置 ====================
    parser.add_argument('--use_gpu', action='store_true', default=True, help='是否使用GPU（默认：开启）')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='禁用GPU（强制使用CPU）')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='GPU类型')  # cuda或mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='是否使用多GPU', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='多GPU的设备ID列表')

    # ==================== 非平稳投影器参数 ====================
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='投影器的隐藏层维度（列表）')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='投影器的隐藏层数量')

    # ==================== 评估指标（DTW）====================
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='是否启用DTW指标（耗时较长；默认：关闭）')

    # ==================== 数据增强配置 ====================
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="数据增强倍数")
    parser.add_argument('--seed', type=int, default=2, help="随机种子")
    parser.add_argument('--jitter', default=False, action="store_true", help="抖动增强")
    parser.add_argument('--scaling', default=False, action="store_true", help="缩放增强")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="等长排列增强")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="随机长度排列增强")
    parser.add_argument('--magwarp', default=False, action="store_true", help="幅度扭曲增强")
    parser.add_argument('--timewarp', default=False, action="store_true", help="时间扭曲增强")
    parser.add_argument('--windowslice', default=False, action="store_true", help="窗口切片增强")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="窗口扭曲增强")
    parser.add_argument('--rotation', default=False, action="store_true", help="旋转增强")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER增强")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW扭曲增强")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="形状DTW扭曲增强")
    parser.add_argument('--wdba', default=False, action="store_true", help="加权DBA增强")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="判别式DTW扭曲增强")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="判别式形状DTW扭曲增强")
    parser.add_argument('--extra_tag', type=str, default="", help="额外标签")

    # ==================== TimeXer模型配置 ====================
    parser.add_argument('--patch_len', type=int, default=16, help='patch长度')

    # ==================== GCN图卷积网络配置 ====================
    parser.add_argument('--node_dim', type=int, default=10, help='每个节点嵌入的维度')
    parser.add_argument('--gcn_depth', type=int, default=2, help='GCN深度')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='GCN dropout率')
    parser.add_argument('--propalpha', type=float, default=0.3, help='传播alpha参数')
    parser.add_argument('--conv_channel', type=int, default=32, help='卷积通道数')
    parser.add_argument('--skip_channel', type=int, default=32, help='跳跃连接通道数')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear模型：是否为每个变量(通道)单独使用线性层')

    # ==================== TimeFilter模型配置 ====================
    parser.add_argument('--alpha', type=float, default=0.1, help='图构建的KNN参数')
    parser.add_argument('--top_p', type=float, default=0.5, help='MoE中的动态路由参数')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='位置编码。设置为0或1')

    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据GPU可用性设置设备
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    # 如果使用多GPU，解析设备ID列表
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # 打印实验参数
    print('Args in experiment:')
    print_args(args)

    # ==================== 根据任务名称选择对应的实验类 ====================
    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    # ==================== 执行训练或测试 ====================
    if args.is_training:
        # 训练模式：根据itr参数重复实验多次
        for ii in range(args.itr):
            # 初始化实验实例
            exp = Exp(args)  # 设置实验
            
            # 生成实验设置名称（用于保存检查点和日志）
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            
            # 针对特定模型覆盖设置名称，确保正确的检查点命名和日志记录
            if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
                setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                        + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                        + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{ii}'

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            
            # 清理GPU缓存
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        # 测试模式：仅进行测试，不训练
        exp = Exp(args)  # 设置实验
        ii = 0
        
        # 生成实验设置名称
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        
        # 针对特定模型覆盖设置名称，确保正确的检查点命名和日志记录
        if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
            setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                    + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                    + f'_tvdt{args.tv_dt}_tvB{args.tv_B}_tvC{args.tv_C}_useD{int(args.use_D)}_{args.des}_{ii}'

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        
        # 清理GPU缓存
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
