# 导入各种数据集加载器
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
# 导入UEA数据集的collate函数（用于处理变长序列）
from data_provider.uea import collate_fn
# 导入PyTorch数据加载器
from torch.utils.data import DataLoader

# 数据集名称到数据集类的映射字典
# 键：数据集名称（命令行参数）
# 值：对应的数据集类
data_dict = {
    'ETTh1': Dataset_ETT_hour,      # ETT小时级数据集1
    'ETTh2': Dataset_ETT_hour,      # ETT小时级数据集2
    'ETTm1': Dataset_ETT_minute,    # ETT分钟级数据集1
    'ETTm2': Dataset_ETT_minute,    # ETT分钟级数据集2
    'custom': Dataset_Custom,       # 自定义数据集
    'm4': Dataset_M4,               # M4竞赛数据集
    'PSM': PSMSegLoader,            # PSM异常检测数据集
    'MSL': MSLSegLoader,            # MSL异常检测数据集
    'SMAP': SMAPSegLoader,          # SMAP异常检测数据集
    'SMD': SMDSegLoader,            # SMD异常检测数据集
    'SWAT': SWATSegLoader,          # SWAT异常检测数据集
    'UEA': UEAloader                # UEA分类数据集
}


def data_provider(args, flag):
    """
    数据提供者函数：根据任务类型和数据集名称创建相应的DataLoader
    
    参数:
        args: 命令行参数对象，包含所有配置信息
        flag: 数据集标志，'train'、'val'或'test'，用于区分训练集、验证集和测试集
    
    返回:
        data_set: 数据集对象
        data_loader: PyTorch DataLoader对象
    """
    # 根据args.data从字典中获取对应的数据集类
    Data = data_dict[args.data]
    
    # 时间特征编码方式：0表示不使用时间特征，1表示使用时间特征编码
    timeenc = 0 if args.embed != 'timeF' else 1

    # 测试集不打乱顺序，训练集和验证集打乱顺序
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False  # 是否丢弃最后一个不完整的batch
    batch_size = args.batch_size  # 批次大小
    freq = args.freq  # 时间频率

    # ==================== 异常检测任务 ====================
    if args.task_name == 'anomaly_detection':
        drop_last = False  # 异常检测不丢弃最后一个batch
        # 创建异常检测数据集
        data_set = Data(
            args = args,
            root_path=args.root_path,   # 数据根路径
            win_size=args.seq_len,      # 窗口大小（序列长度）
            flag=flag,                  # 数据集标志（train/val/test）
        )
        print(flag, len(data_set))      # 打印数据集大小
        
        # 创建DataLoader
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,      # 批次大小
            shuffle=shuffle_flag,       # 是否打乱数据
            num_workers=args.num_workers,  # 数据加载的工作进程数
            drop_last=drop_last)        # 是否丢弃最后一个不完整的batch
        return data_set, data_loader
    # ==================== 分类任务 ====================
    elif args.task_name == 'classification':
        drop_last = False  # 分类任务不丢弃最后一个batch
        # 创建分类数据集
        data_set = Data(
            args = args,
            root_path=args.root_path,   # 数据根路径
            flag=flag,                  # 数据集标志（train/val/test）
        )

        # 创建DataLoader，使用自定义的collate_fn处理变长序列
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,      # 批次大小
            shuffle=shuffle_flag,       # 是否打乱数据
            num_workers=args.num_workers,  # 数据加载的工作进程数
            drop_last=drop_last,        # 是否丢弃最后一个不完整的batch
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)  # 自定义collate函数，填充到最大长度
        )
        return data_set, data_loader
    # ==================== 其他任务（预测、填补等）====================
    else:
        # M4数据集不丢弃最后一个batch
        if args.data == 'm4':
            drop_last = False
        
        # 创建数据集对象
        data_set = Data(
            args = args,
            root_path=args.root_path,       # 数据根路径
            data_path=args.data_path,       # 数据文件路径
            flag=flag,                      # 数据集标志（train/val/test）
            size=[args.seq_len, args.label_len, args.pred_len],  # [输入序列长度, 标签长度, 预测长度]
            features=args.features,         # 特征类型（M/S/MS）
            target=args.target,             # 目标变量
            timeenc=timeenc,                # 时间特征编码方式
            freq=freq,                      # 时间频率
            seasonal_patterns=args.seasonal_patterns  # 季节性模式（用于M4）
        )
        print(flag, len(data_set))          # 打印数据集大小
        
        # 创建DataLoader
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,          # 批次大小
            shuffle=shuffle_flag,           # 是否打乱数据
            num_workers=args.num_workers,   # 数据加载的工作进程数
            drop_last=drop_last)            # 是否丢弃最后一个不完整的batch
        return data_set, data_loader
