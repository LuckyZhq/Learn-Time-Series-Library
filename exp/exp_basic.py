import os
import torch
import importlib
import pkgutil


# 只需要把模型文件放到 models/ 文件夹下
# 例如: models/Transformer.py, models/LSTM.py 等
# 所有模型都会被自动扫描，并且可以通过模型名直接调用


class Exp_Basic(object):
    """
    实验基类

    主要功能：
    1. 保存参数 args
    2. 自动扫描 models 目录，建立模型名称到模块路径的映射
    3. 初始化懒加载模型字典
    4. 获取运行设备（CPU / CUDA / MPS）
    5. 构建模型并将其移动到指定设备
    """

    def __init__(self, args):
        self.args = args

        # -------------------------------------------------------
        # 自动生成模型映射表
        # -------------------------------------------------------
        model_map = self._scan_models_directory()

        # 使用支持懒加载的字典
        # 只有真正访问某个模型时，才会动态 import 该模型模块
        self.model_dict = LazyModelDict(model_map)

        # 获取设备
        self.device = self._acquire_device()

        # 构建模型并放到对应设备上
        self.model = self._build_model().to(self.device)

    def _scan_models_directory(self):
        """
        自动扫描 models 文件夹下所有 .py 文件

        返回:
            model_map: dict
                例如:
                {
                    'Transformer': 'models.Transformer',
                    'LSTM': 'models.LSTM'
                }
        """
        model_map = {}
        models_dir = 'models'

        # 遍历 models 目录下的所有文件
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                # 忽略 __init__.py 和非 .py 文件
                if filename.endswith('.py') and filename != '__init__.py':
                    # 去掉 .py 后缀，得到模块名
                    module_name = filename[:-3]

                    # 拼接完整导入路径
                    full_path = f"{models_dir}.{module_name}"

                    # 构建映射字典
                    # 例如 {'Transformer': 'models.Transformer'}
                    model_map[module_name] = full_path

        return model_map

    def _build_model(self):
        """
        构建模型

        该方法需要在子类中重写。
        """
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """
        获取运行设备

        支持：
        1. CUDA GPU
        2. Apple MPS
        3. CPU

        返回:
            device: torch.device
        """
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            # 设置可见 GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices

            # 指定当前使用的 CUDA 设备
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))

        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            # Apple Silicon GPU 加速
            device = torch.device('mps')
            print('Use GPU: mps')

        else:
            # 默认使用 CPU
            device = torch.device('cpu')
            print('Use CPU')

        return device

    def _get_data(self):
        """
        获取数据
        需要在子类中实现
        """
        pass

    def vali(self):
        """
        验证过程
        需要在子类中实现
        """
        pass

    def train(self):
        """
        训练过程
        需要在子类中实现
        """
        pass

    def test(self):
        """
        测试过程
        需要在子类中实现
        """
        pass


class LazyModelDict(dict):
    """
    智能懒加载字典

    功能：
    - 不在一开始导入所有模型
    - 当通过 key 访问某个模型时，才动态导入对应模块
    - 导入后缓存到字典中，避免重复导入

    例如：
        self.model_dict['Transformer']
    第一次访问时会自动 import models.Transformer
    """

    def __init__(self, model_map):
        # model_map 保存 模型名 -> 模块路径 的映射关系
        self.model_map = model_map
        super().__init__()

    def __getitem__(self, key):
        """
        根据模型名获取模型类

        参数:
            key: str
                模型名称，例如 'Transformer'

        返回:
            model_class: class
                对应的模型类
        """
        # 如果已经加载过，直接返回缓存结果
        if key in self:
            return super().__getitem__(key)

        # 如果模型名不存在于扫描结果中，则报错
        if key not in self.model_map:
            raise NotImplementedError(f"Model [{key}] not found in 'models' directory.")

        # 获取模块导入路径
        module_path = self.model_map[key]

        try:
            # 动态导入模块
            print(f"🚀 Lazy Loading: {key} ...")
            module = importlib.import_module(module_path)
        except ImportError as e:
            # 如果导入失败，提示可能缺少依赖
            print(f"❌ Error: Failed to import model [{key}]. Dependencies missing?")
            raise e

        # 优先查找标准类名 Model
        if hasattr(module, 'Model'):
            model_class = module.Model
        # 如果没有 Model，则尝试查找和文件名同名的类
        elif hasattr(module, key):
            model_class = getattr(module, key)
        else:
            # 如果都没有找到，说明模块结构不符合约定
            raise AttributeError(f"Module {module_path} has no class 'Model' or '{key}'")

        # 将加载后的模型类缓存到字典中
        self[key] = model_class

        return model_class