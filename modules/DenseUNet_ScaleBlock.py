import torch
import torch.nn as nn
import torch.nn.functional as F


class Scale(nn.Module):
    '''DenseNet中用于BatchNormalization的自定义层。

    学习一组用于缩放输入数据的权重和偏置。
    输出简单地由输入的逐元素乘法和一组常数的总和组成:

        out = in * gamma + beta,

    其中'gamma'和'beta'是学到的权重和偏置。

    # 参数
        axis: 整数，模式0中要沿其标准化的轴。例如，
            如果您的输入张量的形状为(samples, channels, rows, cols)，
            将轴设置为1以按特征图(channels轴)进行标准化。
        momentum: 在计算数据的均值和标准差的
            指数平均值时的动量，用于特征方向的标准化。
        weights: 初始化权重。
            2个Numpy数组的列表，形状为:
            `[(input_shape,), (input_shape,)]`
        beta_init: 偏移参数的初始化函数名称
            或者，用于权重初始化的Theano/TensorFlow函数。
            仅当您不传递`weights`参数时，此参数才相关。
        gamma_init: 缩放参数的初始化函数名称
            或者，用于权重初始化的Theano/TensorFlow函数。
            仅当您不传递`weights`参数时，此参数才相关。
    '''

    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero', gamma_init='one'):
        super(Scale, self).__init__()
        self.momentum = momentum
        self.axis = axis
        self.beta_init = self._get_initializer(beta_init)
        self.gamma_init = self._get_initializer(gamma_init)
        self.initial_weights = weights
        self.weights_initialized = False


    def _get_initializer(self, init_name):
        # 返回PyTorch的初始化方法
        if init_name == 'zero':
            return nn.init.zeros_
        elif init_name == 'one':
            return nn.init.ones_
        # 在此处添加其他初始化方法
        else:
            raise ValueError(f"Unknown initializer: {init_name}")

    def build(self, input_shape):
        shape = (int(input_shape[self.axis]),)
        self.gamma = nn.Parameter(self.gamma_init(torch.empty(shape)))
        self.beta = nn.Parameter(self.beta_init(torch.empty(shape)))

        if self.initial_weights is not None:
            assert len(self.initial_weights) == 2, "Expected initial_weights to be a list of length 2"
            self.gamma.data = torch.tensor(self.initial_weights[0], dtype=torch.float32)
            self.beta.data = torch.tensor(self.initial_weights[1], dtype=torch.float32)

    def forward(self, x):
        if not self.weights_initialized:
            shape = (x.shape[self.axis],)
            self.gamma = nn.Parameter(self.gamma_init(shape))
            self.beta = nn.Parameter(self.beta_init(shape))
            if self.initial_weights is not None:
                assert len(self.initial_weights) == 2, "Expected initial_weights to be a list of length 2"
                self.gamma.data = torch.tensor(self.initial_weights[0], dtype=torch.float32)
                self.beta.data = torch.tensor(self.initial_weights[1], dtype=torch.float32)
            self.weights_initialized = True
        input_shape = x.shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = self.gamma.view(broadcast_shape) * x + self.beta.view(broadcast_shape)
        return out

    def get_config(self):
        # 获取模型的配置
        config = {"momentum": self.momentum, "axis": self.axis}
        return config
