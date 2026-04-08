import copy
import random
from functools import wraps
from torchvision import models
from torchvision import transforms as T
# from contrast_weak_augment import weak_augment
# from informer_stack.model2 import informerStack_Dlstm
# from contrast.BYOL.informer.RC_PInformer import *
# from contrast.BYOL.informer.patch_informer import *
from contrast.BYOL.informer.GAF_Informer_Res import *
# from contrast.mask import mask
# helper functions
# import pandas as pd
def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    # y = y.detach()
    # print(x)
    # print(y)
    x = F.normalize(x, dim=-1, p=2) # 对某一个维度进行L2范式处理
    # print(x)
    y = F.normalize(y, dim=-1, p=2)
    # print(y)
    # print(2 - 2 * (x * y).sum(dim=-1))
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size=128):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),  # projection_size自定义的大小为256，和图片的size一致
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=64):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

# class NetWrapper(nn.Module):
#     def __init__(self, net, projection_size, projection_hidden_size, layer=-2, use_simsiam_mlp=False):
#         super().__init__()
#         self.net = net
#         self.layer = layer
#
#         self.projector = None
#         self.projection_size = projection_size
#         self.projection_hidden_size = projection_hidden_size
#
#         self.use_simsiam_mlp = use_simsiam_mlp
#
#         self.hidden = {}
#         self.hook_registered = False
#
#     def _find_layer(self):
#         if type(self.layer) == str:
#             modules = dict([*self.net.named_modules()])  # 如果layer名称是字符串，则使用named_modules()取出ResNet中所有的层
#             return modules.get(self.layer, None)
#         elif type(self.layer) == int:
#             children = [*self.net.children()]  # children()只取出子层，子层下包含的层不会取出
#             return children[self.layer]
#         return None
#
#     def _hook(self, _, input, output):
#         device = input[0].device
#         self.hidden[device] = flatten(output)
#
#     def _register_hook(self):
#         layer = self._find_layer()
#         assert layer is not None, f'hidden layer ({self.layer}) not found'
#         handle = layer.register_forward_hook(self._hook)  # register_forward_hook()是在不改动网络结构的情况下获取网络中间层输出
#         self.hook_registered = True
#
#     @singleton('projector')
#     def _get_projector(self, hidden):
#         _, dim = hidden.shape
#         create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
#         projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size)
#         return projector.to(hidden)
#
#     def get_representation(self, x):
#         if self.layer == -1:
#             return self.net(x)
#
#         if not self.hook_registered:
#             self._register_hook()  # 获取ResNet中间层，即进行1000类别分类前的网络（此时的网络输出为2048）
#
#         self.hidden.clear()  # clear()函数：清除hidden列表中的所有元素
#         _ = self.net(x)      # self.net()函数是指将输入数据进行编码的编码器
#         # print('encoder output:', _.shape)
#         hidden = self.hidden[x.device]
#         # print('flatten后的encoder output:', hidden.shape)
#         self.hidden.clear()
#         assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
#         return hidden    # 得到了编码后摊平的输出
#
#     def forward(self, x, return_projection = True):
#         representation = self.get_representation(x)
#
#         if not return_projection:
#             return representation
#
#         projector = self._get_projector(representation)
#         projection = projector(representation)  # 上面两句话表示映射层选择MLP还是simsamMLP
#         return projection, representation

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        # net,
        image_size,
        hidden_layer = -2,
        projection_size = 21,
        projection_hidden_size=64,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True
    ):
        super().__init__()
        # self.net = net

        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        # self.augment1 = default(augment_fn, DEFAULT_AUG)
        # self.augment2 = default(augment_fn2, self.augment1)

        # self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=not use_momentum)
        # self.online_encoder = nn.LSTM(input_size=21, hidden_size=21, num_layers=2)
        # self.online_encoder = RNN(factors=21, batch_size=100, device='cuda', drop_ratio=0.)
        # self.online_encoder = LSTM(factors=23, batch_size=500, device="cuda")
        # self.online_encoder = informerStack_Dlstm(dropout=0.4)
        self.online_encoder = informerStack_ResCNN(23)
        # self.online_encoder = informerStack_Dlstm(c_out=2, out_len=1, input_size=23, hidden_size=64, dropout=0.1)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)  # EMA这种方式，可以有效保持两个网络是不一样的

        # self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        # self.online_predictor = SimSiamMLP(23, 23, projection_hidden_size)
        # self.online_predictor = MLP(1472, 1472, 512)
        # self.online_predictor = MLP(1344, 1344, 512)
        self.online_predictor = MLP(4608, 4608, 1024)
        # self.online_predictor = MLP(3872, 3872, 1024)
        # self.online_predictor = MLP(23, 23, 64)

        # get device of network and make wrapper same device
        # device = get_module_device(net)
        # self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        # self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        # target_encoder = copy.deepcopy(self.online_predictor)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x, y, z,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            # print(11)
            return self.online_encoder(x, return_projection=return_projection)
        # print(x, y)

        # image_one, image_two = weak_augment(pd.DataFrame(x.numpy()), 1.1), weak_augment(pd.DataFrame(x.numpy()), 7.5)
        # print(image_one.shape, image_two.shape)
        # y = mask(y, ratios=0.1)
        # online_proj_one, _ = self.online_encoder(x.unsqueeze(1))
        # online_proj_two, _ = self.online_encoder(y.unsqueeze(1))
        online_proj_one = self.online_encoder(x)
        online_proj_two = self.online_encoder(y)
        # print(online_proj_one, online_proj_two)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
        # online_pred_one = self.online_predictor(online_proj_one.squeeze(0))
        # online_pred_two = self.online_predictor(online_proj_two.squeeze(0))
        # print(online_pred_one, online_pred_two)
        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            # target_proj_one, _ = target_encoder(x.unsqueeze(1))
            # target_proj_two, _ = target_encoder(y.unsqueeze(1))
            # target_proj_one = target_encoder(x)
            # target_proj_two = target_encoder(y)
            target_proj_three = target_encoder(z)
            # target_proj_three = target_encoder(z.squeeze(0))
            # print(target_proj_one, target_proj_two)
            # target_proj_one.detach()
            # target_proj_two.detach()
            target_proj_three.detach()

        # loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        # loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        loss_one = loss_fn(online_pred_one, target_proj_three.detach())
        loss_two = loss_fn(online_pred_two, target_proj_three.detach())

        loss = loss_one + loss_two
        return loss.mean()


if __name__ == '__main__':
    resnet = models.resnet50(pretrained=True)

    learner = BYOL(
        resnet,
        image_size=256,
        hidden_layer=-1
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)


    def sample_unlabelled_images():
        # return torch.randn(20, 3, 256, 256)
        return torch.randn(400, 21)


    for _ in range(100):
        images = sample_unlabelled_images()
        loss = learner(images)
        print(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()
