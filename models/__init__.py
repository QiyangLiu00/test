from torchvision import models

from .FPNCL import FPNCL
from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18, resnet50, SwinTransformer, resnet18_cifar_variant1, resnet50_fpn_backbone


def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")
    # if backbone == 'resnet50_fpn_backbone':

    # castrate参数表示是否对模型进行修改。
    # 如果castrate为True，
    # 则将模型的fc层替换为一个torch.nn.Identity()层，即一个什么也不做的层，
    # 并将模型的输出维度设置为fc层的输入维度，即backbone.fc.in_features。
    # 最后返回修改后的模型对象。
    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg):
    if model_cfg.name == 'simsiam':
        model = SimSiam(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'FPNCL':
        model = FPNCL(resnet50_fpn_backbone())
        weights_path = 'F:/LQY/workspace/SimSiam/models/backbones/retinanet_resnet50_fpn_coco-eeacb38b.pth'
        # assert os.path.exists(weights_path), "{} is not exist.".format(weights_path)
        weights = torch.load(weights_path)  # 读取预训练模型权重
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in weights.items() if k in model_dict and (v.shape == model_dict[k].shape)}
        model.load_state_dict(weights, strict=False)
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model
