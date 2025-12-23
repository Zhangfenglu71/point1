# models/radar_cls_resnet.py
import torch.nn as nn
import torchvision.models as models


class RadarResNet18(nn.Module):
    """
    简单的 ResNet-18 动作分类器：
      - 输入: (B, 3, H, W)，这里 H=W=120
      - 输出: (B, num_classes)，默认 4 类 [box, jump, run, walk]
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        # 不用预训练，避免联网下载权重
        backbone = models.resnet18(weights=None)

        # 原本 conv1 就是 3 通道，这里只是显式写一下
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
