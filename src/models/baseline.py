import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class Baseline(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Baseline, self).__init__()

        # Load ResNet-50 backbone
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            backbone = resnet50(weights=weights)
        else:
            backbone = resnet50(weights=None)

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Global Average Pool
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 2048-d embedding
        self.embedding = nn.Linear(2048, 2048)

        # BNNeck
        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)

        # Classifier (for training only)
        self.classifier = nn.Linear(2048, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.embedding.weight, mode='fan_out')
        nn.init.constant_(self.embedding.bias, 0)

        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x, return_embedding=False):
        x = self.backbone(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        feat = self.embedding(x)

        bn_feat = self.bnneck(feat)

        if return_embedding:
            # L2 normalized embedding for inference
            return F.normalize(bn_feat, p=2, dim=1)

        logits = self.classifier(bn_feat)

        return logits, feat
