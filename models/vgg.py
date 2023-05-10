import torch.nn as nn
from .blocks import NormalizeLayer, ConvBlock, LinearBlock
from core.dataloader import set_mean_std


class VGG16(nn.Module):
    def __init__(self, act, dataset):
        super().__init__()
        mean, std = set_mean_std(dataset)
        self.norm_layer = NormalizeLayer(mean, std)
        self.layers = nn.Sequential(
            *self._build_features(act),
            *self._build_classifier(act, dataset)
        )

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.layers(x)
        return x

    def _build_features(self, act):
        return [
            *self._build_block(3, 64, 2, act),
            *self._build_block(64, 128, 2, act),
            *self._build_block(128, 256, 3, act),
            *self._build_block(256, 512, 3, act),
            *self._build_block(512, 512, 3, act)
        ]

    @staticmethod
    def _build_classifier(act, dataset):
        if dataset.lower() == 'cifar10':
            output_size, num_cls = 1, 10
        elif dataset.lower() == 'cifar100':
            output_size, num_cls = 1, 100
        elif dataset.lower() == 'imagenet':
            output_size, num_cls = 7, 1000
        else:
            raise NameError()
        return [nn.AdaptiveAvgPool2d((output_size, output_size)),
                nn.Flatten(),
                LinearBlock(512 * output_size * output_size, 4096, bn=True, act=act),
                # nn.Dropout(p=0.5),
                LinearBlock(4096, 4096, bn=True, act=act),
                # nn.Dropout(p=0.5),
                LinearBlock(4096, num_cls, bn=True, act=None),
                ]

    @staticmethod
    def _build_block(in_channel, out_channel, num_layers, act):
        layers = [ConvBlock(in_channel, out_channel, act=act)]
        layers += [ConvBlock(out_channel, out_channel, act=act) for _ in range(num_layers - 1)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return layers
