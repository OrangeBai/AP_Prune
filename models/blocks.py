import torch
from torch import nn as nn, Tensor
from core.utils import set_activation, set_gamma
import numpy as np
from torch.nn import init


class BaseBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weight(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    init.normal_(module.bias.data)
            elif isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    init.normal_(module.bias.data)

    def create_mask(self):
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                module.register_buffer('_mask', torch.ones_like(module.weight))

    def clean_grad(self):
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                device = module.weight.grad.data.device
                mask = module.get_buffer('_mask')
                grad_tensor = module.weight.grad.data
                grad_tensor[mask == 0] = 0
                module.weight.grad.data = grad_tensor.detach().clone().to(device)

    def compute_mask(self, amount):
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                assert hasattr(module, '_im_score') and hasattr(module, '_mask')
                im_score = module.get_buffer('_im_score')
                mask = module.get_buffer('_mask')
                mask[im_score == 0] = 0
                alive = im_score[mask != 0]

                threshold = torch.quantile(alive.abs(), amount / 100, interpolation='linear')
                # indices of units less than threshold
                less_idx = abs(im_score) < threshold
                # how many units remains
                # all_equal_indices = np.where(abs(im_score) == threshold)
                # selected = np.random.choice(len(all_equal_indices[0]), equal_to, replace=False)

                mask[less_idx] = 0
                module.register_buffer('_mask', mask)
                tensor = module.weight.data.cpu().numpy()
                device = module.weight.data.device
                # Apply new weight and mask
                module.weight.data = torch.from_numpy(tensor * mask.cpu().numpy()).to(device)

    @property
    def sparsity(self):
        alive = 0
        num_element = 0
        for name, module in self.named_modules():
            if hasattr(module, '_mask'):
                alive += getattr(module, '_mask').sum()
                num_element += getattr(module, '_mask').nelement()
        return (1 - alive / num_element).cpu().numpy() * 100

    @property
    def num_element(self):
        num_element = 0
        for name, module in self.named_modules():
            if hasattr(module, '_mask'):
                num_element += getattr(module, '_mask').nelement()
        return num_element.cpu().numpy()


class LinearBlock(BaseBlock):
    def __init__(self, in_channels, out_channels, bn=True, act='relu'):
        super().__init__()

        self.LT = nn.Linear(in_channels, out_channels)
        self.BN = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.Act = set_activation(act)
        self.init_weight()

    def forward(self, x):
        x = self.LT(x)
        x = self.BN(x)
        x = self.Act(x)
        return x


class ConvBlock(BaseBlock):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, act='relu',
                 **kwargs):
        super().__init__()
        self.LT = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=padding, stride=stride, bias=False)
        self.BN = nn.BatchNorm2d(out_channels)
        self.Act = set_activation(act)
        self.init_weight()

    def forward(self, x):
        x = self.LT(x)
        x = self.BN(x)
        x = self.Act(x)
        return x


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, x: torch.tensor):
        device = x.device
        (batch_size, num_channels, height, width) = x.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(device)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(device)
        return (x - means) / sds

    # def save(self, ):
    #     torch.save(self.model)
