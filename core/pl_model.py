import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
from core.prune import EntropyHook
from core.dataloader import set_dataloader
import torch
import math
from models import build_model
from core.utils import accuracy, init_optimizer, init_scheduler, set_gamma
from torchvision import transforms
from models.blocks import ConvBlock, LinearBlock
import wandb
from argparse import Namespace


class BaseModel(pl.LightningModule):
    def __init__(self, args: Namespace):
        # model, dataset, batch_size, num_workers, act, optimizer, lr, lr_scheduler):
        """
        init base trainer
        """
        super().__init__()
        self.args = args
        self.model = build_model(args.net, args.act, args.dataset)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.train_loader, self.val_loader = set_dataloader(
            args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        num_step = self.trainer.max_epochs * len(self.train_loader)
        optimizer = init_optimizer(self.model, self.args.optimizer, lr=self.args.lr)

        lr_scheduler = init_scheduler(self.args.lr, self.args.lr_scheduler, num_step, optimizer=optimizer)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        top1, top5 = accuracy(outputs, labels)

        self.log('train/loss', loss, sync_dist=True)
        self.log('train/top1', top1, sync_dist=True)
        self.log('lr', self.lr, sync_dist=True)
        return loss

    @property
    def lr(self):
        return self.optimizers().optimizer.param_groups[0]['lr']

    def validation_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        pred = self.model(images)
        top1, top5 = accuracy(pred, labels)
        loss = self.loss_function(pred, labels)
        self.log('val/loss', loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val/top1', top1, sync_dist=True, on_step=False, on_epoch=True)
        return

    def valid_blocks(self):
        for name, block in self.named_modules():
            if self.check_valid(block):
                yield name, block

    def check_valid(self, block):
        return block != self.model.layers[-1] and isinstance(block, (ConvBlock, LinearBlock))

    def save_model(self):
        exp = getattr(self.logger, "experiment")
        torch.save(self.model, os.path.join(exp.dir, "model.pth"))


class PruneModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_hook = EntropyHook(self, set_gamma(args.act), ratio=0.25)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            prune_milestone = [i for i in range(0, self.trainer.max_epochs - self.args.fine_tune, self.args.skip)]
            prune_dict = {num_epoch: self.args.amount * (i + 1) / len(prune_milestone) for i, num_epoch
                          in enumerate(prune_milestone)}
            setattr(self, 'prune_dict', prune_dict)

    def on_train_start(self) -> None:
        for name, block in self.valid_blocks():
            block.create_mask()

    def on_after_backward(self) -> None:
        for name, block in self.valid_blocks():
            block.clean_grad()
        # for name, parameters in self.named_parameters():
        #     parameters.grad[ == 0] = 0

    def on_train_epoch_start(self) -> None:
        if self.current_epoch in self.prune_dict.keys():
            self.model_hook.set_up()

    def on_train_epoch_end(self) -> None:
        info = {'step': self.global_step, 'epoch': self.current_epoch}
        if self.current_epoch in self.prune_dict.keys():
            global_entropy = self.model_hook.retrieve()

            # compute and apply mask
            for name, block in self.valid_blocks():
                compute_im_score(block, global_entropy[name], self.args.method)

            adjusted_amount = compute_amount(global_entropy, self.args.amount_setting)
            for i, (name, block) in enumerate(self.valid_blocks()):
                block.compute_mask(adjusted_amount[name] * self.prune_dict[self.current_epoch])
                info['sparsity/layer_{0}'.format(i)] = block.sparsity()
            info['sparsity/global'] = self.global_sparsity
            wandb.log(info)
        self.model_hook.remove()

    @property
    def global_sparsity(self):
        n_pruned = 0
        n_element = 0
        for i, (name, block) in enumerate(self.valid_blocks()):
            n_pruned += block.num_pruned()
            n_element += block.num_element()
        return n_pruned / n_element

    def register_mask(self, block):
        pass


def compute_im_score(block, entropy, method):
    if isinstance(block, (LinearBlock, ConvBlock)):
        num_dim = len(entropy['act'].shape)  # num of dimensions
        channel_entropy = entropy['act'].mean(tuple(range(1, num_dim)))
        lt_weights = getattr(block.LT, 'weight').detach()
        lt_im_score = compute_importance(lt_weights, channel_entropy, method)
        block.LT.register_buffer('_im_score', lt_im_score)

        bn_weights = getattr(block.BN, 'weight').detach()
        bn_im_score = compute_importance(bn_weights, channel_entropy, method)
        block.BN.register_buffer('_im_score', bn_im_score)
    return


def compute_amount(global_entropy, amount_setting):
    total_score = 0
    allocation = {}
    if amount_setting == 0:
        for block_name in global_entropy.keys():
            allocation[block_name] = 1
    else:
        for block_name, block_entropy in global_entropy.items():
            layer_avg = sum([layer_entropy.mean() for layer_entropy in block_entropy.values()])
            total_score += math.sqrt(layer_avg)
        for block_name, block_entropy in global_entropy.items():
            layer_avg = sum([layer_entropy.mean() for layer_entropy in block_entropy.values()])
            allocation[block_name] = total_score / len(global_entropy) / math.sqrt(layer_avg)
    return allocation


def compute_importance(weight, channel_entropy, eta):
    """
    Compute the importance score based on weight and entropy of a channel
    :param weight:  Weight of the module, shape as:
                    ConvBlock: in_channels * out_channels * kernel_size_1 * kernel_size_2
                    LinearBlock: in_channels * out_channels
    :param channel_entropy: The averaged entropy of each channel, shape as in_channels * 1 * (1 * 1)
    :param eta: the importance of entropy in pruning,
                0:      prune by weight
                1:      prune by channel_entropy
                2:      prune by weight * entropy
                3:
                else:   eta * channel_entropy * weight
    :return:    The importance_scores
    """
    assert weight.shape[0] == channel_entropy.shape[0] and channel_entropy.ndim == 1
    e_new_shape = (-1,) + (1,) * (weight.dim() - 1)
    channel_entropy = torch.tensor(channel_entropy).view(e_new_shape).cuda()
    if eta == 0:
        importance_scores = weight
    elif eta == 1:
        importance_scores = channel_entropy * torch.ones_like(weight)
    elif eta == 2:
        importance_scores = channel_entropy * weight
    elif eta == 3:
        importance_scores = 1 / (1 / (channel_entropy + 1e-8) + 1 / (weight + 1e-4))
    elif eta == 4:
        normed_entropy = (channel_entropy - channel_entropy.mean()) / channel_entropy.std()
        normed_weight = (weight - weight.mean()) / weight.std()
        importance_scores = normed_entropy * normed_weight
    elif eta == 5:
        normed_entropy = (channel_entropy - channel_entropy.mean()) / channel_entropy.std()
        normed_weight = (weight - weight.mean()) / weight.std()
        importance_scores = normed_entropy + normed_weight
    else:
        raise ValueError()
    return importance_scores
