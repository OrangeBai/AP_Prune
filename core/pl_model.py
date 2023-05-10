import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
from core.prune import EntropyHook
from core.dataloader import set_dataloader
import torch
from models import build_model
from core.utils import accuracy, init_optimizer, init_scheduler, set_gamma
from torchvision import transforms
from models.blocks import ConvBlock, LinearBlock


class BaseModel(pl.LightningModule):
    def __init__(self, model, dataset, batch_size, num_workers, act, optimizer, lr, lr_scheduler):
        """
        init base trainer
        """
        super().__init__()
        self.model = build_model(model, act, dataset)
        self.optimizer = optimizer
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.train_loader, self.val_loader = set_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        num_step = self.trainer.max_epochs * len(self.train_loader) / self.trainer.accumulate_grad_batches
        optimizer = init_optimizer(self.model, self.optimizer, lr=self.lr)

        lr_scheduler = init_scheduler(self.lr, self.lr_scheduler, num_step, optimizer=optimizer)
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
    def __init__(self, model, dataset, batch_size, num_workers, act, optimizer, lr, lr_scheduler, method,
                 skip, amount):
        super().__init__(model, dataset, batch_size, num_workers, act, optimizer, lr, lr_scheduler)
        self.model_hook = EntropyHook(self, set_gamma(act), ratio=0.25)
        self.method = method
        self.amount = amount
        self.skip = skip

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            setattr(self, 'prune_milestones', [i for i in range(0, self.trainer.max_epochs - 1, self.skip)])

    def on_train_start(self) -> None:
        for name, block in self.valid_blocks():
            block.create_mask()

    def on_after_backward(self) -> None:
        # for name, block in self.valid_blocks():
        #     block.clean_grad()
        for name, parameters in self.named_parameters():
            parameters.grad[parameters == 0] = 0

    def on_train_epoch_start(self) -> None:
        if self.current_epoch in self.prune_milestones:
            self.model_hook.set_up()

    def on_train_epoch_end(self) -> None:
        if self.current_epoch in self.prune_milestones:
            global_entropy = self.model_hook.retrieve()

            # compute and apply mask
            for name, block in self.valid_blocks():
                compute_im_score(block, global_entropy[name], self.method)

            computer_sparsity()
            adjusted_amount = compute_amount(global_entropy)
            for i, (name, block) in enumerate(self.valid_blocks()):
                block.compute_mask(adjusted_amount[name] * self.amount / len(self.prune_milestones))
                print(f"sparsity of block is {block.sparsity:.2f}")

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


def compute_amount(global_entropy):
    total_score = 0
    allocation = {}
    for block_name, block_entropy in global_entropy.items():
        layer_avg = sum([layer_entropy.mean() for layer_entropy in block_entropy.values()])
        total_score += 1 / layer_avg
    for block_name, block_entropy in global_entropy.items():
        layer_avg = sum([layer_entropy.mean() for layer_entropy in block_entropy.values()])
        allocation[block_name] = (1 / layer_avg) / (total_score / len(global_entropy))
    return allocation


def computer_sparsity():
    pass


def compute_mask(block, entropy, amount):
    pass


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
