from pytorch_lightning.callbacks import ModelCheckpoint, ModelPruning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import os
from core.pl_model import BaseModel, PruneModel
import torch
from config import *
from settings import TrainParser
import pytorch_lightning as pl

if __name__ == '__main__':
    args = TrainParser().get_args()
    model_dir = os.path.join(MODEL_PATH, args.dataset, args.net, args.project)
    os.makedirs(model_dir, exist_ok=True)

    logtool = WandbLogger(name=args.name, save_dir=model_dir, project=args.project,
                          config=args)

    if args.skip == 0:
        model = BaseModel(args)
    else:
        model = PruneModel(args)

    callbacks = [
        ModelCheckpoint(monitor='val/top1', save_top_k=1, mode="max", save_on_train_epoch_end=False,
                        dirpath=logtool.experiment.dir, filename="best"),
    ]

    trainer = pl.Trainer(devices="auto",
                         precision=32,
                         amp_backend="native",
                         accelerator="cuda",
                         strategy='dp',
                         callbacks=callbacks,
                         max_epochs=args.num_epoch,
                         logger=logtool,
                         enable_progress_bar=args.npbar,
                         inference_mode=False,
                         )
    trainer.fit(model)
    model.save_model()
