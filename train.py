import glob
import os

import certifi
import hydra
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from transformers import AutoTokenizer

from data import get_dataloaders
from SCISOR.faesm import FAESM_Base
from SCISOR.shortening_scud import ShorteningSCUD

os.environ["SSL_CERT_FILE"] = certifi.where()


@hydra.main(version_base=None, config_path="configs", config_name="basic")
def train(cfg: DictConfig) -> None:
    @rank_zero_only
    def init_wandb():
        wandb.login()
        wandb.init(config=OmegaConf.to_container(cfg))

    init_wandb()
    ##### Load data
    pl.seed_everything(cfg.model.seed, workers=True)
    mp.set_start_method("spawn", force=True)
    print("Getting dataloaders.")
    train_dataloader, test_dataloader = get_dataloaders(cfg)
    tokenizer = (
        train_dataloader.tokenizer if hasattr(train_dataloader, "tokenizer") else None
    )

    ##### Setup x0_model
    print("Setting up model.")

    print(cfg)
    ckpt_dir = cfg.train.checkpoint_dir + "/" if "checkpoint_dir" in cfg.train else ""

    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    ##### Pick model
    if not cfg.model.restart:
        nn_params = cfg.architecture.nn_params
        nn_params = (
            OmegaConf.to_container(nn_params, resolve=True)
            if nn_params is not None
            else {}
        )

        model = ShorteningSCUD(
            FAESM_Base,
            nn_params,
            num_classes=len(tokenizer) if tokenizer else cfg.data.N,
            forward_kwargs=OmegaConf.to_container(
                cfg.model.forward_kwargs, resolve=True
            ),
            gen_trans_step=cfg.sampling.gen_trans_step,
            t_max=cfg.model.t_max,
            seed=cfg.model.seed,
            tokenizer=AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D"),
            **OmegaConf.to_container(cfg.train, resolve=True),
        )
        ckpt_path = None
    else:
        ckpt_path = f"{ckpt_dir}checkpoints/{cfg.model.restart}"
        ckpt_path = max(
            glob.glob(os.path.join(ckpt_path, "*.ckpt")), key=os.path.getmtime
        )
        model = ShorteningSCUD.load_from_checkpoint(ckpt_path)

    # Set up model
    model.p0 = torch.load("p0.pt")
    rate = 1 / cfg.model.forward_kwargs.alpha_rate
    model.alpha = lambda t: (1 - t) ** rate
    model.beta = lambda t: rate / (1 - t)

    ##### Train
    # wandb.init()
    wandb_logger = WandbLogger(project="debugging")
    lightning_model = model
    torch.set_float32_matmul_precision("high")

    @rank_zero_only
    def update_wandb_config():
        wandb.config.update(lightning_model.hparams)

    update_wandb_config()

    val_check_interval = 2 * (210000 // cfg.train.batch_size)

    expected_batch_size = {"uniref50": 256, "uniref90": 512}
    assert (
        expected_batch_size[cfg.data.data]
        == cfg.train.batch_size * cfg.train.accumulate
    )

    num_devices = torch.cuda.device_count()
    trainer = Trainer(
        max_epochs=cfg.train.n_epoch,
        accelerator="auto",
        devices=num_devices,
        logger=wandb_logger,
        strategy=DDPStrategy(broadcast_buffers=True, find_unused_parameters=True),
        callbacks=[
            ModelCheckpoint(
                dirpath=f"{ckpt_dir}checkpoints/{wandb_logger.experiment.name}",
                save_on_train_epoch_end=False,
                save_top_k=3,
                monitor="val_l01_epoch",
            )
        ],
        val_check_interval=cfg.train.val_check_interval
        if "val_check_interval" in cfg.train
        else val_check_interval,
        limit_val_batches=cfg.train.limit_val_batches
        if "limit_val_batches" in cfg.train
        else 1.0,
        limit_train_batches=cfg.train.limit_train_batches
        if "limit_train_batches" in cfg.train
        else 1.0,
        accumulate_grad_batches=cfg.train.accumulate // num_devices,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(lightning_model, train_dataloader, test_dataloader, ckpt_path=ckpt_path)
    wandb.finish()


if __name__ == "__main__":
    train()
