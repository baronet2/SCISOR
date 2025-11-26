import tempfile

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from PIL import Image
from pytorch_lightning.utilities import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid


def get_gif(sample_x, sample_a, model, gen_trans_step, batch_size):
    # save images
    p = model.get_stationary()
    samples = torch.multinomial(
        p, num_samples=batch_size * sample_x.shape[1:].numel(), replacement=True
    )
    init_noise = samples.reshape((batch_size,) + sample_x.shape[1:]).to(sample_x.device)
    if sample_a is not None:
        attn_mask = sample_a.repeat(batch_size, *[1] * (sample_a.dim() - 1))
    else:
        attn_mask = None
    images = model.sample_sequence(
        init_noise,
        attn_mask,
        stride=3,
        n_T=gen_trans_step,
    )
    if images is not None:
        # image sequences to gif
        gif = []
        for image in images:
            x_as_image = make_grid(image.float() / (model.num_classes - 1), nrow=2)
            img = x_as_image.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            gif.append(Image.fromarray(img))

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp_file:
            gif[0].save(
                temp_file.name,
                format="GIF",
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file_img:
            last_img = gif[-1]
            last_img.save(temp_file_img)
        return temp_file.name, temp_file_img.name
    else:
        return None, None


def get_text(
    sample_x,
    sample_a,
    model,
    gen_trans_step,
    batch_size,
    tokenizer,
    temperature=1,
    K=0,
    r=1,
):
    if gen_trans_step == 0:
        return None, None
    print("Sampling sequences...")
    p = model.get_stationary()
    init_noise = (
        torch.multinomial(p, batch_size * sample_x.shape[1], replacement=True)
        .reshape(batch_size, sample_x.shape[1])
        .to(sample_x.device)
    )
    init_noise[sample_x == model.tokenizer.pad_token_id] = model.tokenizer.pad_token_id
    init_noise[sample_x == model.tokenizer.cls_token_id] = model.tokenizer.cls_token_id
    init_noise[sample_x == model.tokenizer.eos_token_id] = model.tokenizer.eos_token_id
    if sample_a is not None:
        attn_mask = sample_a.repeat(batch_size, *[1] * (sample_a.dim() - 1))
    else:
        attn_mask = None
    tokens = model.sample_sequence(
        init_noise, attn_mask, n_T=gen_trans_step, temperature=temperature, K=K, r=r
    )
    if tokens is not None:
        tokenizer.decode(tokens[0][0]).replace(" ", "")
        full = [[tokenizer.decode(s).replace(" ", "") for s in t] for t in tokens]
        return full[-1], full
    else:
        return None, None


class DiffusionTrainer(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        gen_trans_step=1000,
        n_gen_images=4,
        grad_clip_val=1,
        weight_decay=0,
        seed=0,
        n_stat_samples=2e6,
        tokenizer=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.weight_decay = weight_decay
        # logging
        self.sample_x = None
        self.validation_step_outputs = []
        self.gen_trans_step = gen_trans_step
        self.n_gen_images = n_gen_images
        self.n_stat_samples = n_stat_samples
        self.tokenizer = tokenizer

        self.warmup_steps = getattr(self.hparams, "lr_warmup", 0)
        # self.to(torch.float32)

    def forward(self, x):
        return NotImplementedError

    def get_kl_t1(self, x):
        return NotImplementedError

    def training_step(self, batch, batch_idx):
        if batch[0].size(0) == 0:
            # Create a dummy loss that still requires grad
            dummy_param = next(self.parameters())
            loss = dummy_param.sum() * 0.0  # requires grad, produces 0.0
            info = {
                "vb_loss": loss.detach().item(),
                "t": torch.tensor(0.0, device=self.device),
                "x_t_length": torch.tensor(0.0, device=self.device),
            }
        else:
            if len(batch) == 2:
                x, attn_mask = batch
                loss, info = self.forward(x.long(), attn_mask)
            elif len(batch) == 5:
                x, t, S, x_t, log_alignments = batch
                loss, info = self.forward_preprocessed(
                    x.long(), t.float(), S, x_t.int(), log_alignments.float()
                )
                attn_mask = None
            if self.sample_x is None:
                self.sample_x = x[:1]
                self.sample_a = None if attn_mask is None else attn_mask[:1]

            if torch.isnan(loss):
                print("Loss is NaN! Exiting!")
                exit()

        self.log("train_loss", info["vb_loss"], sync_dist=True)
        # self.log('train_ce_loss', info['ce_loss'], sync_dist=True)
        if "t" in info:
            self.log("batch_t", info["t"], sync_dist=True)
        if "x_t_length" in info:
            self.log("x_t_length", info["x_t_length"], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch[0].size(0) == 0:
            # Don't return yet because we need to log for sync_dist = True
            info = {
                "vb_loss": torch.tensor(0.0, device=self.device),
                "x_0_length": torch.tensor(0.0, device=self.device),
            }
            kl_1 = 0.0
        elif len(batch) == 2:
            x, attn_mask = batch
            loss, info = self.forward(x.long(), attn_mask)
            kl_1 = self.get_kl_t1(x.long()).detach().item()
        elif len(batch) == 5:
            x, t, S, x_t, log_alignments = batch
            loss, info = self.forward_preprocessed(
                x.long(), t.float(), S, x_t.int(), log_alignments.float()
            )
            kl_1 = self.get_kl_t1(x.long()).detach().item()

        self.log(
            "val_l01", info["vb_loss"], on_step=True, on_epoch=True, sync_dist=True
        )
        self.log("val_l1", kl_1, on_step=True, on_epoch=True, sync_dist=True)
        self.log(
            "val_len_x", info["x_0_length"], on_step=True, on_epoch=True, sync_dist=True
        )
        # self.log('val_ce_loss', info['ce_loss'], on_step=False, on_epoch=True, sync_dist=True)
        loss_dict = {
            # "val_ce_loss": info['ce_loss'],
            "val_l01": info["vb_loss"],
            "val_l1": kl_1,
        }
        return loss_dict

    @rank_zero_only
    def on_validation_epoch_end(
        self,
    ):
        # generate image
        if self.sample_x is not None:
            with torch.no_grad():
                if self.tokenizer is None:
                    gif_fname, img_fname = get_gif(
                        self.sample_x,
                        self.sample_a,
                        self,
                        self.gen_trans_step,
                        self.n_gen_images,
                    )
                    if gif_fname is not None:
                        if isinstance(self.logger, pl.loggers.WandbLogger):
                            wandb.log({"sample_gif": wandb.Image(gif_fname)})
                            wandb.log({"sample_gif_last": wandb.Image(img_fname)})
                else:
                    last_text, gen_text = get_text(
                        self.sample_x,
                        self.sample_a,
                        self,
                        self.gen_trans_step,
                        self.n_gen_images,
                        self.tokenizer,
                    )
                    if last_text is not None:
                        if isinstance(self.logger, pl.loggers.WandbLogger):
                            joined_text = "\n".join(last_text)
                            wandb.log(
                                {
                                    f"sample_text_{self.current_epoch}": wandb.Table(
                                        columns=["text"], data=[[joined_text]]
                                    )
                                }
                            )
                            joined_text_gen = ["\n".join(t) for t in gen_text]
                            wandb.log(
                                {
                                    f"sample_text_process_{self.current_epoch}": wandb.Table(
                                        columns=["text"],
                                        data=[[jt] for jt in joined_text_gen],
                                    )
                                }
                            )
        torch.cuda.empty_cache()

    # def on_fit_start(self):
    #     if isinstance(self.logger, pl.loggers.WandbLogger):
    #         wandb.config.update(self.hparams)

    def on_before_optimizer_step(self, optimizer):
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_val)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return 1.0  # Keep LR constant after warmup

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "gradient_clip_val": self.grad_clip_val,
            "weight_decay": self.weight_decay,
            "gradient_clip_algorithm": "norm",
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # important: update every step
                "frequency": 1,
            },
        }
