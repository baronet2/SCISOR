import os

import requests
import torch

from SCISOR.faesm import FAESM_Base

from .trainer import DiffusionTrainer


def sample_n_transitions_cont(log_alpha, batch_size, times):
    times = times.reshape(-1)
    log_alpha_t = log_alpha(times).reshape(1, -1).repeat(batch_size, 1)
    transitions = torch.poisson(-log_alpha_t)
    return transitions


class ContinuousTimeDiffusion(DiffusionTrainer):
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes: int = 10,
        schedule_type="cos",
        t_max=0.999,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["x0_model_class"])
        self.hparams.update(x0_model_class=x0_model_class.__name__)
        self.x0_model = x0_model_class(**nn_params)
        self.num_classes = num_classes
        self.t_max = t_max

    def get_stationary(self):
        raise NotImplementedError

    def base_predict(self, x_t, t, attn_mask, S=None):
        return self.x0_model(x_t, t, attn_mask, S).to(torch.float32)

    def model_predict(self, x_t, t, attn_mask, S=None):
        return self.base_predict(x_t, t, attn_mask, S)

    def x_t_sample(self, x_0, t, noise, S=None):
        raise NotImplementedError

    def sample_point(self, x, attn_mask=None, rand_shape=None):
        t = torch.rand(x.shape[0], device=x.device) * self.t_max
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        x_t = self.x_t_sample(
            x,
            t,
            torch.rand(
                (*x.shape, rand_shape if rand_shape is not None else self.num_classes),
                device=x.device,
            ),
            S,
        )
        return t, S, x_t

    def load_state_dict(self, state_dict, strict=False):
        # Call the parent class's load_state_dict method
        missing_keys, unexpected_keys = super().load_state_dict(
            state_dict, strict=False
        )

        # Load the additional state dict variables
        for key in [
            "p0_inds",
            "p0_rank",
            "K",
            "L",
            "K_coo",
            "K_csc",
            "K_T",
            "L_T",
            "stat",
            "stationary",
        ]:
            if key in state_dict:
                setattr(self, key, state_dict[key])
                if key in unexpected_keys:
                    unexpected_keys.remove(key)

        if strict:
            error_msgs = []
            if len(unexpected_keys) > 0:
                error_msgs.append(
                    "unexpected key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in unexpected_keys)
                    )
                )
            if len(missing_keys) > 0:
                error_msgs.append(
                    "missing key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in missing_keys)
                    )
                )

            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        self.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )

        return missing_keys, unexpected_keys

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        """
        Loads a model from a local or remote checkpoint (.ckpt).
        Supports direct Hugging Face or HTTP(S) URLs.
        """
        print(f"Loading checkpoint from: {checkpoint_path}")

        if isinstance(checkpoint_path, str) and checkpoint_path.startswith(
            ("http://", "https://")
        ):
            print("Detected remote checkpoint URL. Downloading ...")

            # Try to reuse cached version
            cache_dir = os.path.expanduser("~/.cache/shortening_scud")
            os.makedirs(cache_dir, exist_ok=True)
            local_name = os.path.join(cache_dir, os.path.basename(checkpoint_path))

            if not os.path.exists(local_name):
                with requests.get(checkpoint_path, stream=True) as r:
                    r.raise_for_status()
                    with open(local_name, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"Checkpoint downloaded and cached at {local_name}")
            else:
                print(f"Using cached checkpoint at {local_name}")

            checkpoint_path = local_name

        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
        print("Checkpoint loaded successfully.")

        hparams = checkpoint["hyper_parameters"]
        hparams["x0_model_class"] = FAESM_Base

        print("Instantiating model ...")
        model = cls(**hparams)
        print("Loading state dict ...")
        model.load_state_dict(checkpoint["state_dict"])

        print("Model loaded and ready.")
        return model
