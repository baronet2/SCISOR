import math

import torch
import torch.nn.functional as F

from .alignments import log_ali
from .continuous_time_diffusion import ContinuousTimeDiffusion


class ShorteningDiffusion(ContinuousTimeDiffusion):
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes: int = 10,
        forward_kwargs={},
        schedule_type="cos",
        gamma=0,
        **kwargs,
    ):
        # Precalculate betas, define model_predict, p_sample
        super().__init__(
            x0_model_class, nn_params, num_classes, schedule_type, **kwargs
        )
        self.pad_token_id = (
            forward_kwargs["pad_token_id"] if "pad_token_id" in forward_kwargs else -1
        )
        self.choices = torch.tensor(forward_kwargs["choices"])
        self.batch_strategy = (
            forward_kwargs["batch_strategy"]
            if "batch_strategy" in forward_kwargs
            else None
        )
        self.n_gen_images = (
            forward_kwargs["n_gen_samples"]
            if "n_gen_samples" in forward_kwargs
            else None
        )
        self.window_size = (
            forward_kwargs["window_size"] if "window_size" in forward_kwargs else None
        )
        self.alpha_rate = (
            forward_kwargs["alpha_rate"] if "alpha_rate" in forward_kwargs else 1
        )
        self.structured_forward_process = (
            forward_kwargs["structured"] if "structured" in forward_kwargs else False
        )
        self.rao_blackwellize = (
            forward_kwargs["rao_blackwellize"]
            if "rao_blackwellize" in forward_kwargs
            else False
        )

    def convert_t_to_tau(self, t):
        return -math.log(self.alpha(t))

    def get_stationary(self):
        if self.structured_forward_process:
            return self.p0
        probs = torch.zeros_like(self.p0)
        probs[self.choices] = 1 / len(self.choices)
        return probs

    def get_kl_t1(self, x):
        S, x_t = self.sample_point_given_t(
            x, torch.ones(x.shape[0], device=x.device) * self.t_max
        )

        Ls = (x != self.pad_token_id).sum(dim=1) - 2
        log_binom = (
            torch.lgamma(Ls + S + 1) - torch.lgamma(Ls + 1) - torch.lgamma(S + 1)
        )
        x_0_counts = F.one_hot(x, num_classes=self.num_classes).sum(dim=1)[
            :, self.choices
        ]
        b_log_pi = x_0_counts.float() @ self.get_stationary()[self.choices].log().to(
            x.device
        )
        kl_estimates = log_ali(x, x_t, self.pad_token_id) - log_binom - b_log_pi

        if self.rao_blackwellize:
            kl_estimates *= 1 - self.alpha(self.t_max) ** (Ls + 1)

        return kl_estimates.mean()

    def sample_point_given_S(self, x, S):
        Ls = (x != self.pad_token_id).sum(dim=1)
        B = x.shape[0]

        max_len = int((S + Ls).max())

        # Set up x_t with random elements
        x_t = (
            torch.multinomial(self.get_stationary(), B * max_len, replacement=True)
            .view(B, max_len)
            .to(x.device)
            .to(x.dtype)
        )
        x_t[:, 0] = self.tokenizer.cls_token_id

        # Insert original sequences from x into randomly-selected indices of x_t
        for i in range(B):
            insertion_indices = (
                1 + torch.randperm(int(S[i] + Ls[i] - 2))[: Ls[i] - 2].sort().values
            )
            x_t[i, insertion_indices] = x[i, 1 : int(Ls[i]) - 1]
            x_t[i, int(S[i] + Ls[i]) - 1] = self.tokenizer.eos_token_id
            x_t[i, int(S[i] + Ls[i]) :] = self.pad_token_id

        return x_t

    def sample_point_given_t(self, x, ts):
        Ls = (x != self.pad_token_id).sum(dim=1) - 2

        dist = torch.distributions.NegativeBinomial(Ls + 1, 1 - self.alpha(ts))
        S = dist.sample()

        if self.rao_blackwellize and (S == 0).any():
            probs = dist.log_prob(
                torch.arange(1, 21, device=x.device).unsqueeze(1)
            ).T.exp()
            cat_dist = torch.distributions.Categorical(probs[S == 0])
            new_S = cat_dist.sample().float() + 1
            S[S == 0] = new_S

        x_t = self.sample_point_given_S(x, S)
        return S, x_t

    def sample_point(self, x, attn_mask=None, rand_shape=None):
        if self.batch_strategy == "fixed_t":
            ts = (
                torch.ones(x.shape[0], device=x.device)
                * torch.rand(1, device=x.device)
                * self.t_max
            )
        else:
            ts = torch.rand(x.shape[0], device=x.device) * self.t_max
        S, x_t = self.sample_point_given_t(x, ts)
        return ts, S, x_t

    def get_window_indices(self, x_t):
        full_lengths = (x_t != self.pad_token_id).sum(dim=1).unsqueeze(1) - 2
        start = torch.rand(full_lengths.shape, device=x_t.device)
        scaling = torch.where(
            full_lengths > self.window_size, full_lengths - self.window_size, 0
        )
        start_idx = (start * scaling).floor().int() + 1
        return start_idx + torch.arange(self.window_size, device=x_t.device)

    def forward(self, x, attn_mask=None):
        raise NotImplementedError

    def sample_sequence(self, x, attn_mask=None, n_T=200):
        raise NotImplementedError

    def base_predict(self, x_t, t, attn_mask, S=None):
        raise NotImplementedError
