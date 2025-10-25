import torch
import torch.nn.functional as F
from torch.amp import autocast

from .alignments import get_deletion_log_alignments
from .shortening_diffusion import ShorteningDiffusion
from .utils import kls_probs, kls_probs_both


class ShorteningSCUD(ShorteningDiffusion):
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
        super().__init__(
            x0_model_class,
            nn_params,
            num_classes,
            forward_kwargs,
            schedule_type,
            gamma,
            **kwargs,
        )

    def q_posterior_log_probs(self, x_0, x_t):
        log_alignments = get_deletion_log_alignments(
            x_0, x_t, pad_id=self.pad_token_id
        )[0]
        log_alignments_sum = torch.logsumexp(log_alignments, dim=1, keepdims=True)
        log_probs = log_alignments - log_alignments_sum
        return log_probs

    def forward_preprocessed(self, x, t, S, x_t, log_alignments):
        log_alignments_sum = torch.logsumexp(log_alignments, dim=1, keepdims=True)
        true_q_posterior_log_probs = log_alignments - log_alignments_sum
        true_q_posterior_probs = torch.nan_to_num(true_q_posterior_log_probs.exp(), 0)

        if self.window_size is not None and x_t.shape[1] > self.window_size:
            full_probs = self.predict_with_windows(x_t, t, S)
            kl = kls_probs_both(true_q_posterior_probs, full_probs)
        else:
            pred_q_posterior_logits = self.model_predict(x_t, t, None, S)
            kl = kls_probs(true_q_posterior_probs, pred_q_posterior_logits)

        weight = self.beta(t) / (1 - self.alpha(t))

        Ls = (x != self.pad_token_id).sum(dim=1) - 2
        weight *= 1 - self.alpha(t) ** (Ls + 1)

        vb_loss = (S * kl * weight).mean() * self.t_max

        return vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": 0,  # TODO Actually calculate this?
            "t": t.mean(),
            "x_t_length": x_t.shape[-1] - 2,
            "x_0_length": (x != self.pad_token_id).sum(dim=1).float().mean() - 2,
        }

    def forward(self, x, attn_mask=None):
        t, S, x_t = self.sample_point(x, attn_mask)
        b, l = x_t.shape

        true_q_posterior_log_probs = self.q_posterior_log_probs(x, x_t)
        true_q_posterior_probs = torch.nan_to_num(true_q_posterior_log_probs.exp(), 0)

        if self.window_size is not None and l > self.window_size:
            full_probs = self.predict_with_windows(x_t, t, S)
            kl = kls_probs_both(true_q_posterior_probs, full_probs)
        else:
            pred_q_posterior_logits = self.model_predict(x_t, t, attn_mask, S)
            kl = kls_probs(true_q_posterior_probs, pred_q_posterior_logits)

        weight = self.beta(t) / (1 - self.alpha(t))

        if self.rao_blackwellize:
            Ls = (x != self.pad_token_id).sum(dim=1) - 2
            weight *= 1 - self.alpha(t) ** (Ls + 1)

        vb_loss = (S * kl * weight).mean() * self.t_max

        return vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": 0,  # TODO Actually calculate this?
            "t": t.mean(),
            "x_t_length": x_t.shape[-1] - 2,
            "x_0_length": (x != self.pad_token_id).sum(dim=1).float().mean() - 2,
        }

    def base_predict(self, x_t, t, attn_mask, S=None):
        # TODO Ensure model is using attn_mask correctly
        attn_mask = (x_t != self.pad_token_id).int()
        with autocast("cuda", dtype=torch.bfloat16):
            preds = self.x0_model(x_t, t, attn_mask, S)
        return preds.to(torch.float32)

    def predict_with_windows(self, x_t, t, S):
        rows = torch.arange(x_t.size(0), device=x_t.device).unsqueeze(1)
        cols = self.get_window_indices(x_t)

        model_preds = self.model_predict(x_t[rows, cols], t, None, S)
        model_pred_probs = F.softmax(model_preds, dim=-1)

        full_lengths = (x_t != self.pad_token_id).sum(dim=1).unsqueeze(1) - 2
        full_probs = torch.ones(x_t.shape, device=x_t.device) / full_lengths
        full_probs[rows, cols] = model_pred_probs * self.window_size / full_lengths
        full_probs[
            (x_t == self.pad_token_id)
            | (x_t == self.tokenizer.cls_token_id)
            | (x_t == self.tokenizer.eos_token_id)
        ] = 0
        full_probs = full_probs / full_probs.sum(dim=1, keepdims=True)
        return full_probs

    def p_sample(self, x, t, attn_mask, S, num_dels, temperature=1):
        if num_dels.sum() == 0:
            return x, []
        # TODO Use Gumbel softmax trick for more numerically-stable sampling?
        b, l = x.shape
        if self.window_size is not None and l > self.window_size:
            model_probs = self.predict_with_windows(x, t, S)
        else:
            model_logits = self.model_predict(x, t, attn_mask, S)
            model_probs = torch.softmax(model_logits, dim=-1)
        model_probs = (
            model_probs
            * (x != self.pad_token_id)
            * (x != self.tokenizer.cls_token_id)
            * (x != self.tokenizer.eos_token_id)
        )

        old_lengths = (x != self.pad_token_id).sum(dim=1)
        new_lengths = l - num_dels

        out = (
            torch.ones((b, int(new_lengths.max())), dtype=torch.int, device=x.device)
            * self.pad_token_id
        )
        deleted_indices = []
        for i in range(b):
            if num_dels[i] > 0:
                if temperature == 0.0:
                    sampled_indices = torch.topk(
                        model_probs[i], int(num_dels[i]), dim=-1
                    ).indices
                else:
                    tempered_probs = torch.softmax(
                        torch.log(model_probs[i]) / temperature, dim=-1
                    )
                    sampled_indices = torch.multinomial(
                        tempered_probs, int(num_dels[i])
                    )
                deleted_indices.append(sampled_indices.tolist())
                out[i, : int(new_lengths[i])] = x[
                    i, ~torch.isin(torch.arange(l, device=x.device), sampled_indices)
                ]
            else:
                deleted_indices.append([])
                out[i, : int(new_lengths[i])] = x[i, : int(new_lengths[i])]

        max_new_len = int(max(old_lengths - num_dels))
        result_lengths = (out != self.pad_token_id).sum(dim=1)
        if ((old_lengths - num_dels) != result_lengths).any():
            print("Deleted the wrong number of tokens!")
        return out[:, :max_new_len], deleted_indices

    def get_insertion_schedule(self, x, max_insertions=10000):
        # Get insertion times in tau space (from 0 to infinity)
        Ls = (x != self.pad_token_id).sum(dim=1)

        insertion_intervals = (
            torch.distributions.Exponential(1.0)
            .sample((x.shape[0], max_insertions))
            .to(self.device)
        )
        insertion_intervals_adj = insertion_intervals / (
            Ls[:, None] + 1 + torch.arange(max_insertions, device=self.device)[None, :]
        )  # Rate of insertions proportional to sequence length
        insertion_times_tau = torch.cumsum(insertion_intervals_adj, dim=-1)

        return insertion_times_tau

    def sample_sequence(self, x, attn_mask=None, n_T=None, temperature=1, K=0, r=1):
        if n_T == 0:
            return None

        with torch.no_grad():
            out = []
            T = self.t_max
            tau_max = self.convert_t_to_tau(T)
            insertion_times_tau = self.get_insertion_schedule(x)
            S = (insertion_times_tau <= tau_max).sum(dim=1)
            x_t = self.sample_point_given_S(x, S)

            if n_T is None:
                while S.sum() > 0:
                    dels_this_interval = (S > 0).int()
                    x_t, _ = self.p_sample(
                        x_t,
                        None,
                        attn_mask,
                        S,
                        dels_this_interval,
                        temperature=temperature,
                    )
                    S = S - dels_this_interval
                    out.append(x_t.clone())

            else:
                delta = T / n_T * (K + 1)
                for t in torch.arange(T, 0, -delta):
                    tau1, tau2 = (
                        self.convert_t_to_tau(t - delta),
                        self.convert_t_to_tau(t),
                    )
                    dels_this_interval = (
                        (tau1 < insertion_times_tau) & (insertion_times_tau <= tau2)
                    ).sum(dim=1)
                    for i in range(K):
                        n_corrector_dels = (dels_this_interval * r).int()
                        x_t = self.sample_point_given_S(x_t, n_corrector_dels)
                        x_t, _ = self.p_sample(
                            x_t,
                            t,
                            None,
                            S + n_corrector_dels,
                            n_corrector_dels,
                            temperature=temperature,
                        )
                    x_t, _ = self.p_sample(
                        x_t, t, None, S, dels_this_interval, temperature=temperature
                    )
                    S = S - dels_this_interval
                    out.append(x_t.clone())

        return out

    def shrink_sequence(self, x, S, temperature=1):
        preserved_indices = [list(range(x.shape[1]))] * x.shape[0]
        with torch.no_grad():
            while S.sum() > 0:
                dels_this_interval = (S > 0).int()
                x, deleted_indices = self.p_sample(
                    x, None, None, S, dels_this_interval, temperature=temperature
                )
                preserved_indices = [
                    l[:i] + l[i + 1 :]
                    for l, i in zip(
                        preserved_indices,
                        [d[0] if d else 10000 for d in deleted_indices],
                    )
                ]
                S = S - dels_this_interval
        return x, preserved_indices
