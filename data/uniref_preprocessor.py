import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from SCISOR.alignments import get_deletion_log_alignments


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class BatchProcessor:
    def __init__(
        self, T=0.9, alpha_rate=1.1, device="cuda", max_len=1024, fixed_t=True
    ):
        self.t_max = T
        self.fixed_t = fixed_t
        rate = 1 / alpha_rate
        self.alpha = lambda t: (1 - t) ** rate
        self.beta = lambda t: rate / (1 - t)
        self.device = device
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.pad_token_id = self.tokenizer.pad_token_id
        self.num_classes = len(self.tokenizer.get_vocab())
        self.choices = list(range(4, 24))

    def tokenize_batch(self, batch):
        tokenized_seqs = self.tokenizer([s[: self.max_len] for s in batch[0]]).input_ids
        x = pad_sequence(
            [torch.tensor(s) for s in tokenized_seqs],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        return x

    def calc_p0(self, dataloader, n_counts=1000):
        p0 = torch.zeros(self.num_classes)
        for batch in dataloader:
            if p0.sum() > n_counts:
                break
            x = self.tokenize_batch(batch)
            new = (
                F.one_hot(x.long(), num_classes=self.num_classes)
                .to(torch.float32)
                .view((-1, self.num_classes))
                .sum(0)
            )
            p0 = p0 + new

        classes_to_exclude = list(
            set(range(self.num_classes)) - set(map(int, self.choices))
        )
        p0[classes_to_exclude] = 0
        self.p0 = p0 / p0.sum()

    def sample_point_given_S(self, x, S):
        Ls = (x != self.pad_token_id).sum(dim=1)
        B = x.shape[0]

        max_len = int((S + Ls).max())

        # Set up x_t with random elements
        x_t = (
            torch.multinomial(self.p0, B * max_len, replacement=True)
            .view(B, max_len)
            .to(self.device)
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

        if (S == 0).any():
            probs = dist.log_prob(
                torch.arange(1, 21, device=self.device).unsqueeze(1)
            ).T.exp()
            cat_dist = torch.distributions.Categorical(probs[S == 0])
            new_S = cat_dist.sample().float() + 1
            S[S == 0] = new_S

        x_t = self.sample_point_given_S(x, S)
        return S, x_t

    def sample_point(self, x):
        if self.fixed_t:
            ts = (
                torch.ones(x.shape[0], device=self.device)
                * torch.rand(1, device=self.device)
                * self.t_max
            )
        else:
            ts = torch.rand(x.shape[0], device=self.device) * self.t_max

        S, x_t = self.sample_point_given_t(x, ts)
        return ts, S, x_t

    def process(self, batch, subbatch_size=64):
        x = self.tokenize_batch(batch).to(self.device)
        t, S, x_t = self.sample_point(x)

        lengths = (x_t != self.pad_token_id).sum(dim=1)
        _, sorted_indices = torch.sort(lengths, descending=True)
        x = x[sorted_indices]
        t = t[sorted_indices]
        S = S[sorted_indices]
        x_t = x_t[sorted_indices]

        log_alignments = torch.full(x_t.shape, float("-inf"), device=self.device)

        for i in range(0, x.shape[0], subbatch_size):
            x_full_dims = (x[i : i + subbatch_size] != self.pad_token_id).any(dim=0)
            x_t_full_dims = x_t[i] != self.pad_token_id

            subbatch_log_aligments = get_deletion_log_alignments(
                x[i : i + subbatch_size, x_full_dims],
                x_t[i : i + subbatch_size, x_t_full_dims],
                pad_id=self.pad_token_id,
            )[0]

            log_alignments[i : i + subbatch_size, x_t_full_dims] = (
                subbatch_log_aligments
            )

        return (
            x.to(torch.int8),
            t.to(torch.float16),
            S.to(torch.int16),
            x_t.to(torch.int8),
            log_alignments.to(torch.float16),
        )


def get_batches_from_sequences(sequences, batch_size, max_length=None):
    """
    Generates batches from a list of sequences.

    Args:
        sequences (List[str]): Protein sequences as strings.
        batch_size (int): Number of sequences per batch.
        max_length (int or None): Optional max length filter.

    Yields:
        List[List[str]]: A list containing one list of sequences (for compatibility).
    """
    standard_aas = set("ACDEFGHIKLMNPQRSTVWY")

    def is_standard(seq):
        return all(residue in standard_aas for residue in seq)

    batch = []
    for seq in sequences:
        if not is_standard(seq):
            continue
        if max_length is not None and len(seq) > max_length:
            continue
        batch.append(seq)
        if len(batch) == batch_size:
            yield [batch]  # Yield a batch wrapped in a list
            batch = []
    if batch:
        yield [batch]
