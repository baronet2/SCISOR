import os

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from SCISOR.shortening_scud import ShorteningSCUD

url = "https://huggingface.co/SCISOR/SCISOR/resolve/main/SCISOR_U90_S.ckpt"
batch_size = 32
input_seqs_path = "sampled_sequences.fasta"
shrink_pct = 10
temperature = 1
save_file_path = "shrunk_sequences.fasta"


def read_fasta_to_df(fasta_file):
    with open(fasta_file, "r") as file:
        content = file.read()

    sequences = []
    entries = content.strip().split(">")

    for entry in entries:
        if not entry:
            continue  # skip any empty entries
        lines = entry.strip().split("\n")
        header = lines[0]
        sequence = "".join(lines[1:]).replace(" ", "").replace("\r", "")
        sequences.append(
            {"Header": header, "Sequence": sequence, "Length": len(sequence)}
        )

    df = pd.DataFrame(sequences).drop_duplicates()
    return df.head(100).query("Length <= 1000").reset_index(drop=True)


def untokenize(seq, tokenizer):
    out = (
        tokenizer.decode(seq)
        .replace(" ", "")
        .replace("<cls>", "")
        .replace("<eos>", "")
        .replace("<pad>", "")
    )
    return out


def save_sequences_to_fasta(fasta_file, seqs, headers):
    if os.path.dirname(fasta_file):
        os.makedirs(os.path.dirname(fasta_file), exist_ok=True)
    with open(fasta_file, "w") as f:
        for header, seq in zip(headers, seqs):
            f.write(f">{header}\n{seq}\n")


device = "cuda" if torch.cuda.is_available() else "cpu"

model = ShorteningSCUD.load_from_checkpoint(url)
model.to(device)
model.eval()
model.p0 = torch.load("p0.pt")
rate = 1 / 1.1
model.alpha = lambda t: (1 - t) ** rate
model.beta = lambda t: rate / (1 - t)


original_protein_df = read_fasta_to_df(input_seqs_path)
seq_lengths = torch.tensor(original_protein_df.Length.values, device=model.device)
num_deletions = torch.ceil(seq_lengths * shrink_pct / 100).int()
print(f"Shrinking {len(original_protein_df)} sequences by {shrink_pct}%")

input_ids = [model.tokenizer(s).input_ids for s in original_protein_df.Sequence]
max_len = max([len(x) for x in input_ids])
x = torch.vstack(
    [
        F.pad(
            torch.tensor(x, device=model.device),
            (0, max_len - len(x)),
            value=model.tokenizer.pad_token_id,
        )
        for x in input_ids
    ]
)

sampled_sequences = []
deleted_indices = []
batch_size = model.hparams.batch_size
for i in tqdm(range(0, len(x), batch_size)):
    sequences, preserved_indices = model.shrink_sequence(
        x[i : i + batch_size],
        num_deletions[i : i + batch_size],
        temperature=temperature,
    )
    del_idx = [
        list(set(range(len(s))) - set([j - 1 for j in p]))
        for s, p in zip(
            original_protein_df.Sequence[i : i + batch_size], preserved_indices
        )
    ]
    decoded_seqs = [untokenize(s, model.tokenizer) for s in sequences]
    sampled_sequences.extend(decoded_seqs)
    deleted_indices.extend(del_idx)

assert all(
    [
        "".join([c for i, c in enumerate(s) if i not in d]) == n
        for s, d, n in zip(
            original_protein_df.Sequence, deleted_indices, sampled_sequences
        )
    ]
)
del_str = [
    ",".join([f"{c}{i}" for i, c in enumerate(s) if i in d])
    for s, d in zip(original_protein_df.Sequence, deleted_indices)
]
new_headers = (
    original_protein_df.Header
    + "|deletions "
    + pd.Series(del_str)
    + f"|percentage {shrink_pct}"
)
save_sequences_to_fasta(save_file_path, sampled_sequences, new_headers)
print(f"Saved shrunk sequences to {save_file_path}")
