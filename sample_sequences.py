import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from SCISOR.shortening_scud import ShorteningSCUD
from SCISOR.trainer import get_text

path = "https://huggingface.co/SCISOR/SCISOR/resolve/main/SCISOR_U90_S.ckpt"
batch_size = 32
n_samples = 10
sample_lengths = 100
num_steps = 10
temperature = 1
K = 0
r = 1
save_file_path = "sampled_sequences.fasta"


def get_sequences_from_model(
    sample_x, model, gen_trans_step, batch_size, temperature=1, K=0, r=1
):
    last_text, _ = get_text(
        sample_x,
        None,
        model,
        gen_trans_step,
        batch_size,
        tokenizer,
        temperature=temperature,
        K=K,
        r=r,
    )
    last_text = [
        s.replace("<cls>", "").replace("<eos>", "").replace("<pad>", "")
        for s in last_text
    ]
    return last_text


device = "cuda" if torch.cuda.is_available() else "cpu"


model = ShorteningSCUD.load_from_checkpoint(path)
model.to(device)
model.eval()
model.p0 = torch.load("p0.pt")
rate = 1 / 1.1
model.alpha = lambda t: (1 - t) ** rate
model.beta = lambda t: rate / (1 - t)

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


sample_x = torch.full((n_samples, sample_lengths + 2), 4, device=device)
sample_x[:, 0] = model.tokenizer.cls_token_id
sample_x[:, -1] = model.tokenizer.eos_token_id

sampled_sequences = []
for i in tqdm(range(0, len(sample_x), batch_size)):
    x = sample_x[i : i + batch_size]
    sequences = get_sequences_from_model(
        x,
        model,
        gen_trans_step=num_steps,
        batch_size=x.shape[0],
        temperature=temperature,
        K=K,
        r=r,
    )
    sampled_sequences.extend(sequences)

padding = len(str(len(sampled_sequences)))  # Number of digits needed
with open(save_file_path, "w") as fasta_file:
    for i, seq in enumerate(sampled_sequences):
        fasta_file.write(f">{str(i).zfill(padding)}\n{seq}\n")
