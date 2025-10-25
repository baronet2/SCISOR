import os
import sys

import torch
import tqdm
from uniref_preprocessor import BatchProcessor, get_batches_from_sequences, set_seed

# === Parameters ===
shard_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
shard_path = f"data/uniref90/shards/shard_{shard_index:02d}.txt"
output_root = "data/uniref90_processed"

batch_size = 512
max_len_keep = None
max_len_trim = 1024
seed = 0

print(f"Processing shard {shard_index} from {shard_path}...")
with open(shard_path) as f:
    sequences = [line.strip() for line in f if line.strip()]

set_seed(seed)

batch_generator = get_batches_from_sequences(
    sequences, batch_size=batch_size, max_length=max_len_keep
)

bp = BatchProcessor(fixed_t=False, device="cuda", max_len=max_len_trim)
bp.p0 = torch.load("p0.pt")

out_dir = os.path.join(output_root, f"{shard_index:03d}")
os.makedirs(out_dir, exist_ok=True)

print("Processing batches...")

for i, batch in tqdm.tqdm(enumerate(batch_generator)):
    save_path = os.path.join(out_dir, f"batch_{i:05d}.pt")

    # Skip processing if the file already exists
    if os.path.exists(save_path):
        print(f"Skipping batch {i:05d}, file already exists.")
        continue

    processed = bp.process(batch, subbatch_size=64)
    torch.save(processed, save_path)

print("Done.")
