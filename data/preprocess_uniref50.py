import os

import torch
import tqdm
from sequence_models.datasets import UniRefDataset
from torch.utils.data import DataLoader
from uniref_preprocessor import BatchProcessor, set_seed

num_train_batches = 50
num_test_batches = 10

batch_size = 256
save_dir = "data/uniref50_processed"
seed = 0

set_seed(seed)

bp = BatchProcessor()
bp.p0 = torch.load("p0.pt")

print("Preparing train dataloader...")
dataset = UniRefDataset("data/uniref50/", "train", structure=False, max_len=10000)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

os.makedirs(save_dir, exist_ok=True)
for i, batch in tqdm.tqdm(enumerate(dataloader)):
    processed = bp.process(batch)
    torch.save(processed, os.path.join(save_dir, f"batch_{i:06d}.pt"))
    if i == num_train_batches:
        break

print("Preparing test dataloader...")
dataset = UniRefDataset("data/uniref50/", "test", structure=False, max_len=10000)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
save_dir = save_dir + "_test"
os.makedirs(save_dir, exist_ok=True)
for i, batch in tqdm.tqdm(enumerate(dataloader)):
    processed = bp.process(batch)
    torch.save(processed, os.path.join(save_dir, f"batch_{i:06d}.pt"))
    if i == num_test_batches:
        break
