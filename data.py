import os
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


class ProcessedBatchDataset(Dataset):
    def __init__(
        self,
        data_dir,
        batch_size=64,
        original_batch_size=256,
        pad_token_id=1,
        shuffle=False,
    ):
        self.data_dir = data_dir
        self.batch_size = original_batch_size
        self.sub_batch_size = batch_size
        self.sub_batch_ratio = int(self.batch_size / self.sub_batch_size)
        self.pad_token_id = pad_token_id

        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])
        self.file_paths = [os.path.join(data_dir, f) for f in self.file_list]
        if shuffle:
            random.shuffle(self.file_paths)

    def __len__(self):
        return len(self.file_paths) * self.sub_batch_ratio

    def trim_sub_batch(self, sub_batch):
        """Drop columns that are fully padding"""
        x, t, S, x_t, log_alignments = sub_batch

        x_full_dims = (x != self.pad_token_id).any(dim=0)
        x = x[:, x_full_dims]

        x_t_full_dims = (x_t != self.pad_token_id).any(dim=0)
        x_t = x_t[:, x_t_full_dims]
        log_alignments = log_alignments[:, x_t_full_dims]

        return x, t, S, x_t, log_alignments

    def __getitem__(self, idx):
        batch_idx = idx // self.sub_batch_ratio
        sub_batch_idx = idx % self.sub_batch_ratio
        full_batch = torch.load(self.file_paths[batch_idx], map_location="cpu")
        start_idx = self.sub_batch_size * sub_batch_idx
        sub_batch = [
            item[start_idx : start_idx + self.sub_batch_size].cpu()
            for item in full_batch
        ]
        return self.trim_sub_batch(sub_batch)


class NestedProcessedBatchDataset(ProcessedBatchDataset):
    def __init__(
        self,
        data_dir,
        batch_size=64,
        original_batch_size=512,
        pad_token_id=1,
        shuffle=True,
        max_segment=None,
        min_segment=None,
    ):
        super().__init__(
            data_dir, batch_size, original_batch_size, pad_token_id, shuffle
        )

        # Normalize max_segment (e.g., "029" stays as a string for string comparison)
        self.file_paths = []

        for root, _, files in os.walk(data_dir):
            # Get relative path from data_dir (e.g., '001', '002/subdir')
            rel_path = os.path.relpath(root, data_dir)
            top_level_dir = rel_path.split(os.sep)[0]

            if max_segment is not None and top_level_dir > max_segment:
                continue
            if min_segment is not None and top_level_dir < min_segment:
                continue

            for file in files:
                if file.endswith(".pt"):
                    self.file_paths.append(os.path.join(root, file))

        if shuffle:
            random.shuffle(self.file_paths)


class ProcessedBatchTestDataset(ProcessedBatchDataset):
    def __init__(
        self,
        data_dir,
        batch_size=64,
        original_batch_size=256,
        pad_token_id=1,
        shuffle=False,
        max_epochs=10,
    ):
        super().__init__(
            data_dir, batch_size, original_batch_size, pad_token_id, shuffle=shuffle
        )
        if "epoch_" in self.file_list[0]:
            self.file_paths = [
                os.path.join(data_dir, f)
                for f in self.file_list
                if int(f.split("epoch_")[1][:2]) < max_epochs
            ]
        else:
            self.file_paths = [os.path.join(data_dir, f) for f in self.file_list]


def get_processed_test_dataloader(
    data_dir, batch_size=64, num_workers=0, shuffle=False, pad_token_id=1, max_epochs=10
):
    dataset = ProcessedBatchTestDataset(
        data_dir,
        batch_size,
        pad_token_id=pad_token_id,
        shuffle=shuffle,
        max_epochs=max_epochs,
    )
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=None,  # Already batched into smaller sub-batches
        num_workers=num_workers,
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,
    )


def get_processed_dataloader(
    data_dir, batch_size=64, num_workers=0, shuffle=False, pad_token_id=1
):
    dataset = ProcessedBatchDataset(
        data_dir, batch_size, pad_token_id=pad_token_id, shuffle=shuffle
    )
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=None,  # Already batched into smaller sub-batches
        num_workers=num_workers,
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,
    )


def get_nested_processed_dataloader(
    data_dir,
    batch_size=64,
    num_workers=0,
    shuffle=False,
    pad_token_id=1,
    max_segment=None,
    min_segment=None,
):
    dataset = NestedProcessedBatchDataset(
        data_dir,
        batch_size,
        pad_token_id=pad_token_id,
        shuffle=shuffle,
        max_segment=max_segment,
        min_segment=min_segment,
    )
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=None,  # Already batched into smaller sub-batches
        num_workers=num_workers,
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,
    )


def get_dataloaders(cfg):
    batch_size = cfg.train.batch_size

    print("Getting the ESM tokenizer!")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    pad_token_id = cfg.model.forward_kwargs.pad_token_id

    num_workers = cfg.num_workers if "num_workers" in cfg else 1
    print(f"Using {num_workers} workers.")

    if cfg.data.data == "uniref50":
        print("Getting Uniref50 dataset")
        train_dataloader = get_processed_dataloader(
            cfg.data.train_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=cfg.train.shuffle_train if "shuffle_train" in cfg.train else False,
            pad_token_id=pad_token_id,
        )
        test_dataloader = get_processed_test_dataloader(
            cfg.data.valid_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=cfg.train.shuffle_valid if "shuffle_valid" in cfg.train else False,
            pad_token_id=pad_token_id,
            max_epochs=cfg.train.n_val_epochs if "n_val_epochs" in cfg.train else 1,
        )

    elif cfg.data.data == "uniref90":
        print("Getting Uniref90 dataset")
        n_val_shards = cfg.data.n_val_shards if "n_val_shards" in cfg.data else 1
        max_train_segment = (
            cfg.data.max_segment
            if "max_segment" in cfg.data
            else f"{30 - n_val_shards:03}"
        )
        min_test_segment = int(max_train_segment) + 1
        max_test_segment = min(29, min_test_segment + n_val_shards - 1)
        print(f"Getting train segments 000 to {max_train_segment}")
        train_dataloader = get_nested_processed_dataloader(
            cfg.data.train_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=cfg.train.shuffle_train if "shuffle_train" in cfg.train else True,
            pad_token_id=pad_token_id,
            max_segment=max_train_segment,
            min_segment="000",
        )
        print(f"Getting test segments {min_test_segment:03} to {max_test_segment:03}")
        test_dataloader = get_nested_processed_dataloader(
            cfg.data.train_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=cfg.train.shuffle_valid if "shuffle_valid" in cfg.train else True,
            pad_token_id=pad_token_id,
            max_segment=f"{max_test_segment:03}",
            min_segment=f"{min_test_segment:03}",
        )

    return train_dataloader, test_dataloader
