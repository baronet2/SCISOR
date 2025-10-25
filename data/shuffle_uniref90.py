import os
import random
import sys

# === Default paths and parameters ===
DEFAULT_INPUT_FASTA = "data/uniref90/uniref90.fasta"
DEFAULT_OUTPUT_DIR = "data/uniref90/shards"
DEFAULT_NUM_SHARDS = 30
DEFAULT_SEED = 0


def read_fasta_seqs_only(path):
    """Reads only protein sequences (as strings) from a FASTA file."""
    sequences = []
    with open(path) as f:
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))
    return sequences


def write_shards_txt(sequences, output_dir, num_shards):
    """Writes sequences to text files, one sequence per line."""
    os.makedirs(output_dir, exist_ok=True)
    shard_size = len(sequences) // num_shards
    for i in range(num_shards):
        start = i * shard_size
        end = len(sequences) if i == num_shards - 1 else (i + 1) * shard_size
        shard_seqs = sequences[start:end]
        out_path = os.path.join(output_dir, f"shard_{i:02d}.txt")
        with open(out_path, "w") as f:
            for seq in shard_seqs:
                f.write(seq + "\n")


if __name__ == "__main__":
    # Use CLI args if provided, else fall back to defaults
    input_fasta = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FASTA
    output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    num_shards = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_NUM_SHARDS

    print(f"Reading sequences from: {input_fasta}")
    sequences = read_fasta_seqs_only(input_fasta)
    print(f"Loaded {len(sequences)} sequences.")

    print(f"Shuffling with seed {DEFAULT_SEED}...")
    random.seed(DEFAULT_SEED)
    random.shuffle(sequences)

    print(f"Writing {num_shards} .txt shards to {output_dir}...")
    write_shards_txt(sequences, output_dir, num_shards)

    print("Done.")
