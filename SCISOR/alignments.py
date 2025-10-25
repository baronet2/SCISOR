import torch


def get_deletion_log_alignments(X_0, X_m, pad_id=-1):
    assert X_0.ndim == 2
    b, m, n = X_0.shape[0], X_0.shape[1], X_m.shape[1]

    # log_matching[b][i][j] = log(X_0[b][i] matches X_m[b][j])
    log_matching = torch.where(X_0.unsqueeze(2) == X_m.unsqueeze(1), 0.0, float("-inf"))

    # prefix_dp[b][i][j] = log(# alignments of X_0[b][:j] with X_m[b][:i])
    log_prefix_dp = torch.full(
        (b, n + 1, m + 1), float("-inf"), dtype=torch.float32, device=X_0.device
    )
    log_prefix_dp[:, :, 0] = 0.0  # log(1)

    for j in range(1, m + 1):
        prev = log_prefix_dp[:, :-1, j - 1] + log_matching[:, j - 1, :]
        log_prefix_dp[:, 1:, j] = torch.logcumsumexp(prev, dim=1)

    # suffix_dp[b][i][j] = log(# alignments of X_0[b][j:] with X_m[b][i:])
    log_suffix_dp = torch.full(
        (b, n + 1, m + 1), 0.0, dtype=torch.float32, device=X_0.device
    )
    mask = X_0 == pad_id
    log_suffix_dp[:, -1, :-1] = torch.where(mask, 0.0, float("-inf"))

    for j in range(m - 1, -1, -1):
        prod = log_suffix_dp[:, 1:, j + 1] + log_matching[:, j, :]
        carry = torch.logcumsumexp(prod.flip(dims=[1]), dim=1).flip(dims=[1])
        log_suffix_dp[:, :-1, j] = torch.where(mask[:, j].unsqueeze(1), 0.0, carry)

    # Combine prefix and suffix to get final deletion alignments
    log_deletion_alignments = torch.logsumexp(
        log_prefix_dp[:, :-1] + log_suffix_dp[:, 1:], dim=-1
    )
    log_deletion_alignments = torch.where(
        X_m != pad_id, log_deletion_alignments, float("-inf")
    )

    M = (X_m != pad_id).sum(dim=1) - (X_0 != pad_id).sum(dim=1)
    return log_deletion_alignments, torch.logsumexp(
        log_deletion_alignments, dim=1
    ) - M.log()


def get_deletion_alignments(X_0, X_m, pad_id=-1):
    log_deletion_alignments, log_full_alignments = get_deletion_log_alignments(
        X_0, X_m, pad_id
    )
    return log_deletion_alignments.to(torch.float64).exp(), log_full_alignments.to(
        torch.float64
    ).exp()


def log_ali(X_0, X_m, pad_id=-1):
    assert X_0.ndim == 2
    b, m, n = X_0.shape[0], X_0.shape[1], X_m.shape[1]

    # log_matching[b][i][j] = log(X_0[b][i] matches X_m[b][j])
    log_matching = torch.where(X_0.unsqueeze(2) == X_m.unsqueeze(1), 0.0, float("-inf"))

    # prefix_dp[b][i][j] = # log alignments of X_0[b][:j] with X_m[b][:i]
    log_prefix_dp = torch.full(
        (b, n + 1, m + 1), float("-inf"), dtype=torch.float32, device=X_0.device
    )
    log_prefix_dp[:, :, 0] = 0.0  # log(1)

    for j in range(1, m + 1):
        prev = log_prefix_dp[:, :-1, j - 1] + log_matching[:, j - 1, :]
        log_prefix_dp[:, 1:, j] = torch.logcumsumexp(prev, dim=1)

    X_0_lengths = (X_0 != pad_id).sum(dim=1)
    X_m_lengths = (X_m != pad_id).sum(dim=1)
    log_alignments = log_prefix_dp[range(b), X_m_lengths, X_0_lengths]
    return log_alignments


def ali(X_0, X_m, pad_id=-1):
    log_alignments = log_ali(X_0, X_m, pad_id)
    return log_alignments.to(torch.float64).exp()


if __name__ == "__main__":
    normalize = lambda arr: arr / arr.sum(dim=-1, keepdims=True)

    test_X_0 = torch.tensor([[0, 1, 2]])
    test_X_t = torch.tensor([[0, 0, 1, 0, 1, 2, 2]])
    result = get_deletion_alignments(test_X_0, test_X_t)
    assert torch.allclose(
        result[0], torch.tensor([6, 6, 6, 8, 4, 5, 5], dtype=torch.float64)
    )
    assert torch.allclose(result[1], torch.tensor([10], dtype=torch.float64))

    test_X_0 = torch.tensor([[0, 1, 2, -1, -1]])
    test_X_t = torch.tensor([[0, 0, 1, 0, 1, 2, 2]])
    result = get_deletion_alignments(test_X_0, test_X_t)
    assert torch.allclose(
        result[0], torch.tensor([6, 6, 6, 8, 4, 5, 5], dtype=torch.float64)
    )
    assert torch.allclose(result[1], torch.tensor([10], dtype=torch.float64))

    test_X_0 = torch.tensor([[0, 1, 2, -1, -1], [0, 1, 2, -1, -1]])
    test_X_t = torch.tensor([[0, 0, 1, 0, 1, 2, 2], [0, 0, 1, 0, 1, 2, 2]])
    result = get_deletion_alignments(test_X_0, test_X_t)
    assert torch.allclose(
        result[0],
        torch.tensor(
            [[6, 6, 6, 8, 4, 5, 5], [6, 6, 6, 8, 4, 5, 5]], dtype=torch.float64
        ),
    )
    assert torch.allclose(result[1], torch.tensor([10, 10], dtype=torch.float64))

    test_X_0 = torch.tensor([[0, 1, 2], [0, 1, -1], [1, 1, 1], [0, 1, -1]])
    test_X_t = torch.tensor(
        [
            [0, 2, 1, 2, -1, -1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 2, 1, 2, -1],
        ]
    )
    expected_output = torch.tensor(
        [[0, 1, 0, 0, 0, 0], [6] * 6, [1] * 3 + [0] * 3, [1, 1, 2, 0, 2, 0]],
        dtype=torch.float64,
    )
    result = get_deletion_alignments(test_X_0, test_X_t)
    assert torch.allclose(result[0], expected_output)
    assert torch.allclose(result[1], torch.tensor([1, 9, 1, 2], dtype=torch.float64))
    assert torch.allclose(
        ali(test_X_0, test_X_t), torch.tensor([1, 9, 1, 2], dtype=torch.float64)
    )

    import random
    import time

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    def generate_random_sequence(length):
        return random.choices(amino_acids, k=length)

    def get_p_protein(str_x, str_y, device="cuda"):
        X_0 = torch.tensor(
            [list(map(amino_acids.index, x_0)) for x_0 in str_x], device=device
        )
        X_m = torch.tensor(
            [list(map(amino_acids.index, x_t)) for x_t in str_y], device=device
        )
        start_time = time.time()
        out = normalize(get_deletion_alignments(X_0, X_m)[0])
        end_time = time.time()
        return out, end_time - start_time

    random.seed(42)
    L = 1000
    m = 3000
    n = 256
    device = "cuda"
    X_0 = ["".join(generate_random_sequence(L)) for _ in range(n)]
    X_m = [generate_random_sequence(L + m) for _ in range(n)]

    for x_0, x_m in zip(X_0, X_m):
        X_0_indices = sorted(random.sample(range(L + m), L))
        for i, idx in enumerate(X_0_indices):
            x_m[idx] = x_0[i]

    X_m = ["".join(x_m) for x_m in X_m]

    result, duration = get_p_protein(X_0, X_m, device)
    print(f"Time taken: {duration:.6f} seconds")
    print(result)
    print(
        result.min().item(),
        result.median().item(),
        result.mean().item(),
        result.max().item(),
    )
