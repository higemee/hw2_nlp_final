"""Tokenized training data and token-budget batching."""
import os

import numpy as np
import torch

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


class TokenData:
    """Flat uint16 token arrays + offsets produced by prepare.py.

    The arrays are read into RAM: a batch gathers a few hundred random rows, and random reads from a
    memory-mapped file on a network filesystem (Kaggle inputs) made training ~20x slower.
    """

    def __init__(self, data_dir, split="train", in_memory=True):
        def load(name):
            path = os.path.join(data_dir, f"{split}_{name}.bin")
            return np.fromfile(path, dtype=np.uint16) if in_memory else np.memmap(path, dtype=np.uint16, mode="r")

        self.src, self.tgt = load("src"), load("tgt")
        self.src_off = np.load(os.path.join(data_dir, f"{split}_src_off.npy"))
        self.tgt_off = np.load(os.path.join(data_dir, f"{split}_tgt_off.npy"))
        self.src_len = np.diff(self.src_off)
        self.tgt_len = np.diff(self.tgt_off) + 1  # + BOS in the decoder input / EOS in the target

    def __len__(self):
        return len(self.src_len)


def _round_up(x, m=8):
    return (x + m - 1) // m * m


def make_batches(data, max_tokens, epoch, seed=0, chunk=200_000, bucket=8):
    """Shuffle, sort each chunk by (target length // bucket, source length) and cut it into batches with
    B * (max_src_len + max_tgt_len) <= max_tokens; the batch order is shuffled again."""
    rng = np.random.default_rng(seed * 1000 + epoch)
    idx = rng.permutation(len(data))
    batches = []
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        part = part[np.lexsort((data.src_len[part], data.tgt_len[part] // bucket))]
        src_len, tgt_len = data.src_len[part].tolist(), data.tgt_len[part].tolist()
        start, max_s, max_t = 0, 0, 0
        for j in range(len(part)):
            new_s, new_t = max(max_s, src_len[j]), max(max_t, tgt_len[j])
            if j > start and (j - start + 1) * (_round_up(new_s) + _round_up(new_t)) > max_tokens:
                batches.append(part[start:j])
                start, new_s, new_t = j, src_len[j], tgt_len[j]
            max_s, max_t = new_s, new_t
        batches.append(part[start:])
    return [batches[i] for i in rng.permutation(len(batches))]


def collate(data, ids):
    B = len(ids)
    S = _round_up(int(data.src_len[ids].max()))
    T = _round_up(int(data.tgt_len[ids].max()))
    src = np.full((B, S), PAD_ID, dtype=np.int64)
    tgt_in = np.full((B, T), PAD_ID, dtype=np.int64)
    tgt_out = np.full((B, T), PAD_ID, dtype=np.int64)
    for r, i in enumerate(ids):
        s = data.src[data.src_off[i]:data.src_off[i + 1]]
        t = data.tgt[data.tgt_off[i]:data.tgt_off[i + 1]]
        src[r, :len(s)] = s
        tgt_in[r, 0] = BOS_ID
        tgt_in[r, 1:len(t) + 1] = t
        tgt_out[r, :len(t)] = t
        tgt_out[r, len(t)] = EOS_ID
    return torch.from_numpy(src), torch.from_numpy(tgt_in), torch.from_numpy(tgt_out)


class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, data, batches):
        self.data, self.batches = data, batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return collate(self.data, self.batches[i])


def pad_batch(seqs, pad_id=PAD_ID, multiple=8):
    length = _round_up(max(1, max(len(s) for s in seqs)), multiple)
    out = torch.full((len(seqs), length), pad_id, dtype=torch.long)
    for r, s in enumerate(seqs):
        out[r, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out
