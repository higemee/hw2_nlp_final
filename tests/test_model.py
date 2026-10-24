import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model import ModelConfig, Seq2Seq  # noqa: E402


def naive_greedy(model, src, bos, eos, max_new):
    """Greedy decoding that recomputes the whole decoder at every step."""
    outs = []
    for b in range(src.size(0)):
        s = src[b:b + 1, :int((src[b] != 0).sum())]
        seq = [bos]
        for _ in range(max_new):
            nxt = int(model(s, torch.tensor([seq]))[0, -1].argmax())
            if nxt == eos:
                break
            seq.append(nxt)
        outs.append(seq[1:])
    return outs


def random_model(seed, **kw):
    torch.manual_seed(seed)
    model = Seq2Seq(ModelConfig(**kw)).eval()
    with torch.no_grad():  # larger weights make an untrained model's outputs depend on the input
        for p in model.parameters():
            if p.dim() > 1:
                p.mul_(8)
    return model


def test_kv_cache_matches_naive_decoding():
    model = random_model(0, vocab_size=50, d_model=64, n_heads=4, n_enc_layers=2, n_dec_layers=2, max_len=128)
    src = torch.randint(3, 50, (9, 11))
    for r, length in enumerate([11, 3, 7, 1, 11, 5, 9, 2, 6]):
        src[r, length:] = 0
    # EOS = a token that some rows emit and others do not, so rows finish at different steps
    probe = model.generate(src, 1, -1, max_new_tokens=25)
    counts = {}
    for row in probe:
        for tok in set(row):
            counts[tok] = counts.get(tok, 0) + 1
    eos = max(counts, key=lambda k: (0 < counts[k] < len(probe), counts[k]))
    fast = model.generate(src, 1, eos, max_new_tokens=25)
    assert fast == naive_greedy(model, src, 1, eos, 25)
    assert len({len(x) for x in fast}) > 1


def test_padding_does_not_change_output():
    model = random_model(1, vocab_size=40, d_model=32, n_heads=2, n_enc_layers=1, n_dec_layers=1, max_len=64)
    src = torch.randint(3, 40, (1, 6))
    padded = torch.nn.functional.pad(src, (0, 10))
    assert model.generate(src, 1, 2, max_new_tokens=10) == model.generate(padded, 1, 2, max_new_tokens=10)


def test_num_samples():
    model = random_model(2, vocab_size=40, d_model=32, n_heads=2, n_enc_layers=1, n_dec_layers=1, max_len=64)
    src = torch.randint(3, 40, (3, 5))
    out = model.generate(src, 1, 2, max_new_tokens=8, temperature=1.0, top_p=0.9, num_samples=4)
    assert len(out) == 12


if __name__ == "__main__":
    test_kv_cache_matches_naive_decoding()
    test_padding_does_not_change_output()
    test_num_samples()
    print("ok")
