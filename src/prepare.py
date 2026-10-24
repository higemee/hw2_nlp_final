"""Data preparation.

1. Downloads SlavaYus/tinysms_qa_12m (60 train shards + test questions).
2. Holds out n_val questions of the last shard as validation (answer = executed reference program).
3. Converts reference programs to the compact target format (codefmt.py) and trains a byte-level BPE.
4. Drops pairs whose program does not return a number, tokenizes the rest into flat uint16 arrays.

Output: tokenizer.json, train_{src,tgt}.bin, train_{src,tgt}_off.npy, val.jsonl, test.jsonl, stats.json.

    python src/prepare.py --out data/prepared
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codefmt import to_program, to_target  # noqa: E402
from execute import run_programs  # noqa: E402

SPECIALS = ["<pad>", "<bos>", "<eos>"]


def build_tokenizer(texts, vocab_size):
    tok = Tokenizer(models.BPE())
    # one token per digit: numbers are copied from the question digit by digit
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=5,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def find_parquets(root, split):
    files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    return [f for f in files if f"/{split}/" in f or os.path.basename(f).startswith(split)]


def encode_flat(tok, texts):
    enc = tok.encode_batch(texts, add_special_tokens=False)
    lens = np.fromiter((len(e.ids) for e in enc), dtype=np.int64, count=len(enc))
    flat = np.fromiter((i for e in enc for i in e.ids), dtype=np.uint16, count=int(lens.sum()))
    return flat, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf_repo", default="SlavaYus/tinysms_qa_12m")
    ap.add_argument("--parquet_dir", default=None, help="local parquet files instead of downloading")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--tok_sample", type=int, default=600_000, help="rows used to train the tokenizer")
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--max_shards", type=int, default=0, help="use only the first N train shards (debug)")
    ap.add_argument("--exec_filter", type=int, default=1)
    ap.add_argument("--max_src", type=int, default=320)
    ap.add_argument("--max_tgt", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    root = args.parquet_dir
    if root is None:
        from huggingface_hub import snapshot_download
        root = snapshot_download(args.hf_repo, repo_type="dataset", allow_patterns=["*.parquet", "*.json"],
                                 max_workers=8)
    train_files = find_parquets(root, "train") or find_parquets(root, "part")
    test_files = find_parquets(root, "test")
    if args.max_shards:
        train_files = train_files[:args.max_shards]
    assert train_files, f"no train parquet files under {root}"
    print(f"train shards: {len(train_files)}, test shards: {len(test_files)}", flush=True)

    # validation: random questions of the last shard with a numeric reference answer
    rng = np.random.default_rng(args.seed)
    last = pq.read_table(train_files[-1], columns=["question", "code"]).to_pandas()
    val = last.iloc[np.sort(rng.choice(len(last), size=min(args.n_val, len(last) // 10), replace=False))]
    answers = run_programs(val["code"].tolist(), timeout=2.0)
    n_val = 0
    with open(os.path.join(args.out, "val.jsonl"), "w") as f:
        for (q, c), a in zip(val[["question", "code"]].itertuples(index=False), answers):
            if a is not None:
                f.write(json.dumps({"question": q, "code": c, "answer": a}) + "\n")
                n_val += 1
    val_questions = set(val["question"].tolist())
    print(f"val: {n_val}/{len(val)} questions with a numeric answer", flush=True)

    # tokenizer on a sample of questions and targets from up to 20 shards
    tok_path = os.path.join(args.out, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = Tokenizer.from_file(tok_path)
    else:
        step = max(1, len(train_files) // 20)
        per_file = max(1, args.tok_sample // len(train_files)) * step
        sample = []
        for path in train_files[::step]:
            df = pq.read_table(path, columns=["question", "code"]).to_pandas()
            df = df.sample(n=min(per_file, len(df)), random_state=args.seed)
            sample += df["question"].tolist() + [t for t in map(to_target, df["code"]) if t is not None]
        tok = build_tokenizer(sample, args.vocab_size)
        tok.save(tok_path)
        print(f"tokenizer: {len(sample)} texts, vocab {tok.get_vocab_size()} ({time.time() - t0:.0f}s)", flush=True)
        del sample
    assert [tok.token_to_id(t) for t in SPECIALS] == [0, 1, 2]

    f_src = open(os.path.join(args.out, "train_src.bin"), "wb")
    f_tgt = open(os.path.join(args.out, "train_tgt.bin"), "wb")
    src_lens, tgt_lens = [], []
    n_rows = n_val_drop = n_fmt_drop = n_exec_drop = n_len_drop = 0
    for fi, path in enumerate(train_files):
        df = pq.read_table(path, columns=["question", "code"]).to_pandas()
        n_rows += len(df)
        keep = ~df["question"].isin(val_questions).to_numpy()
        n_val_drop += int((~keep).sum())
        targets = [to_target(c) for c in df["code"]]
        fmt_ok = np.array([t is not None for t in targets])
        n_fmt_drop += int((keep & ~fmt_ok).sum())
        keep &= fmt_ok
        if args.exec_filter:
            # the program is rebuilt from the target, so the format round trip is checked as well
            sel = np.nonzero(keep)[0]
            ok = np.array([a is not None for a in run_programs([to_program(targets[i]) for i in sel])], dtype=bool)
            n_exec_drop += int((~ok).sum())
            keep[sel[~ok]] = False
        questions = df["question"][keep].tolist()
        targets = [t for t, k in zip(targets, keep) if k]
        s_flat, s_len = encode_flat(tok, questions)
        t_flat, t_len = encode_flat(tok, targets)
        ok = (s_len > 0) & (s_len <= args.max_src) & (t_len > 0) & (t_len + 1 <= args.max_tgt)
        if not ok.all():
            s_off = np.concatenate([[0], np.cumsum(s_len)])
            t_off = np.concatenate([[0], np.cumsum(t_len)])
            sel = np.nonzero(ok)[0]
            s_flat = np.concatenate([s_flat[s_off[i]:s_off[i + 1]] for i in sel])
            t_flat = np.concatenate([t_flat[t_off[i]:t_off[i + 1]] for i in sel])
            n_len_drop += int((~ok).sum())
            s_len, t_len = s_len[ok], t_len[ok]
        f_src.write(s_flat.tobytes())
        f_tgt.write(t_flat.tobytes())
        src_lens.append(s_len)
        tgt_lens.append(t_len)
        print(f"shard {fi + 1}/{len(train_files)}: kept {sum(map(len, src_lens))}/{n_rows} "
              f"({time.time() - t0:.0f}s)", flush=True)
    f_src.close()
    f_tgt.close()
    src_lens, tgt_lens = np.concatenate(src_lens), np.concatenate(tgt_lens)
    np.save(os.path.join(args.out, "train_src_off.npy"), np.concatenate([[0], np.cumsum(src_lens)]).astype(np.int64))
    np.save(os.path.join(args.out, "train_tgt_off.npy"), np.concatenate([[0], np.cumsum(tgt_lens)]).astype(np.int64))

    n_test = 0
    with open(os.path.join(args.out, "test.jsonl"), "w") as f:
        for path in test_files:
            for q in pq.read_table(path, columns=["question"]).column("question").to_pylist():
                f.write(json.dumps({"question": q}) + "\n")
                n_test += 1

    def percentiles(a):
        return {p: int(np.percentile(a, p)) for p in (50, 90, 99, 99.9)}

    stats = {"n_rows": n_rows, "n_train": int(len(src_lens)), "dropped_val_copies": n_val_drop,
             "dropped_format": n_fmt_drop, "dropped_exec": n_exec_drop, "dropped_length": n_len_drop,
             "n_val": n_val, "n_test": n_test, "vocab_size": tok.get_vocab_size(),
             "src_tokens": int(src_lens.sum()), "tgt_tokens": int(tgt_lens.sum()),
             "src_len_pct": percentiles(src_lens), "tgt_len_pct": percentiles(tgt_lens),
             "seconds": int(time.time() - t0)}
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
