"""Inference under a time limit: greedy decoding for every question, then (while time allows) k sampled programs
per question; all candidates are executed and the most frequent numeric answer is chosen (ties -> greedy).

Writes one {"code": ...} per line in the order of the input questions.

    python src/infer.py --model_dir runs/main --input test.parquet --out submission.jsonl --num_samples 8
"""
import argparse
import json
import os
import sys
import time

T0 = time.time()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from execute import answers_equal, majority_vote, run_programs  # noqa: E402
from generation import generate_codes  # noqa: E402
from model import ModelConfig, Seq2Seq  # noqa: E402


def load_rows(path):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="directory with model.pt and tokenizer.json")
    ap.add_argument("--input", required=True, help=".parquet or .jsonl with a 'question' field")
    ap.add_argument("--out", default="submission.jsonl")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--sample_rows", type=int, default=1024, help="decoder rows per sampling batch")
    ap.add_argument("--time_limit", type=float, default=160.0, help="seconds since the start of the script")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    bundle = torch.load(os.path.join(args.model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model = Seq2Seq(ModelConfig(**bundle["cfg"]))
    model.load_state_dict({k: v.float() for k, v in bundle["model"].items()})
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    no_amp = torch.autocast(device.type, enabled=False)

    rows = load_rows(args.input)
    questions = [r["question"] for r in rows]
    gpu = f" ({torch.cuda.device_count()} x {torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"{len(questions)} questions, {model.num_params() / 1e6:.1f}M params, {device}{gpu}, "
          f"setup {time.time() - T0:.1f}s", flush=True)

    t = time.time()
    greedy = generate_codes(model, tok, questions, device, no_amp, batch_size=args.batch_size,
                            max_new_tokens=args.max_new_tokens)
    t_greedy = time.time() - t
    print(f"greedy: {t_greedy:.1f}s", flush=True)

    k = args.num_samples
    samples = {}
    if k > 0:
        order = np.argsort([len(e.ids) for e in tok.encode_batch(questions, add_special_tokens=False)])
        per_batch = max(1, args.sample_rows // k)
        t = time.time()
        for s in range(0, len(order), per_batch):
            # stop if the next batch would not fit into the limit (the first one is estimated from the greedy pass)
            batch_time = (time.time() - t) / s * per_batch if s else t_greedy * per_batch * k / len(questions)
            if time.time() - T0 + batch_time + 8 > args.time_limit:
                break
            idx = order[s:s + per_batch]
            outs = generate_codes(model, tok, [questions[i] for i in idx], device, no_amp, batch_size=per_batch,
                                  max_new_tokens=args.max_new_tokens, num_samples=k, temperature=args.temperature)
            for j, i in enumerate(idx):
                samples[i] = outs[j * k:(j + 1) * k]
        print(f"sampling: {len(samples)}/{len(questions)} questions x {k} in {time.time() - t:.1f}s", flush=True)

    candidates = [[greedy[i]] + samples.get(i, []) for i in range(len(questions))]
    results = run_programs([c for cs in candidates for c in cs])
    final, answers, pos = [], [], 0
    for cs in candidates:
        code, answer = majority_vote(cs, results[pos:pos + len(cs)])
        final.append(code)
        answers.append(answer)
        pos += len(cs)

    with open(args.out, "w") as f:
        for code in final:
            f.write(json.dumps({"code": code}) + "\n")
    print(f"wrote {args.out}, total time {time.time() - T0:.1f}s", flush=True)

    if "answer" in rows[0]:
        gold = [r["answer"] for r in rows]
        greedy_answers = run_programs(greedy)
        print(f"pass@1 greedy: {np.mean([answers_equal(a, b) for a, b in zip(greedy_answers, gold)]):.4f}")
        print(f"pass@1 voting: {np.mean([answers_equal(a, b) for a, b in zip(answers, gold)]):.4f}")


if __name__ == "__main__":
    main()
