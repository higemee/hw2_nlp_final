"""Training: DDP via torchrun, fp16 autocast + GradScaler, token-budget batches, AdamW and a
warmup-stable-decay LR schedule over a wall-clock budget. Periodic validation (teacher-forced loss and greedy
pass@1), checkpoints every 30 minutes, resumable.

    torchrun --nproc_per_node=2 src/train.py --data_dir data/prepared --out_dir runs/main --time_budget_h 10
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codefmt import to_target  # noqa: E402
from data import BOS_ID, EOS_ID, BatchDataset, TokenData, make_batches, pad_batch  # noqa: E402
from execute import answers_equal, run_programs  # noqa: E402
from generation import generate_codes  # noqa: E402
from model import ModelConfig, Seq2Seq  # noqa: E402


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_enc", type=int, default=6)
    ap.add_argument("--n_dec", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=32768, help="padded tokens per GPU batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr_ratio", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--decay_frac", type=float, default=0.25, help="final fraction of the run with LR decay")
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--time_budget_h", type=float, default=10.0)
    ap.add_argument("--max_steps", type=int, default=0, help="if > 0, the schedule is defined by steps")
    ap.add_argument("--eval_every_min", type=float, default=45)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--ckpt_every_min", type=float, default=30)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--resume", default="auto", help="'auto' (out_dir/ckpt.pt if it exists), 'none' or a path")
    ap.add_argument("--num_workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


class Logger:
    def __init__(self, path, enabled):
        self.enabled = enabled
        self.f = open(path, "a") if enabled else None

    def __call__(self, msg, **record):
        if self.enabled:
            print(msg, flush=True)
            if record:
                self.f.write(json.dumps(record) + "\n")
                self.f.flush()


@torch.no_grad()
def evaluate(model, tok, val, device, amp_ctx, n_eval):
    model.eval()
    loss_sum, n_tokens = 0.0, 0
    for s in range(0, len(val), 128):
        part = val[s:s + 128]
        src = pad_batch([tok.encode(r["question"], add_special_tokens=False).ids for r in part])
        tgt = [tok.encode(to_target(r["code"]) or "", add_special_tokens=False).ids for r in part]
        tgt_in = pad_batch([[BOS_ID] + t for t in tgt])
        tgt_out = pad_batch([t + [EOS_ID] for t in tgt], multiple=tgt_in.size(1))
        with amp_ctx:
            loss, n = model(src.to(device), tgt_in.to(device), tgt_out.to(device))
        loss_sum += float(loss)
        n_tokens += int(n)
    part = val[:n_eval]
    t0 = time.time()
    codes = generate_codes(model, tok, [r["question"] for r in part], device, amp_ctx, max_new_tokens=256)
    gen_time = time.time() - t0
    preds = run_programs(codes)
    model.train()
    return {"val_loss": loss_sum / max(n_tokens, 1),
            "pass1": float(np.mean([answers_equal(p, r["answer"]) for p, r in zip(preds, part)])),
            "exec_ok": float(np.mean([p is not None for p in preds])),
            "gen_s": gen_time}


def lr_factor(step, progress, args):
    warmup = min(1.0, (step + 1) / args.warmup_steps)
    decay = min(1.0, max(0.0, (1.0 - progress) / args.decay_frac))
    return warmup * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * decay)


def main():
    args = get_args()
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    cuda = torch.cuda.is_available()
    if ddp:
        dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(minutes=60))
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if cuda:
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if cuda else "cpu")
    master = rank == 0
    os.makedirs(args.out_dir, exist_ok=True)
    log = Logger(os.path.join(args.out_dir, "log.jsonl"), master)
    torch.manual_seed(args.seed + rank)
    amp_ctx = torch.autocast("cuda", dtype=torch.float16) if cuda else torch.autocast("cpu", enabled=False)

    tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    with open(os.path.join(args.data_dir, "val.jsonl")) as f:
        val = [json.loads(line) for line in f]
    data = TokenData(args.data_dir, "train")

    cfg = ModelConfig(vocab_size=tok.get_vocab_size(), d_model=args.d_model, n_heads=args.n_heads,
                      n_enc_layers=args.n_enc, n_dec_layers=args.n_dec)
    model = Seq2Seq(cfg).to(device)
    log(f"params {model.num_params() / 1e6:.1f}M, train rows {len(data)}, world size {world}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=cuda or None)
    scaler = torch.amp.GradScaler(device.type, enabled=cuda)

    step, epoch, batch_pos, elapsed_prev = 0, 0, 0, 0.0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    resume = {"auto": ckpt_path if os.path.exists(ckpt_path) else None, "none": None}.get(args.resume, args.resume)
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, epoch, batch_pos, elapsed_prev = ck["step"], ck["epoch"], ck["batch_pos"], ck["elapsed"]
        log(f"resumed from {resume} at step {step}")
        del ck
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index] if cuda else None) if ddp else model

    t_start = time.time()

    def elapsed():
        return elapsed_prev + time.time() - t_start

    def sync_progress():
        # identical on all ranks, so the LR and the stopping decision are identical too
        if args.max_steps:
            return step / args.max_steps
        t = torch.tensor([elapsed()], device=device, dtype=torch.float64)
        if ddp:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t) / (args.time_budget_h * 3600)

    def save(final=False):
        if not master:
            return
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
                 "epoch": epoch, "batch_pos": batch_pos, "elapsed": elapsed(), "cfg": cfg.to_dict()}
        torch.save(state, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        weights = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
        name = os.path.join(args.out_dir, "model.pt" if final else "model_last.pt")
        torch.save({"model": weights, "cfg": cfg.to_dict(), "step": step}, name + ".tmp")
        os.replace(name + ".tmp", name)
        for fname in ("tokenizer.json", "val.jsonl"):
            shutil.copy(os.path.join(args.data_dir, fname), os.path.join(args.out_dir, fname))

    progress = sync_progress()
    last_eval = last_ckpt = t_log = time.time()
    n_tok, t_wait, loss_acc, n_acc = 0, 0.0, torch.zeros((), device=device), 0
    done = False
    while not done:
        batches = make_batches(data, args.max_tokens, epoch, seed=args.seed)
        per_rank = len(batches) // world
        loader = torch.utils.data.DataLoader(BatchDataset(data, batches[rank::world][batch_pos:per_rank]),
                                             batch_size=None, num_workers=args.num_workers, pin_memory=cuda,
                                             prefetch_factor=8 if args.num_workers else None)
        log(f"epoch {epoch}: {per_rank} batches per rank, starting at {batch_pos}")
        it = iter(loader)
        while not done:
            t_fetch = time.time()
            batch = next(it, None)
            if batch is None:
                break
            t_wait += time.time() - t_fetch
            src, tgt_in, tgt_out = batch
            n_tok += int((src != 0).sum()) + int((tgt_out != 0).sum())
            src, tgt_in, tgt_out = (x.to(device, non_blocking=True) for x in (src, tgt_in, tgt_out))
            batch_pos += 1

            lr = args.lr * lr_factor(step, progress, args)
            for g in opt.param_groups:
                g["lr"] = lr
            with amp_ctx:
                loss_sum, n_target = ddp_model(src, tgt_in, tgt_out)
                loss = loss_sum / n_target
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            loss_acc += loss.detach()
            n_acc += 1
            step += 1
            if step % 20 == 0 or args.max_steps:
                progress = sync_progress()
                done = progress >= 1.0

            if step % args.log_every == 0:
                dt = time.time() - t_log
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0
                train_loss = float(loss_acc) / n_acc
                log(f"step {step} ep {epoch} | loss {train_loss:.4f} | lr {lr:.2e} | gnorm {float(gnorm):.2f} | "
                    f"{progress:.1%} | {elapsed() / 3600:.2f}h | {n_tok * world / dt:.0f} tok/s | "
                    f"data wait {t_wait / dt:.0%} | mem {mem:.1f}G",
                    kind="train", step=step, epoch=epoch, loss=train_loss, lr=lr, elapsed=elapsed(),
                    tok_s=n_tok * world / dt)
                t_log, n_tok, t_wait, n_acc = time.time(), 0, 0.0, 0
                loss_acc.zero_()

            if step % 20 == 0 and not done:
                # rank 0 decides, so that all ranks enter the checkpoint / evaluation branches together
                now = time.time()
                flags = torch.tensor([now - last_eval > args.eval_every_min * 60,
                                      now - last_ckpt > args.ckpt_every_min * 60], device=device, dtype=torch.int32)
                if ddp:
                    dist.broadcast(flags, 0)
                if flags[1]:
                    save()
                    last_ckpt = time.time()
                if flags[0]:
                    if master:
                        r = evaluate(model, tok, val, device, amp_ctx, args.n_eval)
                        log(f"[eval] step {step}: val_loss {r['val_loss']:.4f} pass@1 {r['pass1']:.4f} "
                            f"exec_ok {r['exec_ok']:.3f}", kind="eval", step=step, elapsed=elapsed(), **r)
                    if ddp:
                        dist.barrier()
                    last_eval = time.time()
        if not done:
            epoch += 1
            batch_pos = 0

    save(final=True)
    if master:
        r = evaluate(model, tok, val, device, amp_ctx, len(val))
        log(f"[final] step {step}: val_loss {r['val_loss']:.4f} pass@1 {r['pass1']:.4f} exec_ok {r['exec_ok']:.3f}",
            kind="final", step=step, elapsed=elapsed(), **r)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
