import argparse
import math
import time
import os
import sys
import torch
import torch.nn.functional as F
from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS  = 2000
MAX_PARAMS = 2_000_000

def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i     : i + block    ] for i in ix])
    y = torch.stack([ids[i + 1 : i + block + 1] for i in ix])
    return x.to(device), y.to(device)

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",        required=True)
    ap.add_argument("--steps",       type=int,   default=2000)
    ap.add_argument("--lr",          type=float, default=3e-3)
    ap.add_argument("--min_lr",      type=float, default=1e-4)
    ap.add_argument("--warmup",      type=int,   default=200)
    ap.add_argument("--batch",       type=int,   default=16)
    ap.add_argument("--out",         default="ckpt.pt")
    args = ap.parse_args()

    device = "cpu"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bpe_path   = os.path.join(script_dir, "bpe_merges.json")
    
    print("Loading data & training BPE Tokenizer...", flush=True)
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()

    tok = tokenizer_mod.BPETokenizer()
    if not os.path.exists(bpe_path):
        tok.train(text, num_merges=768)
        tok.save(bpe_path)
    else:
        tok = tokenizer_mod.load(bpe_path)

    tokens = tok.encode(text)
    ids = torch.tensor(tokens, dtype=torch.long)
    print(f"Tokenization complete. Total tokens: {len(ids)}", flush=True)

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    
    n = model.n_params()
    print(f"Model parameters: {n} (cap {MAX_PARAMS})", flush=True)
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS} params"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    model.train()
    t0 = time.time()
    losses = []

    for step in range(args.steps):
        lr = get_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        x, y = get_batch(ids, cfg.block_size, args.batch, device)

        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())

        if step % 200 == 0 or step == args.steps - 1:
            avg = sum(losses[-200:]) / len(losses[-200:])
            print(f"Step {step:04d} | Loss: {avg:.4f} | LR: {lr:.5f}", flush=True)

    print("Training complete. Saving checkpoint...", flush=True)
    torch.save({
        "model": model.state_dict(),
        "config": {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))},
        "steps": args.steps,
        "train_loss_curve": losses,
    }, args.out)

if __name__ == '__main__':
    main()
