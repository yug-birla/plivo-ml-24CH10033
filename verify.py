import torch
from model import GPT, Config

cfg = Config()
model = GPT(cfg)
n = model.n_params()
print(f"Parameter count: {n:,}")
assert n <= 2_000_000, f"OVER BUDGET: {n:,}"
print("PARAM CHECK: OK")

x = torch.randint(0, 256, (2, 64))
logits, loss = model(x, x)
print(f"Forward pass: logits={logits.shape}, loss={loss.item():.4f}")
print("FORWARD CHECK: OK")

import tokenizer as tokenizer_mod
tok = tokenizer_mod.load()
test = "Hello world! नमस्ते दुनिया"
encoded = tok.encode(test)
decoded = tok.decode(encoded)
assert decoded == test, f"Tokenizer roundtrip failed: {decoded!r} != {test!r}"
print(f"Tokenizer roundtrip: OK (vocab={tok.vocab_size})")
