"""
MLX training script for NLQ -> Python code generation.
Runs natively on Apple Silicon. No CUDA needed.

Usage:
    python train_mlx.py

Prerequisites:
    python download_code_data.py   # download dataset
    python prepare.py --num-shards 5  # train tokenizer (uses prepare.py for tokenizer only)
"""

import os
import sys
import math
import time
import json
import pickle
import random
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# Model
VOCAB_SIZE = 2048
MAX_SEQ_LEN = 512
DEPTH = 5
N_EMBD = 320
HEAD_DIM = 64
N_HEAD = N_EMBD // HEAD_DIM  # 5
N_KV_HEAD = N_HEAD
MLP_RATIO = 4
USE_SWIGLU = False
DROPOUT = 0.2

# Training
BATCH_SIZE = 8
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.5
WARMUP_STEPS = 150
TIME_BUDGET = 300
GRAD_CLIP = 1.0
USE_LION = False

# ---------------------------------------------------------------------------
# Tokenizer (reuse the one trained by prepare.py)
# ---------------------------------------------------------------------------

def load_tokenizer():
    path = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    if not os.path.exists(path):
        print("No tokenizer found. Run these first:")
        print("  python download_code_data.py")
        print("  python prepare.py --num-shards 5")
        sys.exit(1)
    with open(path, "rb") as f:
        enc = pickle.load(f)
    return enc

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data(enc):
    """Load all parquet shards into tokenized sequences."""
    parquet_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    if not parquet_files:
        print("No data found. Run: python download_code_data.py")
        sys.exit(1)

    # Split: last shard is val
    val_file = parquet_files[-1]
    train_files = parquet_files[:-1]

    bos_id = enc.encode_single_token("<|reserved_0|>")

    def tokenize_shard(filepath):
        pf = pq.ParquetFile(os.path.join(DATA_DIR, filepath))
        all_tokens = []
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                tokens = [bos_id] + enc.encode_ordinary(text)
                all_tokens.extend(tokens)
        return all_tokens

    print("Tokenizing training data...")
    train_tokens = []
    for f in train_files:
        train_tokens.extend(tokenize_shard(f))
    print(f"  Train: {len(train_tokens):,} tokens")

    print("Tokenizing validation data...")
    val_tokens = tokenize_shard(val_file)
    print(f"  Val: {len(val_tokens):,} tokens")

    return train_tokens, val_tokens


def make_batches(tokens, batch_size, seq_len, shuffle=True):
    """Create batches of (input, target) pairs."""
    tokens = np.array(tokens, dtype=np.int32)
    # Chunk into sequences of seq_len+1
    n_seqs = len(tokens) // (seq_len + 1)
    tokens = tokens[:n_seqs * (seq_len + 1)]
    tokens = tokens.reshape(n_seqs, seq_len + 1)

    indices = list(range(n_seqs))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, n_seqs - batch_size + 1, batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = tokens[batch_idx]
        x = mx.array(batch[:, :-1])
        y = mx.array(batch[:, 1:])
        yield x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    n_layer: int = DEPTH
    n_head: int = N_HEAD
    n_kv_head: int = N_KV_HEAD
    n_embd: int = N_EMBD
    head_dim: int = HEAD_DIM
    seq_len: int = MAX_SEQ_LEN


class RMSNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.weight = mx.ones((dims,))

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, eps=1e-5)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=False)
        self.rope = nn.RoPE(config.head_dim)

    def __call__(self, x, mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        # GQA: repeat k,v if needed
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v

        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_dim = MLP_RATIO * config.n_embd
        if USE_SWIGLU:
            # SwiGLU: gate and up projections, then SiLU gating
            self.gate = nn.Linear(config.n_embd, mlp_dim, bias=False)
            self.up = nn.Linear(config.n_embd, mlp_dim, bias=False)
        else:
            self.up = nn.Linear(config.n_embd, mlp_dim, bias=False)
            self.gate = None
        self.down = nn.Linear(mlp_dim, config.n_embd, bias=False)

    def __call__(self, x):
        if self.gate is not None:
            return self.down(nn.silu(self.gate(x)) * self.up(x))
        else:
            return self.down(mx.maximum(self.up(x), 0) ** 2)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.drop = nn.Dropout(DROPOUT) if DROPOUT > 0 else None

    def __call__(self, x, mask=None, train=False):
        h = self.attn(self.norm1(x), mask)
        if self.drop and train:
            h = self.drop(h)
        x = x + h
        h = self.mlp(self.norm2(x))
        if self.drop and train:
            h = self.drop(h)
        x = x + h
        return x


TIED_EMBEDDINGS = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.norm = RMSNorm(config.n_embd)
        if not TIED_EMBEDDINGS:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.emb_drop = nn.Dropout(DROPOUT) if DROPOUT > 0 else None

    def __call__(self, x, train=False):
        B, T = x.shape
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        mask = mask.astype(self.wte.weight.dtype)

        h = self.wte(x)
        if self.emb_drop and train:
            h = self.emb_drop(h)
        for block in self.blocks:
            h = block(h, mask, train=train)
        h = self.norm(h)
        if TIED_EMBEDDINGS:
            logits = h @ self.wte.weight.T
        else:
            logits = self.lm_head(h)
        return logits


def count_params(model):
    return sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def loss_fn(model, x, y):
    logits = model(x, train=True)
    logits = logits.astype(mx.float32)
    loss = nn.losses.cross_entropy(logits, y, reduction='mean')
    return loss


def train():
    print("=" * 60)
    print("NLQ -> Python Code | MLX Training")
    print("=" * 60)

    # Load tokenizer
    enc = load_tokenizer()
    actual_vocab = enc.n_vocab
    print(f"Tokenizer vocab: {actual_vocab}")

    # Load data
    train_tokens, val_tokens = load_all_data(enc)

    # Build model
    config = GPTConfig(vocab_size=actual_vocab)
    model = GPT(config)
    mx.eval(model.parameters())

    n_params = count_params(model)
    print(f"\nModel: {n_params / 1e6:.1f}M parameters")
    print(f"  Layers: {config.n_layer}, Dim: {config.n_embd}, Heads: {config.n_head}")
    print(f"  Seq len: {config.seq_len}, Vocab: {config.vocab_size}")

    # Optimizer with warmup cosine schedule
    warmup = optim.linear_schedule(1e-7, LEARNING_RATE, steps=WARMUP_STEPS)
    cosine = optim.cosine_decay(LEARNING_RATE, decay_steps=5000)
    schedule = optim.join_schedules([warmup, cosine], [WARMUP_STEPS])
    if USE_LION:
        optimizer = optim.Lion(learning_rate=schedule, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(learning_rate=schedule, weight_decay=WEIGHT_DECAY)

    # Compile loss + grad
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print(f"\nTraining for {TIME_BUDGET}s...")
    print(f"  Batch size: {BATCH_SIZE}, Seq len: {MAX_SEQ_LEN}")
    print()

    step = 0
    epoch = 0
    best_val_loss = float('inf')
    t_start = time.time()
    smooth_loss = 0
    total_tokens = 0

    while True:
        epoch += 1
        batches = list(make_batches(train_tokens, BATCH_SIZE, MAX_SEQ_LEN, shuffle=True))

        for x, y in batches:
            t0 = time.time()

            loss, grads = loss_and_grad(model, x, y)

            # Gradient clipping
            grads, _ = optim.clip_grad_norm(grads, max_norm=GRAD_CLIP)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            dt = time.time() - t0
            loss_val = loss.item()
            total_tokens += BATCH_SIZE * MAX_SEQ_LEN
            step += 1

            # EMA smooth loss
            beta = 0.95
            smooth_loss = beta * smooth_loss + (1 - beta) * loss_val if step > 1 else loss_val
            debiased = smooth_loss / (1 - beta ** step)

            elapsed = time.time() - t_start
            remaining = max(0, TIME_BUDGET - elapsed)
            tok_per_sec = BATCH_SIZE * MAX_SEQ_LEN / dt

            if step % 5 == 0 or step == 1:
                print(f"\r  step {step:04d} | loss: {debiased:.4f} | {tok_per_sec:.0f} tok/s | epoch {epoch} | {remaining:.0f}s left   ", end="", flush=True)

            if math.isnan(loss_val) or loss_val > 50:
                print(f"\n\nLoss exploded ({loss_val:.2f}), aborting.")
                sys.exit(1)

            if elapsed >= TIME_BUDGET:
                break

        if time.time() - t_start >= TIME_BUDGET:
            break

    print()
    total_time = time.time() - t_start
    print(f"\nTraining complete: {step} steps, {total_tokens/1e6:.1f}M tokens, {total_time:.1f}s")

    # Validation
    print("Evaluating on validation set...")
    val_batches = list(make_batches(val_tokens, BATCH_SIZE, MAX_SEQ_LEN, shuffle=False))
    val_losses = []
    for x, y in val_batches[:50]:  # up to 50 batches
        loss = loss_fn(model, x, y)
        mx.eval(loss)
        val_losses.append(loss.item())
    val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
    print(f"  Val loss: {val_loss:.4f}")

    # Save checkpoint
    save_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights using MLX safetensors format
    weights = dict(nn.utils.tree_flatten(model.parameters()))
    ckpt_path = os.path.join(save_dir, "model.safetensors")
    mx.save_safetensors(ckpt_path, weights)
    print(f"Saved weights to {ckpt_path}")

    # Save config separately
    config_path = os.path.join(save_dir, "config.json")
    meta = {
        "config": asdict(config),
        "val_loss": val_loss,
        "n_params_M": n_params / 1e6,
        "step": step,
        "total_tokens_M": total_tokens / 1e6,
        "training_seconds": total_time,
    }
    with open(config_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved config to {config_path}")

    # Quick generation test
    print("\n--- Quick generation test ---")
    test_prompts = [
        "Write a function to reverse a string",
        "Write a function to check if a number is prime",
    ]
    for prompt in test_prompts:
        print(f"\nQ: {prompt}")
        code = generate_text(model, enc, config, prompt, max_tokens=128)
        print(f"A:\n{code}")

    print("\n--- Done! ---")
    print(f"Run the webapp: python app.py")


# ---------------------------------------------------------------------------
# Generation (for testing + webapp)
# ---------------------------------------------------------------------------

def generate_text(model, enc, config, prompt, max_tokens=256, temperature=0.7, top_k=40):
    """Generate code from a natural language prompt."""
    formatted = f"### Question\n{prompt}\n### Python Code\n"
    tokens = enc.encode_ordinary(formatted)

    for i in range(max_tokens):
        # Crop to max seq len
        input_tokens = tokens[-config.seq_len:]
        x = mx.array([input_tokens])

        logits = model(x)
        logits = logits[:, -1, :].astype(mx.float32)
        mx.eval(logits)

        if temperature > 0:
            logits = logits / temperature
            # Top-k
            if top_k > 0:
                top_values = mx.topk(logits.squeeze(0), k=min(top_k, logits.shape[-1]))
                threshold = top_values[-1]
                logits = mx.where(logits < threshold, mx.array(float('-inf')), logits)
            # categorical expects unnormalized logits
            next_token = mx.random.categorical(logits.squeeze(0))
            mx.eval(next_token)
        else:
            next_token = mx.argmax(logits, axis=-1).squeeze(0)
            mx.eval(next_token)

        next_token_id = next_token.item()
        tokens.append(next_token_id)

        # Check for stop every 10 tokens (decoding is expensive)
        if i % 10 == 9:
            decoded = enc.decode(tokens)
            code_part = decoded.split("### Python Code\n", 1)[-1] if "### Python Code\n" in decoded else ""
            if "### Question" in code_part:
                break

    # Extract code
    full_text = enc.decode(tokens)
    if "### Python Code\n" in full_text:
        code = full_text.split("### Python Code\n", 1)[1]
        if "### Question" in code:
            code = code.split("### Question")[0]
    else:
        code = full_text
    # Strip special tokens
    for tok in ["<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>", "<|reserved_3|>"]:
        code = code.replace(tok, "")
    return code.strip()


if __name__ == "__main__":
    train()
