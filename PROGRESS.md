# NLQ -> Python Code Generator

Train a tiny (6.8M parameter) GPT model that takes natural language questions and generates Python code. Runs entirely on a Mac with Apple Silicon using MLX.

---

## Quick Start (3 commands)

```bash
# 1. Download code dataset (~14K examples from CodeAlpaca + synthetic)
python download_code_data.py

# 2. Train a BPE tokenizer on the data
python prepare.py --num-shards 5

# 3. Train the model (5 minutes on Apple Silicon)
python train_mlx.py
```

Then run the webapp:
```bash
pip install flask
python app.py
# Open http://localhost:8080
```

## What it does

You type a natural language question, it generates Python code:

```
Q: Write a function to reverse a string
A: def reverse_string(s):
       return s[::-1]

Q: Write a function to do binary search in a sorted list
A: def binary_search(list, target):
       low = 0
       high = len(list) - 1
       while low <= high:
           mid = (low + high) // 2
           if list[mid] == target:
               return mid
           elif list[mid] < target:
               low = mid + 1
           else:
               high = mid - 1
       return -1

Q: Write a function to find the factorial of a number
A: def factorial(n):
       if n == 0: return 1
       else: return n * factorial(n-1)
```

The webapp also has a "Run Code" button that executes the generated code and shows output.

---

## Architecture

| Component | Value |
|-----------|-------|
| Model type | GPT (decoder-only transformer) |
| Parameters | 6.8M |
| Layers | 5 |
| Embedding dim | 320 |
| Attention heads | 5 (head_dim=64) |
| Vocab size | 2,048 (BPE) |
| Context length | 512 tokens |
| Activation | ReLU squared |
| Positional encoding | RoPE |
| Weight tying | Yes (embedding = output projection) |
| Dropout | 0.1 |
| Framework | MLX (Apple Silicon native) |

## Training details

| Setting | Value |
|---------|-------|
| Dataset | CodeAlpaca-20k (filtered Python) + 52 synthetic examples |
| Unique examples | ~14,500 |
| Training repetition | 5x (to compensate for small dataset) |
| Optimizer | AdamW (lr=3e-3, wd=0.3) |
| Batch size | 32 |
| Training time | 5 minutes |
| Throughput | ~86K tokens/sec on Apple Silicon |
| Final val loss | 1.71 |

---

## Files

| File | What it does |
|------|-------------|
| `download_code_data.py` | Downloads CodeAlpaca dataset, formats as parquet shards |
| `prepare.py` | Trains BPE tokenizer on the data |
| `train_mlx.py` | Defines model + trains it using MLX (Apple Silicon) |
| `app.py` | Flask webapp with generate + execute buttons |
| `train.py` | Original autoresearch training script (CUDA/GPU only, not used) |
| `results.tsv` | Log of all experiments run during autoresearch loop |
| `checkpoint/` | Saved model weights (model.safetensors) + config |

---

## How the autoresearch loop worked

We ran 22 experiments in total, each training for 5 minutes. An AI agent (Claude) autonomously:
1. Modified hyperparameters/architecture in `train_mlx.py`
2. Trained for 5 minutes
3. Checked validation loss
4. Kept improvements, discarded regressions
5. Repeated

### Key findings from the experiment loop

| What we tried | Result | Lesson |
|--------------|--------|--------|
| Baseline (depth=5, lr=3e-4) | val=2.07 | Starting point |
| Bigger model (15M params, 8 layers) | val=2.17 | **Bigger model = more overfitting** with small data |
| SwiGLU activation (LLaMA-style) | val=2.06 | Extra gate params waste parameter budget |
| 8x MLP expansion + GQA | val=2.16 | Fun architecture, but didn't help |
| Label smoothing | val=2.84 | **Destroyed performance** |
| Dropout 0.1 | val=2.01 | Helps with overfitting |
| Tied embeddings | val=2.01 | Same quality, 10% fewer params |
| Learning rate sweep (3e-4 to 5e-3) | **val=1.71** at lr=3e-3 | **Default LR was 10x too low!** |
| Dropout sweep (0.05 to 0.15) | 0.1 is optimal | Sweet spot for this dataset size |
| Larger batch size (64) | val=1.78 | Fewer update steps hurts |

### The improvement trajectory

```
Val Loss   What changed
────────   ────────────────────────────────
2.07       Baseline
2.01       + dropout
2.01       + tied embeddings
1.96       + lr bump (4e-4)
1.93       + lr (7e-4)
1.90       + lr (1e-3)
1.79       + lr (2e-3)
1.71       + lr (3e-3)  ← BEST (17% improvement!)
```

**The single biggest win was learning rate**: going from 3e-4 to 3e-3 (10x higher) dropped val loss from 2.01 to 1.71. This is a common finding — small models on small datasets benefit from higher learning rates.

---

## Things students can try

### Easy experiments (modify `train_mlx.py` constants)
- [ ] Change `TIME_BUDGET` to 600 (10 min) — does longer training help?
- [ ] Change `DEPTH` to 3 or 8 — how does depth affect quality?
- [ ] Change `N_EMBD` to 128 or 512 — width vs depth tradeoff
- [ ] Change `BATCH_SIZE` to 16 or 64 — batch size effects
- [ ] Change `WEIGHT_DECAY` to 0.0 or 0.5 — regularization strength

### Medium experiments (modify model code)
- [ ] Add GQA: set `N_KV_HEAD = 1` (single KV head for all Q heads)
- [ ] Try GELU activation instead of ReLU squared
- [ ] Try SwiGLU: set `USE_SWIGLU = True`
- [ ] Untie embeddings: set `TIED_EMBEDDINGS = False`
- [ ] Add layer norm instead of RMS norm

### Hard experiments (modify data/architecture)
- [ ] Add more training data (scrape simple Python tutorials)
- [ ] Change `VOCAB_SIZE` in prepare.py to 4096 and retrain tokenizer
- [ ] Implement mixture-of-experts (MoE) layers
- [ ] Add a "chain of thought" format: question -> explanation -> code
- [ ] Fine-tune on a specific domain (data science, web dev, etc.)

### Research questions
1. At what dataset size does the bigger model (15M) start winning?
2. Is there a scaling law between params and val_loss for this task?
3. Does data quality matter more than quantity at this scale?
4. How does the model perform on code it has never seen vs memorized patterns?

---

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- `pip install mlx tiktoken rustbpe pyarrow requests flask`

## Based on

[autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy — an autonomous AI research framework where LLM agents independently conduct neural network experiments.
