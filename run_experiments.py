"""
Automated experiment harness for autoresearch.
Runs a batch of experiments, updates results.tsv, regenerates plot, and pushes to GitHub.

Usage:
    python run_experiments.py
"""

import os
import re
import subprocess
import time
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_TSV = os.path.join(BASE_DIR, "results.tsv")
PLOT_PATH = os.path.join(BASE_DIR, "experiments_plot.png")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_mlx.py")

# --- Current best config (baseline for all experiments) ---
BEST_CONFIG = {
    "DEPTH": 5,
    "N_EMBD": 320,
    "HEAD_DIM": 64,
    "MLP_RATIO": 4,
    "USE_SWIGLU": False,
    "DROPOUT": 0.1,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 3e-3,
    "WEIGHT_DECAY": 0.5,
    "WARMUP_STEPS": 150,
    "TIME_BUDGET": 300,
    "GRAD_CLIP": 1.0,
    "USE_LION": False,
    "TIED_EMBEDDINGS": True,
    "MAX_SEQ_LEN": 512,
}

# --- Experiments to run ---
# Each experiment: (description, {overrides})
EXPERIMENTS = [
    ("weight_decay=0.7", {"WEIGHT_DECAY": 0.7}),
    ("weight_decay=1.0", {"WEIGHT_DECAY": 1.0}),
    ("lr=4e-3 wd=0.5", {"LEARNING_RATE": 4e-3}),
    ("lr=2e-3 wd=0.5", {"LEARNING_RATE": 2e-3}),
    ("dropout=0.15 wd=0.5", {"DROPOUT": 0.15}),
    ("dropout=0.2 wd=0.5", {"DROPOUT": 0.2}),
    ("warmup=50 wd=0.5", {"WARMUP_STEPS": 50}),
    ("warmup=300 wd=0.5", {"WARMUP_STEPS": 300}),
    ("6L 320d wd=0.5", {"DEPTH": 6}),
    ("8L 256d wd=0.5", {"DEPTH": 8, "N_EMBD": 256}),
    ("SwiGLU wd=0.5", {"USE_SWIGLU": True}),
    ("bs=16 wd=0.5 (more steps)", {"BATCH_SIZE": 16}),
]


def get_next_run_number():
    """Read results.tsv and return the next experiment number."""
    if not os.path.exists(RESULTS_TSV):
        return 1
    with open(RESULTS_TSV) as f:
        lines = f.readlines()
    # Last non-empty line
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("run"):
            return int(line.split("\t")[0]) + 1
    return 1


def apply_config(overrides):
    """Patch train_mlx.py constants with the given config."""
    config = {**BEST_CONFIG, **overrides}
    with open(TRAIN_SCRIPT) as f:
        src = f.read()

    for key, val in config.items():
        # Match patterns like: DEPTH = 5 or LEARNING_RATE = 3e-3 or USE_SWIGLU = False
        if isinstance(val, bool):
            val_str = str(val)
        elif isinstance(val, float):
            # Keep scientific notation for small floats
            if val < 0.01:
                val_str = f"{val:.0e}".replace("+0", "+").replace("-0", "-").replace("e-", "e-").replace("e+", "e+")
                # Simplify: 3e-03 -> 3e-3
                val_str = re.sub(r'e([+-])0*(\d+)', r'e\1\2', f"{val:e}")
                # Prefer compact: 3.000000e-03 -> 3e-3
                mantissa, exp = val_str.split("e")
                mantissa = mantissa.rstrip("0").rstrip(".")
                val_str = f"{mantissa}e{exp}"
            else:
                val_str = str(val)
        else:
            val_str = str(val)

        pattern = rf'^({key}\s*=\s*).*$'
        replacement = rf'\g<1>{val_str}'
        src, count = re.subn(pattern, replacement, src, count=1, flags=re.MULTILINE)
        if count == 0:
            print(f"  WARNING: Could not find {key} in train_mlx.py")

    with open(TRAIN_SCRIPT, "w") as f:
        f.write(src)


def run_training(timeout=660):
    """Run train_mlx.py and return (val_loss, params_M, output)."""
    try:
        result = subprocess.run(
            ["python", TRAIN_SCRIPT],
            capture_output=True, text=True, timeout=timeout,
            cwd=BASE_DIR,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT"

    # Parse val loss
    val_match = re.search(r'Val loss:\s*([\d.]+)', output)
    val_loss = float(val_match.group(1)) if val_match else None

    # Parse params
    params_match = re.search(r'Model:\s*([\d.]+)M', output)
    params_M = float(params_match.group(1)) if params_match else None

    return val_loss, params_M, output


def append_result(run_num, val_loss, params_M, status, description):
    """Append a row to results.tsv."""
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{run_num:03d}\t{val_loss:.4f}\t{params_M}\t{status}\t{description}\n")


def generate_plot():
    """Regenerate experiments_plot.png from results.tsv."""
    runs, losses, statuses, descs, params = [], [], [], [], []
    with open(RESULTS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            runs.append(int(row["run"]))
            losses.append(float(row["val_loss"]))
            statuses.append(row["status"])
            descs.append(row["description"])
            params.append(float(row["params_M"]))

    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(20, 8),
                                            gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor("#0d1117")

    for ax in (ax_main, ax_zoom):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color="#8b949e")

    # --- LEFT: Full view ---
    keep_x = [r for r, s in zip(runs, statuses) if s == "keep"]
    keep_y = [l for l, s in zip(losses, statuses) if s == "keep"]
    disc_x = [r for r, s in zip(runs, statuses) if s == "discard"]
    disc_y = [l for l, s in zip(losses, statuses) if s == "discard"]

    ax_main.plot(runs, losses, c="#30363d", alpha=0.5, linewidth=1, zorder=1)
    ax_main.scatter(disc_x, disc_y, c="#f85149", s=50, zorder=3, label="discard", alpha=0.6, marker="x", linewidths=2)
    ax_main.scatter(keep_x, keep_y, c="#3fb950", s=90, zorder=4, label="keep", edgecolors="white", linewidth=1)

    # Best envelope
    best_so_far = []
    current_best = float("inf")
    for l in losses:
        current_best = min(current_best, l)
        best_so_far.append(current_best)
    ax_main.plot(runs, best_so_far, c="#3fb950", linewidth=2.5, alpha=0.8, linestyle="--", label="best so far", zorder=2)

    # Annotate only milestones (kept experiments that improved best)
    prev_best = float("inf")
    for r, l, s, d in zip(runs, losses, statuses, descs):
        if s == "keep" and l < prev_best:
            short = d.split(" - ")[0] if " - " in d else d
            short = short[:35]
            ax_main.annotate(
                f"#{r}: {short}\n({l:.4f})",
                (r, l), fontsize=7, fontweight="bold",
                textcoords="offset points", xytext=(8, 12),
                color="#3fb950",
                arrowprops=dict(arrowstyle="-", color="#3fb950", alpha=0.4, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117", ec="#3fb950", alpha=0.7, lw=0.5),
            )
            prev_best = l

    ax_main.set_xlabel("Experiment #", fontsize=12, color="#c9d1d9")
    ax_main.set_ylabel("Validation Loss", fontsize=12, color="#c9d1d9")
    ax_main.set_title("All Experiments", fontsize=14, fontweight="bold", color="#c9d1d9")
    ax_main.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#c9d1d9", fontsize=9)

    # --- RIGHT: Zoomed view (only the competitive region) ---
    # Filter to experiments with loss < 2.0
    zoom_threshold = 1.85
    z_runs = [r for r, l in zip(runs, losses) if l <= zoom_threshold]
    z_losses = [l for r, l in zip(runs, losses) if l <= zoom_threshold]
    z_statuses = [s for l, s in zip(losses, statuses) if l <= zoom_threshold]
    z_descs = [d for l, d in zip(losses, descs) if l <= zoom_threshold]

    zk_x = [r for r, s in zip(z_runs, z_statuses) if s == "keep"]
    zk_y = [l for l, s in zip(z_losses, z_statuses) if s == "keep"]
    zd_x = [r for r, s in zip(z_runs, z_statuses) if s == "discard"]
    zd_y = [l for l, s in zip(z_losses, z_statuses) if s == "discard"]

    ax_zoom.scatter(zd_x, zd_y, c="#f85149", s=60, zorder=3, alpha=0.6, marker="x", linewidths=2)
    ax_zoom.scatter(zk_x, zk_y, c="#3fb950", s=100, zorder=4, edgecolors="white", linewidth=1)

    # Annotate all in zoomed view
    for r, l, s, d in zip(z_runs, z_losses, z_statuses, z_descs):
        short = d.split(" - ")[0] if " - " in d else d
        short = short[:30]
        color = "#3fb950" if s == "keep" else "#f85149"
        ax_zoom.annotate(
            f"{short}",
            (r, l), fontsize=6.5,
            textcoords="offset points", xytext=(6, 6),
            color=color, alpha=0.9, rotation=20,
        )

    # Best line in zoom
    z_best = []
    cb = float("inf")
    for r, l in zip(runs, losses):
        cb = min(cb, l)
        if l <= zoom_threshold:
            z_best.append((r, cb))
    if z_best:
        ax_zoom.plot([x[0] for x in z_best], [x[1] for x in z_best],
                     c="#3fb950", linewidth=2, alpha=0.6, linestyle="--")

    ax_zoom.set_xlabel("Experiment #", fontsize=12, color="#c9d1d9")
    ax_zoom.set_ylabel("Validation Loss", fontsize=12, color="#c9d1d9")
    ax_zoom.set_title(f"Zoomed: Loss < {zoom_threshold}", fontsize=14, fontweight="bold", color="#c9d1d9")

    # Summary stats text box
    best_loss = min(losses)
    best_idx = losses.index(best_loss)
    best_desc = descs[best_idx]
    summary = (
        f"Total experiments: {len(runs)}\n"
        f"Best val_loss: {best_loss:.4f} (#{runs[best_idx]})\n"
        f"Best config: {best_desc[:50]}\n"
        f"Improvement: {((losses[0] - best_loss) / losses[0] * 100):.1f}% from baseline"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10, color="#8b949e",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#161b22", ec="#30363d", alpha=0.9))

    fig.suptitle("Autoresearch: NLQ → Python Code Model", fontsize=16,
                 fontweight="bold", color="#58a6ff", y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved to {PLOT_PATH}")


def git_commit_and_push(run_num, description, val_loss):
    """Commit results + plot + any train_mlx changes, push to GitHub."""
    subprocess.run(["git", "add", "results.tsv", "experiments_plot.png", "train_mlx.py",
                    "checkpoint/config.json"],
                   cwd=BASE_DIR, capture_output=True)

    msg = f"exp {run_num:03d}: {description} (val={val_loss:.4f})"
    subprocess.run(["git", "commit", "-m", msg], cwd=BASE_DIR, capture_output=True)
    result = subprocess.run(["git", "push", "nipun", "master"],
                           cwd=BASE_DIR, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Pushed to GitHub")
    else:
        print(f"  Push failed: {result.stderr[:200]}")


def main():
    print("=" * 60)
    print("AUTORESEARCH EXPERIMENT HARNESS")
    print("=" * 60)

    # Track best val_loss from existing results
    best_val_loss = float("inf")
    if os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["status"] == "keep":
                    best_val_loss = min(best_val_loss, float(row["val_loss"]))
    print(f"\nCurrent best val_loss: {best_val_loss:.4f}")
    print(f"Experiments to run: {len(EXPERIMENTS)}")
    print()

    for desc, overrides in EXPERIMENTS:
        run_num = get_next_run_number()
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {run_num:03d}: {desc}")
        print(f"  Overrides: {overrides}")
        print(f"{'='*60}")

        # Apply config
        apply_config(overrides)
        print(f"  Config applied to train_mlx.py")

        # Train
        t0 = time.time()
        val_loss, params_M, output = run_training(timeout=660)
        elapsed = time.time() - t0

        if val_loss is None:
            print(f"  FAILED (no val_loss parsed). Skipping.")
            # Revert to best config
            apply_config({})
            continue

        # Determine status
        if val_loss < best_val_loss:
            status = "keep"
            best_val_loss = val_loss
            improvement = "NEW BEST!"
        else:
            status = "discard"
            improvement = ""
            # Revert config to best
            apply_config({})

        desc_full = f"{desc} {improvement}".strip()
        print(f"  Val loss: {val_loss:.4f} | Params: {params_M}M | Status: {status} | {elapsed:.0f}s")

        # Update TSV
        append_result(run_num, val_loss, params_M, status, desc_full)
        print(f"  Updated results.tsv")

        # Regenerate plot
        generate_plot()

        # Git commit + push
        git_commit_and_push(run_num, desc_full, val_loss)

    # Final summary
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
