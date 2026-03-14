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
import matplotlib.ticker as mticker


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_TSV = os.path.join(BASE_DIR, "results.tsv")
PLOT_PATH = os.path.join(BASE_DIR, "experiments_plot.png")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_mlx.py")

BASELINE_LOSS = 2.0686  # experiment 001

# --- Current best config (baseline for all experiments) ---
# Updated after exp 049: bs=16 was the big win
BEST_CONFIG = {
    "DEPTH": 5,
    "N_EMBD": 320,
    "HEAD_DIM": 64,
    "MLP_RATIO": 4,
    "USE_SWIGLU": False,
    "DROPOUT": 0.3,
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 3e-3,
    "WEIGHT_DECAY": 0.5,
    "WARMUP_STEPS": 300,
    "TIME_BUDGET": 300,
    "GRAD_CLIP": 1.0,
    "DECAY_STEPS": 5000,
    "USE_LION": False,
    "TIED_EMBEDDINGS": True,
    "MAX_SEQ_LEN": 512,
}

# --- Experiments to run ---
# Each experiment: (description, {overrides})
# Target: 35% improvement = val_loss < 1.345
# Best config: bs=8 + dropout=0.3 + warmup=300 + wd=0.5
# Plateau at ~29%. Need fundamentally different levers.
EXPERIMENTS = [
    # Remaining from round 5 + new ideas
    ("lr=5e-3 decay=2000", {"LEARNING_RATE": 5e-3, "DECAY_STEPS": 2000}),
    ("7min decay=3500", {"TIME_BUDGET": 420, "DECAY_STEPS": 3500}),
    ("7min bs=8 lr=2e-3 decay=3500", {"TIME_BUDGET": 420, "DECAY_STEPS": 3500, "LEARNING_RATE": 2e-3}),
    ("8L 192d (tiny+deep)", {"DEPTH": 8, "N_EMBD": 192}),
    ("lr=8e-3 decay=1500 warm=100", {"LEARNING_RATE": 8e-3, "DECAY_STEPS": 1500, "WARMUP_STEPS": 100}),
    # Gradient accumulation via 2 mini-batches (bs=4 x2 = effective bs=8)
    # Actually just try bs=4 with higher LR to compensate
    ("bs=4 lr=2e-3 decay=3000", {"BATCH_SIZE": 4, "LEARNING_RATE": 2e-3, "DECAY_STEPS": 3000}),
    # MLP ratio 3 (fewer params per layer = faster = more steps)
    ("mlp_ratio=3 bs=8", {"MLP_RATIO": 3}),
    # Wide + shallow (more capacity per step)
    ("3L 448d bs=8", {"DEPTH": 3, "N_EMBD": 448}),
    # 10 min with very slow decay
    ("10min decay=6000", {"TIME_BUDGET": 600, "DECAY_STEPS": 6000}),
    # The nuclear option: everything at once
    ("10min bs=4 lr=2e-3 drop=0.3 wd=0.7", {"TIME_BUDGET": 600, "BATCH_SIZE": 4, "LEARNING_RATE": 2e-3, "WEIGHT_DECAY": 0.7, "DECAY_STEPS": 6000}),
]


def get_next_run_number():
    """Read results.tsv and return the next experiment number."""
    if not os.path.exists(RESULTS_TSV):
        return 1
    with open(RESULTS_TSV) as f:
        lines = f.readlines()
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
        if isinstance(val, bool):
            val_str = str(val)
        elif isinstance(val, float):
            if val < 0.01:
                val_str = re.sub(r'e([+-])0*(\d+)', r'e\1\2', f"{val:e}")
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

    val_match = re.search(r'Val loss:\s*([\d.]+)', output)
    val_loss = float(val_match.group(1)) if val_match else None

    params_match = re.search(r'Model:\s*([\d.]+)M', output)
    params_M = float(params_match.group(1)) if params_match else None

    return val_loss, params_M, output


def append_result(run_num, val_loss, params_M, status, time_s, description):
    """Append a row to results.tsv."""
    improv = (BASELINE_LOSS - val_loss) / BASELINE_LOSS * 100
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{run_num:03d}\t{val_loss:.4f}\t{params_M}\t{status}\t{time_s}\t{improv:.1f}\t{description}\n")


def generate_plot():
    """Regenerate experiments_plot.png from results.tsv."""
    runs, losses, statuses, descs, params, improvs = [], [], [], [], [], []
    with open(RESULTS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            runs.append(int(row["run"]))
            losses.append(float(row["val_loss"]))
            statuses.append(row["status"])
            descs.append(row["description"])
            params.append(float(row["params_M"]))
            improvs.append(float(row["improv_%"]))

    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor("#0d1117")

    # Layout: main plot left (wide), improvement % right (narrow)
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[3, 1],
                          hspace=0.35, wspace=0.3)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    for ax in (ax_main, ax_zoom, ax_bar):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.12, color="#8b949e")

    # ── LEFT: Full view ──
    keep_x = [r for r, s in zip(runs, statuses) if s == "keep"]
    keep_y = [l for l, s in zip(losses, statuses) if s == "keep"]
    disc_x = [r for r, s in zip(runs, statuses) if s == "discard"]
    disc_y = [l for l, s in zip(losses, statuses) if s == "discard"]

    ax_main.plot(runs, losses, c="#30363d", alpha=0.4, linewidth=1, zorder=1)
    ax_main.scatter(disc_x, disc_y, c="#f85149", s=40, zorder=3, alpha=0.5, marker="x", linewidths=1.5)
    ax_main.scatter(keep_x, keep_y, c="#3fb950", s=70, zorder=4, edgecolors="white", linewidth=0.8)

    # Best envelope
    best_so_far = []
    current_best = float("inf")
    for l in losses:
        current_best = min(current_best, l)
        best_so_far.append(current_best)
    ax_main.plot(runs, best_so_far, c="#3fb950", linewidth=2.5, alpha=0.8,
                 linestyle="--", label="best so far", zorder=2)

    # Target line at 35%
    target_35 = BASELINE_LOSS * 0.65
    ax_main.axhline(y=target_35, color="#f0883e", linewidth=1.5, linestyle=":",
                     alpha=0.7, label=f"35% target ({target_35:.3f})", zorder=1)

    # Annotate only milestones that improved best
    prev_best = float("inf")
    milestone_count = 0
    for r, l, s, d in zip(runs, losses, statuses, descs):
        if s == "keep" and l < prev_best:
            short = d.split(" - ")[0] if " - " in d else d
            short = short[:30]
            improv_pct = (BASELINE_LOSS - l) / BASELINE_LOSS * 100
            # Alternate annotation position to avoid overlap
            offset_y = 14 if milestone_count % 2 == 0 else -18
            ax_main.annotate(
                f"#{r} ({improv_pct:.0f}%)",
                (r, l), fontsize=7, fontweight="bold",
                textcoords="offset points", xytext=(6, offset_y),
                color="#3fb950",
                arrowprops=dict(arrowstyle="-", color="#3fb950", alpha=0.3, lw=0.6),
                bbox=dict(boxstyle="round,pad=0.15", fc="#0d1117", ec="#3fb950", alpha=0.6, lw=0.4),
            )
            prev_best = l
            milestone_count += 1

    ax_main.set_xlabel("Experiment #", fontsize=11, color="#c9d1d9")
    ax_main.set_ylabel("Validation Loss", fontsize=11, color="#c9d1d9")
    ax_main.set_title("All Experiments", fontsize=13, fontweight="bold", color="#c9d1d9")
    ax_main.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#c9d1d9", fontsize=8)

    # ── RIGHT: Zoomed (auto-range to competitive region) ──
    best_loss = min(losses)
    zoom_lo = best_loss - 0.03
    zoom_hi = best_loss + 0.15
    z_idx = [i for i, l in enumerate(losses) if zoom_lo <= l <= zoom_hi]

    if len(z_idx) < 5:  # widen if too few points
        zoom_hi = best_loss + 0.25
        z_idx = [i for i, l in enumerate(losses) if zoom_lo <= l <= zoom_hi]

    z_runs = [runs[i] for i in z_idx]
    z_losses = [losses[i] for i in z_idx]
    z_statuses = [statuses[i] for i in z_idx]
    z_descs = [descs[i] for i in z_idx]

    zk = [(r, l) for r, l, s in zip(z_runs, z_losses, z_statuses) if s == "keep"]
    zd = [(r, l) for r, l, s in zip(z_runs, z_losses, z_statuses) if s == "discard"]

    if zd:
        ax_zoom.scatter([x[0] for x in zd], [x[1] for x in zd],
                       c="#f85149", s=50, zorder=3, alpha=0.5, marker="x", linewidths=1.5)
    if zk:
        ax_zoom.scatter([x[0] for x in zk], [x[1] for x in zk],
                       c="#3fb950", s=80, zorder=4, edgecolors="white", linewidth=0.8)

    # Annotate zoomed view - only label each point with short desc
    from matplotlib.patches import FancyBboxPatch
    used_positions = []
    for r, l, s, d in zip(z_runs, z_losses, z_statuses, z_descs):
        short = d.split(" - ")[0] if " - " in d else d
        short = short[:25]
        color = "#3fb950" if s == "keep" else "#f8514966"
        # Only annotate kept ones in zoom to reduce clutter
        if s == "keep":
            ax_zoom.annotate(
                short, (r, l), fontsize=6, color=color,
                textcoords="offset points", xytext=(5, 5), rotation=15,
            )

    ax_zoom.axhline(y=target_35, color="#f0883e", linewidth=1.5, linestyle=":", alpha=0.7)
    ax_zoom.set_ylim(zoom_lo, zoom_hi)
    ax_zoom.set_xlabel("Experiment #", fontsize=11, color="#c9d1d9")
    ax_zoom.set_ylabel("Validation Loss", fontsize=11, color="#c9d1d9")
    ax_zoom.set_title("Zoomed: Competitive Region", fontsize=13, fontweight="bold", color="#c9d1d9")

    # ── BOTTOM: Improvement % bar chart ──
    colors = ["#3fb950" if s == "keep" else "#f8514944" for s in statuses]
    ax_bar.bar(runs, improvs, color=colors, width=0.8, zorder=3)
    ax_bar.axhline(y=0, color="#8b949e", linewidth=0.5)
    ax_bar.axhline(y=35, color="#f0883e", linewidth=1.5, linestyle=":", alpha=0.7, label="35% target")
    ax_bar.set_xlabel("Experiment #", fontsize=11, color="#c9d1d9")
    ax_bar.set_ylabel("Improvement %", fontsize=11, color="#c9d1d9")
    ax_bar.set_title("Improvement over Baseline (higher = better)", fontsize=13,
                     fontweight="bold", color="#c9d1d9")
    ax_bar.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=8)
    ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))

    # Summary
    best_idx = losses.index(best_loss)
    best_improv = (BASELINE_LOSS - best_loss) / BASELINE_LOSS * 100
    summary = (
        f"Experiments: {len(runs)}  |  "
        f"Best: {best_loss:.4f} (#{runs[best_idx]}, {best_improv:.1f}% improvement)  |  "
        f"Target: {target_35:.4f} (35%)"
    )
    fig.text(0.5, 0.005, summary, ha="center", fontsize=11, color="#58a6ff",
             family="monospace", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="#161b22", ec="#58a6ff", alpha=0.9, lw=1))

    fig.suptitle("Autoresearch: NLQ → Python Code Model", fontsize=17,
                 fontweight="bold", color="#58a6ff", y=0.99)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved to {PLOT_PATH}")


def git_commit_and_push(run_num, description, val_loss):
    """Commit results + plot + any train_mlx changes, push to GitHub."""
    subprocess.run(["git", "add", "results.tsv", "experiments_plot.png", "train_mlx.py",
                    "checkpoint/config.json", "index.html"],
                   cwd=BASE_DIR, capture_output=True)

    improv = (BASELINE_LOSS - val_loss) / BASELINE_LOSS * 100
    msg = f"exp {run_num:03d}: {description} (val={val_loss:.4f}, {improv:.1f}%)"
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
    print(f"Target: 35% improvement (val_loss < {BASELINE_LOSS * 0.65:.4f})")
    print("=" * 60)

    # Track best val_loss from existing results
    best_val_loss = float("inf")
    if os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["status"] == "keep":
                    best_val_loss = min(best_val_loss, float(row["val_loss"]))
    best_improv = (BASELINE_LOSS - best_val_loss) / BASELINE_LOSS * 100
    print(f"\nCurrent best val_loss: {best_val_loss:.4f} ({best_improv:.1f}% improvement)")
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
        elapsed = int(time.time() - t0)

        if val_loss is None:
            print(f"  FAILED (no val_loss parsed). Skipping.")
            apply_config({})
            continue

        # Determine status
        improv = (BASELINE_LOSS - val_loss) / BASELINE_LOSS * 100
        if val_loss < best_val_loss:
            status = "keep"
            best_val_loss = val_loss
            tag = "NEW BEST!"
        else:
            status = "discard"
            tag = ""
            apply_config({})

        desc_full = f"{desc} {tag}".strip()
        print(f"  Val loss: {val_loss:.4f} | {improv:.1f}% | Params: {params_M}M | {status} | {elapsed}s")

        # Update TSV
        append_result(run_num, val_loss, params_M, status, elapsed, desc_full)
        print(f"  Updated results.tsv")

        # Regenerate plot
        generate_plot()

        # Git commit + push
        git_commit_and_push(run_num, desc_full, val_loss)

    # Final summary
    best_improv = (BASELINE_LOSS - best_val_loss) / BASELINE_LOSS * 100
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Best val_loss: {best_val_loss:.4f} ({best_improv:.1f}% improvement)")
    print(f"Target was: {BASELINE_LOSS * 0.65:.4f} (35%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
