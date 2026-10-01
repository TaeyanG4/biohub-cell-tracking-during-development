"""Regenerate the figures used in README.md and docs/.

Run from the repository root with the global Python (matplotlib, numpy, zarr):

    python docs/figures/make_figures.py

Inputs:
- docs/data/submissions.csv          our scored submissions (public + private, read after the close)
- docs/data/leaderboard_private.json private leaderboard summary (public information)
- data/train/<movie>.zarr|.geff      competition data, NOT in this repository (download from Kaggle);
                                     the frame figure is skipped when it is missing
- tmp/c023_output/submission.csv     C023's Kaggle output for the 4 visible movies (not in this repository)
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"

# Reference palette (categorical order is fixed; colour follows the entity)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
MUTED = "#a9a8a2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})

MEDALS = {"gold (rank 18)": 0.941, "silver (rank 200)": 0.923, "bronze (rank 401)": 0.920}


def load_submissions():
    rows = list(csv.DictReader(open(ROOT / "docs/data/submissions.csv", encoding="utf-8")))
    seen, out = set(), []
    for r in rows:  # several byte-identical C003/C008 resubmissions: keep the first
        if r["candidate"] in seen and r["candidate"].startswith("C"):
            continue
        seen.add(r["candidate"])
        r["t"] = datetime.fromisoformat(r["date_utc"])
        r["pub"], r["pri"] = float(r["public"]), float(r["private"])
        out.append(r)
    return out


def family(cand: str) -> str:
    """Which coordinate head / base a submission used."""
    if not cand.startswith("C"):
        return "pre-candidate probes"
    n = int(cand[1:])
    if n <= 11:
        return "B0 (public 0.947) tuning"
    if n in (23, 28, 42, 43, 46, 53, 54, 65, 67, 69, 70):
        return "x138 public head (C023 base)"
    return "own head / 2-head ensemble"


FAMILY_COLOR = {
    "pre-candidate probes": MUTED,
    "B0 (public 0.947) tuning": YELLOW,
    "own head / 2-head ensemble": BLUE,
    "x138 public head (C023 base)": ORANGE,
}


def fig_timeline(subs):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    t = [r["t"] for r in subs]
    ax.plot(t, [r["pub"] for r in subs], "o", ms=6, color=BLUE, mec=SURFACE, mew=1.5, label="public LB (embryo fdad)")
    ax.plot(t, [r["pri"] for r in subs], "o", ms=6, color=ORANGE, mec=SURFACE, mew=1.5, label="private LB (embryo ea36)")
    for name, y in MEDALS.items():
        ax.axhline(y, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.text(t[0], y + 0.0007, f"private {name} cut {y:.3f}", color=INK2, fontsize=8, va="bottom")
    notes = {"C004": "C004 0.948\nB0 tuning", "C012": "C012 own head\npriv 0.924 (best)",
             "C023": "C023 x138 head\npub 0.954 / priv 0.917", "C069": "C069 final pick\npriv 0.918"}
    for r in subs:
        if r["candidate"] in notes:
            ax.annotate(notes[r["candidate"]], (r["t"], r["pri"]), xytext=(0, -34), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_ylim(0.895, 0.962)
    ax.set_ylabel("score")
    ax.set_title("Every scored submission: public vs private (read after the close)", loc="left")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "score_timeline.png", dpi=160)
    plt.close(fig)


def fig_public_private(subs):
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    rng = np.random.default_rng(0)
    for fam, col in FAMILY_COLOR.items():
        pts = [r for r in subs if family(r["candidate"]) == fam and r["pub"] > 0.92]
        if not pts:
            continue
        jx = rng.uniform(-0.00025, 0.00025, len(pts))
        ax.scatter([r["pub"] for r in pts] + jx, [r["pri"] for r in pts], s=58, color=col,
                   edgecolor=SURFACE, linewidth=1.5, label=fam, zorder=3)
    for r in subs:
        if r["candidate"] in ("C012", "C023", "C024", "C069"):
            ax.annotate(r["candidate"], (r["pub"], r["pri"]), xytext=(7, 3), textcoords="offset points",
                        fontsize=8.5, color=INK2)
    ax.axhline(MEDALS["silver (rank 200)"], color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.axhline(MEDALS["bronze (rank 401)"], color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.text(0.9435, 0.9233, "silver cut 0.923", fontsize=8, color=INK2)
    ax.text(0.9435, 0.9203, "bronze cut 0.920", fontsize=8, color=INK2)
    ax.set_xlim(0.943, 0.9555)
    ax.set_ylim(0.9125, 0.926)
    ax.set_xlabel("public LB")
    ax.set_ylabel("private LB")
    ax.set_title("Public ties hid a 0.007 private spread", loc="left")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "public_vs_private.png", dpi=160)
    plt.close(fig)


def fig_leaderboard():
    lb = json.load(open(ROOT / "docs/data/leaderboard_private.json", encoding="utf-8"))
    s = np.array(lb["private_scores_top1200"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ranks = np.arange(1, len(s) + 1)
    ax.plot(ranks, s, color=BLUE, lw=2)
    ax.set_xscale("log")
    for name, y in MEDALS.items():
        ax.axhline(y, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=0)
        dy = -0.0042 if name.startswith("bronze") else 0.0012
        ax.text(1.05, y + dy, f"{name}: {y:.3f}", fontsize=8, color=INK2)
    us = lb["our_team"]
    ax.plot([us["private_rank"]], [us["private"]], "o", ms=9, color=ORANGE, mec=SURFACE, mew=2, zorder=4)
    ax.annotate(f"Taeyang: {us['private']:.3f}, rank {us['private_rank']} / {lb['teams']}",
                (us["private_rank"], us["private"]), xytext=(-60, -32), textcoords="offset points",
                ha="right", fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
    for r in lb["top20"][:3]:
        ax.annotate(f"#{r['rank']} {r['private']:.3f}", (r["rank"], r["private"]), xytext=(6, -4),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set_ylim(0.905, 0.982)
    ax.set_xlabel("private rank (log scale)")
    ax.set_ylabel("private score")
    ax.set_title("Private leaderboard, top 1,200 of 4,017 teams", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "leaderboard_private.png", dpi=160)
    plt.close(fig)


def fig_frame(movie="6bba_05db0fb1", t0=40, tail=8):
    img_path = ROOT / "data/train" / f"{movie}.zarr"
    gt_path = ROOT / "data/train" / f"{movie}.geff"
    sub_path = ROOT / "tmp/c023_output/submission.csv"
    if not img_path.exists():
        print("skip frame figure: competition data not present")
        return
    import zarr

    vol = zarr.open_array(str(img_path / "0"), mode="r")[t0].astype(np.float32)
    mip = vol.max(axis=0)
    lo, hi = np.percentile(mip, [1, 99.7])
    mip = np.clip((mip - lo) / (hi - lo), 0, 1)

    g = zarr.open_group(str(gt_path), mode="r")
    ids = g["nodes/ids"][:]
    p = {k: g[f"nodes/props/{k}/values"][:] for k in "tzyx"}
    pos = {int(i): (int(p["t"][j]), p["y"][j], p["x"][j]) for j, i in enumerate(ids)}
    edges = g["edges/ids"][:]

    pred = []
    if sub_path.exists():
        for r in csv.DictReader(open(sub_path, encoding="utf-8")):
            if r["dataset"] == movie and r["row_type"] == "node" and int(r["t"]) == t0:
                pred.append((float(r["y"]), float(r["x"])))
    pred = np.array(pred) if pred else np.zeros((0, 2))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4))
    for ax in axes:
        ax.imshow(mip, cmap="gray", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_title(f"{movie}, frame {t0}: max projection over z", loc="left", fontsize=10)

    ax = axes[1]
    if len(pred):
        ax.scatter(pred[:, 1], pred[:, 0], s=10, facecolor="none", edgecolor=AQUA, linewidth=0.8,
                   label=f"C023 prediction ({len(pred)} nodes)")
    for a, b in edges:
        ta, ya, xa = pos[int(a)]
        tb, yb, xb = pos[int(b)]
        if t0 - tail <= ta < t0 and tb <= t0:
            ax.plot([xa, xb], [ya, yb], color=ORANGE, lw=1.6, solid_capstyle="round")
    gt_now = np.array([(y, x) for (t, y, x) in pos.values() if t == t0])
    if len(gt_now):
        ax.scatter(gt_now[:, 1], gt_now[:, 0], s=44, color=ORANGE, edgecolor=SURFACE, linewidth=1.2,
                   label=f"annotated GT ({len(gt_now)} nodes) + last {tail} frames", zorder=4)
    ax.set_title("Sparse annotation vs dense prediction", loc="left", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, facecolor=SURFACE, framealpha=0.9, frameon=True, edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(OUT / "sample_frame_tracks.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    subs = load_submissions()
    fig_timeline(subs)
    fig_public_private(subs)
    fig_leaderboard()
    fig_frame()
    print("figures written to", OUT)
