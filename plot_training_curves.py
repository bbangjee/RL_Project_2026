"""학습 로그를 5개 시드 평균으로 집계해 발표용 학습 곡선을 생성한다."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

ALGORITHMS = ("dqn", "ppo", "a2c")
LABELS = {"dqn": "DQN", "ppo": "PPO", "a2c": "A2C"}
COLORS = {"dqn": "#2455A4", "ppo": "#A50000", "a2c": "#666666"}
LINE_RE = re.compile(
    r"^step\s+(\d+) \| 점수\s+([0-9.]+) \| 적발\s+([0-9.]+)%"
)


def read_log(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            match = LINE_RE.match(line)
            if match:
                rows.append(tuple(map(float, match.groups())))
    if not rows:
        raise ValueError(f"학습 기록을 찾을 수 없음: {path}")
    return np.asarray(rows)


def aggregate(algo, points=160):
    runs = [read_log(RESULTS / f"{algo}_seed{seed}.log") for seed in range(5)]
    start = max(run[0, 0] for run in runs)
    end = min(run[-1, 0] for run in runs)
    steps = np.linspace(start, end, points)
    scores = np.stack([np.interp(steps, run[:, 0], run[:, 1]) for run in runs])
    caught = np.stack([np.interp(steps, run[:, 0], run[:, 2]) for run in runs])
    return steps, scores.mean(0), scores.std(0), caught.mean(0), caught.std(0)


def style_axis(ax, ylabel, ylim):
    ax.set_xlabel("Environment steps", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0, 2_000_000)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f}M"))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def draw(ax, metric):
    for algo in ALGORITHMS:
        steps, score_mean, score_std, caught_mean, caught_std = aggregate(algo)
        mean, std = (score_mean, score_std) if metric == "score" else (caught_mean, caught_std)
        ax.plot(steps, mean, label=LABELS[algo], color=COLORS[algo], linewidth=2.4)
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[algo], alpha=0.12)


def save_individual(metric, title, ylabel, ylim, filename):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    draw(ax, metric)
    style_axis(ax, ylabel, ylim)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_combined():
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    draw(axes[0], "score")
    style_axis(axes[0], "Average score (last 100 episodes)", (0, 100))
    axes[0].set_title("Average Score During Training", fontsize=15, fontweight="bold", pad=12)

    draw(axes[1], "caught")
    style_axis(axes[1], "Detection rate (%)", (0, 100))
    axes[1].set_title("Detection Rate During Training", fontsize=15, fontweight="bold", pad=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.20, wspace=0.24)
    fig.savefig(FIGURES / "training_curves.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    save_individual(
        "score",
        "Average Score During Training",
        "Average score (last 100 episodes)",
        (0, 100),
        "training_score.png",
    )
    save_individual(
        "caught",
        "Detection Rate During Training",
        "Detection rate (%)",
        (0, 100),
        "training_detection.png",
    )
    save_combined()
    print(f"saved: {FIGURES}")
