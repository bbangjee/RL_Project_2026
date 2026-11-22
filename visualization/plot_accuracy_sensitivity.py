"""E2 AI 정확도 민감도 실험의 발표용 그래프를 생성한다."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

SETTINGS = (
    ("85%", "ppo"),
    ("90%", "ppo_ai4acc0.9"),
    ("95%", "ppo_ai4acc0.95"),
)


def set_korean_font():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_means():
    means = []
    for label, name in SETTINGS:
        runs = [json.loads((RESULTS / f"{name}_seed{seed}.json").read_text()) for seed in range(5)]
        means.append(
            {
                "label": label,
                "score": np.mean([run["score"] for run in runs]),
                "success90": 100 * np.mean([run["success90"] for run in runs]),
                "caught": 100 * np.mean([run["caught"] for run in runs]),
                "ai_frac": 100 * np.mean([run["ai_frac"] for run in runs]),
            }
        )
    return means


if __name__ == "__main__":
    set_korean_font()
    data = load_means()
    labels = [row["label"] for row in data]
    values = [row["success90"] for row in data]
    colors = ["#B7B7B7", "#A50000", "#7B0000"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("4점 문제 AI 정확도에 따른 90점 이상 달성률", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("4점 문제 AI 정확도", fontsize=11)
    ax.set_ylabel("90점 이상 달성률 (%)", fontsize=11)
    ax.set_ylim(0, 35)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.annotate(
        "+14.8%p",
        xy=(2, values[2]),
        xytext=(1.45, 32),
        fontsize=11,
        color="#A50000",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#A50000", "lw": 1.5},
    )

    fig.tight_layout()
    output = FIGURES / "ai_accuracy_success.png"
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {output}")
