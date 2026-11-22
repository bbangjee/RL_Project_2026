"""E3 90점 보너스 민감도 실험의 발표용 그래프를 생성한다."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

SETTINGS = (
    ("0", "ppo_ai4acc0.95"),
    ("0.5", "ppo_ai4acc0.95_bonus0.5"),
    ("1.0", "ppo_ai4acc0.95_bonus1.0"),
    ("2.0", "ppo_ai4acc0.95_bonus2.0"),
)


def set_korean_font():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_means():
    rows = []
    rng = np.random.default_rng(0)
    for label, name in SETTINGS:
        runs = [json.loads((RESULTS / f"{name}_seed{seed}.json").read_text()) for seed in range(5)]
        success = 100 * np.asarray([run["success90"] for run in runs])
        bootstrap = success[rng.integers(0, len(success), (100_000, len(success)))].mean(1)
        ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])
        rows.append(
            {
                "label": label,
                "success90": success.mean(),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "caught": 100 * np.mean([run["caught"] for run in runs]),
            }
        )
    return rows


if __name__ == "__main__":
    set_korean_font()
    data = load_means()
    labels = [row["label"] for row in data]
    success = [row["success90"] for row in data]
    caught = [row["caught"] for row in data]
    success_error = np.asarray(
        [[row["success90"] - row["ci_low"] for row in data],
         [row["ci_high"] - row["success90"] for row in data]]
    )

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    success_bars = ax.bar(
        x - width / 2,
        success,
        width,
        label="90점 이상 달성률 (95% CI)",
        color=["#C98A8A", "#A50000", "#C95C5C", "#C98A8A"],
        yerr=success_error,
        capsize=4,
        error_kw={"ecolor": "#5A0000", "elinewidth": 1.2},
    )
    caught_bars = ax.bar(x + width / 2, caught, width, label="적발률", color="#8A8A8A")
    success_bars[1].set_edgecolor("#4A0000")
    success_bars[1].set_linewidth(2.0)

    ax.set_title("90점 달성 보너스에 따른 성능 변화", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("90점 달성 보너스 λ", fontsize=11)
    ax.set_ylabel("비율 (%)", fontsize=11)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 47)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=10)

    for i, bar in enumerate(success_bars):
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + success_error[1, i] + 0.6,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for bar in caught_bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    output = FIGURES / "reward_bonus_sensitivity.png"
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {output}")
