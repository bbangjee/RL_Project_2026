"""학습 곡선과 에피소드 재생 시각화. 결과는 figures/ 폴더에 저장.
python visualize.py curves                      -> figures/curves_E1.png, curves_E4.png, curves_E5.png
python visualize.py replay dqn_seed0 --seed 3   -> figures/replay_dqn_seed0_ep3.gif (results/dqn_seed0.pt 사용)"""
import argparse
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle, Wedge
import torch
from exam_env import ExamEnv, ANGLES, POINTS, FOV, T
from sweeps import SEEDS, SWEEPS, run_name
from ppo import mlp

# 한글 글꼴 (없으면 기본 글꼴, 한글이 네모로 보일 수 있음)
_fonts = {f.name for f in font_manager.fontManager.ttflist}
for _f in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"]:
    if _f in _fonts:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

RED, GREEN, INK, GRAY = "#cf3445", "#2c7a57", "#1d2230", "#b9bfcb"
HONEST, DESIGNER = 40.5, 57.7   # rules.py 결과 (기준선)
LINE = re.compile(r"step\s+(\d+) \| 점수\s+([\d.]+) \| 적발\s+([\d.]+)%")


# ---------------- 학습 곡선 ----------------
def read_log(path):
    """로그에서 (스텝, 점수, 적발률%) 배열. 학습 중 롤아웃 최근 100판 평균."""
    with open(path, encoding="utf-8") as f:
        return np.array([[float(x) for x in m.groups()] for m in map(LINE.search, f) if m])


def curves(out):
    for group, items in SWEEPS.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
        for algo, ov in items:
            name = run_name(algo, ov)
            logs = [read_log(p) for s in SEEDS if os.path.exists(p := f"results/{name}_seed{s}.log")]
            logs = [l for l in logs if len(l)]
            if not logs:
                continue
            n = min(len(l) for l in logs)
            arr = np.stack([l[:n] for l in logs])          # (seed, 시점, 3)
            x = arr[0, :, 0] / 1e6
            for ax, k in ((ax1, 1), (ax2, 2)):
                line, = ax.plot(x, arr[:, :, k].mean(0), lw=1.8, label=f"{name} (seed {len(logs)}개)")
                ax.fill_between(x, arr[:, :, k].min(0), arr[:, :, k].max(0), color=line.get_color(), alpha=0.15)
        ax1.axhline(HONEST, color=GRAY, ls="--", lw=1)
        ax1.text(0.01, HONEST + 1, "정직 정책 40.5", color="gray", fontsize=9, transform=ax1.get_yaxis_transform())
        ax1.axhline(DESIGNER, color=INK, ls=":", lw=1)
        ax1.text(0.01, DESIGNER + 1, "설계자 규칙 57.7", color=INK, fontsize=9, transform=ax1.get_yaxis_transform())
        ax1.set(title="점수 (학습 중 최근 100판)", xlabel="환경 스텝 (백만)", ylabel="점수", ylim=(0, 70))
        ax2.set(title="적발률", xlabel="환경 스텝 (백만)", ylabel="%", ylim=(0, 100))
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
        ax1.legend(fontsize=9, loc="lower right")
        fig.suptitle(f"{group}  ·  선은 seed 평균, 띠는 최소~최대", fontsize=12)
        fig.tight_layout()
        path = f"{out}/curves_{group.split()[0]}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print("저장:", path)


# ---------------- 에피소드 재생 ----------------
def rollout(model_path, seed):
    """greedy 정책으로 한 판 진행하며 매 분 상태 기록."""
    net = mlp(ExamEnv.n_actions)
    net.load_state_dict(torch.load(model_path))
    env = ExamEnv(seed)
    obs, _ = env.reset()
    frames, done = [], False
    while not done:
        with torch.no_grad():
            a = int(net(torch.as_tensor(obs)).argmax())
        p = POINTS[env.order[env.k]]                      # 이번 분에 푸는 문제 배점
        obs, _, done, _, info = env.step(a)
        frames.append(dict(t=env.t, row=env.row, col=env.col, pc=env.pc, head=env.head, z=env.z, a=a, p=p,
                           seen=env.in_view(), k=env.k, exp=env.expected, caught=info.get("caught", False)))
    return frames, info


def replay(name, seed, out, fps):
    frames, info = rollout(f"results/{name}.pt", seed)
    n = len(frames)
    z = np.array([f["z"] for f in frames])
    acts = np.array([f["a"] for f in frames])
    seen = np.array([f["seen"] for f in frames])

    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1, 1], wspace=0.25, hspace=0.45)
    ax = fig.add_subplot(gs[:, 0])
    axz = fig.add_subplot(gs[0, 1])
    axa = fig.add_subplot(gs[1, 1])

    # 오른쪽 위: 의심도
    axz.set(xlim=(0, T), ylim=(0, 1.05), title="의심도 z", xlabel="시간 (분)")
    axz.grid(alpha=0.3)
    zline, = axz.plot([], [], color=RED, lw=1.8)
    # 오른쪽 아래: 행동 타임라인 (위 줄: 행동, 아래 줄: 감독관 시야 안)
    axa.set(xlim=(0, T), ylim=(-0.6, 1.6), xlabel="시간 (분)", yticks=[0, 1.2], yticklabels=["시야 안", "행동"],
            title="행동: 회색 직접 · 빨강 AI  /  시야 안: 검정")
    bars_a = axa.bar(np.arange(1, n + 1), 0.8, bottom=0.8, width=1.0, color=[RED if a else GRAY for a in acts])
    bars_s = axa.bar(np.arange(1, n + 1), 0.8, bottom=-0.4, width=1.0, color=[INK if s else "#eeeeee" for s in seen])
    for b in (*bars_a, *bars_s):
        b.set_visible(False)

    def draw(i):
        f = frames[i]
        ax.clear()
        ax.set(xlim=(-0.5, 5.5), ylim=(5.5, -0.9), aspect="equal", xticks=range(6), yticks=range(6))
        ax.tick_params(length=0, labelsize=8)
        for r in range(6):
            for c in range(6):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc="#f8e3e6" if r == 0 else "white", ec="#efd2d6"))
        # 감독관 시야: 우리 각도 θ(정면=아래, +는 열 증가 쪽)를 데이터 좌표 각도 90°-θ 로 변환
        phi = ANGLES[f["head"]]
        ax.add_patch(Wedge((f["pc"], 0), 9, 90 - phi - FOV, 90 - phi + FOV, color=RED, alpha=0.13, lw=0))
        ax.add_patch(Circle((f["pc"], 0), 0.3, color=INK))
        ax.plot([f["pc"], f["pc"] + 0.55 * np.sin(np.radians(phi))], [0, 0.55 * np.cos(np.radians(phi))], color=INK, lw=3)
        # 학생: 직접 초록, AI 빨강, 적발 검정 X
        color = INK if f["caught"] else (RED if f["a"] else GREEN)
        ax.add_patch(Rectangle((f["col"] - 0.3, f["row"] - 0.3), 0.6, 0.6, color=color))
        if f["caught"]:
            ax.text(f["col"], f["row"], "적발", color="white", ha="center", va="center", fontsize=9, weight="bold")
        act = "적발!" if f["caught"] else ("AI 안경" if f["a"] else "직접 풀기")
        ax.set_title(f"{name}  ·  {f['t']}분  ·  {f['p']}점 문제  ·  {act}\n"
                     f"완료 {f['k']}/30문항  ·  기대 점수 {f['exp']:.1f}", fontsize=10)
        zline.set_data(np.arange(1, i + 2), z[: i + 1])
        bars_a[i].set_visible(True)
        bars_s[i].set_visible(True)

    end = "적발 (0점)" if info["caught"] else f"정상 종료, 실제 점수 {info['score']}점"
    fig.suptitle(f"에피소드 seed {seed}  ·  결과: {end}  ·  AI 비율 {info['ai_frac']:.0%}", fontsize=12)
    anim = FuncAnimation(fig, draw, frames=n, interval=1000 // fps)
    path = f"{out}/replay_{name}_ep{seed}.gif"
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"저장: {path}  ({n}분, {end})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["curves", "replay"])
    ap.add_argument("name", nargs="?", default="dqn_seed0", help="replay할 모델 이름 (results/<이름>.pt)")
    ap.add_argument("--seed", type=int, default=0, help="replay할 에피소드 seed (자리, 문제 순서, 감독관이 바뀜)")
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()
    os.makedirs("figures", exist_ok=True)
    if args.what == "curves":
        curves("figures")
    else:
        replay(args.name, args.seed, "figures", args.fps)
