"""최소 PPO (직접 구현). 하이퍼파라미터 기본값은 sweeps.py의 BASE["ppo"].
실행: python ppo.py --steps 500000 --seed 0 --ent 0.05"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from exam_env import ExamEnv
from rules import evaluate
from sweeps import BASE


def mlp(out):
    return nn.Sequential(
        nn.Linear(ExamEnv.obs_dim, 64), 
        nn.Tanh(), 
        nn.Linear(64, 64), 
        nn.Tanh(), 
        nn.Linear(64, out))


def finish(net, path, log, env_kwargs=None):
    """모델 저장 후 학습과 분리된 평가 에피소드 2000판으로 greedy 평가. PPO, A2C, DQN 공통."""
    torch.save(net.state_dict(), path)

    @torch.no_grad()
    def greedy(o, e):
        return int(net(torch.as_tensor(o)).argmax())

    s, s90, c, ai = evaluate(greedy, env_kwargs=env_kwargs)
    log(f"\n[평가 2000판, greedy] 점수 {s:.1f} | 90점+ {s90:.1%} | "
        f"적발률 {c:.1%} | AI 비율 {ai:.1%} | 저장: {path}")
    return {"score": float(s), "success90": float(s90), "caught": float(c),
            "ai_frac": float(ai), "model": path}


def train(steps, seed=0, out=".", name="ppo", log=print, *,
          lr, gamma, ent, n_steps, epochs, minibatch, lam, clip, ai_acc_4):
    torch.set_num_threads(1)  # 병렬 실행 시 프로세스마다 스레드 1개
    torch.manual_seed(seed)
    env = ExamEnv(seed, ai_acc_4=ai_acc_4)
    actor, critic = mlp(ExamEnv.n_actions), mlp(1)
    params = [*actor.parameters(), *critic.parameters()]
    opt = torch.optim.Adam(params, lr=lr)

    N = n_steps
    O = np.zeros((N, ExamEnv.obs_dim), np.float32)
    A, LP, R, D, V = (np.zeros(N, np.float32) for _ in range(5))
    obs, _ = env.reset()
    ep_scores, ep_caught = [], []

    for update in range(steps // N):
        # 1) 롤아웃 수집
        for i in range(N):
            with torch.no_grad():
                x = torch.as_tensor(obs)
                dist = torch.distributions.Categorical(logits=actor(x))
                a = dist.sample()
                O[i], A[i], LP[i], V[i] = obs, a, dist.log_prob(a), critic(x)
            obs, R[i], D[i], _, info = env.step(int(a))
            if D[i]:
                ep_scores.append(info["score"]); ep_caught.append(info["caught"])
                obs, _ = env.reset()

        # 2) GAE (종료는 전부 terminated라 truncation 처리 없음)
        with torch.no_grad():
            next_v = critic(torch.as_tensor(obs)).item()
        adv = np.zeros(N, np.float32)
        last = 0.0
        for i in reversed(range(N)):
            nv = next_v if i == N - 1 else V[i + 1]
            nonterm = 1.0 - D[i]
            delta = R[i] + gamma * nv * nonterm - V[i]
            last = delta + gamma * lam * nonterm * last
            adv[i] = last
        ret = adv + V

        # 3) PPO 업데이트
        o, a, lp_old = torch.as_tensor(O), torch.as_tensor(A).long(), torch.as_tensor(LP)
        adv_t, ret_t = torch.as_tensor(adv), torch.as_tensor(ret)
        for _ in range(epochs):
            for idx in torch.randperm(N).split(minibatch):
                dist = torch.distributions.Categorical(logits=actor(o[idx]))
                lp = dist.log_prob(a[idx])
                ratio = (lp - lp_old[idx]).exp()
                ad = adv_t[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pg_loss = -torch.min(ratio * ad, ratio.clamp(1 - clip, 1 + clip) * ad).mean()
                v_loss = ((critic(o[idx]).squeeze(-1) - ret_t[idx]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = pg_loss + 0.5 * v_loss - ent * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()

        if update % 10 == 0 and ep_scores:
            kl = (lp_old[idx] - lp).mean().item()
            clipfrac = ((ratio - 1).abs() > clip).float().mean().item()
            log(f"step {(update + 1) * N:>8} | 점수 {np.mean(ep_scores[-100:]):5.1f} | "
                f"적발 {np.mean(ep_caught[-100:]):5.1%} | ent {entropy.item():.3f} | kl {kl:.4f} | "
                f"clip {clipfrac:.2f} | vloss {v_loss.item():.4f}")

    return finish(actor, f"{out}/{name}_seed{seed}.pt", log,
                  env_kwargs={"ai_acc_4": ai_acc_4})


def cli(algo, train_fn):
    """--steps, --seed 와 sweeps.py BASE의 모든 하이퍼파라미터를 명령행 인자로."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    for k, v in BASE[algo].items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    train_fn(**vars(ap.parse_args()), name=algo)


if __name__ == "__main__":
    cli("ppo", train)
