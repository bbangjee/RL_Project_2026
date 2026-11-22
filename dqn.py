"""최소 DQN (직접 구현, target network + replay buffer). 기본값은 sweeps.py의 BASE["dqn"].
실행: python dqn.py --steps 2000000 --seed 0"""
import copy
import numpy as np
import torch
import torch.nn as nn
from exam_env import ExamEnv
from ppo import mlp, finish, cli


def train(steps, seed=0, out=".", name="dqn", log=print, *,
          lr, gamma, buf_size, batch, warmup, train_every, target_every, eps_frac,
          ai_acc_4, reward_bonus):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = ExamEnv(seed, ai_acc_4=ai_acc_4, reward_bonus=reward_bonus)
    q = mlp(ExamEnv.n_actions)
    q_target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=lr)

    O = np.zeros((buf_size, ExamEnv.obs_dim), np.float32)
    O2 = np.zeros_like(O)
    A = np.zeros(buf_size, np.int64)
    R, D = np.zeros(buf_size, np.float32), np.zeros(buf_size, np.float32)
    obs, _ = env.reset()
    scores, caught, loss = [], [], torch.tensor(0.0)

    for t in range(steps):
        # epsilon: 전체 스텝의 30% 동안 1 -> 0.05 선형 감소
        eps = max(0.05, 1 - 0.95 * t / (eps_frac * steps))
        if rng.random() < eps:
            a = int(rng.integers(ExamEnv.n_actions))
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(obs)).argmax())
        obs2, r, done, _, info = env.step(a)
        i = t % buf_size
        O[i], A[i], R[i], O2[i], D[i] = obs, a, r, obs2, done
        obs = obs2
        if done:
            scores.append(info["score"]); caught.append(info["caught"])
            obs, _ = env.reset()

        if t >= warmup and t % train_every == 0:
            idx = rng.integers(0, min(t + 1, buf_size), batch)
            o, a_, r_, o2, d = (torch.as_tensor(x[idx]) for x in (O, A, R, O2, D))
            with torch.no_grad():
                y = r_ + gamma * (1 - d) * q_target(o2).max(1).values   # 종료는 전부 terminated
            q_sa = q(o).gather(1, a_[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q_sa, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10)
            opt.step()
        if t % target_every == 0:
            q_target.load_state_dict(q.state_dict())

        if (t + 1) % 50_000 == 0 and scores:
            log(f"step {t + 1:>8} | 점수 {np.mean(scores[-100:]):5.1f} | 적발 {np.mean(caught[-100:]):5.1%} | "
                f"eps {eps:.2f} | loss {loss.item():.5f}")

    return finish(q, f"{out}/{name}_seed{seed}.pt", log,
                  env_kwargs={"ai_acc_4": ai_acc_4, "reward_bonus": reward_bonus})


if __name__ == "__main__":
    cli("dqn", train)
