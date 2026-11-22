"""최소 A2C (직접 구현, n-step return). 기본값은 sweeps.py의 BASE["a2c"]. 실행: python a2c.py --steps 2000000 --seed 0"""
import numpy as np
import torch
import torch.nn as nn
from exam_env import ExamEnv
from ppo import mlp, finish, cli


def train(steps, seed=0, out=".", name="a2c", log=print, *, lr, gamma, ent, n_steps):
    n = n_steps
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = ExamEnv(seed)
    actor, critic = mlp(ExamEnv.n_actions), mlp(1)
    params = [*actor.parameters(), *critic.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    obs, _ = env.reset()
    scores, caught = [], []

    for update in range(steps // n):
        # 1) n스텝 수집
        O, A, R, D = [], [], [], []
        for _ in range(n):
            with torch.no_grad():
                a = int(torch.distributions.Categorical(logits=actor(torch.as_tensor(obs))).sample())
            O.append(obs); A.append(a)
            obs, r, done, _, info = env.step(a)
            R.append(r); D.append(done)
            if done:
                scores.append(info["score"]); caught.append(info["caught"])
                obs, _ = env.reset()

        # 2) n-step return (마지막 상태 가치로 bootstrap, 종료 지점에서 끊음)
        with torch.no_grad():
            g = critic(torch.as_tensor(obs)).item()
        ret = np.zeros(n, np.float32)
        for i in reversed(range(n)):
            g = R[i] + gamma * g * (1 - D[i])
            ret[i] = g

        # 3) actor-critic 업데이트
        o, a_t, ret_t = torch.as_tensor(np.array(O)), torch.as_tensor(A), torch.as_tensor(ret)
        v = critic(o).squeeze(-1)
        dist = torch.distributions.Categorical(logits=actor(o))
        adv = ret_t - v.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # PPO와 같은 정규화. 없으면 보상이 작아 entropy 항이 학습을 압도함
        entropy = dist.entropy().mean()
        loss = -(dist.log_prob(a_t) * adv).mean() + 0.5 * ((ret_t - v) ** 2).mean() - ent * entropy
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 0.5)
        opt.step()

        if update % (50_000 // n) == 0 and scores:
            log(f"step {(update + 1) * n:>8} | 점수 {np.mean(scores[-100:]):5.1f} | "
                f"적발 {np.mean(caught[-100:]):5.1%} | ent {entropy.item():.3f}")

    return finish(actor, f"{out}/{name}_seed{seed}.pt", log)


if __name__ == "__main__":
    cli("a2c", train)
