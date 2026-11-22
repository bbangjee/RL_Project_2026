"""하이퍼파라미터와 sweep 정의. 실험 구성은 이 파일만 고치면 된다.
실행 이름 규칙: 알고리즘 이름 + 바꾼 파라미터 (예: ppo, ppo_ent0.01, ppo_gamma0.99, dqn). 결과는 results/<이름>_seed<n>.json"""

SEEDS = [0, 1, 2, 3, 4]
STEPS = 2_000_000          # 모든 방법에 같은 환경 스텝 예산

# 알고리즘별 기본 하이퍼파라미터 (sweep에서 안 바꾸면 이 값)
BASE = {
    "ppo": dict(lr=3e-4, gamma=0.995, ent=0.05, n_steps=2048, epochs=10,
                minibatch=256, lam=0.95, clip=0.2, ai_acc_4=0.85),
    "a2c": dict(lr=7e-4, gamma=0.995, ent=0.05, n_steps=32, ai_acc_4=0.85),
    "dqn": dict(lr=3e-4, gamma=0.995, buf_size=200_000, batch=128, warmup=10_000,
                train_every=4, target_every=2000, eps_frac=0.3, ai_acc_4=0.85),
}

# sweep 그룹: [(알고리즘, 기본값에서 바꿀 파라미터), ...]. 같은 설정은 그룹 사이에서 한 번만 실행됨
SWEEPS = {
    "E1 알고리즘": [("ppo", {}), ("a2c", {}), ("dqn", {})],
    "E2 AI 정확도": [("ppo", {"ai_acc_4": acc}) for acc in (0.85, 0.90, 0.95)],
    "E4 할인율 gamma": [("ppo", {"gamma": g}) for g in (0.99, 0.995, 1.0)],
    "E5 entropy 계수": [("ppo", {"ent": e}) for e in (0.01, 0.05, 0.1)],
}


def run_name(algo, override):
    """기본값과 같은 파라미터는 이름에서 뺀다. ppo + {gamma: 0.995} -> 'ppo'"""
    diff = {k: v for k, v in override.items() if BASE[algo][k] != v}
    labels = {"ai_acc_4": "ai4acc"}
    return "_".join([algo, *(f"{labels.get(k, k)}{v}" for k, v in diff.items())])
