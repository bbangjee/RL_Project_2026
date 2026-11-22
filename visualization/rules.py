"""규칙 정책으로 환경 수치 점검. 실행: python -m visualization.rules"""
import numpy as np
from environment.exam_env import ANGLES, FOV, ExamEnv


def evaluate(policy, n=2000, seed=12345, env_kwargs=None):
    """policy(obs, env) -> action.

    평균 점수, 90점 이상 달성률, 적발률, AI 사용 비율을 반환한다.
    90점 이상 달성은 적발되지 않고 최종 점수가 90점 이상인 경우다.
    """
    env = ExamEnv(seed, **(env_kwargs or {}))
    scores, success90, caught, ai = [], [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy(obs, env))
        scores.append(info["score"])
        success90.append(not info["caught"] and info["score"] >= 90)
        caught.append(info["caught"])
        ai.append(info["ai_frac"])
    return np.mean(scores), np.mean(success90), np.mean(caught), np.mean(ai)


def safe_next_step(env):
    """감독관이 다음 스텝에 어떻게 움직여도(한 칸, 한 구역) 시야 밖이면 True. 감독관 규칙을 아는 설계자 규칙."""
    for pc in (env.pc - 1, env.pc, env.pc + 1):
        for h in (env.head - 1, env.head, env.head + 1):
            if 0 <= pc <= 5 and 0 <= h <= 4:
                ang = np.degrees(np.arctan2(env.col - pc, env.row))
                if abs(ang - ANGLES[h]) <= FOV:
                    return False
    return True


rng = np.random.default_rng(0)
RULES = {
    "랜덤": lambda o, e: int(rng.random() < 0.5),
    "정직 (항상 직접)": lambda o, e: 0,
    "항상 AI": lambda o, e: 1,
    "현재 시야 밖이면 AI": lambda o, e: int(not e.in_view()),
    "안전 타이밍 정책": lambda o, e: int(safe_next_step(e)),
}

if __name__ == "__main__":
    print(f"{'정책':<20}{'점수':>8}{'90점+':>8}{'적발률':>8}{'AI비율':>8}")
    for name, pol in RULES.items():
        s, s90, c, a = evaluate(pol)
        print(f"{name:<20}{s:8.1f}{s90:8.1%}{c:8.1%}{a:8.1%}")
