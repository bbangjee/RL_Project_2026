"""AI 안경 시험장 환경 (문서 5, 6장 정의). Gymnasium 스타일: reset() -> (obs, info), step(a) -> (obs, r, terminated, truncated, info)."""
import numpy as np

T = 100                                # 시험 시간 100분 (1스텝 = 1분)
GAMMA = 0.995
POINTS = np.array([2, 3, 4])           # 배점 인덱스 0, 1, 2
COUNT = np.array([3, 14, 13])          # 배점별 문항 수 (합 30, 100점)
SOLVE_TIME = np.array([2, 3, 6])       # 직접 풀기 시간
SOLVE_ACC = np.array([0.83, 0.68, 0.35])
AI_TIME = np.array([1, 1, 2])          # AI 안경 사용 시간
AI_ACC = np.array([0.98, 0.95, 0.85])
ANGLES = np.array([-60, -30, 0, 30, 60])  # 감독관 고개 5구역
FOV = 30                               # 시야: 각도 차이 30도 이내


class ExamEnv:
    obs_dim = 17
    n_actions = 2  # 0: 직접 풀기, 1: AI 안경 사용하기 

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        self.row, self.col = int(r.integers(1, 6)), int(r.integers(0, 6))  # 학생 자리
        self.pc, self.w = int(r.integers(0, 6)), int(r.choice([-1, 1]))     # 감독관 열, 걷는 방향
        self.head, self.omega = int(r.integers(0, 5)), int(r.choice([-1, 1]))  # 고개 구역, 훑는 방향
        self.z = 0.0          # 의심도
        self.t = 0
        self.order = r.permutation(np.repeat([0, 1, 2], COUNT))  # 문제 순서 (배점 인덱스)
        self.k = 0            # 현재 문제 번호
        self.prog = [0, 0]    # [직접 진행도, AI 진행도]
        self.expected = 0.0   # 기대 점수 (state용)
        self.score = 0        # 실제 점수 (보상용)
        self.ai_steps = 0
        self.left = COUNT.copy()
        self.left[self.order[0]] -= 1  # 현재 문제 제외한 남은 문제 수
        return self._obs(), {}

    def in_view(self):
        ang = np.degrees(np.arctan2(self.col - self.pc, self.row))  # 감독관 정면(0도) 기준 학생 방향
        return abs(ang - ANGLES[self.head]) <= FOV

    def step(self, a):
        r = self.rng
        self.t += 1
        # 감독관 걷기: 0.1 확률로 방향 반전, 0.5 확률로 한 칸 이동 (벽이면 반대로)
        if r.random() < 0.1:
            self.w = -self.w
        if r.random() < 0.5:
            if not 0 <= self.pc + self.w <= 5:
                self.w = -self.w
            self.pc += self.w
        # 고개: 0.5 확률로 훑는 방향 쪽 옆 구역으로 (끝이면 방향 반전)
        if r.random() < 0.5:
            if not 0 <= self.head + self.omega <= 4:
                self.omega = -self.omega
            self.head += self.omega

        # 행동 처리: AI 사용 중 시야 안이면 적발 판정, 아니면 의심도 감소
        if a == 1:
            self.ai_steps += 1
        if a == 1 and self.in_view():
            d = np.hypot(self.row, self.col - self.pc)
            p0, inc = (0.5, 0.4) if d <= 2 else (0.3, 0.25) if d <= 4 else (0.1, 0.1)
            if r.random() < min(1.0, p0 + self.z):
                return self._obs(), 0.0, True, False, self._info(caught=True)
            self.z = min(1.0, self.z + inc)
        else:
            self.z = max(0.0, self.z - 0.05)

        # 진행도 +1, 다 차면 그 방법의 정답률로 채점하고 다음 문제로
        p = self.order[self.k]
        self.prog[a] += 1
        if self.prog[a] >= (SOLVE_TIME, AI_TIME)[a][p]:
            acc = (SOLVE_ACC, AI_ACC)[a][p]
            self.expected += POINTS[p] * acc
            self.score += POINTS[p] * int(r.random() < acc)
            self.k += 1
            self.prog = [0, 0]
            if self.k < 30:
                self.left[self.order[self.k]] -= 1

        done = self.t >= T or self.k >= 30
        reward = self.score / 100 if done else 0.0
        return self._obs(), reward, done, False, self._info(caught=False) if done else {}

    def _info(self, caught):
        return {"caught": caught, "score": 0 if caught else self.score, "ai_frac": self.ai_steps / self.t}

    def _obs(self):
        p = self.order[min(self.k, 29)]
        return np.array([
            self.row / 5, self.col / 5,            # 1, 2 내 자리
            self.pc / 5, self.w,                   # 3, 4 감독관 위치, 걷는 방향
            ANGLES[self.head] / 60, self.omega,    # 5, 6 고개 각도, 훑는 방향
            self.z,                                # 7 의심도
            *(np.arange(3) == p),                  # 8~10 현재 문제 배점 one-hot
            self.prog[0] / SOLVE_TIME[p],          # 11 직접 진행도
            self.prog[1] / AI_TIME[p],             # 12 AI 진행도
            1 - self.t / T,                        # 13 남은 시간
            self.expected / 100,                   # 14 기대 점수
            *(self.left / COUNT),                  # 15~17 배점별 남은 문제 수
        ], dtype=np.float32)
