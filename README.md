# AI Cheating Strategy in a 2D Exam Room

서강대학교 2026년 2학기 강화학습 프로젝트입니다. 2D 시험실에서 학생 에이전트가 매 분 `직접 풀이`와 `AI 사용` 중 하나를 선택하고, 감독관에게 적발되지 않으면서 90점 이상을 얻는 전략을 학습합니다.

## Team

| 이름 | 학번 |
|---|---:|
| 방지혁 | 12026***1 |
| 김태환 | 12026***1 |

- GitHub: <https://github.com/bbangjee/RL_Project_2026>

## Problem Setup

- 시험 시간: 100분, 1 step = 1분
- 시험 문제: 수능 수학 형식 30문항, 100점
- 행동 공간: `0 = 직접 풀이`, `1 = AI 사용`
- 직접 풀이 시간: 2점 2분, 3점 4분, 4점 10분
- AI 사용 시간: 모든 문제 1분
- 적발: 감독관의 시야, 거리, 누적 의심도로 확률 계산
- 보상: 중간 보상은 0, 정상 종료 시 `최종 점수 / 100`, 적발 시 0
- 성공 지표: 적발되지 않고 최종 점수 90점 이상

에이전트는 16차원 observation을 사용합니다. 실제 정답 여부와 실제 누적 점수는 환경 내부에서만 관리하며 observation에 포함하지 않습니다.

## Algorithms

다음 알고리즘을 PyTorch로 직접 구현했습니다.

- DQN: replay buffer와 target network 사용
- PPO: clipped policy update와 GAE 사용
- A2C: n-step return 사용

모든 알고리즘은 두 개의 은닉층 `(64, 64)`과 `Tanh` 활성화 함수를 사용하는 MLP를 사용합니다.

## Installation

Python 3.10 이상을 권장합니다.

```bash
git clone https://github.com/bbangjee/RL_Project_2026.git
cd RL_Project_2026

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Windows PowerShell에서는 가상환경을 다음 명령으로 활성화합니다.

```powershell
.venv\Scripts\Activate.ps1
```

## Quick Start

### 1. Baseline policies

```bash
python3 rules.py
```

다음 고정 정책의 평균 점수, 90점 이상 달성률, 적발률과 AI 사용 비율을 평가합니다.

- 무작위 선택
- 항상 직접 풀이
- 항상 AI 사용
- 현재 감독관 시야 밖이면 AI 사용
- 안전 타이밍 정책

### 2. Algorithm comparison

DQN, PPO, A2C를 각각 5개 random seed에서 2,000,000 environment steps 동안 학습합니다.

```bash
python3 run_experiments.py --only E1 --workers 3
```

### 3. AI accuracy sensitivity

4점 문제의 AI 정확도를 85%, 90%, 95%로 변경하여 PPO를 학습합니다. 기존 결과가 있으면 자동으로 건너뜁니다.

```bash
python3 run_experiments.py --only E2 --workers 3
```

### 4. 90-point reward bonus sensitivity

90점 이상 달성 보너스를 비교합니다. 4점 문제 AI 정확도는 95%로 고정합니다.

```bash
python3 run_experiments.py --only E3 --workers 3
```

### 5. PPO hyperparameter sweeps

```bash
# discount factor: 0.99, 0.995, 1.0
python3 run_experiments.py --only E4 --workers 3

# entropy coefficient: 0.01, 0.05, 0.1
python3 run_experiments.py --only E5 --workers 3

# 두 실험을 한 번에 실행
python3 run_experiments.py --only E4 E5 --workers 3
```

`--workers`는 동시에 실행할 CPU process 수입니다. 각 학습 process는 PyTorch thread 하나를 사용합니다.

### 6. Train one model

```bash
python3 ppo.py --steps 2000000 --seed 0
python3 a2c.py --steps 2000000 --seed 0
python3 dqn.py --steps 2000000 --seed 0
```

## Evaluation

학습된 각 모델을 학습과 분리된 2,000개 에피소드에서 greedy policy로 평가합니다. 각 실험은 5개 random seed의 평균을 사용하고 평균 점수에는 bootstrap 95% confidence interval을 계산합니다.

평가 지표는 다음과 같습니다.

- 평균 최종 점수
- 90점 이상 달성률
- 적발률
- AI 사용 비율

## Results

### Algorithm comparison

| Algorithm | Average score | 95% CI | Score >= 90 | Detection rate | AI usage |
|---|---:|---:|---:|---:|---:|
| DQN | **68.3** | [65.9, 70.1] | 9.5% | **10.4%** | 24.1% |
| PPO | 60.5 | [57.7, 63.2] | **12.9%** | 23.9% | 31.8% |
| A2C | 30.7 | [25.4, 35.9] | 11.2% | 64.2% | 44.5% |

### AI accuracy sensitivity with PPO

| 4-point AI accuracy | Average score | 95% CI | Score >= 90 | Detection rate | AI usage |
|---:|---:|---:|---:|---:|---:|
| 85% | 60.5 | [57.7, 63.2] | 12.9% | 23.9% | 31.8% |
| 90% | **64.1** | [62.9, 65.5] | 20.7% | **22.2%** | 31.9% |
| 95% | 63.0 | [60.0, 65.9] | **27.7%** | 25.3% | 32.4% |

## Saved Models and Logs

학습된 모델과 평가 결과는 [`results/`](./results)에서 확인할 수 있습니다.

| Extension | Description |
|---|---|
| `.pt` | PyTorch model weights |
| `.json` | Final evaluation metrics |
| `.log` | Training log |

파일 이름은 `algorithm_setting_seedN` 형식을 사용합니다.

```text
results/
├── dqn_seed0.pt
├── ppo_seed0.pt
├── a2c_seed0.pt
├── ppo_ai4acc0.9_seed0.pt
└── ppo_ai4acc0.95_seed0.pt
```

GitHub에서 `.pt` 파일을 선택한 뒤 `Download raw file`을 누르면 학습된 모델을 다운로드할 수 있습니다.

## Visualization

학습 곡선을 다시 생성합니다.

```bash
python3 plot_training_curves.py
```

결과:

- `figures/training_curves.png`
- `figures/training_score.png`
- `figures/training_detection.png`

저장된 모델의 에피소드 행동을 GIF로 시각화합니다.

```bash
python3 visualize.py replay dqn_seed0 --seed 3
```

결과는 `figures/replay_dqn_seed0_ep3.gif`에 저장됩니다.

## Project Structure

```text
.
├── exam_env.py              # 2D 시험실 환경
├── rules.py                 # baseline policies 및 평가
├── dqn.py                   # DQN 구현
├── ppo.py                   # PPO 구현
├── a2c.py                   # A2C 구현
├── sweeps.py                # 실험 설정과 hyperparameters
├── run_experiments.py       # 병렬 학습 및 결과 집계
├── plot_training_curves.py  # 발표용 학습 곡선 생성
├── visualize.py             # 학습 곡선과 에피소드 GIF 생성
├── results/                 # model, metric, training log
└── figures/                 # graph 및 animation
```

## Reproducibility

- 학습 seed: `0, 1, 2, 3, 4`
- 기본 학습 예산: 알고리즘별 2,000,000 environment steps
- 공통 할인율: `gamma = 0.995`
- 기본 결과와 sweep 설정: [`sweeps.py`](./sweeps.py)

동일한 명령을 다시 실행하면 이미 완료된 `.json` 결과는 건너뛰고 미완료 작업만 실행합니다.
