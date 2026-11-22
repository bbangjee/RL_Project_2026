"""sweeps.py에 정의된 실험을 CPU 코어 병렬로 실행. 실행: python run_experiments.py --workers 24
결과: results/<이름>_seed<n>.json, .log, .pt. 이미 끝난 실행은 건너뜀(중단 후 재실행 가능).
특정 그룹만: python run_experiments.py --only E1"""
import argparse
import importlib
import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import numpy as np
from sweeps import BASE, SEEDS, STEPS, SWEEPS, run_name
from tqdm import tqdm

OUT = "results"
STEP_RE = re.compile(r"^step\s+(\d+)")


def run(algo, override, seed, steps):
    name = run_name(algo, override)
    train = importlib.import_module(algo).train
    with open(f"{OUT}/{name}_seed{seed}.log", "w", encoding="utf-8") as f:
        res = train(steps, seed, OUT, name, lambda s: (f.write(s + "\n"), f.flush()),
                    **{**BASE[algo], **override})
    res.update(name=name, algo=algo, seed=seed, steps=steps, **override)
    with open(f"{OUT}/{name}_seed{seed}.json", "w") as f:
        json.dump(res, f)
    return f"{name}_seed{seed}", res


def bootstrap_ci(x, n=10000, seed=0):
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def latest_step(name, seed, started_at):
    """현재 실행에서 log에 기록된 마지막 환경 step을 반환한다."""
    path = f"{OUT}/{name}_seed{seed}.log"
    if not os.path.exists(path) or os.path.getmtime(path) < started_at - 1:
        return 0
    step = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if match := STEP_RE.match(line):
                step = int(match.group(1))
    return step


def summary(groups):
    for g in groups:
        print(f"\n[{g}]\n{'실행':<18}{'n':>3}{'점수 평균':>10}{'95% CI':>18}"
              f"{'90점+':>8}{'적발률':>8}{'AI비율':>8}")
        for algo, ov in SWEEPS[g]:
            name = run_name(algo, ov)
            rs = [json.load(open(p)) for s in SEEDS if os.path.exists(p := f"{OUT}/{name}_seed{s}.json")]
            if not rs:
                continue
            sc = np.array([r["score"] for r in rs])
            lo, hi = bootstrap_ci(sc)
            print(f"{name:<18}{len(rs):>3}{sc.mean():10.1f}   [{lo:5.1f}, {hi:5.1f}]"
                  f"{np.mean([r['success90'] for r in rs]):8.1%}"
                  f"{np.mean([r['caught'] for r in rs]):8.1%}{np.mean([r['ai_frac'] for r in rs]):8.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--only", nargs="*", default=None, help="그룹 이름 앞부분 (예: E1 E5)")
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS,
                    help="실행할 seed 목록 (기본값: 0 1 2 3 4)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    groups = [g for g in SWEEPS if args.only is None or any(g.startswith(o) for o in args.only)]
    jobs = {}  # 같은 설정이 여러 그룹에 있으면 한 번만
    for g in groups:
        for algo, ov in SWEEPS[g]:
            for s in args.seeds:
                key = (run_name(algo, ov), s)
                if key not in jobs and not os.path.exists(f"{OUT}/{key[0]}_seed{s}.json"):
                    jobs[key] = (algo, ov, s)
    print(f"실행할 작업 {len(jobs)}개, 워커 {args.workers}개")
    if jobs:
        started_at = time.time()
        with ProcessPoolExecutor(args.workers) as pool:
            futures = {
                pool.submit(run, algo, ov, seed, args.steps): key
                for key, (algo, ov, seed) in jobs.items()
            }
            pending = set(futures)
            finished = set()
            shown_steps = 0
            total_steps = len(jobs) * args.steps

            with tqdm(total=total_steps, desc="전체 학습", unit="step", unit_scale=True) as bar:
                while pending:
                    done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for fut in done:
                        key = futures[fut]
                        finished.add(key)
                        name, result = fut.result()
                        tqdm.write(
                            f"[{len(finished)}/{len(jobs)}] {name}: 점수 {result['score']:.1f}, "
                            f"90점+ {result['success90']:.1%}, 적발률 {result['caught']:.1%}"
                        )

                    current_steps = 0
                    for name, seed in jobs:
                        if (name, seed) in finished:
                            current_steps += args.steps
                        else:
                            current_steps += min(args.steps, latest_step(name, seed, started_at))
                    if current_steps > shown_steps:
                        bar.update(current_steps - shown_steps)
                        shown_steps = current_steps
                    bar.set_postfix_str(f"완료 {len(finished)}/{len(jobs)}")

                if shown_steps < total_steps:
                    bar.update(total_steps - shown_steps)
    summary(groups)
