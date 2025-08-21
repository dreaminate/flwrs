from __future__ import annotations
import argparse, os, time, json, yaml, subprocess

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="src/bench/experiments.yaml")
    return p.parse_args()

def run_one(exp: dict):
    name = exp["name"]
    strategy = exp["strategy"]
    models = ",".join(exp["models"])
    rounds  = int(exp.get("rounds", 20))
    clients = int(exp.get("clients", 30))
    params  = exp.get("params", {})
    cmd = [
        "python", "src/run_multimodel_sim.py",
        "--strategy", strategy,
        "--models", models,
        "--rounds", str(rounds),
        "--clients", str(clients),
        "--min_fit", str(max(2, int(clients*0.3))),
    ]
    for k, v in params.items():
        cmd.extend([f"--{k}", str(v)])
    print(f"[bench] running {name}: {' '.join(cmd)}")
    code = subprocess.call(cmd)
    return code

def main():
    args = parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    exps = cfg.get("experiments", [])
    os.makedirs("bench_results", exist_ok=True)
    summary = []
    for exp in exps:
        t0 = time.time()
        code = run_one(exp)
        dt = time.time() - t0
        summary.append({"name": exp["name"], "retcode": code, "sec": round(dt, 2)})
    with open("bench_results/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[bench] summary:", summary)

if __name__ == "__main__":
    main()
