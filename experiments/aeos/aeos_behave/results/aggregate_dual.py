import json, glob, os

results_dir = os.path.dirname(os.path.abspath(__file__))
# Check both the old flat directory and the new tabular2 subdirectory
files = sorted(glob.glob(os.path.join(results_dir, "exp2_*_tabular*.json")) + glob.glob(os.path.join(results_dir, "tabular2", "exp2_*.json")))

print(f"{'Reviewer':<22} | {'Coder':<22} | {'Acc':>5} | {'Iters':>5} | {'BestIt':>6} | {'Sunk':>4} | {'Time':>6} | Stop Reason")
print("-" * 130)

for f in files:
    with open(f, "r") as fh:
        d = json.load(fh)
    rev = d.get("reviewer_model", "?")
    cod = d.get("coder_model", "?")
    acc = d.get("best_accuracy") or 0
    iters = d.get("total_iterations") or 0
    best_it = d.get("best_iteration") or 0
    sunk = d.get("sunk_cost_episodes") or 0
    t = d.get("total_time_seconds") or 0
    stop = d.get("stop_reason", "?")
    print(f"{rev:<22} | {cod:<22} | {acc:>5.3f} | {iters:>5} | {best_it:>6} | {sunk:>4} | {t:>5.0f}s | {stop}")
