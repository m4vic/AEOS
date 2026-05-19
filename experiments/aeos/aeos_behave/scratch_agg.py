import json
import glob
import os
from collections import defaultdict

results_dir = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos\aeos_behave\results"
datasets = ["tabular2", "vision", "text"]

output = []

def process_exp1(dataset):
    path = os.path.join(results_dir, dataset, "exp1_*.json")
    files = glob.glob(path)
    
    models = defaultdict(lambda: {"acc": [], "iters": [], "sunk": [], "time": []})
    
    for f in files:
        with open(f, "r") as fh:
            d = json.load(fh)
        model = d.get("model", "unknown")
        # In single agent, model might be missing, try to infer from filename
        if model == "unknown":
            basename = os.path.basename(f)
            model = basename.split("_")[1]
            
        acc = d.get("best_accuracy", 0)
        iters = d.get("total_iterations", 0)
        sunk = d.get("sunk_cost_episodes", 0)
        t = d.get("total_time_seconds", 0)
        
        models[model]["acc"].append(acc)
        models[model]["iters"].append(iters)
        models[model]["sunk"].append(sunk)
        models[model]["time"].append(t)
        
    res = []
    for m, data in models.items():
        n = len(data["acc"])
        if n == 0: continue
        avg_acc = sum(data["acc"])/n
        avg_iters = sum(data["iters"])/n
        avg_sunk = sum(data["sunk"])/n
        avg_time = sum(data["time"])/n
        res.append((m, avg_acc, avg_iters, avg_sunk, avg_time, n))
        
    res.sort(key=lambda x: x[1], reverse=True)
    return res

def process_exp2(dataset):
    path = os.path.join(results_dir, dataset, "exp2_*.json")
    files = glob.glob(path)
    
    pairings = defaultdict(lambda: {"acc": [], "iters": [], "sunk": [], "time": []})
    
    for f in files:
        with open(f, "r") as fh:
            d = json.load(fh)
            
        rev = d.get("reviewer_model", "unknown")
        cod = d.get("coder_model", "unknown")
        if rev == "unknown":
            basename = os.path.basename(f)
            parts = basename.split("_")
            if len(parts) >= 3:
                rev = parts[1]
                cod = parts[2]
                
        key = f"{rev} + {cod}"
        
        acc = d.get("best_accuracy", 0)
        iters = d.get("total_iterations", 0)
        sunk = d.get("sunk_cost_episodes", 0)
        t = d.get("total_time_seconds", 0)
        
        pairings[key]["acc"].append(acc)
        pairings[key]["iters"].append(iters)
        pairings[key]["sunk"].append(sunk)
        pairings[key]["time"].append(t)
        
    res = []
    for p, data in pairings.items():
        n = len(data["acc"])
        if n == 0: continue
        avg_acc = sum(data["acc"])/n
        avg_iters = sum(data["iters"])/n
        avg_sunk = sum(data["sunk"])/n
        avg_time = sum(data["time"])/n
        res.append((p, avg_acc, avg_iters, avg_sunk, avg_time, n))
        
    res.sort(key=lambda x: x[1], reverse=True)
    return res

for ds in datasets:
    output.append(f"## Dataset: {ds.upper()}")
    
    # Single
    single = process_exp1(ds)
    output.append("### Single-Agent (Exp 1)")
    output.append("| Model | Accuracy | Total Iters | Sunk Cost Episodes | Time (s) | Runs |")
    output.append("|---|---|---|---|---|---|")
    for m, acc, iters, sunk, t, n in single:
        output.append(f"| {m} | {acc:.3f} | {iters:.1f} | {sunk:.1f} | {t:.0f} | {n} |")
        
    output.append("")
    
    # Dual
    dual = process_exp2(ds)
    output.append("### Dual-Agent (Exp 2)")
    output.append("| Reviewer + Coder | Accuracy | Total Iters | Sunk Cost Episodes | Time (s) | Runs |")
    output.append("|---|---|---|---|---|---|")
    for p, acc, iters, sunk, t, n in dual:
        output.append(f"| {p} | {acc:.3f} | {iters:.1f} | {sunk:.1f} | {t:.0f} | {n} |")
        
    output.append("\n---\n")

with open(r"C:\Users\ZENITH\.gemini\antigravity\brain\06b86e6a-e82b-4e0f-80a2-24da4771d50b\scratch\paper3_tables.md", "w") as f:
    f.write("\n".join(output))

print("Aggregated tables written to paper3_tables.md")
