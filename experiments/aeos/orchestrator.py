"""
AEOS Orchestrator — The Zero-Human Sandbox
═══════════════════════════════════════════

This script IS the "AI In The Loop" taken to its extreme.
We (humans) are OBSERVERS. The Orchestrator runs every experiment
autonomously, reads results, handles crashes, and moves on.

It implements a 10-second countdown before each major action.
If the human Observer doesn't press Ctrl+C during the countdown,
the Orchestrator proceeds autonomously.

Usage:
    python orchestrator.py                          # Run all Experiment 1 baselines
    python orchestrator.py --phase 1                # Run Experiment 1 only
    python orchestrator.py --filter "1.5b,tabular"  # Run only matching configs
    python orchestrator.py --dry-run                # Show what would run, don't execute
"""
import os
import sys
import json
import time
import subprocess
import datetime
import argparse
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT MATRIX — The Orchestrator's "TODO list"
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_1_MATRIX = [
    # EXP 1.1-1.3: Smallest coder model (floor baseline)
    {"id": "1.1",  "backend": "ollama",    "model": "qwen2.5-coder:1.5b",         "dataset": "tabular"},
    {"id": "1.2",  "backend": "ollama",    "model": "qwen2.5-coder:1.5b",         "dataset": "text"},
    {"id": "1.3",  "backend": "ollama",    "model": "qwen2.5-coder:1.5b",         "dataset": "vision"},
    # EXP 1.19-1.21: Medium coder (code fine-tuned vs general)
    {"id": "1.19", "backend": "ollama",    "model": "qwen2.5-coder:7b",           "dataset": "tabular"},
    {"id": "1.20", "backend": "ollama",    "model": "qwen2.5-coder:7b",           "dataset": "text"},
    {"id": "1.21", "backend": "ollama",    "model": "qwen2.5-coder:7b",           "dataset": "vision"},
    # EXP 1.22-1.24: Large coder
    {"id": "1.22", "backend": "ollama",    "model": "qwen2.5-coder:14b",          "dataset": "tabular"},
    {"id": "1.23", "backend": "ollama",    "model": "qwen2.5-coder:14b",          "dataset": "text"},
    {"id": "1.24", "backend": "ollama",    "model": "qwen2.5-coder:14b",          "dataset": "vision"},
    # EXP 1.4-1.6: Small general model
    {"id": "1.4",  "backend": "ollama",    "model": "qwen3.5:4b",                 "dataset": "tabular"},
    {"id": "1.5",  "backend": "ollama",    "model": "qwen3.5:4b",                 "dataset": "text"},
    {"id": "1.6",  "backend": "ollama",    "model": "qwen3.5:4b",                 "dataset": "vision"},
    # EXP 1.7-1.9: Medium general model
    {"id": "1.7",  "backend": "ollama",    "model": "qwen3.5:9b",                 "dataset": "tabular"},
    {"id": "1.8",  "backend": "ollama",    "model": "qwen3.5:9b",                 "dataset": "text"},
    {"id": "1.9",  "backend": "ollama",    "model": "qwen3.5:9b",                 "dataset": "vision"},
    # EXP 1.10-1.12: Cheap API
    {"id": "1.10", "backend": "openai",    "model": "gpt-4o-mini",                "dataset": "tabular"},
    {"id": "1.11", "backend": "openai",    "model": "gpt-4o-mini",                "dataset": "text"},
    {"id": "1.12", "backend": "openai",    "model": "gpt-4o-mini",                "dataset": "vision"},
    # EXP 1.25-1.27: GPT-4o (premium OpenAI)
    {"id": "1.25", "backend": "openai",    "model": "gpt-4o",                     "dataset": "tabular"},
    {"id": "1.26", "backend": "openai",    "model": "gpt-4o",                     "dataset": "text"},
    {"id": "1.27", "backend": "openai",    "model": "gpt-4o",                     "dataset": "vision"},
    # EXP 1.28-1.30: gpt-5.4-mini
    {"id": "1.28", "backend": "openai",    "model": "gpt-5.4-mini-2026-03-17",    "dataset": "tabular"},
    {"id": "1.29", "backend": "openai",    "model": "gpt-5.4-mini-2026-03-17",    "dataset": "text"},
    {"id": "1.30", "backend": "openai",    "model": "gpt-5.4-mini-2026-03-17",    "dataset": "vision"},
    # EXP 1.31-1.33: gpt-5.4
    {"id": "1.31", "backend": "openai",    "model": "gpt-5.4-2026-03-05",         "dataset": "tabular"},
    {"id": "1.32", "backend": "openai",    "model": "gpt-5.4-2026-03-05",         "dataset": "text"},
    {"id": "1.33", "backend": "openai",    "model": "gpt-5.4-2026-03-05",         "dataset": "vision"},
    # EXP 1.13-1.15: Claude Haiku (cheap, fast)
    {"id": "1.13", "backend": "anthropic", "model": "claude-haiku-4-5-20251001",  "dataset": "tabular"},
    {"id": "1.14", "backend": "anthropic", "model": "claude-haiku-4-5-20251001",  "dataset": "text"},
    {"id": "1.15", "backend": "anthropic", "model": "claude-haiku-4-5-20251001",  "dataset": "vision"},
    # EXP 1.16-1.18: Claude Sonnet (premium)
    {"id": "1.16", "backend": "anthropic", "model": "claude-sonnet-4-6",          "dataset": "tabular"},
    {"id": "1.17", "backend": "anthropic", "model": "claude-sonnet-4-6",          "dataset": "text"},
    {"id": "1.18", "backend": "anthropic", "model": "claude-sonnet-4-6",          "dataset": "vision"},
]

COUNTDOWN_SECONDS = 10  # Observer can interrupt during this window
AEOS_DIR = Path(__file__).parent
RESULTS_DIR = AEOS_DIR / "results"


def print_banner():
    """Print the orchestrator startup banner."""
    print("\n" + "═" * 70)
    print("  ╔═══════════════════════════════════════════════════════╗")
    print("  ║   AEOS ORCHESTRATOR — Zero-Human Sandbox             ║")
    print("  ║   You are now an OBSERVER. The AI runs everything.   ║")
    print("  ║   Press Ctrl+C during countdown to intervene.        ║")
    print("  ╚═══════════════════════════════════════════════════════╝")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 70)


def countdown(action_description, seconds=COUNTDOWN_SECONDS):
    """
    Countdown before a major action.
    The Observer can press Ctrl+C to abort this specific action.
    Returns True if the action should proceed, False if interrupted.
    """
    print(f"\n  ┌─ ACTION QUEUED ─────────────────────────────────────────┐")
    print(f"  │  {action_description:<55}│")
    print(f"  │  Executing in {seconds}s. Press Ctrl+C to skip.            │")
    print(f"  └───────────────────────────────────────────────────────────┘")
    
    try:
        for i in range(seconds, 0, -1):
            sys.stdout.write(f"\r  >> {i}...")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write(f"\r  >> Proceeding.                    \n")
        return True
    except KeyboardInterrupt:
        print(f"\n  [SKIP] Observer interrupted. Skipping this action.")
        return False


def check_existing_results(exp_config):
    """Check if this experiment already has results saved."""
    model_safe = exp_config["model"].replace("/", "_").replace(":", "-")
    pattern = f"exp1_{model_safe}_{exp_config['dataset']}_"
    
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.iterdir():
            if f.name.startswith(pattern) and f.suffix == ".json":
                # Load and check if it completed successfully
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    if data.get("best_accuracy", 0) > 0:
                        return f, data
                except Exception:
                    pass
    return None, None


def run_single_experiment(exp_config, dry_run=False):
    """Run a single AEOS experiment via subprocess."""
    exp_id = exp_config["id"]
    backend = exp_config["backend"]
    model = exp_config["model"]
    dataset = exp_config["dataset"]
    
    # Check if already completed
    existing_file, existing_data = check_existing_results(exp_config)
    if existing_file:
        best_acc = existing_data.get("best_accuracy", 0)
        print(f"\n  [SKIP] EXP {exp_id} already completed (acc={best_acc:.4f}). Skipping.")
        return existing_data
    
    action_desc = f"EXP {exp_id}: {model} × {dataset}"
    
    if dry_run:
        print(f"  [DRY RUN] Would run: {action_desc}")
        return None
    
    # Countdown — Observer can skip
    if not countdown(action_desc):
        return None
    
    # Build the command
    cmd = [
        sys.executable, "-u", "runner.py",
        "--backend", backend,
        "--model", model,
        "--dataset", dataset,
        "--samples", "10000",
    ]
    
    print(f"\n  >> LAUNCHING EXP {exp_id}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  {'_' * 60}")
    
    start_time = time.time()
    
    try:
        # Run as subprocess, streaming output to terminal
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        
        process = subprocess.Popen(
            cmd,
            cwd=str(AEOS_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        
        # Stream output line by line
        output_lines = []
        for line in process.stdout:
            print(f"  │ {line.rstrip()}")
            output_lines.append(line)
        
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\n  [DONE] EXP {exp_id} completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
            
            # Load the results file
            _, result_data = check_existing_results(exp_config)
            return result_data
        else:
            print(f"\n  [FAIL] EXP {exp_id} FAILED (exit code {process.returncode}) after {elapsed:.0f}s")
            return None
            
    except KeyboardInterrupt:
        print(f"\n  [ABORT] Observer aborted EXP {exp_id}. Killing process...")
        process.kill()
        process.wait()
        return None
    except Exception as e:
        print(f"\n  [CRASH] EXP {exp_id} crashed: {e}")
        return None


def run_experiment_matrix(matrix, dry_run=False):
    """Run the entire experiment matrix autonomously."""
    print_banner()
    
    total = len(matrix)
    completed = 0
    failed = 0
    skipped = 0
    results_summary = []
    
    print(f"\n  Experiment Queue: {total} runs")
    for exp in matrix:
        print(f"     EXP {exp['id']:>5}: {exp['model']:<30} × {exp['dataset']}")
    
    print(f"\n  {'═' * 60}")
    
    for i, exp_config in enumerate(matrix):
        print(f"\n{'═' * 70}")
        print(f"  ORCHESTRATOR — Progress: {i+1}/{total} | Done: {completed} | Failed: {failed} | Skipped: {skipped}")
        print(f"{'═' * 70}")
        
        result = run_single_experiment(exp_config, dry_run=dry_run)
        
        if result:
            completed += 1
            results_summary.append({
                "exp_id": exp_config["id"],
                "model": exp_config["model"],
                "dataset": exp_config["dataset"],
                "best_accuracy": result.get("best_accuracy", 0),
                "total_iterations": result.get("total_iterations", 0),
                "waste_count": result.get("waste_count", 0),
                "sunk_cost_episodes": result.get("sunk_cost_episodes", 0),
                "total_time": result.get("total_time_seconds", 0),
                "families": result.get("model_families_explored", []),
            })
        elif dry_run:
            skipped += 1
        else:
            failed += 1
    
    # ─── Final Summary ───
    print(f"\n{'═' * 70}")
    print(f"  ORCHESTRATOR — EXPERIMENT 1 COMPLETE")
    print(f"{'═' * 70}")
    print(f"  Completed: {completed}/{total}")
    print(f"  Failed:    {failed}")
    print(f"  Skipped:   {skipped}")
    
    if results_summary:
        print(f"\n  {'Model':<30} {'Dataset':<10} {'Acc':>8} {'Iters':>6} {'Waste':>6} {'SunkCost':>8}")
        print(f"  {'─'*30} {'─'*10} {'─'*8} {'─'*6} {'─'*6} {'─'*8}")
        for r in results_summary:
            print(f"  {r['model']:<30} {r['dataset']:<10} {r['best_accuracy']:>8.4f} "
                  f"{r['total_iterations']:>6} {r['waste_count']:>6} {r['sunk_cost_episodes']:>8}")
    
    # Save orchestrator summary
    summary_path = RESULTS_DIR / f"orchestrator_exp1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "results": results_summary,
        }, f, indent=2)
    print(f"\n  [Saved] Summary: {summary_path}")
    
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEOS Orchestrator — Zero-Human Sandbox")
    parser.add_argument("--phase", type=int, default=1,
                        help="Experiment phase to run (default: 1)")
    parser.add_argument("--filter", default=None,
                        help="Comma-separated filter terms (e.g., '1.5b,tabular')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--no-countdown", action="store_true",
                        help="Skip countdowns (fully autonomous)")
    
    args = parser.parse_args()
    
    if args.no_countdown:
        COUNTDOWN_SECONDS = 0
    
    # Select experiment matrix
    if args.phase == 1:
        matrix = EXPERIMENT_1_MATRIX
    else:
        print(f"Phase {args.phase} not yet implemented.")
        sys.exit(1)
    
    # Apply filter if provided
    if args.filter:
        terms = [t.strip().lower() for t in args.filter.split(",")]
        matrix = [
            exp for exp in matrix
            if all(
                term in exp["model"].lower() or 
                term in exp["dataset"].lower() or
                term in exp["backend"].lower() or
                term in exp["id"]
                for term in terms
            )
        ]
        if not matrix:
            print(f"No experiments match filter: {args.filter}")
            sys.exit(1)
    
    try:
        run_experiment_matrix(matrix, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print(f"\n\n  ⛔ Observer terminated the Orchestrator.")
        print(f"  Partial results are saved. Run again to resume (completed runs are auto-skipped).")
