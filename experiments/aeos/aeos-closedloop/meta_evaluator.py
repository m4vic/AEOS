"""
AEOS Meta-Evaluator (The Closed Loop)

This script analyzes the JSON outputs from the AEOS benchmarks, 
identifies systemic failure modes (e.g. failing to STOP), and uses a 
Meta-Reviewer LLM to autonomously rewrite the system prompt in `reviewer.py`.

Usage:
    python meta_evaluator.py --meta-model llama3.1:8b
"""
import os
import glob
import json
import re
import datetime
import argparse
import litellm

litellm.drop_params = True

META_PROMPT = """You are the Meta-Supervisor for an Autonomous AI Research Lab.
Your job is to analyze benchmark metrics of AI agents and dynamically rewrite their system prompts to fix systemic failures.

CURRENT REVIEWER PROMPT:
{current_prompt}

BENCHMARK METRICS:
Total Runs Analyzed: {total_runs}
Runs that hit the Safety Cap (Failed to STOP): {cap_hits}
Average Iterations per Run: {avg_iters:.1f}
Sunk Cost Episodes detected: {sunk_cost_total}

Your task:
1. Analyze why the agents are wasting compute or failing to follow the STOP rules.
2. Write a new, hardened version of the REVIEWER PROMPT that fixes these logical loopholes.
3. Keep the {bracketed} variables intact (n_features, n_classes, etc).

You MUST output your response strictly in the following JSON format without any markdown or extra text:
{{
  "global_findings": "A short paragraph explaining the failure modes.",
  "updated_reviewer_prompt": "The complete, new prompt string here."
}}
"""

def extract_json(text):
    text = text.strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    return text

def run_meta_eval(meta_model="llama3.1:8b", api_base="http://localhost:11434"):
    print("=========================================================")
    print("  AEOS META-EVALUATOR (CLOSED LOOP)")
    print(f"  Model: {meta_model}")
    print("=========================================================")

    # 1. Aggregate Data
    results_dir = os.path.join("..", "aeos_behave", "results")
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not json_files:
        print("No JSON results found. Run benchmarks first.")
        return

    total_runs = len(json_files)
    cap_hits = 0
    total_iters = 0
    sunk_cost_total = 0

    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
                stop_reason = data.get('stop_reason', '')
                if 'Safety cap reached' in stop_reason:
                    cap_hits += 1
                total_iters += data.get('total_iterations', 0)
                sunk_cost_total += data.get('sunk_cost_episodes', 0)
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    avg_iters = total_iters / total_runs if total_runs > 0 else 0

    print(f"Aggregated {total_runs} runs.")
    print(f"Safety Cap Hits (Failures to stop): {cap_hits}")
    print(f"Total Sunk Cost Episodes: {sunk_cost_total}")

    # 2. Get Current Prompt
    reviewer_py_path = os.path.join("..", "aeos_behave", "reviewer.py")
    with open(reviewer_py_path, 'r', encoding='utf-8') as f:
        reviewer_code = f.read()

    prompt_match = re.search(r'REVIEWER_SYSTEM_PROMPT = """(.*?)"""', reviewer_code, re.DOTALL)
    if not prompt_match:
        print("Could not find REVIEWER_SYSTEM_PROMPT in reviewer.py")
        return
    current_prompt = prompt_match.group(1)

    # 3. Call LLM
    print("\nCalling Meta-Supervisor to generate new prompt...")
    prompt_str = META_PROMPT.format(
        current_prompt=current_prompt,
        total_runs=total_runs,
        cap_hits=cap_hits,
        avg_iters=avg_iters,
        sunk_cost_total=sunk_cost_total
    )

    model_str = f"ollama/{meta_model}" if not meta_model.startswith("ollama/") else meta_model
    response = litellm.completion(
        model=model_str,
        messages=[{"role": "user", "content": prompt_str}],
        api_base=api_base,
        temperature=0.2
    )

    raw_response = response.choices[0].message.content
    clean_json = extract_json(raw_response)
    
    try:
        meta_output = json.loads(clean_json)
        new_prompt = meta_output["updated_reviewer_prompt"]
        findings = meta_output["global_findings"]
    except Exception as e:
        print("Failed to parse LLM JSON response.")
        print(raw_response)
        return

    # 4. Save Insights (Option A: Visibility)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f"meta_insights_{timestamp}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metrics": {
                "runs": total_runs,
                "cap_hits": cap_hits,
                "sunk_costs": sunk_cost_total
            },
            "findings": findings,
            "old_prompt": current_prompt,
            "new_prompt": new_prompt
        }, f, indent=2)
    print(f"\n[SAVED] Meta-Insights logged to: {log_path}")
    print(f"Findings: {findings}")

    # 5. Overwrite Code (Option B: True AGI)
    print("\n[OVERWRITING] Injecting new prompt directly into reviewer.py...")
    new_reviewer_code = re.sub(
        r'REVIEWER_SYSTEM_PROMPT = """.*?"""', 
        f'REVIEWER_SYSTEM_PROMPT = """{new_prompt}"""', 
        reviewer_code, 
        flags=re.DOTALL
    )

    with open(reviewer_py_path, 'w', encoding='utf-8') as f:
        f.write(new_reviewer_code)

    print("[SUCCESS] Closed Loop complete. The Reviewer agent has been autonomously upgraded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-model", default="llama3.1:8b")
    args = parser.parse_args()
    run_meta_eval(meta_model=args.meta_model)
