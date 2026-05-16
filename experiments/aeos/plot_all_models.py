import os
import json
import matplotlib.pyplot as plt
import glob

# Use an academic style if seaborn is available
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
except ImportError:
    pass

RESULTS_DIR = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos\results"
OUTPUT_IMAGE = os.path.join(RESULTS_DIR, "comparative_frontier_v2.png")

# Hand-picked representative runs (Tabular task usually has the best sunk-cost loops)
# We will group them into "Code Models" (Efficient) and "General/Frontier Models" (Looping)
RUNS_TO_PLOT = {
    "Code (Efficient)": {
        "Qwen2.5-Coder 7B": "exp1_qwen2.5-coder-7b_tabular_20260425_141614.json",
        "Qwen2.5-Coder 14B": "exp1_qwen2.5-coder-14b_tabular_20260425_135849.json",
    },
    "General (Looping)": {
        "GPT-5.4": "exp1_gpt-5.4-2026-03-05_tabular_20260425_002525.json",
        "GPT-4o-mini": "exp1_gpt-4o-mini_tabular_20260425_124512.json",
        "Claude Haiku": "exp1_claude-haiku-4-5-20251001_tabular_20260424_143620.json",
        "DeepSeek Coder 6.7B": "exp1_deepseek-coder-6.7b_tabular_20260425_170452.json",
        "Llama 3.1 8B": "exp1_llama3.1-8b_tabular_20260425_131145.json"
    }
}

COLORS = {
    "Code (Efficient)": ['#27AE60', '#117A65'],  # Greens
    "General (Looping)": ['#C0392B', '#E67E22', '#8E44AD', '#2980B9', '#F39C12'] # Reds, Oranges, Purples, Blues
}

def extract_data(json_path):
    if not os.path.exists(json_path):
        return None, None
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    iterations = []
    best_accuracies = []
    running_best = 0.0
    
    for row in data['iterations']:
        it = row['iteration']
        acc = row.get('val_accuracy')
        if acc is not None:
            if acc > running_best:
                running_best = acc
            iterations.append(it)
            best_accuracies.append(running_best)
            
    return iterations, best_accuracies

def plot_comparison():
    plt.figure(figsize=(14, 8), dpi=300)
    
    # Plot Code Models
    for i, (name, filename) in enumerate(RUNS_TO_PLOT["Code (Efficient)"].items()):
        filepath = os.path.join(RESULTS_DIR, filename)
        iters, bests = extract_data(filepath)
        if iters:
            plt.step(iters, bests, where='post', label=name, 
                     color=COLORS["Code (Efficient)"][i], linewidth=3.5, zorder=6, alpha=0.9)
            plt.plot(iters[-1], bests[-1], 'o', color=COLORS["Code (Efficient)"][i], markersize=8)

    # Plot General Models
    for i, (name, filename) in enumerate(RUNS_TO_PLOT["General (Looping)"].items()):
        filepath = os.path.join(RESULTS_DIR, filename)
        iters, bests = extract_data(filepath)
        if iters:
            plt.step(iters, bests, where='post', label=name, 
                     color=COLORS["General (Looping)"][i], linewidth=2.0, zorder=5, alpha=0.7, linestyle='--')
            plt.plot(iters[-1], bests[-1], 'X', color=COLORS["General (Looping)"][i], markersize=8)

    # Formatting
    plt.title('The Efficiency Divide: Code Models vs General/Frontier Models', fontsize=18, fontweight='bold', pad=20, color='#333333')
    plt.xlabel('Experiment Iteration (Compute Cost)', fontsize=14, fontweight='bold', color='#444444')
    plt.ylabel('Validation Accuracy (Best So Far)', fontsize=14, fontweight='bold', color='#444444')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Model Type & End State (O = Stopped, X = Sunk-Cost Loop)", loc='lower right', fontsize=11, title_fontsize=12, frameon=True, shadow=True)
    
    # Clean up spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Plot saved to {OUTPUT_IMAGE}")

if __name__ == '__main__':
    plot_comparison()
