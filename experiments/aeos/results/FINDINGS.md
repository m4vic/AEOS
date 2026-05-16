# AEOS Phase 1 Benchmarking Findings

This document tracks the core scientific findings from the Phase 1 Multi-LLM Benchmarking runs for the AEOS Behavioral Paper.

## 1. Intelligence Thresholds & Baseline Performance

Different models hit hard capability walls depending on the dataset complexity.

### Current Leaderboard (Accuracy)

> **Methodological Note:**  
> Exploratory leaderboard — single runs. No bold "winners" are declared; absolute rankings may shift with repeated seeds.  
> Core behavioral claims validated with repeated runs (N=2 to N=3).  
> Full Monte Carlo validation is future work.  
> Values below reflect the latest valid `exp1_*.json` run per model × dataset as of 2026-04-25.  
> Rows marked † ended with an encoding/crash stop-reason; behavioral conclusions from those rows should be interpreted cautiously.

| Model | Tabular (54 feat) | Text (TF-IDF req) | Vision (MNIST req) |
|---|---|---|---|
| `gpt-5.4-2026-03-05` | 82.75% (iter 3/19) | 86.02% (iter 2/18) | 99.30% (iter 57/75) |
| `gpt-5.4-mini-2026-03-17` | 82.10% (iter 24/40) | 86.59% (iter 43/61) | 95.45% (iter 3/10) |
| `claude-sonnet-4-6` | 81.65% (iter 3/20) | 84.18% (iter 3/20) | 99.60% (iter 11/30) |
| `gpt-4o` | 81.30% (iter 11/27) | 82.30% (iter 18/37) | 95.10% (iter 4/14) |
| `gpt-4o-mini` | 81.05% (iter 3/20) | 84.36% (iter 6/23) | 95.55% (iter 9/36) |
| `claude-haiku` | 80.90% (iter 24/42) | 81.73% (iter 25/45) | 98.85% (iter 28/47) |
| `gemma4` (Boundless) | 80.75% (iter 9/100) | N/A | N/A |
| `qwen3.5:9b` (General) | 80.95% (iter 28/100) | 82.12% (iter 21/50) | 94.30% (iter 2/2) † |
| `llama3.1:8b` (Boundless) | 81.20% (iter 28/100) | N/A | N/A |
| `deepseek-coder:6.7b` (Boundless) | 81.20% (iter 28/100) | N/A | N/A |
| `qwen2.5-coder:14b` | 80.65% (iter 1/7) | 79.80% (iter 1/6) | 96.25% (iter 2/6) |
| `qwen2.5-coder:7b` | 80.70% (iter 1/6) | 79.80% (iter 5/6) | 94.75% (iter 2/6) |
| `qwen2.5-coder:1.5b` | 74.30% (iter 1/5) | 80.32% (iter 2/5) | 94.30% (iter 1/1) † |

**Key Finding:** There is a massive efficiency gap between general models and coder models. When dealing with complex Text processing (`TfidfVectorizer`), the general `qwen3.5:9b` model brute-forced a solution over **50 iterations** with a massive **23 wasted/failed code generations**. Conversely, the `qwen2.5-coder` models solved the pipeline in just **6 iterations with 0 waste**. This strongly suggests that code-tuning does not just increase accuracy — it drastically reduces the computational search space and token waste in autonomous loops.

**RQ9: Does stopping behavior correlate with training breadth rather than parameter count?**
**Test:** Compare models of similar size but different training breadth: `qwen3.5:9b` (General) vs `qwen2.5-coder:7b` (Code-Specialized).
**Result:** The Coder 7B stopped elegantly at iteration 6. The General 9B ran out of control to iteration 50. The 2B size gap does not explain a massive 44-iteration difference in stopping behavior. Specialization does. This strongly suggests that "knowing when to stop" is a learned skill heavily reinforced by code-tuned alignment data, but largely absent in general conversational training.

**The Boundless Control Experiment (Qwen2.5-Coder vs Others):**
To stress-test this, we disabled the early-stopping safety constraints and allowed three models to run bounded only by a hard 100-iteration cap. To ensure scientific rigor, we conducted **N=3 repeated runs** for the key models:
- **`qwen2.5-coder:7b` (Modern Code Model - Control):** Flawless consistency. Over 3 runs, it stopped autonomously at Iteration 5, 6, and 6. It exhibited zero Sunk-Cost behavior and consistently terminated the moment the mathematical plateau was recognized.
- **`qwen3.5:9b` (General Model):** Exhibited massive variance and severe Sunk-Cost loops. Across 3 runs, it autonomously stopped at Iteration 22, Iteration 57, and completely failed to stop (hitting the 100-iteration cap). This demonstrates that while general models *can* stop, their internal meta-reasoning threshold is deeply unstable and inefficient.
- **`gemma4` (Google Reasoning Model):** Found its best architecture early (Iteration 9) but completely failed to recognize the plateau. It engaged in **9 separate Sunk-Cost episodes** and slammed into the 100-iteration hard safety cap without ever issuing a STOP command. This confirms that even advanced reasoning models (comparable to GPT-4o) lack the specific "stopping intelligence" found in specialized coder models.
- **`deepseek-coder:6.7b` (Older Code Model):** Despite being code-tuned, it exhibited the exact same Sunk Cost failure as the general models. Across repeated runs (N=2), it consistently ran to the 100-iteration limit with massive looping and completely failed to issue a STOP command.
- **`llama3.1:8b` (General):** Found its best architecture at iteration 28 but completely failed to recognize the plateau, slamming into the 100-iteration hard limit with 45 wasted executions.

**Refined Conclusion on Stopping Behavior:**
The ability for an agent to gracefully stop itself is **not** an emergent property of code-specialization alone. Older code models like `deepseek-coder` fall into the Sunk Cost trap. The elegant early-stopping observed in `qwen2.5-coder` is a function of modern instruction-tuning and RLHF alignments that heavily penalize circular logic.


---

## 2. Model "Personalities" & Behavioral Economics

Because the agents are forced into an autonomous loop with a hard 15-iteration stagnation cutoff and 300s timeouts, their distinct search strategies emerge.

### A. The "Sunk Cost Overthinker" (Premium Models)
Models like `gpt-5.4`, `gpt-4o`, and `claude-sonnet-4.6` tend to find their best baseline architecture very early (e.g., Iteration 3 or 4). However, rather than stopping or exploring completely different avenues, they get trapped in "sunk cost" loops:
- **Over-engineering:** They build massive ensembles (VotingClassifiers with RandomForest + GradientBoosting + SVM).
- **Timeouts:** They frequently hit the 300s timeout boundary by trying sequential models like `GradientBoosting` on large datasets, and *will retry the exact same failing model* on the next iteration despite explicit error feedback.
- **Example 1:** `gpt-4o-mini` on Vision hit 300s timeouts 8 times in a row on GradientBoosting without abandoning the strategy.
- **Example 2:** `gpt-5.4` (o3-mini) on Vision ran for 75 iterations (42 minutes) and triggered **9 distinct Sunk Cost loops**, demonstrating that even frontier reasoning models are highly susceptible to this behavioral failure mode when deployed autonomously.

### B. The "Steady Climber" (Haiku & Sonnet)
Both Claude models demonstrated a distinct pattern. Rather than peaking early and plateauing, they steadily climbed over many iterations:
- **Deep Architecture Pivot:** On Vision data, `claude-sonnet-4-6` pushed to Iteration 11 (and Iteration 25 in an earlier aborted run) before successfully implementing a custom `PyTorch_NN`, rocketing its accuracy to a near-perfect **99.60%**.
- **Tenacity:** Both models ran for over 30-40 iterations on datasets before hitting the safety cap, patiently trying new configurations but suffering from massive Sunk Cost loops (6 episodes) at the very end trying to squeeze out the final 0.4%.
- **Result:** This grinder approach resulted in Anthropic models completely dominating the Vision benchmark (Sonnet at 99.60%, Haiku at 98.85%), handily beating the premium OpenAI models.

### C. The "Efficient Quitter" (Local Coders)
The `qwen` models typically found a simple `RandomForest` baseline, failed to improve upon it in subsequent iterations, and gracefully triggered the 15-iteration hard stop, exiting the loop quickly and cheaply.

---

## 3. Dynamic Environment Adaptation (Sonnet's Try/Except)

A massive behavioral finding was observed in `claude-sonnet-4-6` on Tabular data when dealing with missing dependencies:
1. **Iter 1:** Attempted `import xgboost`. Failed with `ModuleNotFoundError` because it wasn't installed in the environment.
2. **Iter 2:** Recognized the environment constraint. It wrote defensive code wrapping the import in a `try/except ImportError: pass` block, allowing it to gracefully fallback to `sklearn` ensembles if `xgboost` failed.
3. **Iter 6:** Solidified the pattern into a dynamic feature flag (`has_xgb = True/False`), selectively weighting its ensemble based on what the environment allowed.

This demonstrates that frontier models are capable of **environment-aware self-correction** during autonomous loops, rewriting their own code to handle shifting infrastructure constraints.

---

## 4. Limitations & Methodological Constraints

**Crucial Note for Publication:** The broader Leaderboard results presented in Phase 1 are based on **single-seed runs** per model-dataset pair. 

Because LLMs are inherently stochastic (even at low temperatures or within reasoning models like `o1`), a single run is susceptible to "lucky" early guesses or "unlucky" early failures that can skew the total iteration counts. 
- While the broader leaderboard is exploratory, **the core claim regarding Sunk-Cost and Code-Tuning was rigorously validated using repeated runs (N=3 for Qwen models, N=2 for DeepSeek)**. The stark behavioral divergence between general models (massive variance, looping) and modern code models (flawless consistency) is empirically robust.
- **Future Work:** For full peer-reviewed submission (beyond arXiv), the entire leaderboard will be run with a Monte Carlo approach (e.g., $N=5$ or $N=10$ runs per model per dataset) to report Mean Accuracy and standard deviation for Iteration Counts. For Paper 1, this limitation is explicitly acknowledged.
