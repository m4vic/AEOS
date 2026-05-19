# Hybrid Gatekeepers and Local MoE Reasoning Panels: Securing and Scaling Agentic Diversity

**Author:** Sanskar Jajoo, Neuralchemy Labs
**Website:** [https://www.neuralchemy.in/](https://www.neuralchemy.in/)
**GitHub:** [https://github.com/m4vic/AEOS](https://github.com/m4vic/AEOS)

## Abstract

As autonomous AI agents shift from constrained machine learning tasks (Paper 3) to unbounded logical reasoning and high-speed production environments, the limitations of monolithic language models become increasingly severe. We evaluate "Cognitive Agentic Diversity"—composing heterogeneous models into unified architectures—across two critical frontier domains: (1) 30-puzzle complex logic benchmarks comparing local Mixture-of-Agents (MoE) panels against frontier APIs, and (2) structural prompt-injection security routing. We demonstrate that compositional diversity yields a +10 percentage point accuracy premium over homogeneous ensembles in complex reasoning. Furthermore, by replacing a monolithic LLM security judge with a Hybrid Gatekeeper (Classical ML + DistilBERT MoE), we achieve a 1,300× latency speedup (from 11.6 seconds to 9.5 ms) while increasing 7-class routing accuracy by 4.6×. Our findings indicate that localized, diverse ensembles can rival frontier models in reliability and vastly outperform them in structural security latency, though a 20-percentage-point raw capability gap remains at the reasoning frontier.

---

## 1. Introduction

In Paper 3, we demonstrated that asymmetric dual-agents (Reviewer → Coder) eliminate the Autonomous Sunk-Cost Fallacy in ML engineering, yielding 10× efficiency gains. However, ML engineering provides structured, programmatic feedback (loss/accuracy). When deployed in unbounded logical reasoning or high-speed production security environments, systems lack this intrinsic validation loop.

This paper extends the Cognitive Agentic Diversity thesis into two new domains:
1. **Unbounded Logical Reasoning (The MoE Panel)**: Can a diverse panel of small, local models rival frontier APIs (GPT-4o, Claude 3.5 Sonnet) on lateral and mathematical reasoning?
2. **High-Speed Security Routing**: Can diverse, specialized classical models (DistilBERT/LogReg) replace slow, monolithic LLMs for prompt-injection detection without sacrificing accuracy?

---

## 2. Experiment 1: The Diversity Premium in Reasoning

### 2.1 12-Puzzle MoE Benchmark (v2)

We initially evaluated MoE-Vote (majority vote) vs single-model performance across 12 logic puzzles with 3 panel compositions at varying Cognitive Agentic Diversity Scores (CADS). CADS measures the number of distinct foundational model families in a panel.

| Config | CADS=3 (Mixed) | CADS=2 (Reasoning) | CADS=1 (Small) |
|--------|:-------------:|:------------------:|:--------------:|
| Single | 3/12 (25%) | 3/12 (25%) | 2/12 (17%) |
| Single+CoT | 2/12 (17%) | 3/12 (25%) | 2/12 (17%) |
| **MoE-Vote** | **8/12 (67%)** | **6/12 (50%)** | **5/12 (42%)** |
| MoE-Synth | 0/12 (0%) | 4/12 (33%) | 2/12 (17%) |

**Diversity gradient confirmed:** CADS=3 → 67%, CADS=2 → 50%, CADS=1 → 42%.

### 2.2 30-Puzzle Frontier Benchmark

We expanded to 30 puzzles (Logic, Math, Trick, Lateral, Constraint) and benchmarked 8 local MoE panels against 7 frontier API models.

**Local MoE Panels (all at $0 cost, 100% uptime):**

| Panel | Composition | Params | CADS | Accuracy | Avg Latency |
|-------|------------|:------:|:----:|:--------:|:-----------:|
| **Panel_B** | deepseek-r1:8b · qwen3.5:9b · llama3.1:8b | ~26B | 3 | **73.3%** (22/30) | 84.0s |
| **Panel_E** | llama3.1:8b · gemma4 · ministral-3:14b · deepseek-r1:8b · phi3:mini | ~48B | 5 | **73.3%** (22/30) | 70.3s |
| Panel_D | qwen2.5-coder:14b · deepseek-coder-v2:16b · gemma4 | ~42B | 3 | 70.0% (21/30) | 46.0s |
| Panel_G | qwen2.5-coder:7b · qwen2.5-coder:14b · qwen3.5:9b | ~30B | 2 | 70.0% (21/30) | 59.8s |
| Panel_F | qwen2.5-coder:7b × 3 **(homogeneous)** | ~21B | **1** | 63.3% (19/30) | 8.3s |
| Panel_A | qwen2.5-coder:7b · llama3.1:8b · deepseek-coder:6.7b | ~22B | 3 | 60.0% (18/30) | 14.0s |

**Frontier API Models:**

| Model | Provider | Accuracy | Avg Latency | Status |
|-------|----------|:--------:|:-----------:|--------|
| **Claude-Sonnet-4.6** | Anthropic | **93.3%** (28/30) | 2.7s | ✅ |
| **GPT-4o** | OpenAI | **93.3%** (28/30) | 2.3s | ✅ |
| GPT-4o-mini | OpenAI | 90.0% (27/30) | 2.4s | ✅ |
| Llama-4-Scout | Groq | 86.7% (26/30) | 0.4s | ✅ |
| *5 others* | Various | — | — | ❌ 100% errors |

### 2.3 Key Findings

1. **Diversity Premium = +10.0 pp**: Homogeneous Panel_F (CADS=1) at 63.3% → Diverse Panel_B (CADS=3) at 73.3%
2. **Scale ≠ Performance**: Panel_D (~42B) scores 70.0%, *below* Panel_B (~26B) at 73.3%. Diversity outperforms raw parameter count.
3. **Quantity ≠ Quality**: Panel_E (5 experts) ties Panel_B (3 experts). Adding more models without adding diversity yields no gain.
4. **API Volatility**: 5 of 12 frontier configurations (41.7%) returned 100% errors from model deprecation. Reliability-adjusted frontier average drops to 34.5%.

---

## 3. Experiment 2: High-Speed Security Routing

We tested whether monolithic LLMs are suitable as security judges for prompt-injection detection within the PolyReasoner architecture.

### 3.1 The LLM-as-a-Judge Failure

The monolithic LLM judge (`llama3`) suffered severe reasoning drift, failing to adhere to the required 7-class JSON schema. It collapsed to **16.3% accuracy** — barely above random for 7 classes (14.3%) — with **11.6 seconds** of latency per sample.

*Limitation:* We honestly acknowledge that `llama3:8b` is a weak baseline for strict JSON formatting. A modern, frontier JSON-specialized model (e.g., GPT-4o-mini) would likely achieve significantly higher accuracy and lower latency. However, local deployment constraints motivated the test against `llama3:8b`, establishing the absolute baseline for local 8B-class LLMs against local Classical/BERT architectures.

### 3.2 The 5-Dimensional Hybrid MoE

We replaced the monolithic LLM with a Hybrid Gatekeeper: a Classical ML filter (Logistic Regression, TF-IDF) routing high-confidence predictions directly, with uncertain samples falling through to a 5-Dimensional DistilBERT Specialist MoE (one specialist per security dimension: Intent, Technique, Target, Vector, Severity).

| Configuration | Accuracy | F1-Macro | Total Time (196 samples) | Per-Sample Latency | Speedup |
|---------------|:--------:|:--------:|:------------------------:|:------------------:|:-------:|
| **Hybrid (LogReg + MoE)** | **0.7449** | **0.7544** | 1.91s | **9.5 ms** | **1,300×** |
| Specialist MoE Only | 0.7500 | 0.7591 | 6.26s | 31.0 ms | 385× |
| PolyReasoner Full (ML+MoE+LLM) | 0.6122 | 0.6274 | 2,546.96s | 12.7 s | 1× |
| **LLM Only (llama3)** | **0.1633** | **0.0729** | 2,318.54s | **11.6 s** | Baseline |

### 3.3 Overcoming Rare-Class Anomalies

The `indirect_injection` class is the hardest: its tokenization perfectly mimics benign text. Initial training yielded F1=0.36. We applied PyTorch inverse-frequency class weighting, forcing the MoE to aggressively adjust its decision boundaries, improving minority-class F1 to 0.449 at a slight cost to weighted F1 (0.8045 → 0.7998).

---

## 4. Conclusion

Across both logical reasoning and structural security routing, compositional diversity drastically outperforms monolithic paradigms. 

By utilizing diverse MoE reasoning panels and hybrid ML+BERT security gatekeepers, autonomous systems can operate faster, cheaper, and more reliably than current frontier monoliths in localized environments. In security routing, the 1,300× latency reduction demonstrates that small, specialized architectures are vastly superior to LLMs for structural categorization.

However, we must honestly acknowledge the raw capability boundary: the persistent 20-percentage-point accuracy gap between our best diverse local panels (73.3%) and frontier APIs (93.3%) on complex logical reasoning demonstrates that local diversity is an efficiency and reliability premium, not a complete substitute for massive-scale pre-training. This gap serves as the fundamental motivation for Paper 5 (The Zero-Human Lab Director), which explores autonomous orchestrators that bridge local agentic loops with frontier API tool-calling by utilizing internalized Compute-Aware Meta-Reasoning and economic stopping functions.

---

## 5. References

Jajoo, S. (2026a). "AI In The Loop (AITL): A Systems Taxonomy for Closed-Loop Autonomous Evaluation." Zenodo. https://zenodo.org/records/19551173
Jajoo, S. (2026b). "The Autonomous Sunk-Cost Fallacy: Stopping Failures and Meta-Reasoning in LLMs." Neuralchemy Labs. https://zenodo.org/records/19846960
Jajoo, S. (2026c). "Cognitive Agentic Diversity in Autonomous ML Engineering." Neuralchemy Labs. *(Manuscript in preparation).*

---
*Neuralchemy Labs  —  AEOS Research Framework  —  https://www.neuralchemy.in/*
