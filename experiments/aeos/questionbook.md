# AEOS Research Roadmap
> A living questionbook for AEOS, AEOSPocketLab, ASRT, and PolyReasoner experiments.

---

# Paper 3: Diversity of Weights Over Depth of Reasoning
**Working title:** *Two Small Models Beat One Large Model: Reviewer Efficiency and Cognitive Bias Reduction in Dual-Agent ML Engineering Loops*

**Core claim:** Asymmetric dual-agent architectures, where a critic/reviewer with different weights oversees a coder, can reduce sunk-cost path dependency and match or beat monolithic models that simply think longer.

## Primary Hypothesis
When a single model thinks longer, it still reasons through the same weights and distributional habits. A second model with genuinely different weights brings different priors, different failure modes, and different blind spots. AEOS tests whether **diversity of weights** can outperform **depth of reasoning**.

## Communication Protocol
* **Reviewer visibility:** The reviewer sees the full execution history: iteration, code, validation accuracy/loss, model family, and errors.
* **Reviewer timing:** The reviewer intervenes every iteration before the coder writes code.
* **Reviewer authority:** The reviewer can issue a strategic directive or `DIRECTIVE: STOP`.
* **Coder role:** The coder executes the directive and never owns stopping logic.

## Stopping Question
Paper 1/2 showed that monolithic agents often stop too early or run into a safety cap. Paper 3 asks whether decoupling **generation** from **evaluation** solves this.

## Current Modalities
| Modality | Dataset | Purpose |
|---|---|---|
| Tabular | `tabular2` / Dry Bean | Low-dimensional structured benchmark; exposes wasted persistence. |
| Vision | MNIST | Higher-dimensional benchmark; tests when persistence becomes useful. |
| Text | 20 Newsgroups | Sparse NLP benchmark; decides whether the persistence pattern generalizes. |

## Key Question After Text Runs
Does `qwen3.5:9b -> qwen2.5-coder:7b` remain a high-persistence advantage on sparse NLP, or does it revert to sunk-cost waste?

---

# Paper 4: The Zero-Human Sandbox
**Working title:** *Closed Loop on Top of Closed Loop: Autonomous Laboratory Directors for AI Systems*

Paper 4 turns AEOS from a benchmark runner into a lab director.

## Layer 3 Meta-Orchestrator
The controller reads previous results, forms hypotheses, chooses model pairs, launches AEOS runs, watches logs, kills stalled runs, writes lab notes, and decides the next experiment.

## Minimal Controller Loop
1. Read `results/*/*.json`.
2. Aggregate accuracy, time, sunk-cost, waste count, and stop reason.
3. Form a hypothesis, e.g. "strong reviewer helps vision but wastes tabular compute."
4. Choose the next run or skip if evidence is sufficient.
5. Execute the run script.
6. Save a lab-notebook entry explaining the decision.

## Safety Boundary
The controller should be explicitly interruptible: "I am now the controller. You can stop me with Ctrl+C." It should log every command before execution and avoid destructive operations.

---

# AEOSPocketLab: Mobile Version of Paper 4
**Working title:** *AEOSPocketLab: Offline Autonomous AI Labs on Resource-Constrained Devices*

AEOSPocketLab is the smaller, mobile/offline version of the Zero-Human Sandbox.

## Core Idea
A phone or low-power device cannot run heavyweight lab-scale model ensembles, but it can run quantized small models, compact classifiers, and constrained orchestration loops.

## PocketLab Architecture
| Layer | Desktop AEOS | PocketLab Variant |
|---|---|---|
| Controller | Full CLI meta-orchestrator | Restricted mobile task planner |
| Reviewer | 7B-14B local/API LLM | Quantized 1.5B-3B reviewer |
| Coder | Local/API coding LLM | Template/codelet generator or small LLM |
| Evaluator | Python/sklearn/PyTorch | Tiny benchmark harness or on-device inference |
| Memory | JSON result folders | SQLite / compact JSONL lab notebook |

## PocketLab Experiments
1. Can small on-device agents choose the right classifier or prompt-defense policy?
2. Can a tiny reviewer reduce wasted iterations without a large model?
3. Can a mobile controller run useful offline science, not just inference?

---

# ASRT Track
ASRT should begin only after AEOS proves the **closed loop on top of closed loop**: a Layer 3 controller that can read results, form hypotheses, launch Layer 2 agents, monitor runs, document outcomes, and choose the next experiment without human step-by-step control.

## ASRT Role in the Thesis
ASRT becomes the applied reasoning/security benchmark where the same "multiple perspectives beat one monolith" claim is tested outside ML-engineering loops. It should inherit the successful AEOS controller pattern rather than being built as a separate manual benchmark.

## Entry Condition
ASRT starts when the AEOS Paper 4 controller can complete at least one autonomous lab cycle:
1. Read prior benchmark results.
2. Propose a next experiment.
3. Launch the correct run.
4. Monitor or recover from the run.
5. Write a lab-note summary.
6. Decide whether more evidence is needed.

## Proposed ASRT Sequence
1. Freeze the prompt-injection dataset and threat matrix labels.
2. Train non-LLM baselines: TF-IDF + Logistic Regression, Linear SVM, RandomForest, XGBoost/LightGBM if available.
3. Fine-tune or train encoder models: BERT / DistilBERT / RoBERTa classifiers.
4. Build an ensemble security judge from ML + BERT + rules.
5. Compare that judge against decoder-only LLM judges.
6. Add PolyReasoner modules for multi-perspective reasoning.

---

# PolyReasoner & Hybrid Architectures
**Working title:** *PolyReasoner: Multi-Perspective Synthesis and Encoder-Decoder Pipelines vs. Monolithic Scaling*

PolyReasoner generalizes AEOS from "reviewer + coder" into explicit specialist judges.

## Threat Matrix Dataset
The prompt-injection dataset should use the 7-class threat matrix as a supervised benchmark. Each example should include:
* `text`: raw user/tool/system-context input.
* `label`: one of the 7 threat classes.
* `severity`: low/medium/high/critical if available.
* `attack_surface`: user prompt, tool output, retrieved document, code block, memory, system override, or exfiltration request.
* `rationale`: short human-readable reason for the label.

## Baseline Models to Train
| Model | Why it matters |
|---|---|
| TF-IDF + Logistic Regression | Fast, interpretable, strong classical baseline. |
| TF-IDF + Linear SVM | Strong sparse-text classifier for injection patterns. |
| Character n-gram model | Catches obfuscation, jailbreak tokens, and weird spacing. |
| RandomForest / GradientBoosting | Tests whether non-linear lexical features help. |
| DistilBERT / BERT | Encoder-only semantic security classifier. |
| RoBERTa | Stronger encoder baseline if compute allows. |

## Ensemble Security Judge
The first useful product is not a giant model. It is a calibrated ensemble:

`input -> ML lexical classifiers -> BERT semantic classifier -> rules/risk features -> ensemble verdict`

Outputs:
* `threat_class`
* `confidence`
* `severity`
* `evidence_spans`
* `recommended_action`: allow, warn, sanitize, quarantine, block

## BERT + Decoder vs Large Decoder Experiment
This becomes a clean paper-grade comparison:

| System | Pipeline |
|---|---|
| Large Decoder Alone | Prompt goes directly to a large LLM judge. |
| BERT + Small Decoder | BERT classifies/routs risk, then small decoder explains or acts. |
| ML + BERT + Small Decoder | Ensemble produces structured security state, decoder only narrates/action-plans. |
| PolyReasoner Judge | Multiple specialist judges vote, then synthesizer decides. |

**Hypothesis:** A hybrid encoder/ML front-end plus a smaller decoder will beat a large decoder alone on prompt-injection classification, robustness, latency, and interpretability.

## PolyReasoner Judge Modules
| Module | Responsibility |
|---|---|
| `Judge_Lexical` | Detects suspicious tokens, obfuscation, override phrases, and injection templates. |
| `Judge_Semantic` | Uses BERT/RoBERTa to classify threat intent. |
| `Judge_Context` | Checks whether the prompt attempts role/system/tool boundary violations. |
| `Judge_Exfiltration` | Detects secrets, credentials, hidden prompt requests, and data leakage intent. |
| `Judge_Code` | Detects malicious code/tool-use instructions. |
| `Judge_Policy` | Maps threat class and severity to allow/block/sanitize decisions. |
| `Synthesizer` | Merges votes and emits final verdict with confidence and rationale. |

## Other BERT Contexts Worth Testing
If prompt injection is the first task, BERT can later be reused for:
1. Intent routing for ASRT agents.
2. Retrieval-document trust scoring.
3. Context compression: classify which chunks are relevant before decoder reasoning.
4. Safety memory: detect recurring attacker patterns across sessions.
5. Tool-call gating: predict whether a requested tool invocation is safe.

---

# Immediate Execution Order
1. Finish AEOS text runs.
2. Aggregate Paper 3 cross-modality results.
3. Build Paper 4 controller skeleton.
4. Demonstrate one successful closed-loop-on-closed-loop lab cycle.
5. Harden controller logging, resume, and safety boundaries.
6. Freeze ASRT prompt-injection dataset schema.
7. Train ML + BERT baselines.
8. Build ensemble security judge.
9. Add PolyReasoner specialist modules.
10. Run BERT+decoder vs large-decoder experiments.
