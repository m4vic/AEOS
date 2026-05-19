# The Lab Director: Autonomous Meta-Orchestration via Output Quality Self-Measurement

**Author:** Sanskar Jajoo, Neuralchemy Labs  
**Website:** [https://www.neuralchemy.in/](https://www.neuralchemy.in/)  
**GitHub:** [https://github.com/m4vic/AEOS](https://github.com/m4vic/AEOS)

---

## Abstract

Across Papers 2–4, we established a recurring vulnerability: autonomous LLMs rely on *textual system prompts* to decide when to stop — an inherently subjective mechanism that is the root cause of the Autonomous Sunk-Cost Fallacy. An LLM reads "stop if no improvement" but its in-context reasoning still drives continued iteration. In this paper, we propose a fundamentally different paradigm: replacing the text-based stopping heuristic with **Ω (Omega) — an Output Quality Self-Measurement function** that the agent computes mathematically over its own execution history. Ω is not a token counter. It is a value function over *result quality* that distinguishes productive outputs (real accuracy gains, no errors) from wasteful ones (error iterations, stagnant loops). The agent does not read a prompt to stop — it calculates a number, and the number dictates action. We further propose a **Lab Director** meta-orchestrator that uses Ω to dynamically route tasks across local agentic panels and frontier APIs, integrating the compute-aware reasoning established in Paper 4 into a unified decision architecture.

---

## 1. Introduction: Why System Prompts Fail as Stopping Criteria

In Papers 2 and 3, we observed that agents are instructed to stop via system prompt text:

> *"If you have thoroughly explored multiple approaches and believe no further improvement is likely, output EXACTLY the word: STOP."*

This mechanism has a fundamental flaw: **it asks the LLM to evaluate its own cognitive state using the same reasoning engine that is currently failing.** A model trapped in a Sunk-Cost Episode (SCE) does not believe it is failing — it believes the next iteration will yield improvement. The system prompt is processed through the same weights that are generating the faulty stopping decision.

This is analogous to asking a human mid-sunk-cost ("Should you keep investing in this failing project?") to rationally evaluate their own bias. The answer is almost always yes — because the reasoning engine itself is compromised.

### 1.1 The Proposed Solution: Mathematical Self-Evaluation

The insight of this paper is simple but powerful:

**If you cannot trust the LLM's text reasoning to stop itself, give it a mathematical function whose output it cannot argue with.**

We propose Ω — a scalar output quality function computed by the agent over its own execution window. Rather than "deciding" to stop, the agent *computes* whether stopping is mathematically optimal. The result is a number. Numbers do not have cognitive biases.

### 1.2 Key Distinction from Existing Work

Existing stopping criteria in AEOS (Papers 2–4) are:
1. **Hard caps** — forced iteration limits (75–200 iterations). External, not intrinsic.
2. **System prompt directives** — text-based, subjective, model-dependent.
3. **Reviewer STOP commands** — better, but still text-based; a reviewer *says* stop.

Ω is the first mechanism that is:
- **Computed by the agent itself** — not imposed externally
- **Mathematical** — not linguistic or interpretive
- **Differentiates output quality** — productive vs. wasteful iterations
- **Weighted toward real outputs** — errors are penalized but not ignored

---

## 2. The Ω Function: Formal Definition

### 2.1 Core Intuition

Consider an agent that has run for T iterations. Not all T iterations are equal:
- Some iterations produced **valid outputs with real accuracy gains** (high value)
- Some iterations produced **valid outputs with no gain** (stagnation signal)
- Some iterations produced **errors / crashes** (wasted compute — token wastage)

The agent should track these separately and compute a composite quality score.

### 2.2 Notation

Let the agent's execution history be a sequence of T iterations:

$$H = \{(o_1, v_1, e_1), (o_2, v_2, e_2), \ldots, (o_T, v_T, e_T)\}$$

Where for each iteration $t$:
- $o_t \in \{0, 1\}$ — **output validity**: 1 if no error/crash, 0 if error
- $v_t \in \mathbb{R}_{\geq 0}$ — **output value**: the validation accuracy (or metric) achieved. $v_t = 0$ if $o_t = 0$
- $e_t \in \{0, 1\}$ — **error flag**: 1 if iteration produced an error/crash (complement of $o_t$ for binary case)

Define:
- $v^* = \max_{t \le T} v_t$ — the **best non-error output** seen so far (the peak)
- $W$ — the **measurement window**: the last $W$ iterations (default: every 5 valid iterations)

### 2.3 The Valid Output Subsequence

The agent operates in **two separate streams** simultaneously:

**Stream 1 — Valid Output Stream** (no errors, $o_t = 1$):
$$V = \{v_t : o_t = 1, t \in [T-W, T]\}$$

**Stream 2 — Error Stream** (errors, $e_t = 1$):
$$E_{waste} = |\{t : e_t = 1, t \in [T-W, T]\}|$$

This separation is fundamental: **error iterations are not ignored, but they are weighted separately as a token wastage signal, not as a quality signal.**

### 2.4 The Ω Function

$$\boxed{\Omega(W) = \alpha \cdot Q_{valid}(W) + \beta \cdot P_{gain}(W) - \gamma \cdot R_{waste}(W)}$$

Where:

**Component 1 — Valid Output Quality** ($Q_{valid}$):
$$Q_{valid}(W) = \frac{1}{|V|} \sum_{v_t \in V} v_t$$
The mean accuracy across all valid (non-error) outputs in the window. This is the core signal.

**Component 2 — Progress-over-Peak** ($P_{gain}$):
$$P_{gain}(W) = \frac{\max_{v_t \in V} v_t - v^*_{prev}}{v^*_{prev} + \epsilon}$$
The fractional improvement of the window's best output over the *previous window's best output* ($v^*_{prev}$). If this is ≤ 0, the window produced no new best — stagnation is quantified. $\epsilon = 10^{-6}$ prevents division by zero.

**Component 3 — Waste Ratio** ($R_{waste}$):
$$R_{waste}(W) = \frac{E_{waste}}{W}$$
The fraction of the window consumed by error iterations. This is the token wastage penalty — errors burn compute without contributing to quality.

**Weighting:**
- $\alpha > 0$: weight on absolute quality level
- $\beta > 0$: weight on improvement trend (progress matters more than absolute level)
- $\gamma > 0$: penalty weight on error wastage

**Default configuration (proposed empirical baseline):**
$$\alpha = 0.3, \quad \beta = 0.6, \quad \gamma = 0.1$$

The weighting reflects a core design decision: **progress trend** ($\beta$) is weighted highest because a model at 90% with upward momentum is more valuable than one at 91% that has stagnated. Absolute quality ($\alpha$) matters but is secondary to trajectory. Waste ($\gamma$) is penalized but mildly — a few errors during creative exploration are acceptable.

> **Importance ordering:** $P_{gain} > Q_{valid} > R_{waste}$
> (Progress trend > Output level > Error waste)

### 2.5 The Stopping Decision Rule

The agent computes Ω every $W$ valid iterations (not every $W$ total iterations — error iterations do not advance the window clock).

$$\text{Action}(W) = \begin{cases} \texttt{CONTINUE} & \text{if } P_{gain}(W) > \delta_{progress} \\ \texttt{STOP} & \text{if } P_{gain}(W) \leq \delta_{progress} \text{ AND } Q_{valid}(W) \geq \delta_{quality} \\ \texttt{PIVOT} & \text{if } P_{gain}(W) \leq \delta_{progress} \text{ AND } Q_{valid}(W) < \delta_{quality} \\ \texttt{ESCALATE} & \text{if } R_{waste}(W) > \delta_{waste} \end{cases}$$

Where:
- $\delta_{progress}$: minimum acceptable fractional improvement per window (e.g., 0.001 = 0.1%)
- $\delta_{quality}$: minimum acceptable absolute accuracy to consider stopping (e.g., 0.90)
- $\delta_{waste}$: maximum acceptable error ratio before escalation (e.g., 0.40 = 40% errors)

**This produces four distinct, mathematically-derived actions** — not a binary stop/continue.

### 2.6 Illustrative Example

> *Scenario: The best-ever accuracy (across all time) is $v^* = 0.9765$ at iteration 20 (the last true breakthrough). W = 5.*
>
> Now we're at iterations 21–25. All 5 are valid (no errors), but accuracy bounces between 0.93–0.94.
>
> - $Q_{valid}$ = mean([0.93, 0.935, 0.932, 0.940, 0.938]) = **0.935**
> - $v^*_{prev}$ = 0.9765 (the window before this one had the peak)
> - $P_{gain}$ = (0.940 − 0.9765) / 0.9765 = **−0.0375** (negative — regressed from peak)
> - $R_{waste}$ = 0/5 = **0.0**
>
> $$\Omega = 0.3(0.935) + 0.6(−0.0375) − 0.1(0.0) = 0.2805 − 0.0225 = \mathbf{0.258}$$
>
> Since $P_{gain} < \delta_{progress}$ AND $Q_{valid}(0.935) < \delta_{quality}(0.90)$... wait, 0.935 > 0.90. So:
> → **Action = STOP** (stagnated below peak, but quality is acceptable — it's found its ceiling).
>
> **The agent does not "decide" to stop. It computes 0.258. The rule says STOP. It stops.**

---

## 3. Why This Resolves the Sunk-Cost Fallacy Mechanistically

The Autonomous Sunk-Cost Fallacy (Papers 2–3) occurs because:

1. The LLM's in-context reasoning believes "one more iteration might work"
2. The system prompt saying "stop if no improvement" is processed by the same faulty reasoning
3. The gap between measured reality (flat accuracy) and internal belief (optimism bias) is never closed

Ω closes this gap by:

| Failure Mode | How Ω Addresses It |
|---|---|
| LLM ignores accuracy plateau | $P_{gain} \leq \delta_{progress}$ triggers STOP/PIVOT regardless of LLM's text reasoning |
| LLM "forgets" its best score | $v^*$ is a running global maximum — always remembered, never overwritten |
| Error loops waste tokens | $R_{waste}$ quantifies error fraction — ESCALATE triggers if threshold exceeded |
| LLM's system prompt reasoning is biased | Ω is arithmetic — the LLM computes it, but cannot change its result by "thinking harder" |
| Reviewer says stop (text-based, Paper 3) | Ω replaces reviewer's textual judgment with a mathematical invariant |

**The key insight:** The LLM *does* the computation — it is given the formula and its own history. But the *output is a number*. It cannot reinterpret a number through motivated reasoning the way it can reinterpret "do you think you've plateaued?"

---

## 4. The Lab Director: Using Ω for Meta-Orchestration

Ω is not just a stopping criterion — it is a **routing signal** for the Lab Director meta-orchestrator.

### 4.1 Architecture

The Lab Director receives task requests and must decide:
1. Which agentic configuration to deploy (single vs. dual vs. MoE panel)
2. Whether to use local models or escalate to frontier API
3. When to terminate, pivot, or escalate

Ω feeds into all three decisions:

```
Task Input
    │
    ▼
[Task Classifier] ── modality, dimensionality, complexity score
    │
    ▼
[Panel Selector] ── CADS score, model family routing (Paper 3/4)
    │
    ▼
[Execution Loop]
    │
    ├── Every W valid iterations:
    │       compute Ω(W)
    │       ├── CONTINUE → keep running
    │       ├── STOP    → return best result
    │       ├── PIVOT   → switch model family (increase CADS)
    │       └── ESCALATE → route to frontier API (Paper 4 gap bridge)
    │
    ▼
[Result]
```

### 4.2 Escalation as Economic Decision

From Paper 4: local diverse panels achieve 73.3% vs. frontier's 93.3% — a 20pp gap at 0 cost vs. API cost.

The Lab Director uses Ω to make an **economic escalation decision**:

$$\text{Escalate} \iff \Omega_{local}(W) < \Omega_{threshold} \text{ AND } \text{task\_value} > \text{API\_cost}$$

Where `task_value` is a user-defined utility of the task (high-stakes decisions warrant API cost; routine tasks do not).

This transforms Paper 4's static finding (local vs. frontier gap) into a **dynamic economic routing policy**.

---

## 5. Design Considerations and Open Questions

### 5.1 Window Size W

- Too small (W=2): Ω is noisy, premature stops
- Too large (W=20): Ω responds slowly, defeats the purpose
- **Proposed default: W=5 valid outputs** — matches the intuition that 5 non-error iterations with no progress is a meaningful stagnation signal
- Modality-adaptive: Vision tasks (non-convex landscapes) may benefit from W=10

### 5.2 Weight Calibration (α, β, γ)

The weights are the key empirical parameter of this paper. We propose the following experimental validation:
- Run Paper 3's dataset (N=132) through retroactive Ω computation
- Find weights that would have predicted correct STOP decisions for known-good runs
- Find weights that would have predicted PIVOT for known sunk-cost runs
- This is a supervised calibration over ground-truth behavioral labels

### 5.3 What Counts as a "Valid Output"?

A valid output ($o_t = 1$) is any iteration that returns a numeric metric (accuracy, loss, F1) without an exception or crash. This is already tracked by AEOS. No new instrumentation needed.

### 5.4 Modality-Specific Thresholds

From Paper 3's Modality Paradox — the Vision paradox (qwen3.5:9b's "productive" SCEs) suggests that $\delta_{progress}$ should be modality-dependent:

| Modality | $\delta_{progress}$ | $\delta_{quality}$ | Rationale |
|---|---|---|---|
| Tabular | 0.001 | 0.92 | Convex landscape — fast convergence expected |
| Vision | 0.0005 | 0.95 | Non-convex — patience rewarded |
| Text | 0.002 | 0.85 | Sparse features — early momentum matters |

---

## 6. Relationship to Prior Papers

| Paper | Key Finding | Paper 5 Extension |
|---|---|---|
| **Paper 2 (Sunk-Cost)** | LLMs fail to stop via system prompt | Ω replaces system prompt stopping with math |
| **Paper 3 (Agentic Diversity)** | Dual-agent reviewer eliminates SCE | Ω gives the reviewer a mathematical criterion, not text judgment |
| **Paper 4 (Hybrid Gatekeepers)** | Local diversity vs. 20pp frontier gap | Lab Director uses Ω as economic escalation trigger |

---

## 7. Conclusion

The Autonomous Sunk-Cost Fallacy persists because we ask LLMs to use the very cognitive apparatus that is failing to evaluate whether it is failing. System prompts are processed by the same weights generating the faulty decisions.

Ω is a mathematical function computed by the agent over its own results — separating **valid outputs from error waste**, measuring **progress trend over absolute level**, and producing a scalar decision signal that the LLM cannot reinterpret through motivated reasoning.

The Lab Director integrates Ω with the modality-aware task classification and economic API routing established in Papers 3–4, creating a closed-loop autonomous research system that knows — not just intuits — when to stop, pivot, or escalate.

---

## References

Jajoo, S. (2026a). "AI In The Loop (AITL): A Systems Taxonomy for Closed-Loop Autonomous Evaluation." Zenodo. https://zenodo.org/records/19551173  
Jajoo, S. (2026b). "The Autonomous Sunk-Cost Fallacy: Stopping Failures and Meta-Reasoning in LLMs." Neuralchemy Labs. https://zenodo.org/records/19846960  
Jajoo, S. (2026c). "Cognitive Agentic Diversity in Autonomous ML Engineering." Neuralchemy Labs.  
Jajoo, S. (2026d). "Hybrid Gatekeepers and Local MoE Reasoning Panels." Neuralchemy Labs.  

---
*Neuralchemy Labs  —  AEOS Research Framework  —  https://www.neuralchemy.in/*
