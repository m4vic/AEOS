# AEOS Paper Structure — The Unified Narrative

## The Single Thesis
> "Autonomous LLM agents exhibit measurable cognitive biases and failure modes when engineering alone. Multi-agent collaboration and hybrid architectures systematically eliminate these failures."

Every experiment proves one piece of this thesis.

---

## How Each Section Creates The Next

**Section 1** asks: "Can a single LLM do autonomous ML engineering?"
→ Answer: Yes, but we observe problems. This raises the question...

**Section 2** asks: "What exactly goes wrong?" (Sunk cost, wasted iterations, narrow strategy)
→ We quantify the failures with data. Now the reader wants to know...

**Section 3** asks: "Can a second LLM fix these failures?"
→ Yes, a Reviewer agent reduces waste. A Debate setup breaks sunk cost anchoring. But this is expensive (2x LLM calls). So...

**Section 4** asks: "Can a cheap specialized model (BERT) achieve the same benefit?"
→ Yes, hybrid LLM+BERT outperforms LLM-alone at near-zero extra cost. But we're still paying for a big LLM. So...

**Section 5** asks: "Can we chain cheap→expensive models to minimize total cost?"
→ Yes, escalation chains achieve premium accuracy at budget cost.

**Conclusion:** The optimal AITL agent is not one big model — it's an orchestrated system of specialized components.

The reader never feels like they're reading disconnected experiments — they're following a logical investigation where each finding motivates the next.

---

## Concrete Outline

### Title
**"AEOS: Behavioral Analysis and Optimization of Autonomous LLM Agents in Self-Directed ML Engineering"**

### Abstract
LLMs can autonomously engineer ML solutions, but they exhibit cognitive biases (sunk cost, anchoring). We show that multi-agent and hybrid architectures fix this while reducing cost.

### 1. Introduction
- AITL recap (reference your published paper)
- AEOS as the experimental testbed
- Central question: "We study HOW LLMs behave autonomously, not just WHETHER they can"

### 2. Related Work
- AutoML vs AEOS (AEOS is NOT AutoML — the LLM writes arbitrary code, not searching a predefined space)
- LLM-as-coder (Devin, SWE-Bench, AlphaCode)
- Multi-agent systems (CAMEL, AutoGen, CrewAI)
- Our contribution: behavioral analysis + optimization in engineering loops

### 3. AEOS Architecture
- The loop: generate → execute → evaluate → feedback → repeat
- The pivot mechanism and STOP signal
- Metrics framework (all 12 metrics from the questionbook)

### 4. Experiment 1 — Single-Agent Baselines (RQ1, RQ2, RQ7)
- Run each LLM independently on 3 datasets (Covtype, MNIST, IMDB)
- **Finding 1:** Sunk cost behavior — LLMs anchor to first successful strategy
- **Finding 2:** Engineering personalities — some are conservative (sklearn only), some are adventurous (try PyTorch)
- **Finding 3:** Self-satisfaction varies — some models never STOP, some stop too early
- Tables: accuracy, iterations, waste rate, families explored per model

### 5. Experiment 2 — Multi-Agent Collaboration (RQ3, RQ4)
- Coder + Reviewer configurations (M1, M2, M3)
- **Finding 4:** Reviewer reduces wasted iterations by X%
- **Finding 5:** Two 3B models debating can match a single 9B (or not — either result is publishable)
- **Finding 6:** The collaboration tax — does the accuracy gain justify 2x inference time?

### 6. Experiment 3 — Hybrid LLM + BERT (RQ5)
- Fine-tuned DistilBERT as a "sixth sense" for the LLM agent
- BERT scores injected into LLM prompt before each iteration
- **Finding 7:** Hybrid reduces sunk-cost episodes by Y%
- **Finding 8:** 5ms BERT inference vs 5-second LLM inference — cost/benefit analysis

### 7. Experiment 4 — Escalation Chains (RQ6)
- Cheap model → expensive model with history transfer
- **Finding 9:** Claude with inherited history converges in 3 iterations vs 10 from scratch
- **Finding 10:** 70% cost reduction for equivalent final accuracy
- Cost-efficiency frontier plots

### 8. Discussion
- LLM cognitive biases: parallels to human psychology
- Practical implications: how should companies architect agentic AI systems?
- The case for orchestration over monolithic models

### 9. Conclusion
- The optimal autonomous agent is not one big model — it's a system
- Open-source release of AEOS framework
- Future work: the Zero-Human Sandbox (Orchestrator agent)

---

## The Connecting Thread (For your reference)

The paper tells ONE story: **"We gave LLMs full autonomy, watched them fail in predictable ways, and engineered solutions."**

| Section | Discovery | Leads to |
|---|---|---|
| Exp 1 (Single Agent) | LLMs have biases and waste iterations | "Can we fix this with collaboration?" |
| Exp 2 (Multi-Agent) | Yes, but it's expensive (2x calls) | "Can we do it cheaper?" |
| Exp 3 (Hybrid BERT) | Yes, BERT is nearly free and helps | "Can we also reduce the base LLM cost?" |
| Exp 4 (Escalation) | Yes, start cheap and escalate only when needed | "The optimal system is orchestrated" |

Each experiment is a direct response to the limitation found in the previous one. That's what makes it one paper, not four.
