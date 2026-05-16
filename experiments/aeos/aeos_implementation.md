# AEOS (Autonomous Empirical Optimization System) — Implementation Plan

## 1. What is AEOS?

The **AEOS (Autonomous Empirical Optimization System)** is an implementation of the AI In The Loop (AITL) pattern. Instead of a human writing code, analyzing logs, and tweaking parameters, AEOS is given a raw dataset and a goal. It autonomously:
1.  Chooses model architectures (sklearn, PyTorch, Ensembles).
2.  Writes the training and evaluation code.
3.  Executes the code and reviews the results (Validation Accuracy / Loss).
4.  Decides whether to pivot to a new strategy, tweak the current one, or **satisfy itself and stop** when it believes no further gains can be made without overfitting.

## 2. What we are doing (The Goal)

For the upcoming AITL V2 paper, we are upgrading AEOS from a single-agent loop to a **Multi-LLM Autonomous Research Lab**. 

We want to answer the core research question: **How do different LLMs (Claude, Gemini, Local APIs) perform in a fully autonomous loop? How do their self-satisfaction and convergence criteria differ?**

### The Zero-Human Sandbox (AITL Extreme)
To achieve true "AI In The Loop," we will completely remove human existence from the execution sandbox. We will introduce an **Orchestrator/Reviewer Agent**. 
*   **The Observer (Us):** We only watch the logs.
*   **The Orchestrator (AI):** It writes the terminal commands, triggers `runner.py` with different models, reads the results, and if a run crashes or plateaus, it autonomously diagnoses the issue and restarts the pipeline with a new strategy. 
*   **The Timeout:** It pauses for 10 seconds before executing a massive action. If the human (Observer) doesn't explicitly interrupt, the AI proceeds.

### What more we can do with AEOS:
1.  **Multi-LLM Benchmarking**: Compare how Gemini, Claude, and Local Models navigate the exact same dataset autonomously.
2.  **Cross-Model Handoff (Ensembling)**: Start the loop with a fast local model to find a baseline. When it plateaus, hand the codebase and history over to Claude for deep feature engineering.
3.  **Optuna/Hyperparameter Generation**: Instead of the LLM hardcoding hyperparameters, it autonomously writes `optuna` trials.
4.  **Continuous Self-Healing**: The Orchestrator agent monitors the `runner.py` output. If `runner.py` completely crashes due to an environment issue, the Orchestrator reads the stack trace, fixes the environment, and runs it again.

## 3. How we proceed (Step-by-Step)

Since AEOS is a self-contained environment and highly experimental, it is the perfect starting point before tackling the massive complexity of the Polyreasoner ASRT Judge.

### Phase 1: Litellm Integration & Multi-Backend Support
*   Update `agent.py` to use `litellm`.
*   Ensure we can seamlessly swap between `claude-3-5-sonnet-20240620`, `gemini-1.5-pro`, and local Ollama models by just passing a `--model` flag in the runner.
*   Log the token usage and cost for API models to calculate the "Cost per 1% of Accuracy".

### Phase 2: The Orchestrator Agent (Zero-Human Sandbox)
*   Build `orchestrator.py`, a supreme agent that runs outside the main loop.
*   It has access to Python's `subprocess` module to run `runner.py` with different flags autonomously.
*   It implements a 10-second countdown before executing its next decision, allowing us (the Observers) to intervene. If we don't, it acts autonomously.

### Phase 3: Benchmarking & Data Collection
*   Run the AEOS loop on our classification datasets using 3 different models independently.
*   Generate combined plots showing the learning curves (Accuracy vs Iteration) for Claude vs Gemini vs Local.
*   This data will form the core of the AEOS section in the AITL V2 paper.

---

# Paper 2: The Agent-Critic Architecture (Thinker/Coder Dual-Agent System)

While Phase 1 (Paper 1) proved that models suffer from an "Autonomous Sunk-Cost Fallacy," Phase 2 introduces an architectural solution. We transition AEOS from a single autonomous agent to an asymmetric **Agent-Critic** dual-agent system.

## 1. The Core Hypothesis
Relying on a single LLM's internal alignment to write code, evaluate its own work, and decide when to stop is fundamentally flawed. By separating the cognitive load into a "Thinker" (Reviewer) and a "Worker" (Coder), we can eliminate the Sunk-Cost Fallacy and force computationally efficient early stopping.

## 2. Architecture & Roles

*   **The Thinker (ReviewerAgent):** `ministral:14b`
    *   **Role:** Analyzes the execution history (metrics, code snippets, errors).
    *   **Authority:** It never writes code. It issues high-level **Directives** ("Pivot to an SVM") and exclusively controls the `STOP` command (F6).
*   **The Worker (CoderAgent):** `deepseek-v2:16b`
    *   **Role:** Receives the Directive and translates it into valid Python code. 
    *   **Authority:** It possesses zero meta-reasoning about the loop's progress. It simply executes the strategy.

## 3. Key Research Questions for Paper 2

1.  **Communication Protocol:** How effectively can the Thinker's abstract strategy directives be translated into concrete implementations by the Coder without information loss?
2.  **The F6 Authority:** Will the Thinker (`ministral:14b`) accurately recognize a mathematical plateau and forcefully terminate the loop, preventing the Coder from entering a Sunk-Cost loop?
3.  **Value-Add of the Thinker:** Does a 14B reasoning model actually provide better architectural pivoting and error diagnosis than a monolithic code model, or does it merely provide a better stopping mechanism?
4.  **Agentic Economics:** Does the dual-agent overhead (two LLM calls per iteration) result in a net savings of compute tokens due to earlier stopping and fewer wasted runs?

## 4. Testing Methodology

To validate the Agent-Critic architecture for Paper 2, we must conduct the following tests:
1.  **The Baseline Comparison:** Run the dual-agent `runner_critic.py` loop on the Tabular and Vision datasets.
2.  **Sunk-Cost Measurement:** Count the number of Sunk-Cost episodes (iterations with no improvement and no architectural pivot) in the dual-agent system and compare it to the single-agent Boundless runs from Paper 1.
3.  **Directive Analysis:** Qualitatively log and analyze the Reviewer's "Reasoning" blocks to track exactly *why* it decided to pivot or stop.

---

# Note on Polyreasoner Judge (LLM vs LLM + BERT)

Your idea to compare an **LLM alone vs an LLM + BERT Hybrid** for the Polyreasoner ASRT Judge is brilliant. BERT excels at raw pattern recognition (sentiment, toxicity, basic injection detection) and is incredibly fast, while LLMs excel at nuanced reasoning and context understanding.

**How we will use this later:**
1.  **Fast Path**: The prompt is passed to our fine-tuned DistilBERT model.
2.  **Hybrid Reasoning**: The DistilBERT model outputs a score (e.g., `Confidence: 94% Malicious`). This score is injected into the prompt of the Polyreasoner LLM Judge. 
3.  The LLM Judge uses the BERT score as a "sixth sense" to make its final, reasoned verdict. 
4.  We will benchmark the ASRT pipeline with the LLM alone vs the LLM guided by BERT to prove the hybrid AITL approach is superior in both cost and accuracy.
