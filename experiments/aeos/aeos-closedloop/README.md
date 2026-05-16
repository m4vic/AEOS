# AEOS Closed-Loop Framework

## Vision
This directory is reserved for future expansion to design a **Closed Loop** system built *on top* of the existing AEOS framework. 

While the current AEOS handles autonomous evaluation (generating, testing, and logging models), the ultimate goal of AITL (AI-In-The-Loop) is to feed these results back into a higher-order system.

## Capabilities
- **Meta-Learning:** Automatically analyzing the `aeos_behave/results/` JSON outputs to identify systemic failure modes across all tested LLMs.
- **Visibility:** Logging the Meta-LLM's reasoning and structural prompt changes to `meta_insights.json` for human oversight.
- **Self-Correction (True AGI Mode):** Overwriting the source code (`reviewer.py`) with updated agent constraints dynamically across subsequent benchmark campaigns without human intervention.

## System Architecture

```mermaid
graph TD
    %% Define Nodes
    AEOS["AEOS Benchmark Loop<br>(runner.py / runner_critic.py)"]
    JSON["results/*.json<br>(Execution Logs & Metrics)"]
    MetaEval["meta_evaluator.py<br>(Closed-Loop Orchestrator)"]
    LLM["Meta-Reviewer LLM<br>(e.g., llama3.1:8b)"]
    Insights["meta_insights.json<br>(Visibility Log)"]
    Reviewer["aeos_behave/reviewer.py<br>(System Prompts)"]

    %% Define Flow
    AEOS -->|Outputs metrics| JSON
    JSON -->|Aggregates data| MetaEval
    MetaEval -->|Requests analysis| LLM
    LLM -->|Generates fixed prompts| MetaEval
    MetaEval -->|Logs reasoning Visibility| Insights
    MetaEval -->|Regex overwrites prompt Autonomy| Reviewer
    Reviewer -->|New constraints applied| AEOS

    %% Styling
    style AEOS fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff
    style JSON fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff
    style MetaEval fill:#c0392b,stroke:#e74c3c,stroke-width:2px,color:#fff
    style LLM fill:#8e44ad,stroke:#9b59b6,stroke-width:2px,color:#fff
    style Reviewer fill:#2980b9,stroke:#3498db,stroke-width:2px,color:#fff
```
