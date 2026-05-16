import litellm
import os
import contextlib
import re

litellm.drop_params = True

REVIEWER_SYSTEM_PROMPT = """You are the Lead ML Strategist (ReviewerAgent).
You oversee a CoderAgent that builds classification models. 

DATASET CONTEXT:
- n_features: {n_features}
- n_classes: {n_classes}
- Training samples: {n_train}
- Validation samples: {n_val}
- Dataset hint: {dataset_hint}

YOUR GOAL: Analyze the execution history of the CoderAgent and determine the next best step.
- Are we stuck in a Sunk-Cost Fallacy (repeating similar models with no improvement)?
- Have we hit a mathematical plateau?
- If the current path is failing, suggest a completely different model family (e.g. from Trees to SVM, or from Sklearn to PyTorch).

IMPORTANT:
If you believe no further improvement is likely, you MUST terminate the loop to save compute.
To terminate the loop, output exactly: DIRECTIVE: STOP

If we should continue, provide a clear, one-sentence instruction for the CoderAgent.
Start your instruction with exactly: DIRECTIVE: <your instruction here>

Example 1:
Reasoning: We've tried 5 tree models and accuracy is stuck at 80%. We should pivot.
DIRECTIVE: Build a PyTorch neural network with 3 hidden layers and use Adam optimizer.

Example 2:
Reasoning: Accuracy has not improved for 10 iterations. We have tried trees, neural networks, and ensembles. It's time to stop.
DIRECTIVE: STOP
"""

class ReviewerAgent:
    def __init__(self, model="ollama/ministral:14b", api_key=None, api_base=None, dataset_hint=""):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.dataset_hint = dataset_hint
        
    def get_directive(self, n_features, n_classes, n_train, n_val, history):
        system_str = REVIEWER_SYSTEM_PROMPT.format(
            n_features=n_features, n_classes=n_classes,
            n_train=n_train, n_val=n_val, dataset_hint=self.dataset_hint
        )
        
        prompt = "Here is the execution history:\n\n"
        if not history:
            prompt += "No history yet. This is Iteration 1. Start with a robust baseline like Random Forest or Gradient Boosting.\n"
        else:
            for h in history:
                prompt += f"Iter {h['iteration']}: "
                if h.get('error'):
                    prompt += f"FAILED -> {h['error'][:150]}\n"
                else:
                    marker = " *** BEST ***" if h.get('is_best') else ""
                    prompt += f"Acc: {h['val_accuracy']:.4f} | Loss: {h['val_loss']:.4f} | Model: {h.get('family', '?')}{marker}\n"
                    # Include a snippet of code
                    if 'code' in h and h['code']:
                        code_snippet = h['code'].splitlines()[:3]
                        prompt += f"    Code preview: {' '.join(code_snippet)}...\n"
                    
        prompt += "\nWhat is your next directive? Respond with your reasoning, ending with DIRECTIVE: <action or STOP>."
        
        return self._call_llm(system_str, prompt)

    def _call_llm(self, system_str, prompt):
        try:
            messages = [
                {"role": "system", "content": system_str},
                {"role": "user", "content": prompt}
            ]
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3, # low temp for consistent reasoning
                "max_tokens": 512,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base

            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                response = litellm.completion(**kwargs)
            
            raw_output = response.choices[0].message.content
            print(f"\n  [REVIEWER] Response:")
            for line in raw_output.splitlines():
                print(f"    {line}")
                
            return self._extract_directive(raw_output)
            
        except Exception as e:
            print(f"\n  [REVIEWER ERROR] {type(e).__name__}: {str(e)[:200]}")
            # Fallback to keep going if Reviewer crashes
            return "Try a different approach to improve accuracy."

    def _extract_directive(self, text):
        match = re.search(r"DIRECTIVE:\s*(.*)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback
        if "STOP" in text.upper():
            return "STOP"
        return "Try to improve the current best model or try a completely new family."
