"""
AEOS — BERT-Specialist CoderAgent
A CoderAgent variant whose system prompt CONSTRAINS it to always use
HuggingFace transformers (BERT / DistilBERT) for feature extraction.
Used in the Tri-Agent competitive generation to test domain-expert
constrained generation vs. free-form LLM generation.
"""
import litellm
import os
import contextlib
import re

litellm.drop_params = True

BERT_CODER_SYSTEM_PROMPT = """You are a BERT-Specialist CoderAgent (ML Engineer).
You have a classification dataset. Here is everything you know:
- n_features: {n_features}
- n_classes: {n_classes}
- Training samples: {n_train}
- Validation samples: {n_val}

DATASET TYPE: {dataset_hint}

YOUR SPECIALIZATION: You are a **BERT / Transformer specialist**.
You MUST use HuggingFace `transformers` library for feature extraction or fine-tuning.

MANDATORY APPROACH:
1. Use `transformers` (e.g. DistilBertModel, BertModel, AutoModel) for embedding extraction or fine-tuning.
2. For TEXT data: tokenize with AutoTokenizer, extract [CLS] embeddings, then classify.
3. For TABULAR/VISION data: use a simple transformer encoder or adapt BERT's architecture.
4. You may combine BERT embeddings with a sklearn classifier (LogisticRegression, SVM, etc.) on top.
5. Prefer `distilbert-base-uncased` for speed (timeout constraint).

YOUR TASK: Write a Python function `solve(X_train, y_train, X_val, y_val)` that:
1. Uses transformers for feature extraction or fine-tuning.
2. Returns predictions as a numpy array of shape (n_val,) with integer class labels.

Available in your namespace: numpy (as np), sklearn submodules, torch, nn, optim, transformers.

RULES:
1. You MUST define: def solve(X_train, y_train, X_val, y_val): ... return predictions
2. predictions must be a numpy array of integers in range [0, {max_class}]
3. Your code has a {timeout}-second time limit. Be efficient — prefer DistilBERT over full BERT.
4. Output ONLY the code inside ```python ... ```. No explanations.
5. If the data is already numeric (not raw text), you can still use a small transformer encoder architecture.
"""


class BertCoderAgent:
    """CoderAgent that is constrained to always use BERT/transformers."""
    
    def __init__(self, model="ollama/deepseek-coder:6.7b", api_key=None, api_base=None, dataset_hint=""):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.dataset_hint = dataset_hint

    def _detect_model_family(self, code):
        """Detect the model family from generated code — BERT-specific families."""
        code_lower = code.lower()
        if 'distilbert' in code_lower: return 'DistilBERT'
        if 'bertmodel' in code_lower or 'bert' in code_lower and 'transformers' in code_lower: return 'BERT'
        if 'automodel' in code_lower: return 'HF_AutoModel'
        if 'roberta' in code_lower: return 'RoBERTa'
        if 'nn.transformerencoder' in code_lower: return 'PyTorch_Transformer'
        if 'nn.module' in code_lower or 'nn.linear' in code_lower: return 'PyTorch_NN'
        if 'randomforest' in code_lower: return 'RandomForest'
        if 'svc' in code_lower or 'svm' in code_lower: return 'SVM'
        if 'logisticregression' in code_lower: return 'LogisticRegression'
        if 'mlpclassifier' in code_lower: return 'sklearn_MLP'
        return 'Unknown_BERT'

    def generate_code(self, n_features, n_classes, n_train, n_val, directive, history, best_code=None, timeout=120):
        system_str = BERT_CODER_SYSTEM_PROMPT.format(
            n_features=n_features, n_classes=n_classes,
            n_train=n_train, n_val=n_val,
            max_class=n_classes - 1, timeout=timeout,
            dataset_hint=self.dataset_hint
        )
        
        prompt = f"DIRECTIVE FROM REVIEWER: {directive}\n\n"
        prompt += "REMINDER: You MUST use HuggingFace transformers (BERT/DistilBERT) in your solution.\n\n"
        
        if history and history[-1].get('error'):
            prompt += f"WARNING: The previous attempt resulted in an error:\n{history[-1]['error']}\nPlease fix this if applicable.\n\n"
        
        if best_code:
            prompt += "For context, here is the current best working code:\n```python\n"
            prompt += best_code + "\n```\n\n"
            
        prompt += "Now, output the new BERT-based code inside ```python ... ```."

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
                "temperature": 0.7,
                "max_tokens": 2048,
            }
            if self.api_key: kwargs["api_key"] = self.api_key
            if self.api_base: kwargs["api_base"] = self.api_base

            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                response = litellm.completion(**kwargs)
            
            raw_output = response.choices[0].message.content
            return self._extract_code(raw_output)
            
        except Exception as e:
            print(f"\n  [BERT CODER ERROR] {type(e).__name__}: {str(e)[:200]}")
            raise e

    def _extract_code(self, text):
        stripped = text.strip()
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
        
        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match: return match.group(1).strip()
        
        code = stripped
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"```\s*$", "", code)
        return code.strip()
