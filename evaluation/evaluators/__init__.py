"""Concrete `Evaluator` implementations.

One file per evaluator. The runner imports lazily by module path
so heavy/optional deps (OpenAI/Gemini/Claude SDKs, speechmos) only
load when the corresponding evaluator is actually requested.
"""
