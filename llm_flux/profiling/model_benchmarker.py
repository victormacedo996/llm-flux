"""
profiling/model_benchmarker.py — Model accuracy / perplexity domain service.

Extends the original design with:
  - A clear separation between PerplexityTests and AccuracyTests.
  - Auto-discovery of test methods via the method-registry pattern.
  - A `register()` class method so tests can be added at runtime without
    subclassing (alternative extensibility path).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Set, get_args

import torch
from datasets import Dataset
from loguru import logger
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm_flux.profiling.types.benchmark import (
    AccuracyTestResult,
    ComputePerplexityForBatchReturn,
    ComputePerplexityForDatasetReturn,
    PerplexityTestResult,
)

# ── Test registries ───────────────────────────────────────────────────────────
# Add new Literal values here to declare new tests.
# The corresponding method must share the exact same name (underscored).

PERPLEXITY_TESTS = Literal["lambada"]
ACCURACY_TESTS = Literal["arc_c"]

AVAILABLE_TESTS: List[str] = list(get_args(PERPLEXITY_TESTS) + get_args(ACCURACY_TESTS))

TestResult = PerplexityTestResult | AccuracyTestResult


class ModelPerformanceBenchmarker:
    """
    Runs a configurable set of model benchmarks and returns structured results.

    **Extending with new tests (two options):**

    Option A — subclass::

        class MyBenchmarker(ModelPerformanceBenchmarker):
            def arc_c(self, tokenizer, model, batch_size=16) -> AccuracyTestResult:
                ...

    Option B — runtime registration::

        bench = ModelPerformanceBenchmarker()
        bench.register("my_test", my_test_fn)

    In both cases the method is auto-discovered by the registry and becomes
    available via ``benchmark(..., tests={"my_test"})``.
    """

    def __init__(self) -> None:
        self._test_methods: Dict[
            str, Callable[[PreTrainedTokenizerBase, PreTrainedModel, int], TestResult]
        ] = {}
        # Auto-discover methods matching declared test names
        for test in AVAILABLE_TESTS:
            if hasattr(self, test):
                self._test_methods[test] = getattr(self, test)

    def register(
        self,
        name: str,
        fn: Callable[[PreTrainedTokenizerBase, PreTrainedModel, int], TestResult],
    ) -> None:
        """Register a custom test function at runtime."""
        self._test_methods[name] = fn

    # ── Core perplexity infrastructure ───────────────────────────────────────

    def _compute_perplexity_for_batch(
        self,
        input_texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
    ) -> ComputePerplexityForBatchReturn:
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        input_ids: torch.Tensor = inputs["input_ids"]
        attention_mask: torch.Tensor = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits: torch.Tensor = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].to(logits.dtype)

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        target_log_probs = (
            log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        )
        target_log_probs = target_log_probs * shift_mask

        nll = -target_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1)
        perplexities = torch.exp(nll)

        return ComputePerplexityForBatchReturn(
            perplexities=perplexities.tolist(),
            mean_perplexity=perplexities.mean().item(),
        )

    def evaluate_perplexity(
        self,
        dataset: Dataset,
        text_column: str,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> ComputePerplexityForDatasetReturn:
        all_perplexities: List[float] = []
        effective_size = (
            min(len(dataset), num_examples) if num_examples else len(dataset)
        )

        for i in range(0, effective_size, batch_size):
            batch_texts: List[str] = dataset[i : min(i + batch_size, effective_size)][
                text_column
            ]
            result = self._compute_perplexity_for_batch(batch_texts, tokenizer, model)
            all_perplexities.extend(result.perplexities)

        mean = sum(all_perplexities) / len(all_perplexities) if all_perplexities else 0.0
        return ComputePerplexityForDatasetReturn(
            all_perplexities=all_perplexities, mean_perplexity=mean
        )

    # ── Built-in tests ────────────────────────────────────────────────────────

    def lambada(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> PerplexityTestResult:
        """Perplexity on the LAMBADA dataset (last-word prediction)."""
        from datasets import load_dataset

        dataset = load_dataset("cimec/lambada", split="test")
        result = self.evaluate_perplexity(
            dataset=dataset,
            text_column="text",
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
            num_examples=num_examples,
        )
        return PerplexityTestResult(result=result, test_name="lambada")

    # arc_c is declared but not implemented here — it's intentionally left as
    # an example of how to extend via subclassing or register().

    # ── Public benchmark API ──────────────────────────────────────────────────

    def benchmark(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        tests: Set[str],
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> List[TestResult]:
        if not tests:
            raise ValueError("Provide at least one test name.")

        results: List[TestResult] = []
        for test in tests:
            method = self._test_methods.get(test)
            if method is None:
                logger.warning(f"Test '{test}' is not implemented — skipping.")
                continue
            
            # Use introspection to see if the custom method supports num_examples
            import inspect
            sig = inspect.signature(method)
            if "num_examples" in sig.parameters:
                results.append(method(tokenizer, model, batch_size, num_examples=num_examples))
            else:
                results.append(method(tokenizer, model, batch_size))
                
        return results
