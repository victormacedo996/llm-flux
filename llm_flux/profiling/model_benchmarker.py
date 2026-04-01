"""Model benchmarker for perplexity and task metrics across common LLM datasets."""
from __future__ import annotations

import inspect
import math
import re
import string
from typing import Any, Callable, Dict, List, Literal, Set, Sequence, get_args

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
from llm_flux.profiling.types.stats import DescriptiveStats


PERPLEXITY_TESTS = Literal["lambada", "wikitext_103_v1", "c4"]
ACCURACY_TESTS = Literal[
    "mmlu_pro",
    "gsm8k",
    "math_500",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "piqa",
    "sciq",
    "big_bench_hard",
    "truthfulqa_mc2",
    "squad",
    "humaneval_pass1",
    "mbpp_pass1",
]


_DEFAULT_MAX_NEW_TOKENS = 64
_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)

AVAILABLE_TESTS: List[str] = list(get_args(PERPLEXITY_TESTS) + get_args(ACCURACY_TESTS))

TestResult = PerplexityTestResult | AccuracyTestResult


class ModelPerformanceBenchmarker:
    """Runs a configurable set of model benchmarks and returns structured results."""

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
            all_perplexities=all_perplexities,
            mean_perplexity=mean,
            stats=DescriptiveStats.from_values(all_perplexities),
        )

    @staticmethod
    def _load_dataset(
        path: str,
        split: str,
        subset: str | None = None,
    ) -> Dataset:
        from datasets import load_dataset

        kwargs: dict[str, Any] = {"path": path, "split": split}
        if subset:
            kwargs["name"] = subset
        try:
            return load_dataset(**kwargs)
        except Exception:
            fallback_splits = ["validation", "test", "train"]
            for candidate in fallback_splits:
                if candidate == split:
                    continue
                try:
                    kwargs["split"] = candidate
                    return load_dataset(**kwargs)
                except Exception:
                    continue
            raise

    @staticmethod
    def _first_n(dataset: Dataset, num_examples: int | None) -> Dataset:
        if num_examples is None:
            return dataset
        return dataset.select(range(min(num_examples, len(dataset))))

    @staticmethod
    def _normalize_text(value: str) -> str:
        lowered = value.lower().strip()
        no_punct = lowered.translate(_PUNCT_TRANSLATION)
        return " ".join(no_punct.split())

    @classmethod
    def _token_f1(cls, prediction: str, reference: str) -> float:
        pred_tokens = cls._normalize_text(prediction).split()
        ref_tokens = cls._normalize_text(reference).split()
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        common: dict[str, int] = {}
        for t in pred_tokens:
            common[t] = common.get(t, 0) + 1
        overlap = 0
        for t in ref_tokens:
            count = common.get(t, 0)
            if count > 0:
                overlap += 1
                common[t] = count - 1
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _extract_last_number(text: str) -> str:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return numbers[-1] if numbers else ""

    def _generate_text(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        prompt: str,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    ) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(
            model.device
        )
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _choice_logprob(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        prompt: str,
        choice: str,
    ) -> float:
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(
            model.device
        )["input_ids"]
        full_ids = tokenizer(
            prompt + choice,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(model.device)["input_ids"]

        with torch.no_grad():
            logits = model(full_ids).logits[:, :-1, :]

        targets = full_ids[:, 1:]
        prompt_len = max(1, prompt_ids.shape[1] - 1)
        if prompt_len >= targets.shape[1]:
            return float("-inf")

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        choice_region = target_log_probs[:, prompt_len:]
        return float(choice_region.sum().item())

    def _multiple_choice_accuracy(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        prompt_builder: Callable[[dict[str, Any]], str],
        choices_builder: Callable[[dict[str, Any]], Sequence[str]],
        gold_index_builder: Callable[[dict[str, Any]], int],
    ) -> tuple[float, int, list[float]]:
        correct = 0
        total = 0
        per_example_scores: list[float] = []
        for row in dataset:
            choices = list(choices_builder(row))
            if len(choices) < 2:
                continue
            try:
                gold_idx = gold_index_builder(row)
            except Exception:
                continue
            if gold_idx < 0 or gold_idx >= len(choices):
                continue

            prompt = prompt_builder(row)
            scores = [self._choice_logprob(tokenizer, model, prompt, f" {c}") for c in choices]
            pred_idx = max(range(len(scores)), key=lambda idx: scores[idx])
            total += 1
            hit = 1.0 if pred_idx == gold_idx else 0.0
            per_example_scores.append(hit)
            if hit:
                correct += 1

        return (correct / total if total else 0.0, total, per_example_scores)

    def _exact_match_generation_accuracy(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        prompt_builder: Callable[[dict[str, Any]], str],
        reference_builder: Callable[[dict[str, Any]], str],
        parser: Callable[[str], str] | None = None,
    ) -> tuple[float, int, list[float]]:
        parser_fn = parser or self._normalize_text
        correct = 0
        total = 0
        per_example_scores: list[float] = []
        for row in dataset:
            prompt = prompt_builder(row)
            reference = parser_fn(reference_builder(row))
            if not reference:
                continue
            prediction = parser_fn(self._generate_text(tokenizer, model, prompt))
            total += 1
            hit = 1.0 if prediction == reference else 0.0
            per_example_scores.append(hit)
            if hit:
                correct += 1
        return (correct / total if total else 0.0, total, per_example_scores)

    def _squad_f1(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
    ) -> tuple[float, int, list[float]]:
        scores: list[float] = []
        for row in dataset:
            answers = row.get("answers", {})
            gold_texts = answers.get("text", []) if isinstance(answers, dict) else []
            if not gold_texts:
                continue
            context = str(row.get("context", ""))
            question = str(row.get("question", ""))
            prompt = (
                "Read the context and answer the question with a short span.\n"
                f"Context: {context}\n"
                f"Question: {question}\n"
                "Answer:"
            )
            prediction = self._generate_text(tokenizer, model, prompt)
            f1 = max(self._token_f1(prediction, gold) for gold in gold_texts)
            scores.append(f1)
        if not scores:
            return 0.0, 0, []
        return float(sum(scores) / len(scores)), len(scores), scores

    def _truthfulqa_mc2(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
    ) -> tuple[float, int, list[float]]:
        example_scores: list[float] = []

        for row in dataset:
            question = str(row.get("question", "")).strip()
            if not question:
                continue

            true_answers = row.get("correct_answers") or row.get("mc2_true_answers") or []
            false_answers = row.get("incorrect_answers") or row.get("mc2_false_answers") or []

            if not true_answers and "mc2_targets" in row and isinstance(row["mc2_targets"], dict):
                mc2_targets: dict[str, Any] = row["mc2_targets"]
                true_answers = [k for k, v in mc2_targets.items() if int(v) == 1]
                false_answers = [k for k, v in mc2_targets.items() if int(v) == 0]

            if isinstance(true_answers, str):
                true_answers = [true_answers]
            if isinstance(false_answers, str):
                false_answers = [false_answers]

            if not true_answers or not false_answers:
                continue

            prompt = f"Question: {question}\nAnswer:"
            true_scores = [
                self._choice_logprob(tokenizer, model, prompt, f" {str(ans).strip()}")
                for ans in true_answers
            ]
            false_scores = [
                self._choice_logprob(tokenizer, model, prompt, f" {str(ans).strip()}")
                for ans in false_answers
            ]

            all_scores = true_scores + false_scores
            max_logp = max(all_scores)
            true_mass = sum(math.exp(s - max_logp) for s in true_scores)
            total_mass = true_mass + sum(math.exp(s - max_logp) for s in false_scores)
            if total_mass <= 0:
                continue
            example_scores.append(true_mass / total_mass)

        if not example_scores:
            return 0.0, 0, []
        return float(sum(example_scores) / len(example_scores)), len(example_scores), example_scores

    @staticmethod
    def _label_to_index(label: str, num_choices: int) -> int:
        value = label.strip()
        if value.isdigit():
            idx = int(value)
            if 0 <= idx < num_choices:
                return idx
            if 1 <= idx <= num_choices:
                return idx - 1
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        upper = value.upper()
        if upper in letters[:num_choices]:
            return letters.index(upper)
        raise ValueError(f"Cannot parse choice label '{label}' for {num_choices} choices")

    @staticmethod
    def _eval_pass_at_1_python(
        generated_code: str,
        tests: list[str],
    ) -> bool:
        namespace: dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(generated_code, namespace, namespace)
            for t in tests:
                exec(t, namespace, namespace)
            return True
        except Exception:
            return False

    def _humaneval_pass_at_1(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
    ) -> tuple[float, int, list[float]]:
        passed = 0
        total = 0
        per_example_scores: list[float] = []
        for row in dataset:
            prompt = str(row.get("prompt", ""))
            test_code = str(row.get("test", "")).strip()
            entry_point = str(row.get("entry_point", "")).strip()
            if not prompt or not test_code or not entry_point:
                continue

            completion = self._generate_text(tokenizer, model, prompt, max_new_tokens=256)
            candidate = f"{prompt}{completion}"
            runnable_tests = [test_code, f"check({entry_point})"]
            total += 1
            hit = 1.0 if self._eval_pass_at_1_python(candidate, runnable_tests) else 0.0
            per_example_scores.append(hit)
            if hit:
                passed += 1

        return (passed / total if total else 0.0, total, per_example_scores)

    def _mbpp_pass_at_1(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
    ) -> tuple[float, int, list[float]]:
        passed = 0
        total = 0
        per_example_scores: list[float] = []
        for row in dataset:
            problem = str(row.get("text", "")).strip()
            tests = row.get("test_list")
            if not problem or not isinstance(tests, list) or not tests:
                continue

            prompt = (
                "Write Python code that solves the following task. "
                "Return only Python code.\n"
                f"Task: {problem}\n"
            )
            generated = self._generate_text(tokenizer, model, prompt, max_new_tokens=256)
            total += 1
            hit = 1.0 if self._eval_pass_at_1_python(generated, [str(t) for t in tests]) else 0.0
            per_example_scores.append(hit)
            if hit:
                passed += 1
        return (passed / total if total else 0.0, total, per_example_scores)

    # ── Built-in tests ───────────────────────────────────────────────────────

    def lambada(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> PerplexityTestResult:
        """Perplexity on the LAMBADA dataset (last-word prediction)."""
        dataset = self._load_dataset(path="cimec/lambada", split="test")
        dataset = self._first_n(dataset, num_examples)
        result = self.evaluate_perplexity(
            dataset=dataset,
            text_column="text",
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
        )
        return PerplexityTestResult(result=result, test_name="lambada")

    def wikitext_103_v1(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> PerplexityTestResult:
        dataset = self._load_dataset(path="Salesforce/wikitext", subset="wikitext-103-v1", split="test")
        dataset = self._first_n(dataset, num_examples)
        result = self.evaluate_perplexity(
            dataset=dataset,
            text_column="text",
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
        )
        return PerplexityTestResult(result=result, test_name="wikitext_103_v1")

    def c4(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> PerplexityTestResult:
        dataset = self._load_dataset(path="allenai/c4", subset="en", split="validation")
        dataset = self._first_n(dataset, num_examples)
        result = self.evaluate_perplexity(
            dataset=dataset,
            text_column="text",
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
        )
        return PerplexityTestResult(result=result, test_name="c4")

    def mmlu_pro(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="TIGER-Lab/MMLU-Pro", split="test")
        dataset = self._first_n(dataset, num_examples)

        def prompt_builder(row: dict[str, Any]) -> str:
            q = str(row.get("question", ""))
            options = row.get("options", row.get("choices", []))
            lines = [f"{chr(65+i)}. {str(c)}" for i, c in enumerate(options)]
            return f"Question: {q}\n" + "\n".join(lines) + "\nAnswer:"

        def choices_builder(row: dict[str, Any]) -> Sequence[str]:
            return [str(c) for c in row.get("options", row.get("choices", []))]

        def gold_builder(row: dict[str, Any]) -> int:
            answer = row.get("answer", row.get("answer_index", 0))
            if isinstance(answer, int):
                return answer
            return self._label_to_index(str(answer), len(choices_builder(row)))

        score, used, per_example = self._multiple_choice_accuracy(
            dataset, tokenizer, model, prompt_builder, choices_builder, gold_builder
        )
        return AccuracyTestResult(
            test_name="mmlu_pro",
            metric_name="accuracy",
            accuracy=score,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
        )

    def gsm8k(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="openai/gsm8k", subset="main", split="test")
        dataset = self._first_n(dataset, num_examples)
        score, used, per_example = self._exact_match_generation_accuracy(
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            prompt_builder=lambda r: (
                "Solve the following math word problem and provide only the final number.\n"
                f"Problem: {str(r.get('question', ''))}\nFinal answer:"
            ),
            reference_builder=lambda r: str(r.get("answer", "")),
            parser=self._extract_last_number,
        )
        return AccuracyTestResult(
            test_name="gsm8k",
            metric_name="number_extraction_accuracy",
            accuracy=score,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
        )

    def math_500(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="HuggingFaceH4/MATH-500", split="test")
        dataset = self._first_n(dataset, num_examples)
        score, used, per_example = self._exact_match_generation_accuracy(
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            prompt_builder=lambda r: (
                "Solve the problem and output only the final answer.\n"
                f"Problem: {str(r.get('problem', r.get('question', '')))}\nAnswer:"
            ),
            reference_builder=lambda r: str(r.get("answer", r.get("solution", ""))),
            parser=self._extract_last_number,
        )
        return AccuracyTestResult(
            test_name="math_500",
            metric_name="accuracy",
            accuracy=score,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
        )

    def arc_challenge(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="allenai/ai2_arc", subset="ARC-Challenge", split="test")
        dataset = self._first_n(dataset, num_examples)

        def _choices(row: dict[str, Any]) -> list[str]:
            choices = row.get("choices", {})
            text = choices.get("text", []) if isinstance(choices, dict) else []
            return [str(c) for c in text]

        score, used, per_example = self._multiple_choice_accuracy(
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            prompt_builder=lambda r: (
                f"Question: {str(r.get('question', ''))}\n"
                + "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(_choices(r)))
                + "\nAnswer:"
            ),
            choices_builder=_choices,
            gold_index_builder=lambda r: self._label_to_index(
                str(r.get("answerKey", "A")),
                len(_choices(r)),
            ),
        )
        return AccuracyTestResult(
            test_name="arc_challenge",
            metric_name="accuracy",
            accuracy=score,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
        )

    def hellaswag(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="Rowan/hellaswag", split="validation")
        dataset = self._first_n(dataset, num_examples)
        score, used, per_example = self._multiple_choice_accuracy(
            dataset,
            tokenizer,
            model,
            prompt_builder=lambda r: (
                f"Context: {str(r.get('ctx', r.get('context', '')))}\n"
                "Choose the best ending.\nAnswer:"
            ),
            choices_builder=lambda r: [str(c) for c in r.get("endings", [])],
            gold_index_builder=lambda r: int(r.get("label", 0)),
        )
        return AccuracyTestResult(
            test_name="hellaswag",
            metric_name="accuracy",
            accuracy=score,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
        )

    def winogrande(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="allenai/winogrande", subset="winogrande_xl", split="validation")
        dataset = self._first_n(dataset, num_examples)
        score, used, per_example = self._multiple_choice_accuracy(
            dataset,
            tokenizer,
            model,
            prompt_builder=lambda r: (
                "Fill in the blank with the best option.\n"
                f"Sentence: {str(r.get('sentence', ''))}\nAnswer:"
            ),
            choices_builder=lambda r: [str(r.get("option1", "")), str(r.get("option2", ""))],
            gold_index_builder=lambda r: int(str(r.get("answer", "1"))) - 1,
        )
        return AccuracyTestResult(
            test_name="winogrande",
            metric_name="accuracy",
            accuracy=score,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
        )

    def piqa(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="ybisk/piqa", split="validation")
        dataset = self._first_n(dataset, num_examples)
        score, used, per_example = self._multiple_choice_accuracy(
            dataset,
            tokenizer,
            model,
            prompt_builder=lambda r: f"Goal: {str(r.get('goal', ''))}\nBest solution:",
            choices_builder=lambda r: [str(r.get("sol1", "")), str(r.get("sol2", ""))],
            gold_index_builder=lambda r: int(r.get("label", 0)),
        )
        return AccuracyTestResult(
            test_name="piqa",
            metric_name="accuracy",
            accuracy=score,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
        )

    def sciq(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="allenai/sciq", split="test")
        dataset = self._first_n(dataset, num_examples)

        def choices(row: dict[str, Any]) -> list[str]:
            return [
                str(row.get("distractor1", "")),
                str(row.get("distractor2", "")),
                str(row.get("distractor3", "")),
                str(row.get("correct_answer", "")),
            ]

        score, used, per_example = self._multiple_choice_accuracy(
            dataset,
            tokenizer,
            model,
            prompt_builder=lambda r: f"Question: {str(r.get('question', ''))}\nAnswer:",
            choices_builder=choices,
            gold_index_builder=lambda r: 3,
        )
        return AccuracyTestResult(
            test_name="sciq",
            metric_name="accuracy",
            accuracy=score,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
        )

    def big_bench_hard(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="maveriq/bigbenchhard", split="test")
        dataset = self._first_n(dataset, num_examples)
        score, used, per_example = self._exact_match_generation_accuracy(
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            prompt_builder=lambda r: (
                f"Task: {str(r.get('task', ''))}\n"
                f"Question: {str(r.get('input', r.get('question', '')))}\nAnswer:"
            ),
            reference_builder=lambda r: str(
                r.get("target", r.get("answer", r.get("output", "")))
            ),
        )
        return AccuracyTestResult(
            test_name="big_bench_hard",
            metric_name="accuracy",
            accuracy=score,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
        )

    def truthfulqa_mc2(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="domenicrosati/TruthfulQA", split="validation")
        dataset = self._first_n(dataset, num_examples)
        score, used, per_example = self._truthfulqa_mc2(dataset, tokenizer, model)
        return AccuracyTestResult(
            test_name="truthfulqa_mc2",
            metric_name="mc2",
            accuracy=score,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
            details={
                "definition": (
                    "Normalized probability mass assigned to true answers "
                    "among true+false reference sets."
                )
            },
        )

    def squad(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="rajpurkar/squad", split="validation")
        dataset = self._first_n(dataset, num_examples)
        f1, used, per_example = self._squad_f1(dataset, tokenizer, model)
        return AccuracyTestResult(
            test_name="squad",
            metric_name="f1",
            accuracy=f1,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
        )

    def humaneval_pass1(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="openai/openai_humaneval", split="test")
        dataset = self._first_n(dataset, num_examples)
        score, used, per_example = self._humaneval_pass_at_1(dataset, tokenizer, model)
        return AccuracyTestResult(
            test_name="humaneval_pass1",
            metric_name="pass@1",
            accuracy=score,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
        )

    def mbpp_pass1(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        batch_size: int = 16,
        num_examples: int | None = None,
    ) -> AccuracyTestResult:
        del batch_size
        dataset = self._load_dataset(path="Muennighoff/mbpp", split="test")
        dataset = self._first_n(dataset, num_examples)
        score, used, per_example = self._mbpp_pass_at_1(dataset, tokenizer, model)
        return AccuracyTestResult(
            test_name="mbpp_pass1",
            metric_name="pass@1",
            accuracy=score,
            num_examples=used,
            per_example_scores=per_example,
            stats=DescriptiveStats.from_values(per_example) if per_example else None,
        )

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

            sig = inspect.signature(method)
            if "num_examples" in sig.parameters:
                results.append(method(tokenizer, model, batch_size, num_examples=num_examples))
            else:
                results.append(method(tokenizer, model, batch_size))

        return results
