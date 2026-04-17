"""
profiling/llm_profiler.py — Deep LLM structural profiling domain service.

Provides: parameter counting, architecture analysis, attention layer analysis,
connection analysis (fx / hooks / basic fallback), and memory estimation.

This is a domain service class — it is framework-agnostic and has no
dependency on ModelForge ports.  The ComprehensiveProfilingAdapter wires
it to the pipeline.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from llm_flux.profiling.types.llm import (
    AnalyzeConnections,
    ArchitectureInfo,
    AttentionLayerAnalysisInfo,
    ConnectionAnalysisInfo,
    EstimateMemory,
    LayerConnectionInfo,
    LLMInfo,
    MemoryEstimate,
    MemoryEstimationInfo,
    ModelSummary,
    ParameterInfo,
    PrecisionType,
)


class LLMProfiler:
    """
    Deep profiler for PyTorch LLMs.

    Combines static analysis (parameter counting, architecture) with optional
    dynamic analysis (connection tracing via torch.fx or hooks) and
    theoretical memory estimation.
    """

    PRECISION_BYTES: dict[PrecisionType, float] = {
        PrecisionType.FP32: 4.0,
        PrecisionType.FP16: 2.0,
        PrecisionType.BFLOAT16: 2.0,
        PrecisionType.INT8: 1.0,
        PrecisionType.INT4: 0.5,
    }

    def __init__(self, model: nn.Module, tokenizer: Any, verbose: bool = True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.device = self._detect_device()
        self.profile_data: LLMInfo | None = None

    # ── Public entry points ───────────────────────────────────────────────────

    def profile_complete(
        self,
        analyze_connections: AnalyzeConnections | None = None,
        estimate_memory: EstimateMemory | None = None,
    ) -> LLMInfo:
        """Run all static analyses and optionally dynamic ones."""
        if self.verbose:
            logger.info("📊 Counting parameters...")
        parameters_info = self.count_parameters()

        if self.verbose:
            logger.info("🏗️  Analyzing architecture...")
        architecture_info = self.analyze_architecture()

        if self.verbose:
            logger.info("🎯 Analyzing attention layers...")
        attention_layer_info = self.analyze_attention_layers()

        if analyze_connections:
            conn = self.analyze_connections(
                input_shape=analyze_connections.input_shape,
                input_sample=analyze_connections.sample_input,
            )
            architecture_info.connections = conn

        memory_info = None
        if estimate_memory:
            if self.verbose:
                logger.info("💾 Estimating memory requirements...")
            memory_info = self.estimate_memory_requirements(
                sequence_length=estimate_memory.sequence_length,
                batch_size=estimate_memory.batch_size,
            )

        self.profile_data = LLMInfo(
            architecture=architecture_info,
            attention_layers=attention_layer_info,
            parameters=parameters_info,
            summary=self.get_model_summary(),
            memory_estimation=memory_info,
        )
        if self.verbose:
            logger.info("✅ Profiling complete!")
        return self.profile_data

    def print_summary(self) -> None:
        if self.profile_data:
            print(self.profile_data.model_dump_json(indent=2))

    # ── Parameter analysis ────────────────────────────────────────────────────

    def count_parameters(self) -> ParameterInfo:
        total = trainable = non_trainable = 0
        dtype_counts: defaultdict[str, int] = defaultdict(int)
        dtype_bytes: defaultdict[str, int] = defaultdict(int)

        for param in self.model.parameters():
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
            else:
                non_trainable += n
            key = str(param.dtype)
            dtype_counts[key] += n
            dtype_bytes[key] += n * param.element_size()

        return ParameterInfo(
            total=total, trainable=trainable, non_trainable=non_trainable,
            total_millions=round(total / 1e6, 2),
            total_billions=round(total / 1e9, 3),
            by_dtype_counts=dict(dtype_counts),
            by_dtype_bytes={k: int(v) for k, v in dtype_bytes.items()},
        )

    # ── Architecture analysis ─────────────────────────────────────────────────

    def get_model_summary(self) -> ModelSummary:
        return ModelSummary(
            device=str(self.device),
            model_class=type(self.model).__name__,
            pytorch_version=torch.__version__,
        )

    def analyze_architecture(self) -> ArchitectureInfo:
        layer_types: defaultdict[str, int] = defaultdict(int)
        layer_details: list[dict[str, Any]] = []

        for name, module in self.model.named_modules():
            if list(module.children()):
                continue  # skip container modules; only leaf nodes
            layer_type = type(module).__name__
            layer_types[layer_type] += 1
            info: dict[str, Any] = {
                "name": name or layer_type,
                "type": layer_type,
                "parameters": sum(p.numel() for p in module.parameters()),
                "depth": name.count(".") if name else 0,
            }
            self._extract_layer_attributes(module, info)
            layer_details.append(info)

        max_depth = max((ld["depth"] for ld in layer_details), default=0)
        return ArchitectureInfo(
            total_layers=len(layer_details),
            layer_types_count=dict(layer_types),
            layer_details=layer_details,
            max_depth=max_depth,
        )

    def _extract_layer_attributes(self, module: nn.Module, info: dict[str, Any]) -> None:
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            info["input_size"] = module.in_features
            info["output_size"] = module.out_features
        if hasattr(module, "num_embeddings") and hasattr(module, "embedding_dim"):
            info["vocab_size"] = module.num_embeddings
            info["embedding_dim"] = module.embedding_dim
        for attr in ("num_heads", "num_attention_heads", "head_dim", "embed_dim"):
            if hasattr(module, attr):
                info[attr] = getattr(module, attr)

    # ── Attention layer analysis ──────────────────────────────────────────────

    def analyze_attention_layers(self) -> AttentionLayerAnalysisInfo:
        candidates = {
            name: mod
            for name, mod in self.model.named_modules()
            if self._looks_like_attention(mod)
        }
        leaves = self._filter_leaf_candidates(list(candidates.keys()))
        layers = [
            {
                "name": n, "type": type(candidates[n]).__name__,
                **{
                    attr: getattr(candidates[n], attr)
                    for attr in ("num_attention_heads", "num_heads", "embed_dim", "head_dim")
                    if hasattr(candidates[n], attr)
                },
            }
            for n in sorted(leaves)
        ]
        return AttentionLayerAnalysisInfo(num_attention_layers=len(layers), attention_layers=layers)

    def _looks_like_attention(self, module: nn.Module) -> bool:
        if isinstance(module, nn.MultiheadAttention):
            return True
        if any(hasattr(module, a) for a in ("num_attention_heads", "num_heads", "head_dim")):
            return True
        cls = type(module).__name__.lower()
        return any(kw in cls for kw in ("attention", "attn", "multihead"))

    def _filter_leaf_candidates(self, names: list[str]) -> list[str]:
        return [n for n in names if not any(o != n and o.startswith(n + ".") for o in names)]

    # ── Memory estimation ─────────────────────────────────────────────────────

    def estimate_memory_requirements(
        self,
        sequence_length: int = 2048,
        batch_size: int = 1,
        include_kv_cache: bool = True,
        include_activations: bool = True,
        gradient_accumulation_steps: int = 1,
        optimizer_type: str = "adamw",
    ) -> MemoryEstimationInfo:
        param_info = self.count_parameters()
        arch_info = self.analyze_architecture()
        attention_info = self.analyze_attention_layers()
        total_params = param_info.total

        estimates: dict[str, MemoryEstimate] = {}
        for precision in PrecisionType:
            bpp = self.PRECISION_BYTES[precision]
            weights_mb = (total_params * bpp) / (1024 ** 2)

            kv_mb = (
                self._estimate_kv_cache_memory(attention_info, sequence_length, batch_size, bpp)
                if include_kv_cache and attention_info.num_attention_layers > 0
                else None
            )
            act_mb = (
                self._estimate_activation_memory(arch_info, sequence_length, batch_size, bpp)
                if include_activations
                else None
            )
            training = self._estimate_training_memory(
                total_params, bpp, optimizer_type, gradient_accumulation_steps, precision
            )

            inference_mb = weights_mb + (kv_mb or 0) + (act_mb or 0)
            total_inference_mb = inference_mb * 1.2

            training_mb = weights_mb + (act_mb or 0) + training["total"]
            total_training_mb = training_mb * 1.3

            estimates[precision.value] = MemoryEstimate(
                precision=precision.value,
                bytes_per_parameter=bpp,
                model_weights_mb=round(weights_mb, 2),
                kv_cache_mb=round(kv_mb, 2) if kv_mb is not None else None,
                activation_memory_mb=round(act_mb, 2) if act_mb is not None else None,
                gradient_memory_mb=round(training["gradients"], 2),
                optimizer_memory_mb=round(training["optimizer"], 2),
                training_memory_mb=round(training["total"], 2),
                total_memory_mb=round(total_training_mb, 2),
                total_memory_gb=round(total_training_mb / 1024, 3),
                total_inference_memory_mb=round(total_inference_mb, 2),
            )
        return MemoryEstimationInfo(estimates=estimates, base_parameters=total_params)

    def _estimate_training_memory(
        self, total: int, bpp: float, optimizer: str, grad_accum: int, precision: PrecisionType
    ) -> dict[str, float]:
        grad_mb = (total * bpp) / (1024 ** 2) * grad_accum
        opt_bytes = {"adamw": 8, "sgd": bpp, "adafactor": 4}.get(optimizer.lower(), 8)
        opt_mb = (total * opt_bytes) / (1024 ** 2)
        if precision in (PrecisionType.FP16, PrecisionType.BFLOAT16):
            opt_mb += (total * 4) / (1024 ** 2)  # master weights
        return {"gradients": grad_mb, "optimizer": opt_mb, "total": grad_mb + opt_mb}

    def _estimate_kv_cache_memory(
        self, attn: AttentionLayerAnalysisInfo, seq: int, batch: int, bpp: float
    ) -> float:
        total = 0.0
        for layer in attn.attention_layers:
            heads = layer.get("num_attention_heads") or layer.get("num_heads") or 12
            head_dim = layer.get("head_dim") or (layer.get("embed_dim", 768) // heads)
            total += 2 * batch * heads * seq * head_dim * bpp
        return total / (1024 ** 2)

    def _estimate_activation_memory(
        self, arch: ArchitectureInfo, seq: int, batch: int, bpp: float
    ) -> float:
        embed_dim = next(
            (ld["embedding_dim"] for ld in arch.layer_details if ld.get("embedding_dim")), 768
        )
        attn_layers = sum(
            1 for ld in arch.layer_details if "attention" in ld.get("type", "").lower()
        ) or 12
        return (batch * seq * embed_dim * attn_layers * 4 * bpp) / (1024 ** 2)

    # ── Connection analysis ───────────────────────────────────────────────────

    def analyze_connections(
        self,
        sample_input: Any | None = None,
        input_shape: tuple[int, ...] | None = None,
        input_sample: Callable[[], Any] | None = None,
    ) -> ConnectionAnalysisInfo:
        if self.verbose:
            logger.info("🔗 Analyzing layer connections...")

        inp = sample_input or self._prepare_inference_input(input_shape, input_sample)

        result = (
            self._analyze_connections_fx(inp)
            or self._analyze_connections_hooks(inp)
            or self._analyze_connections_basic()
        )
        return result

    def _prepare_inference_input(
        self,
        input_shape: tuple[int, ...] | None,
        input_sample: Callable[[], Any] | None,
    ) -> Any:
        if input_sample is not None:
            return self._move_to_device(input_sample())
        if input_shape is None:
            raise ValueError("Provide input_shape or input_sample for connection analysis.")
        if len(input_shape) == 2:
            b, s = input_shape
            return torch.randint(0, 1000, (b, s), dtype=torch.long, device=self.device)
        return torch.randn(input_shape, device=self.device)

    def _move_to_device(self, inp: Any) -> Any:
        if isinstance(inp, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inp.items()}
        if isinstance(inp, (tuple, list)):
            return tuple(v.to(self.device) if torch.is_tensor(v) else v for v in inp)
        if torch.is_tensor(inp):
            return inp.to(self.device)
        return inp

    def _forward_pass(self, inp: Any) -> Any:
        if isinstance(inp, dict):
            return self.model(**inp)
        if isinstance(inp, (tuple, list)):
            return self.model(*inp)
        return self.model(inp)

    def _build_empty_graph(self) -> dict[str, LayerConnectionInfo]:
        return {
            (name or type(m).__name__): LayerConnectionInfo(
                name=name or type(m).__name__,
                type=type(m).__name__,
                parameters=sum(p.numel() for p in m.parameters()),
                depth=name.count(".") if name else 0,
            )
            for name, m in self.model.named_modules()
            if not list(m.children())
        }

    def _analyze_connections_fx(self, inp: Any) -> ConnectionAnalysisInfo | None:
        try:
            import torch.fx
            traced = torch.fx.symbolic_trace(self.model)
            graph: dict[str, LayerConnectionInfo] = {}
            node_to_name: dict[Any, str] = {}

            for node in traced.graph.nodes:
                if node.op != "call_module":
                    continue
                mod = dict(traced.named_modules())[node.target]
                info = LayerConnectionInfo(
                    name=node.target, type=type(mod).__name__,
                    parameters=sum(p.numel() for p in mod.parameters()),
                    depth=node.target.count("."),
                )
                self._extract_layer_attributes(mod, info.__dict__)
                graph[node.target] = info
                node_to_name[node] = node.target

            for node in traced.graph.nodes:
                if node.op != "call_module" or node.target not in graph:
                    continue
                cur = graph[node.target]
                for arg in node.args:
                    if hasattr(arg, "target") and arg.target in graph:
                        cur.input_layers.append(arg.target)
                        graph[arg.target].output_layers.append(node.target)

            self._attach_shapes_via_hooks(inp, graph)
            return self._analyze_connection_patterns(graph, "torch_fx")
        except Exception as e:
            if self.verbose:
                logger.debug(f"torch.fx analysis failed: {e}")
            return None

    def _analyze_connections_hooks(self, inp: Any) -> ConnectionAnalysisInfo | None:
        try:
            graph = self._build_empty_graph()
            execution_order: list[str] = []
            shape_info: dict[str, dict[str, Any]] = {}
            hooks = []

            def make_hook(layer_name: str):
                def hook(module, input, output):
                    execution_order.append(layer_name)
                    i0 = input[0] if isinstance(input, (tuple, list)) and input else None
                    shape_info[layer_name] = {
                        "input_shape": list(i0.shape) if torch.is_tensor(i0) else None,
                        "output_shape": list(output.shape) if torch.is_tensor(output) else None,
                    }
                return hook

            for name, mod in self.model.named_modules():
                if not list(mod.children()):
                    key = name or type(mod).__name__
                    if key in graph:
                        hooks.append(mod.register_forward_hook(make_hook(key)))

            with torch.no_grad():
                self._forward_pass(inp)
            for h in hooks:
                h.remove()

            for i, name in enumerate(execution_order):
                if name not in graph:
                    continue
                graph[name].input_shape = shape_info.get(name, {}).get("input_shape")
                graph[name].output_shape = shape_info.get(name, {}).get("output_shape")
                if i > 0:
                    prev = execution_order[i - 1]
                    if prev in graph and prev not in graph[name].input_layers:
                        graph[name].input_layers.append(prev)
                        graph[prev].output_layers.append(name)

            return self._analyze_connection_patterns(graph, "hooks")
        except Exception as e:
            if self.verbose:
                logger.debug(f"Hooks analysis failed: {e}")
            return None

    def _analyze_connections_basic(self) -> ConnectionAnalysisInfo:
        return self._analyze_connection_patterns(self._build_empty_graph(), "basic")

    def _attach_shapes_via_hooks(
        self, inp: Any, graph: dict[str, LayerConnectionInfo]
    ) -> None:
        shape_info: dict[str, dict[str, list[int] | None]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, input, output):
                i0 = input[0] if isinstance(input, (tuple, list)) and input else None
                shape_info[name] = {
                    "input_shape": list(i0.shape) if torch.is_tensor(i0) else None,
                    "output_shape": list(output.shape) if torch.is_tensor(output) else None,
                }
            return hook

        for name, mod in self.model.named_modules():
            if name in graph:
                hooks.append(mod.register_forward_hook(make_hook(name)))
        try:
            with torch.no_grad():
                self._forward_pass(inp)
        finally:
            for h in hooks:
                h.remove()

        for name, shapes in shape_info.items():
            if name in graph:
                graph[name].input_shape = shapes["input_shape"]
                graph[name].output_shape = shapes["output_shape"]

    def _analyze_connection_patterns(
        self, graph: dict[str, LayerConnectionInfo], method: str
    ) -> ConnectionAnalysisInfo:
        return ConnectionAnalysisInfo(
            total_connections=sum(len(v.output_layers) for v in graph.values()),
            connection_graph=graph,
            analysis_method=method,
            has_skip_connections=any(len(v.input_layers) > 1 for v in graph.values()),
            max_fan_in=max((len(v.input_layers) for v in graph.values()), default=0),
            max_fan_out=max((len(v.output_layers) for v in graph.values()), default=0),
        )

    # ── Device detection ──────────────────────────────────────────────────────

    def _detect_device(self) -> torch.device:
        p = next(iter(self.model.parameters()), None)
        if p is not None:
            return p.device
        b = next(iter(self.model.buffers()), None)
        if b is not None:
            return b.device
        return torch.device("cpu")
