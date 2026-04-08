from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from uuid import uuid4

import networkx as nx
import torch

from .graphrag import GraphTaskPlan
from .discovery import ObservedInteractionFeature, discover_interaction_features
from .models import DEFAULT_MAX_NEW_TOKENS, get_last_token_hidden, generate
from .steering import generate_with_decaying_steering, generate_with_steering

UNKNOWN_CONTROLLER_SORT_VALUE = float("-inf")
RUNTIME_ARTIFACT_FORMAT_VERSION = 1
MIN_TRUNCATE_LENGTH = 1
TRUNCATION_SUFFIX = "..."
SVG_MIN_WIDTH = 960
SVG_WIDTH_PER_NODE = 200
SVG_MIN_HEIGHT = 720
SVG_HEIGHT_PER_NODE = 120
SVG_MAX_WIDTH = 2400
SVG_MAX_HEIGHT = 1800
SCALE_EPSILON = 1e-10


def _require_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _coerce_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _coerce_optional_non_negative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    # Reject bool explicitly so True/False do not silently become 1/0 for schedules.
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer when provided.")
    if isinstance(value, int):
        coerced = value
    else:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a non-negative integer when provided.") from exc
        if not numeric_value.is_integer():
            raise ValueError(f"{field_name} must be a non-negative integer when provided.")
        coerced = int(numeric_value)
    if coerced < 0:
        raise ValueError(f"{field_name} must be a non-negative integer when provided.")
    return coerced


@dataclass
class PlannerDecision:
    """Planner output for routing one task through the hybrid agent."""

    task_type: str
    needs_retrieval: bool = False
    controller_id: str | None = None
    reasoning_effort: str = "medium"
    use_steering: bool = True
    allow_fallback: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task_type = _require_text(self.task_type, "task_type")
        self.reasoning_effort = _require_text(self.reasoning_effort, "reasoning_effort")
        self.metadata = _coerce_metadata(self.metadata)


@dataclass
class SteeringController:
    """One reusable steering policy built from persistent activation data."""

    controller_id: str
    feature_name: str
    layer_idx: int
    vector: torch.Tensor
    alpha: float = 1.5
    decay: float = 1.0
    max_steps: int | None = None
    task_types: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.controller_id = _require_text(self.controller_id, "controller_id")
        self.feature_name = _require_text(self.feature_name, "feature_name")
        if self.layer_idx < 0:
            raise ValueError("layer_idx must be non-negative.")
        self.vector = torch.as_tensor(self.vector, dtype=torch.float32).detach().cpu()
        self.max_steps = _coerce_optional_non_negative_int(self.max_steps, "max_steps")
        normalized_task_types = []
        for task_type in self.task_types:
            normalized_task_types.append(_require_text(task_type, "task_types"))
        self.task_types = tuple(normalized_task_types)
        self.metadata = _coerce_metadata(self.metadata)

    def scale_at_step(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be non-negative.")
        if self.max_steps is not None and step >= self.max_steps:
            return 0.0
        return float(self.alpha) * (float(self.decay) ** step)


@dataclass
class SteeringFeatureScore:
    feature_id: str
    score: float


@dataclass
class ActivationTrace:
    """A lightweight persistent trace of prompt-time steering feature scores."""

    model_name: str
    controller_id: str | None
    layer_idx: int | None
    prompt: str
    top_feature_scores: list[SteeringFeatureScore]
    output_text: str = ""
    prompt_hidden_norm: float | None = None
    observed_feature_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model_name = _require_text(self.model_name, "model_name")
        self.prompt = _require_text(self.prompt, "prompt")
        self.output_text = str(self.output_text or "")
        self.observed_feature_ids = [str(feature_id) for feature_id in self.observed_feature_ids]
        self.metadata = _coerce_metadata(self.metadata)


@dataclass
class ExecutorResult:
    prompt: str
    output_text: str
    controller_id: str | None
    activation_trace: ActivationTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prompt = _require_text(self.prompt, "prompt")
        self.output_text = _require_text(self.output_text, "output_text")
        self.metadata = _coerce_metadata(self.metadata)


@dataclass
class VerifierResult:
    passed: bool
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = float(self.confidence)
        self.issues = [str(issue) for issue in self.issues]
        self.metadata = _coerce_metadata(self.metadata)


@dataclass
class HybridAgentRun:
    task: str
    plan: PlannerDecision
    context: list[str]
    draft: ExecutorResult
    verdict: VerifierResult
    selected_controller_id: str | None
    fallback_used: bool = False
    path_context: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task = _require_text(self.task, "task")
        self.context = [str(item) for item in self.context]
        self.metadata = _coerce_metadata(self.metadata)


def _coerce_controller_payloads(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "feature_vectors" in payload:
        return list(payload["feature_vectors"])
    if "controllers" in payload:
        return list(payload["controllers"])
    raise ValueError("Expected a payload with either 'feature_vectors' or 'controllers'.")


def load_steering_controllers(
    input_path: str | Path,
    default_alpha: float = 1.5,
    default_decay: float = 1.0,
    task_types_by_controller: Mapping[str, Sequence[str]] | None = None,
) -> list[SteeringController]:
    """Load reusable steering controllers from persisted discovered feature vectors."""
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    task_types_by_controller = task_types_by_controller or {}
    controllers = []
    for entry in _coerce_controller_payloads(payload):
        controller_id_value = entry.get("controller_id", entry.get("name"))
        feature_name_value = entry.get("feature_name", entry.get("name"))
        if controller_id_value is None or feature_name_value is None:
            raise ValueError(
                "Each persisted controller entry must include 'controller_id' or 'name'."
            )
        controller_id = str(controller_id_value)
        feature_name = str(feature_name_value)
        task_types = tuple(task_types_by_controller.get(controller_id, ()))
        controllers.append(
            SteeringController(
                controller_id=controller_id,
                feature_name=feature_name,
                layer_idx=int(entry["layer_idx"]),
                vector=torch.tensor(entry["vector"], dtype=torch.float32),
                alpha=float(entry.get("alpha", default_alpha)),
                decay=float(entry.get("decay", default_decay)),
                max_steps=entry.get("max_steps"),
                task_types=task_types,
                metadata={
                    "category": entry.get("category"),
                    "summary": entry.get("summary"),
                    "model_name": entry.get("model_name"),
                    **dict(entry.get("metadata") or {}),
                },
            )
        )
    return controllers


def build_executor_prompt(task: str, context: Sequence[str], plan: PlannerDecision) -> str:
    """Compose a compact executor prompt from planner output plus retrieved context."""
    sections = [f"Task type: {plan.task_type}", f"Reasoning effort: {plan.reasoning_effort}"]
    if context:
        sections.append("Context:\n" + "\n".join(str(item) for item in context))
    sections.append("Task:\n" + _require_text(task, "task"))
    return "\n\n".join(sections)


def collect_controller_trace(
    prompt: str,
    controllers: Sequence[SteeringController],
    model,
    tokenizer,
    device: str | torch.device,
    model_name: str,
    top_k: int = 3,
    layer_idx: int | None = None,
) -> ActivationTrace | None:
    """Score the prompt hidden state against persisted steering vectors."""
    if not controllers:
        return None
    if layer_idx is None:
        layer_indices = {controller.layer_idx for controller in controllers}
        if len(layer_indices) != 1:
            raise ValueError(
                "collect_controller_trace() requires all controllers to target the same layer; "
                f"received layers: {sorted(layer_indices)}"
            )
        trace_controllers = list(controllers)
        resolved_layer_idx = controllers[0].layer_idx
    else:
        resolved_layer_idx = layer_idx
        trace_controllers = [
            controller for controller in controllers if controller.layer_idx == resolved_layer_idx
        ]
        if not trace_controllers:
            return None
    hidden = get_last_token_hidden(prompt, resolved_layer_idx, model, tokenizer, device)
    scores = []
    skipped_controllers = []
    for controller in trace_controllers:
        if controller.vector.shape != hidden.shape:
            skipped_controllers.append(
                {
                    "controller_id": controller.controller_id,
                    "reason": "shape_mismatch",
                    "controller_shape": tuple(controller.vector.shape),
                    "hidden_shape": tuple(hidden.shape),
                }
            )
            continue
        score = float(torch.dot(hidden, controller.vector).item())
        scores.append(SteeringFeatureScore(feature_id=controller.controller_id, score=score))
    if not scores:
        return None
    top_scores = sorted(scores, key=lambda item: abs(item.score), reverse=True)[:top_k]
    return ActivationTrace(
        model_name=model_name,
        controller_id=None,
        layer_idx=resolved_layer_idx,
        prompt=prompt,
        top_feature_scores=top_scores,
        prompt_hidden_norm=float(hidden.norm().item()),
        metadata={"skipped_controllers": skipped_controllers},
    )


class SteeredExecutor:
    """Executor that optionally applies one persisted steering controller during generation."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str | torch.device = "cpu",
        model_name: str = "unknown-model",
        top_k_trace_features: int = 3,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = _require_text(model_name, "model_name")
        self.top_k_trace_features = max(int(top_k_trace_features), 1)

    def execute(
        self,
        task: str,
        plan: PlannerDecision,
        context: Sequence[str],
        controller: SteeringController | None = None,
        controllers: Sequence[SteeringController] = (),
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> ExecutorResult:
        prompt = build_executor_prompt(task, context, plan)
        trace_layer_idx = controller.layer_idx if controller is not None else None
        activation_trace = None
        controller_list = list(controllers)
        if controller_list:
            if trace_layer_idx is None:
                controller_layers = {candidate.layer_idx for candidate in controller_list}
                if len(controller_layers) == 1:
                    trace_layer_idx = next(iter(controller_layers))
            if trace_layer_idx is not None:
                activation_trace = collect_controller_trace(
                    prompt=prompt,
                    controllers=controller_list,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    model_name=self.model_name,
                    top_k=self.top_k_trace_features,
                    layer_idx=trace_layer_idx,
                )
        if activation_trace is not None:
            if controller is not None:
                activation_trace.controller_id = controller.controller_id
            else:
                activation_trace.controller_id = None

        if controller is None:
            output_text = generate(
                prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                max_new_tokens=max_new_tokens,
            )
            if activation_trace is None:
                activation_trace = ActivationTrace(
                    model_name=self.model_name,
                    controller_id=None,
                    layer_idx=trace_layer_idx,
                    prompt=prompt,
                    top_feature_scores=[],
                    output_text=output_text,
                )
            else:
                activation_trace.output_text = output_text
            return ExecutorResult(
                prompt=prompt,
                output_text=output_text,
                controller_id=None,
                activation_trace=activation_trace,
            )

        if controller.decay == 1.0 and controller.max_steps is None:
            output_text = generate_with_steering(
                prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                layer_idx=controller.layer_idx,
                steering_vector=controller.vector,
                device=self.device,
                alpha=controller.alpha,
                max_new_tokens=max_new_tokens,
            )
        else:
            output_text = generate_with_decaying_steering(
                prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                layer_idx=controller.layer_idx,
                steering_vector=controller.vector,
                device=self.device,
                alpha=controller.alpha,
                decay=controller.decay,
                max_steps=controller.max_steps,
                max_new_tokens=max_new_tokens,
            )
        if activation_trace is None:
            activation_trace = ActivationTrace(
                model_name=self.model_name,
                controller_id=controller.controller_id,
                layer_idx=controller.layer_idx,
                prompt=prompt,
                top_feature_scores=[],
                output_text=output_text,
            )
        else:
            activation_trace.output_text = output_text
        return ExecutorResult(
            prompt=prompt,
            output_text=output_text,
            controller_id=controller.controller_id,
            activation_trace=activation_trace,
        )


class InMemorySteeringMemory:
    """Minimal structured store for controllers, traces, runs, and per-task outcomes."""

    def __init__(self, controllers: Iterable[SteeringController] = ()) -> None:
        self._controllers: dict[str, SteeringController] = {}
        self.run_history: list[HybridAgentRun] = []
        self.activation_traces: list[ActivationTrace] = []
        self._observed_trace_ids: set[str] = set()
        self._dynamic_features: dict[str, dict[str, ObservedInteractionFeature]] = {}
        self._stats: dict[tuple[str, str], dict[str, int]] = {}
        self.register_controllers(controllers)

    def register_controllers(self, controllers: Iterable[SteeringController]) -> None:
        for controller in controllers:
            self._controllers[controller.controller_id] = controller

    def list_controllers(self) -> list[SteeringController]:
        return list(self._controllers.values())

    def get_controller(self, controller_id: str) -> SteeringController:
        try:
            return self._controllers[controller_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._controllers))
            raise ValueError(
                f"Unknown controller_id {controller_id!r}; choose from: {available}."
            ) from exc

    def observe_interaction(self, trace: ActivationTrace) -> list[ObservedInteractionFeature]:
        trace_id = str(trace.metadata.get("interaction_trace_id") or f"trace-{uuid4().hex}")
        trace.metadata["interaction_trace_id"] = trace_id
        if trace_id in self._observed_trace_ids:
            observed_feature_ids = trace.metadata.get(
                "observed_feature_ids",
                getattr(trace, "observed_feature_ids", []),
            )
            trace.observed_feature_ids = [str(feature_id) for feature_id in observed_feature_ids]
            model_features = self._dynamic_features.get(trace.model_name, {})
            existing_features: list[ObservedInteractionFeature] = []
            for feature_id in trace.observed_feature_ids:
                feature = model_features.get(feature_id)
                if feature is not None:
                    existing_features.append(feature)
            return existing_features
        self.activation_traces.append(trace)
        self._observed_trace_ids.add(trace_id)
        observed_features = discover_interaction_features([trace])
        stored_features: list[ObservedInteractionFeature] = []
        for feature in observed_features:
            model_features = self._dynamic_features.setdefault(feature.model_name, {})
            existing_feature = model_features.get(feature.feature_id)
            if existing_feature is None:
                model_features[feature.feature_id] = feature
                stored_features.append(feature)
                continue
            existing_feature.observation_count += feature.observation_count
            existing_feature.summary = feature.summary
            existing_feature.input_example = feature.input_example
            existing_feature.output_example = feature.output_example
            existing_feature.metadata.update(feature.metadata)
            existing_feature.metadata["latest_input_example"] = feature.input_example
            existing_feature.metadata["latest_output_example"] = feature.output_example
            stored_features.append(existing_feature)
        trace.observed_feature_ids = [feature.feature_id for feature in stored_features]
        if stored_features:
            trace.metadata["observed_feature_ids"] = list(trace.observed_feature_ids)
        return stored_features

    def list_dynamic_features(self, model_name: str | None = None) -> list[ObservedInteractionFeature]:
        if model_name is not None:
            return list(self._dynamic_features.get(model_name, {}).values())
        return [
            feature
            for features_by_name in self._dynamic_features.values()
            for feature in features_by_name.values()
        ]

    def controller_success_rate(self, task_type: str, controller_id: str) -> float | None:
        stats = self._stats.get((task_type, controller_id))
        if not stats:
            return None
        return stats["success"] / stats["total"]

    def select_controller(
        self,
        task_type: str,
        requested_controller_id: str | None = None,
    ) -> SteeringController | None:
        if requested_controller_id is not None:
            return self.get_controller(requested_controller_id)

        candidates = [
            controller
            for controller in self._controllers.values()
            if not controller.task_types or task_type in controller.task_types
        ]
        if not candidates:
            return None

        def sort_key(controller: SteeringController) -> tuple[float, str]:
            rate = self.controller_success_rate(task_type, controller.controller_id)
            return (
                rate if rate is not None else UNKNOWN_CONTROLLER_SORT_VALUE,
                controller.controller_id,
            )

        return max(candidates, key=sort_key)

    def controller_stats(self) -> list[dict[str, Any]]:
        return [
            {
                "task_type": task_type,
                "controller_id": controller_id,
                "success": values["success"],
                "total": values["total"],
                "success_rate": (values["success"] / values["total"]) if values["total"] else None,
            }
            for (task_type, controller_id), values in sorted(self._stats.items())
        ]

    def record_run(self, run: HybridAgentRun) -> None:
        self.run_history.append(run)
        if run.draft.activation_trace is not None:
            self.observe_interaction(run.draft.activation_trace)
        if run.selected_controller_id is None:
            return
        key = (run.plan.task_type, run.selected_controller_id)
        stats = self._stats.setdefault(key, {"success": 0, "total": 0})
        stats["total"] += 1
        fallback_used = getattr(run, "fallback_used", False)
        if run.verdict.passed and not fallback_used:
            stats["success"] += 1


class PlannerProtocol(Protocol):
    def plan(self, task: str, memory: InMemorySteeringMemory) -> PlannerDecision: ...


class RetrieverProtocol(Protocol):
    def retrieve(self, task: str, plan: PlannerDecision) -> Any: ...


class ToolRouterProtocol(Protocol):
    def route(self, task: str, plan: PlannerDecision, context: Sequence[str]) -> Sequence[str]: ...


class VerifierProtocol(Protocol):
    def verify(
        self,
        task: str,
        draft: ExecutorResult,
        context: Sequence[str],
        plan: PlannerDecision,
    ) -> VerifierResult: ...


def _call_component(
    component: Any,
    method_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    method = getattr(component, method_name, None)
    if callable(method):
        return method(*args, **kwargs)
    if callable(component):
        return component(*args, **kwargs)
    raise TypeError(f"Component must be callable or provide a .{method_name}(...) method.")


def _coerce_context_payload(payload: Any) -> tuple[list[str], Any | None]:
    if payload is None:
        return [], None
    to_prompt_sections = getattr(payload, "to_prompt_sections", None)
    if callable(to_prompt_sections):
        return [str(section) for section in to_prompt_sections()], payload
    if isinstance(payload, str):
        return [payload], None
    return [str(item) for item in payload], None


def _drift_score_from_verifier(verdict: VerifierResult) -> float:
    return max(0.0, 1.0 - float(verdict.confidence))


def _truncate_text(value: Any, limit: int = 72) -> str:
    normalized = " ".join(str(value).split())
    limit = max(limit, 0)
    if limit == 0:
        return ""
    if len(normalized) <= limit:
        return normalized
    if limit == MIN_TRUNCATE_LENGTH:
        return normalized[:limit]
    if limit <= len(TRUNCATION_SUFFIX):
        return normalized[:limit]
    return normalized[: limit - len(TRUNCATION_SUFFIX)].rstrip() + TRUNCATION_SUFFIX


def _serialize_task_plan(task_plan: GraphTaskPlan) -> dict[str, Any]:
    return {
        "task_id": task_plan.task_id,
        "user_query": task_plan.user_query,
        "intent_id": task_plan.intent_id,
        "intent_text": task_plan.intent_text,
        "goal_type": task_plan.goal_type,
        "priority": task_plan.priority,
        "risk_level": task_plan.risk_level,
        "active_subgoal_id": task_plan.active_subgoal.subgoal_id,
        "constraints": [
            {
                "constraint_id": constraint.constraint_id,
                "type": constraint.type,
                "value": constraint.value,
            }
            for constraint in task_plan.constraints
        ],
        "subgoals": [
            {
                "subgoal_id": subgoal.subgoal_id,
                "text": subgoal.text,
                "order": subgoal.order,
                "status": subgoal.status,
                "metadata": dict(subgoal.metadata),
            }
            for subgoal in task_plan.subgoals
        ],
        "metadata": dict(task_plan.metadata),
    }


def _serialize_feature_vector(controller: SteeringController) -> dict[str, Any]:
    metadata = dict(controller.metadata)
    return {
        "name": controller.feature_name,
        "feature_name": controller.feature_name,
        "controller_id": controller.controller_id,
        "model_name": metadata.get("model_name"),
        "category": metadata.get("category"),
        "summary": metadata.get("summary"),
        "layer_idx": controller.layer_idx,
        "vector": controller.vector.tolist(),
        "vector_norm": float(controller.vector.norm().item()),
        "vector_size": int(controller.vector.numel()),
        "positive_example_count": int(metadata.get("positive_example_count", 0)),
        "negative_example_count": int(metadata.get("negative_example_count", 0)),
        "test_case_count": int(metadata.get("test_case_count", 0)),
        "evaluation_criteria": list(metadata.get("evaluation_criteria", ())),
        "alpha": float(controller.alpha),
        "decay": float(controller.decay),
        "max_steps": controller.max_steps,
        "task_types": list(controller.task_types),
        "metadata": metadata,
    }


def _serialize_activation_trace(trace: ActivationTrace) -> dict[str, Any]:
    return {
        "model_name": trace.model_name,
        "controller_id": trace.controller_id,
        "layer_idx": trace.layer_idx,
        "prompt": trace.prompt,
        "prompt_hidden_norm": trace.prompt_hidden_norm,
        "top_feature_scores": [
            {"feature_id": score.feature_id, "score": score.score}
            for score in trace.top_feature_scores
        ],
        "metadata": dict(trace.metadata),
    }


def _serialize_verdict(verdict: VerifierResult) -> dict[str, Any]:
    return {
        "passed": bool(verdict.passed),
        "confidence": float(verdict.confidence),
        "issues": list(verdict.issues),
        "metadata": dict(verdict.metadata),
    }


def _build_runtime_discoveries_payload(memory: InMemorySteeringMemory) -> dict[str, Any]:
    controllers = sorted(memory.list_controllers(), key=lambda item: item.controller_id)
    activation_traces = [_serialize_activation_trace(trace) for trace in memory.activation_traces]
    return {
        "format_version": RUNTIME_ARTIFACT_FORMAT_VERSION,
        "run_count": len(memory.run_history),
        "feature_vector_count": len(controllers),
        "feature_vectors": [_serialize_feature_vector(controller) for controller in controllers],
        "activation_trace_count": len(activation_traces),
        "activation_traces": activation_traces,
        "controller_stats": memory.controller_stats(),
    }


def _add_graph_node(
    nodes: dict[str, dict[str, Any]],
    *,
    node_id: str,
    kind: str,
    label: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    nodes[node_id] = {
        "id": node_id,
        "kind": kind,
        "label": label,
        "metadata": dict(metadata or {}),
    }


def _add_graph_edge(
    edges: dict[tuple[str, str, str], dict[str, str]],
    *,
    source: str,
    target: str,
    edge_type: str,
) -> None:
    edges[(source, target, edge_type)] = {
        "source": source,
        "target": target,
        "type": edge_type,
    }


def _build_runtime_graph_payload(run_history: Sequence[HybridAgentRun]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, str]] = {}
    runs = []

    for index, run in enumerate(run_history, start=1):
        task_plan = GraphTaskPlan.from_task_and_plan(run.task, run.plan)
        run_id = str(run.metadata.get("graph_run_id") or f"session-run-{index}")
        run_summary = {
            "run_id": run_id,
            "task": run.task,
            "selected_controller_id": run.selected_controller_id,
            "fallback_used": bool(run.fallback_used),
            "task_plan": _serialize_task_plan(task_plan),
            "draft": {
                "prompt": run.draft.prompt,
                "output_text": run.draft.output_text,
                "controller_id": run.draft.controller_id,
                "activation_trace": (
                    _serialize_activation_trace(run.draft.activation_trace)
                    if run.draft.activation_trace is not None
                    else None
                ),
                "metadata": dict(run.draft.metadata),
            },
            "verdict": _serialize_verdict(run.verdict),
            "metadata": dict(run.metadata),
        }
        runs.append(run_summary)

        _add_graph_node(
            nodes,
            node_id=task_plan.task_id,
            kind="Task",
            label=_truncate_text(task_plan.user_query),
            metadata={"risk_level": task_plan.risk_level, "goal_type": task_plan.goal_type},
        )
        _add_graph_node(
            nodes,
            node_id=task_plan.intent_id,
            kind="Intent",
            label=_truncate_text(task_plan.intent_text),
            metadata={"priority": task_plan.priority},
        )
        _add_graph_edge(edges, source=task_plan.task_id, target=task_plan.intent_id, edge_type="SEEKS")
        for subgoal in task_plan.subgoals:
            _add_graph_node(
                nodes,
                node_id=subgoal.subgoal_id,
                kind="Subgoal",
                label=_truncate_text(subgoal.text),
                metadata={"order": subgoal.order, "status": subgoal.status},
            )
            _add_graph_edge(
                edges,
                source=task_plan.task_id,
                target=subgoal.subgoal_id,
                edge_type="DECOMPOSED_INTO",
            )
            _add_graph_edge(
                edges,
                source=subgoal.subgoal_id,
                target=task_plan.intent_id,
                edge_type="SUPPORTS_INTENT",
            )
        _add_graph_node(nodes, node_id=run_id, kind="Run", label=run_id, metadata={"index": index})
        _add_graph_edge(edges, source=run_id, target=task_plan.task_id, edge_type="HAS_TASK")

        if run.selected_controller_id is not None:
            _add_graph_node(
                nodes,
                node_id=f"controller:{run.selected_controller_id}",
                kind="Controller",
                label=_truncate_text(run.selected_controller_id),
                metadata={},
            )
            _add_graph_edge(
                edges,
                source=run_id,
                target=f"controller:{run.selected_controller_id}",
                edge_type="SELECTED_CONTROLLER",
            )

        draft_node_id = f"{run_id}:draft"
        _add_graph_node(
            nodes,
            node_id=draft_node_id,
            kind="Draft",
            label=_truncate_text(run.draft.output_text),
            metadata={"controller_id": run.draft.controller_id},
        )
        _add_graph_edge(edges, source=run_id, target=draft_node_id, edge_type="HAS_DRAFT")

        verdict_node_id = f"{run_id}:verdict"
        _add_graph_node(
            nodes,
            node_id=verdict_node_id,
            kind="Verdict",
            label="passed" if run.verdict.passed else "failed",
            metadata={
                "confidence": float(run.verdict.confidence),
                "issues": list(run.verdict.issues),
            },
        )
        _add_graph_edge(edges, source=draft_node_id, target=verdict_node_id, edge_type="EVALUATED_AS")

        if run.path_context is not None:
            path_groups = (
                ("evidence_paths", run.path_context.evidence_paths, "EVIDENCE_PATH"),
                ("analogous_prior_paths", run.path_context.analogous_prior_paths, "ANALOGOUS_PATH"),
                ("correction_paths", run.path_context.correction_paths, "CORRECTION_PATH"),
            )
            for _, paths, edge_type in path_groups:
                for path in paths:
                    _add_graph_node(
                        nodes,
                        node_id=path.path_id,
                        kind="RetrievedPath",
                        label=_truncate_text(path.path_text),
                        metadata={
                            "path_kind": path.path_kind,
                            "score": float(path.score),
                            "root_anchor": path.root_anchor,
                            "supporting_node_ids": list(path.supporting_node_ids),
                            "metadata": dict(path.metadata),
                        },
                    )
                    _add_graph_edge(edges, source=run_id, target=path.path_id, edge_type=edge_type)
                    _add_graph_edge(
                        edges,
                        source=path.path_id,
                        target=task_plan.intent_id,
                        edge_type="SERVES",
                    )
                    _add_graph_edge(
                        edges,
                        source=path.path_id,
                        target=task_plan.active_subgoal.subgoal_id,
                        edge_type="ANCHORS",
                    )

        if run.fallback_used:
            drift_node_id = f"{run_id}:drift"
            correction_node_id = f"{run_id}:correction"
            _add_graph_node(
                nodes,
                node_id=drift_node_id,
                kind="DriftEvent",
                label="verifier_rejected_steered_draft",
                metadata={"score": _drift_score_from_verifier(run.verdict)},
            )
            _add_graph_node(
                nodes,
                node_id=correction_node_id,
                kind="Correction",
                label="fallback_to_unsteered_execution",
                metadata={"outcome": "retrying_without_steering"},
            )
            _add_graph_edge(edges, source=draft_node_id, target=drift_node_id, edge_type="TRIGGERED")
            _add_graph_edge(
                edges,
                source=drift_node_id,
                target=correction_node_id,
                edge_type="CORRECTED_BY",
            )
            _add_graph_edge(
                edges,
                source=correction_node_id,
                target=task_plan.active_subgoal.subgoal_id,
                edge_type="RESTORED",
            )

    return {
        "format_version": RUNTIME_ARTIFACT_FORMAT_VERSION,
        "graph_type": "hybrid_meta_cognition_runtime",
        "run_count": len(runs),
        "runs": runs,
        "nodes": sorted(nodes.values(), key=lambda item: (item["kind"], item["id"])),
        "edges": sorted(edges.values(), key=lambda item: (item["source"], item["target"], item["type"])),
    }


def _write_json_artifact(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _scale_coordinate(value: float, lower: float, upper: float, size: float, padding: float) -> float:
    if abs(upper - lower) < SCALE_EPSILON:
        return padding + (size - 2 * padding) / 2
    return padding + ((value - lower) / (upper - lower)) * (size - 2 * padding)


def _write_graph_visualization_artifact(path: Path, graph_payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.DiGraph()
    for node in graph_payload.get("nodes", ()):
        graph.add_node(node["id"], label=node["label"], kind=node["kind"])
    for edge in graph_payload.get("edges", ()):
        graph.add_edge(edge["source"], edge["target"], relation=edge["type"])

    if not graph.nodes:
        path.write_text(
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_MIN_WIDTH}" height="200">'
                '<rect width="100%" height="100%" fill="#f8fafc" />'
                f'<text x="{SVG_MIN_WIDTH / 2}" y="100" text-anchor="middle" font-size="18" '
                'font-family="Arial, sans-serif" fill="#0f172a">No graph state recorded.</text>'
                "</svg>"
            ),
            encoding="utf-8",
        )
        return path

    positions = nx.spring_layout(graph, seed=42)
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    width = min(
        SVG_MAX_WIDTH,
        max(SVG_MIN_WIDTH, SVG_WIDTH_PER_NODE * max(len(graph.nodes), 1)),
    )
    height = min(
        SVG_MAX_HEIGHT,
        max(SVG_MIN_HEIGHT, SVG_HEIGHT_PER_NODE * max(len(graph.nodes), 1)),
    )
    padding = 80.0
    scaled_positions = {
        node_id: (
            _scale_coordinate(point[0], min(xs), max(xs), float(width), padding),
            _scale_coordinate(point[1], min(ys), max(ys), float(height), padding),
        )
        for node_id, point in positions.items()
    }

    color_by_kind = {
        "Task": "#dbeafe",
        "Intent": "#fde68a",
        "Subgoal": "#dcfce7",
        "Run": "#e9d5ff",
        "Controller": "#fbcfe8",
        "Draft": "#fecaca",
        "Verdict": "#bfdbfe",
        "RetrievedPath": "#fed7aa",
        "DriftEvent": "#fecdd3",
        "Correction": "#c7d2fe",
    }
    node_width = 180.0
    node_height = 52.0
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" /></marker></defs>',
        '<rect width="100%" height="100%" fill="#f8fafc" />',
    ]
    for source_id, target_id, data in graph.edges(data=True):
        source_x, source_y = scaled_positions[source_id]
        target_x, target_y = scaled_positions[target_id]
        svg_lines.append(
            f'<line x1="{source_x:.2f}" y1="{source_y:.2f}" x2="{target_x:.2f}" y2="{target_y:.2f}" '
            'stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)" />'
        )
        label_x = (source_x + target_x) / 2
        label_y = (source_y + target_y) / 2 - 6
        svg_lines.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="middle" '
            'font-size="11" font-family="Arial, sans-serif" fill="#475569">'
            f'{escape(str(data["relation"]))}</text>'
        )
    for node_id, data in graph.nodes(data=True):
        x, y = scaled_positions[node_id]
        fill = color_by_kind.get(data["kind"], "#e2e8f0")
        svg_lines.append(
            f'<rect x="{x - node_width / 2:.2f}" y="{y - node_height / 2:.2f}" '
            f'width="{node_width:.2f}" height="{node_height:.2f}" rx="12" ry="12" '
            f'fill="{fill}" stroke="#334155" stroke-width="1.5" />'
        )
        svg_lines.append(
            f'<text x="{x:.2f}" y="{y - 4:.2f}" text-anchor="middle" '
            'font-size="12" font-weight="bold" font-family="Arial, sans-serif" fill="#0f172a">'
            f'{escape(str(data["kind"]))}</text>'
        )
        svg_lines.append(
            f'<text x="{x:.2f}" y="{y + 14:.2f}" text-anchor="middle" '
            'font-size="11" font-family="Arial, sans-serif" fill="#334155">'
            f'{escape(_truncate_text(data["label"], limit=28))}</text>'
        )
    svg_lines.append("</svg>")
    path.write_text("\n".join(svg_lines), encoding="utf-8")
    return path


def _record_graph_state(
    graph_store: Any,
    run_handle: Any,
    *,
    observed_features: Sequence[ObservedInteractionFeature] | None = None,
    **kwargs: Any,
) -> Any:
    record_state = getattr(graph_store, "record_state", None)
    if not callable(record_state):
        return _call_component(graph_store, "record_state", run_handle, **kwargs)
    if observed_features is not None:
        try:
            signature = inspect.signature(record_state)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            accepts_var_keyword = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if "observed_features" in signature.parameters or accepts_var_keyword:
                return record_state(run_handle, observed_features=observed_features, **kwargs)
        return record_state(run_handle, **kwargs)
    return record_state(run_handle, **kwargs)


def _ensure_activation_trace(
    draft: ExecutorResult,
    *,
    model_name: str,
    controller: SteeringController | None,
) -> ActivationTrace:
    if draft.activation_trace is None:
        draft.activation_trace = ActivationTrace(
            model_name=model_name,
            controller_id=draft.controller_id,
            layer_idx=controller.layer_idx if controller is not None else None,
            prompt=draft.prompt,
            output_text=draft.output_text,
            top_feature_scores=[],
        )
        return draft.activation_trace
    if not draft.activation_trace.output_text:
        draft.activation_trace.output_text = draft.output_text
    if draft.activation_trace.controller_id is None:
        draft.activation_trace.controller_id = draft.controller_id
    return draft.activation_trace


class HybridMetaCognitionAgent:
    """Composable planner/retriever/executor/verifier loop with memory-backed controller routing."""

    def __init__(
        self,
        planner: PlannerProtocol | Callable[[str, InMemorySteeringMemory], PlannerDecision],
        executor: SteeredExecutor | Any,
        verifier: VerifierProtocol
        | Callable[[str, ExecutorResult, Sequence[str], PlannerDecision], VerifierResult],
        memory: InMemorySteeringMemory,
        retriever: RetrieverProtocol
        | Callable[[str, PlannerDecision], Sequence[str]]
        | None = None,
        tool_router: ToolRouterProtocol
        | Callable[[str, PlannerDecision, Sequence[str]], Sequence[str]]
        | None = None,
        graph_store: Any | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.tool_router = tool_router
        self.executor = executor
        self.verifier = verifier
        self.memory = memory
        self.graph_store = graph_store
        self.max_new_tokens = max_new_tokens
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None

    def __enter__(self) -> HybridMetaCognitionAgent:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.close()
            return None
        try:
            self.close()
        except Exception:
            pass
        return None

    def persist_artifacts(self, artifact_dir: str | Path | None = None) -> dict[str, Path] | None:
        destination = self.artifact_dir if artifact_dir is None else Path(artifact_dir)
        close_graph_store = getattr(self.graph_store, "close", None)
        if destination is None:
            if callable(close_graph_store):
                close_graph_store()
            return None
        try:
            destination.mkdir(parents=True, exist_ok=True)
            runtime_discoveries = _build_runtime_discoveries_payload(self.memory)
            graph_payload = _build_runtime_graph_payload(self.memory.run_history)
            artifacts = {
                "adaptive_discoveries": _write_json_artifact(
                    destination / "adaptive_discoveries.json",
                    runtime_discoveries,
                ),
                "graph_state": _write_json_artifact(destination / "graph_state.json", graph_payload),
                "graph_visualization": _write_graph_visualization_artifact(
                    destination / "graph_state.svg",
                    graph_payload,
                ),
            }
        except Exception as persistence_error:
            if callable(close_graph_store):
                try:
                    close_graph_store()
                except Exception as close_exc:
                    persistence_error.__context__ = close_exc
                    raise persistence_error
            raise
        if callable(close_graph_store):
            close_graph_store()
        return artifacts

    def close(self) -> dict[str, Path] | None:
        return self.persist_artifacts()

    def run(self, task: str) -> HybridAgentRun:
        normalized_task = _require_text(task, "task")
        plan = _call_component(self.planner, "plan", normalized_task, self.memory)
        graph_run = None
        if self.graph_store is not None:
            graph_run = _call_component(self.graph_store, "start_run", normalized_task, plan)
        context: list[str] = []
        path_context = None
        if plan.needs_retrieval and self.retriever is not None:
            retrieved = _call_component(self.retriever, "retrieve", normalized_task, plan) or ()
            rendered_context, path_context = _coerce_context_payload(retrieved)
            context.extend(rendered_context)
            if graph_run is not None:
                _record_graph_state(
                    self.graph_store,
                    graph_run,
                    step=1,
                    text="Retrieved graph-native context" if path_context is not None else "Retrieved context",
                    state_type="retrieval",
                    path_context=path_context,
                    metadata={"task_type": plan.task_type},
                )
        if self.tool_router is not None:
            tool_outputs = _call_component(self.tool_router, "route", normalized_task, plan, context) or ()
            context.extend(str(item) for item in tool_outputs)

        selected_controller = None
        if plan.use_steering:
            selected_controller = self.memory.select_controller(
                task_type=plan.task_type,
                requested_controller_id=plan.controller_id,
            )

        draft = self.executor.execute(
            normalized_task,
            plan,
            context,
            controller=selected_controller,
            controllers=self.memory.list_controllers(),
            max_new_tokens=self.max_new_tokens,
        )
        model_name = str(getattr(self.executor, "model_name", "unknown-model"))
        initial_trace = _ensure_activation_trace(
            draft,
            model_name=model_name,
            controller=selected_controller,
        )
        observed_features = self.memory.observe_interaction(initial_trace)
        draft_state_id = None
        if graph_run is not None:
            draft_state_id = _record_graph_state(
                self.graph_store,
                graph_run,
                step=2,
                text=draft.output_text,
                state_type="draft",
                path_context=path_context,
                observed_features=observed_features,
                metadata={
                    "prompt": draft.prompt,
                    "controller_id": draft.controller_id,
                },
            )
        verdict = _call_component(self.verifier, "verify", normalized_task, draft, context, plan)
        if graph_run is not None and draft_state_id is not None:
            _call_component(
                self.graph_store,
                "record_verifier_result",
                graph_run,
                state_id=draft_state_id,
                verdict=verdict,
            )
        fallback_used = False

        if selected_controller is not None and plan.allow_fallback and not verdict.passed:
            if graph_run is not None and draft_state_id is not None:
                _call_component(
                    self.graph_store,
                    "record_drift_and_correction",
                    graph_run,
                    state_id=draft_state_id,
                    step=2,
                    drift_kind="verifier_rejected_steered_draft",
                    score=_drift_score_from_verifier(verdict),
                    description="Verifier rejected the steered draft, so the agent re-anchored with an unsteered fallback.",
                    correction_kind="fallback",
                    action="fallback_to_unsteered_execution",
                    outcome="retrying_without_steering",
                )
            draft = self.executor.execute(
                normalized_task,
                plan,
                context,
                controller=None,
                controllers=self.memory.list_controllers(),
                max_new_tokens=self.max_new_tokens,
            )
            fallback_trace = _ensure_activation_trace(
                draft,
                model_name=model_name,
                controller=None,
            )
            observed_features = self.memory.observe_interaction(fallback_trace)
            if graph_run is not None:
                draft_state_id = _record_graph_state(
                    self.graph_store,
                    graph_run,
                    step=3,
                    text=draft.output_text,
                    state_type="corrected_draft",
                    path_context=path_context,
                    observed_features=observed_features,
                    metadata={
                        "prompt": draft.prompt,
                        "controller_id": draft.controller_id,
                    },
                )
            verdict = _call_component(self.verifier, "verify", normalized_task, draft, context, plan)
            if graph_run is not None and draft_state_id is not None:
                _call_component(
                    self.graph_store,
                    "record_verifier_result",
                    graph_run,
                    state_id=draft_state_id,
                    verdict=verdict,
                )
            fallback_used = True

        run = HybridAgentRun(
            task=normalized_task,
            plan=plan,
            context=context,
            draft=draft,
            verdict=verdict,
            selected_controller_id=selected_controller.controller_id if selected_controller else None,
            fallback_used=fallback_used,
            path_context=path_context,
            metadata=(
                {"graph_run_id": graph_run.run_id}
                if graph_run is not None
                else {}
            ),
        )
        self.memory.record_run(run)
        if graph_run is not None:
            _call_component(
                self.graph_store,
                "record_outcome",
                graph_run,
                verdict=verdict,
                fallback_used=fallback_used,
                notes="; ".join(verdict.issues) if verdict.issues else None,
            )
        return run
