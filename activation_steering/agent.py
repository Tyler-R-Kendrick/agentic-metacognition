from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import torch

from .models import DEFAULT_MAX_NEW_TOKENS, get_last_token_hidden, generate
from .steering import generate_with_decaying_steering, generate_with_steering

UNKNOWN_CONTROLLER_SORT_VALUE = float("-inf")


def _require_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _coerce_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


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
    prompt_hidden_norm: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model_name = _require_text(self.model_name, "model_name")
        self.prompt = _require_text(self.prompt, "prompt")
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
) -> ActivationTrace | None:
    """Score the prompt hidden state against persisted steering vectors."""
    if not controllers:
        return None
    layer_indices = {controller.layer_idx for controller in controllers}
    if len(layer_indices) != 1:
        raise ValueError(
            "collect_controller_trace() requires all controllers to target the same layer; "
            f"received layers: {sorted(layer_indices)}"
        )
    layer_idx = controllers[0].layer_idx
    hidden = get_last_token_hidden(prompt, layer_idx, model, tokenizer, device)
    scores = []
    skipped_controllers = []
    for controller in controllers:
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
        layer_idx=layer_idx,
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
        activation_trace = collect_controller_trace(
            prompt=prompt,
            controllers=list(controllers),
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            model_name=self.model_name,
            top_k=self.top_k_trace_features,
        )
        if activation_trace is not None:
            if controller is not None:
                activation_trace.controller_id = controller.controller_id
                activation_trace.layer_idx = controller.layer_idx
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

    def record_run(self, run: HybridAgentRun) -> None:
        self.run_history.append(run)
        if run.draft.activation_trace is not None:
            self.activation_traces.append(run.draft.activation_trace)
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
    def retrieve(self, task: str, plan: PlannerDecision) -> Sequence[str]: ...


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
) -> Any:
    method = getattr(component, method_name, None)
    if callable(method):
        return method(*args)
    if callable(component):
        return component(*args)
    raise TypeError(f"Component must be callable or provide a .{method_name}(...) method.")


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
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.tool_router = tool_router
        self.executor = executor
        self.verifier = verifier
        self.memory = memory
        self.max_new_tokens = max_new_tokens

    def run(self, task: str) -> HybridAgentRun:
        normalized_task = _require_text(task, "task")
        plan = _call_component(self.planner, "plan", normalized_task, self.memory)
        context: list[str] = []
        if plan.needs_retrieval and self.retriever is not None:
            retrieved = _call_component(self.retriever, "retrieve", normalized_task, plan) or ()
            context.extend(str(item) for item in retrieved)
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
        verdict = _call_component(self.verifier, "verify", normalized_task, draft, context, plan)
        fallback_used = False

        if selected_controller is not None and plan.allow_fallback and not verdict.passed:
            draft = self.executor.execute(
                normalized_task,
                plan,
                context,
                controller=None,
                controllers=self.memory.list_controllers(),
                max_new_tokens=self.max_new_tokens,
            )
            verdict = _call_component(self.verifier, "verify", normalized_task, draft, context, plan)
            fallback_used = True

        run = HybridAgentRun(
            task=normalized_task,
            plan=plan,
            context=context,
            draft=draft,
            verdict=verdict,
            selected_controller_id=selected_controller.controller_id if selected_controller else None,
            fallback_used=fallback_used,
        )
        self.memory.record_run(run)
        return run
