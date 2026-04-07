from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


def _require_text(value: Any, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _coerce_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _validate_similarity_function(similarity_function: str) -> str:
    normalized = _require_text(similarity_function, "similarity_function").lower()
    allowed = {"cosine", "euclidean"}
    if normalized not in allowed:
        raise ValueError(
            "similarity_function must be one of: " + ", ".join(sorted(allowed)) + "."
        )
    return normalized


ISSUE_PENALTY_PER_ITEM = 0.25


def _verifier_completeness_from_issues(issues: Sequence[str]) -> float:
    if not issues:
        return 1.0
    return max(0.0, 1.0 - ISSUE_PENALTY_PER_ITEM * len(issues))


def _validate_neo4j_driver(driver: Any) -> Any:
    if driver is None:
        raise ValueError("driver must provide a .session(...) method.")
    session = getattr(driver, "session", None)
    if not callable(session):
        raise ValueError("driver must provide a .session(...) method.")
    return driver


@dataclass(frozen=True)
class GraphConstraint:
    constraint_id: str
    type: str
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_id", _require_text(self.constraint_id, "constraint_id"))
        object.__setattr__(self, "type", _require_text(self.type, "type"))
        object.__setattr__(self, "value", _require_text(self.value, "value"))


@dataclass(frozen=True)
class GraphSubgoal:
    subgoal_id: str
    text: str
    order: int
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subgoal_id", _require_text(self.subgoal_id, "subgoal_id"))
        object.__setattr__(self, "text", _require_text(self.text, "text"))
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "status", _require_text(self.status, "status"))
        object.__setattr__(self, "metadata", _coerce_metadata(self.metadata))


@dataclass(frozen=True)
class GraphTaskPlan:
    task_id: str
    user_query: str
    intent_id: str
    intent_text: str
    goal_type: str = "answer"
    priority: float = 1.0
    risk_level: str = "medium"
    constraints: tuple[GraphConstraint, ...] = ()
    subgoals: tuple[GraphSubgoal, ...] = ()
    active_subgoal_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _require_text(self.task_id, "task_id"))
        object.__setattr__(self, "user_query", _require_text(self.user_query, "user_query"))
        object.__setattr__(self, "intent_id", _require_text(self.intent_id, "intent_id"))
        object.__setattr__(self, "intent_text", _require_text(self.intent_text, "intent_text"))
        object.__setattr__(self, "goal_type", _require_text(self.goal_type, "goal_type"))
        object.__setattr__(self, "priority", float(self.priority))
        object.__setattr__(self, "risk_level", _require_text(self.risk_level, "risk_level"))
        object.__setattr__(self, "metadata", _coerce_metadata(self.metadata))

    @property
    def active_subgoal(self) -> GraphSubgoal:
        if self.active_subgoal_id is not None:
            for subgoal in self.subgoals:
                if subgoal.subgoal_id == self.active_subgoal_id:
                    return subgoal
        if self.subgoals:
            return self.subgoals[0]
        return GraphSubgoal(
            subgoal_id=f"{self.task_id}-subgoal-1",
            text=self.user_query,
            order=1,
        )

    @classmethod
    def from_task_and_plan(cls, task: str, plan: Any) -> GraphTaskPlan:
        normalized_task = _require_text(task, "task")
        metadata = _coerce_metadata(getattr(plan, "metadata", None))
        task_id = str(metadata.get("task_id") or f"task-{uuid4().hex}")
        intent_id = str(metadata.get("intent_id") or f"intent-{uuid4().hex}")
        intent_text = str(metadata.get("intent_text") or metadata.get("intent") or normalized_task)

        constraints_payload = metadata.get("constraints") or ()
        constraints: list[GraphConstraint] = []
        for index, item in enumerate(constraints_payload, start=1):
            if isinstance(item, Mapping):
                constraint_type = item.get("type", "constraint")
                constraint_value = item.get("value", item.get("text", ""))
                constraint_id = item.get("constraint_id", f"{task_id}-constraint-{index}")
            else:
                constraint_type = "constraint"
                constraint_value = item
                constraint_id = f"{task_id}-constraint-{index}"
            constraints.append(
                GraphConstraint(
                    constraint_id=str(constraint_id),
                    type=str(constraint_type),
                    value=str(constraint_value),
                )
            )

        subgoals_payload = metadata.get("subgoals") or (normalized_task,)
        subgoals: list[GraphSubgoal] = []
        for index, item in enumerate(subgoals_payload, start=1):
            if isinstance(item, Mapping):
                subgoal_text = item.get("text", item.get("subgoal_text", normalized_task))
                subgoal_id = item.get("subgoal_id", f"{task_id}-subgoal-{index}")
                order = item.get("order", index)
                status = item.get("status", "pending")
                subgoal_metadata = item.get("metadata")
            else:
                subgoal_text = item
                subgoal_id = f"{task_id}-subgoal-{index}"
                order = index
                status = "pending"
                subgoal_metadata = None
            subgoals.append(
                GraphSubgoal(
                    subgoal_id=str(subgoal_id),
                    text=str(subgoal_text),
                    order=int(order),
                    status=str(status),
                    metadata=_coerce_metadata(subgoal_metadata),
                )
            )

        active_subgoal_id = metadata.get("active_subgoal_id")
        if active_subgoal_id is None and subgoals:
            active_subgoal_id = subgoals[0].subgoal_id
        return cls(
            task_id=task_id,
            user_query=normalized_task,
            intent_id=intent_id,
            intent_text=intent_text,
            goal_type=str(metadata.get("goal_type", getattr(plan, "task_type", "answer"))),
            priority=float(metadata.get("priority", 1.0)),
            risk_level=str(metadata.get("risk_level", "medium")),
            constraints=tuple(constraints),
            subgoals=tuple(sorted(subgoals, key=lambda item: item.order)),
            active_subgoal_id=str(active_subgoal_id) if active_subgoal_id is not None else None,
            metadata=metadata,
        )


@dataclass(frozen=True)
class RetrievedPath:
    path_id: str
    path_kind: str
    path_text: str
    score: float = 0.0
    redundancy_penalty: float = 0.0
    root_anchor: str | None = None
    supporting_node_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_id", _require_text(self.path_id, "path_id"))
        object.__setattr__(self, "path_kind", _require_text(self.path_kind, "path_kind"))
        object.__setattr__(self, "path_text", _require_text(self.path_text, "path_text"))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "redundancy_penalty", float(self.redundancy_penalty))
        object.__setattr__(
            self,
            "supporting_node_ids",
            tuple(str(node_id) for node_id in self.supporting_node_ids),
        )
        object.__setattr__(self, "metadata", _coerce_metadata(self.metadata))

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RetrievedPath:
        return cls(
            path_id=str(record.get("path_id") or f"path-{uuid4().hex}"),
            path_kind=str(record.get("path_kind", "evidence")),
            path_text=str(record.get("path_text", "")),
            score=float(record.get("score", 0.0)),
            redundancy_penalty=float(record.get("redundancy_penalty", 0.0)),
            root_anchor=record.get("root_anchor"),
            supporting_node_ids=tuple(record.get("supporting_node_ids") or ()),
            metadata=dict(record.get("metadata") or {}),
        )


@dataclass
class PathRAGContext:
    current_intent: str
    active_subgoal: str
    evidence_paths: list[RetrievedPath] = field(default_factory=list)
    analogous_prior_paths: list[RetrievedPath] = field(default_factory=list)
    correction_paths: list[RetrievedPath] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.current_intent = _require_text(self.current_intent, "current_intent")
        self.active_subgoal = _require_text(self.active_subgoal, "active_subgoal")
        self.metadata = _coerce_metadata(self.metadata)

    def to_prompt_sections(self) -> list[str]:
        return serialize_path_rag_context(self)


def serialize_path_rag_context(context: PathRAGContext) -> list[str]:
    def render_paths(title: str, paths: Sequence[RetrievedPath]) -> str:
        if not paths:
            return f"{title}\n- none"
        lines = [title]
        for index, path in enumerate(paths, start=1):
            lines.append(f"{index}. {path.path_text}")
            if path.root_anchor:
                lines.append(f"   root: {path.root_anchor}")
            if path.supporting_node_ids:
                lines.append(f"   nodes: {', '.join(path.supporting_node_ids)}")
            support = path.metadata.get("support")
            if support:
                lines.append(f"   support: {support}")
            lines.append(f"   score: {path.score:.3f}")
        return "\n".join(lines)

    return [
        f"CURRENT_INTENT\n- {context.current_intent}",
        f"ACTIVE_SUBGOAL\n- {context.active_subgoal}",
        render_paths("EVIDENCE_PATHS", context.evidence_paths),
        render_paths("ANALOGOUS_PRIOR_PATHS", context.analogous_prior_paths),
        render_paths("CORRECTION_PATHS", context.correction_paths),
    ]


@dataclass(frozen=True)
class GraphRunHandle:
    run_id: str
    task_plan: GraphTaskPlan


class Neo4jGraphStore:
    """Minimal Neo4j persistence layer for graph-native task, state, and correction memory."""

    _UNIQUE_CONSTRAINTS = {
        "Task": "task_id",
        "Intent": "intent_id",
        "Subgoal": "subgoal_id",
        "Model": "model_name",
        "Document": "doc_id",
        "Chunk": "chunk_id",
        "Entity": "entity_id",
        "Claim": "claim_id",
        "Evidence": "evidence_id",
        "Run": "run_id",
        "State": "state_id",
        "InteractionFeature": "feature_id",
        "DriftEvent": "event_id",
        "Correction": "correction_id",
    }
    _NORMAL_INDEXES = {
        "Task": "status",
        "Subgoal": "status",
        "Document": "updated_at",
        "DriftEvent": "kind",
        "CorrectionPattern": "drift_kind",
    }
    _VECTOR_INDEXES = {
        "Intent": "embedding",
        "Subgoal": "embedding",
        "Chunk": "embedding",
        "Entity": "embedding",
        "Claim": "embedding",
        "State": "embedding",
    }

    def __init__(self, driver: Any, database: str = "neo4j") -> None:
        self.driver = _validate_neo4j_driver(driver)
        self.database = _require_text(database, "database")

    def close(self) -> None:
        close = getattr(self.driver, "close", None)
        if callable(close):
            close()

    def _run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **parameters)
            data = []
            for record in result:
                if hasattr(record, "data"):
                    data.append(record.data())
                else:
                    data.append(dict(record))
            return data

    def ensure_schema(
        self,
        embedding_dimensions: Mapping[str, int] | None = None,
        similarity_function: str = "cosine",
    ) -> None:
        similarity_function = _validate_similarity_function(similarity_function)
        for label, property_name in self._UNIQUE_CONSTRAINTS.items():
            self._run(
                f"CREATE CONSTRAINT {label.lower()}_{property_name}_unique IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
            )
        for label, property_name in self._NORMAL_INDEXES.items():
            self._run(
                f"CREATE INDEX {label.lower()}_{property_name}_index IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.{property_name})"
            )
        for label, property_name in self._VECTOR_INDEXES.items():
            dimensions = None if embedding_dimensions is None else embedding_dimensions.get(label)
            if dimensions is None:
                continue
            self._run(
                f"CREATE VECTOR INDEX {label.lower()}_{property_name}_index IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.{property_name}) "
                "OPTIONS {indexConfig: {"
                f"`vector.dimensions`: {int(dimensions)}, "
                f"`vector.similarity_function`: '{similarity_function}'"
                "}}"
            )

    def start_run(self, task: str, plan: Any) -> GraphRunHandle:
        task_plan = GraphTaskPlan.from_task_and_plan(task, plan)
        run_id = str(task_plan.metadata.get("run_id") or f"run-{uuid4().hex}")
        self._run(
            """
            MERGE (t:Task {task_id: $task_id})
            SET t.user_query = $user_query,
                t.created_at = coalesce(t.created_at, datetime()),
                t.status = 'active',
                t.risk_level = $risk_level

            MERGE (i:Intent {intent_id: $intent_id})
            SET i.text = $intent_text,
                i.goal_type = $goal_type,
                i.priority = $priority

            MERGE (t)-[:SEEKS]->(i)

            MERGE (r:Run {run_id: $run_id})
            SET r.started_at = coalesce(r.started_at, datetime()),
                r.status = 'active',
                r.agent_version = $agent_version

            MERGE (r)-[:HAS_TASK]->(t)
            """,
            task_id=task_plan.task_id,
            user_query=task_plan.user_query,
            risk_level=task_plan.risk_level,
            intent_id=task_plan.intent_id,
            intent_text=task_plan.intent_text,
            goal_type=task_plan.goal_type,
            priority=task_plan.priority,
            run_id=run_id,
            agent_version=str(task_plan.metadata.get("agent_version", "hybrid-agent")),
        )
        if task_plan.constraints:
            self._run(
                """
                MATCH (i:Intent {intent_id: $intent_id})
                UNWIND $constraints AS constraint
                MERGE (c:Constraint {constraint_id: constraint.constraint_id})
                SET c.type = constraint.type,
                    c.value = constraint.value
                MERGE (i)-[:CONSTRAINED_BY]->(c)
                """,
                intent_id=task_plan.intent_id,
                constraints=[constraint.__dict__ for constraint in task_plan.constraints],
            )
        if task_plan.subgoals:
            subgoals_payload = [
                {
                    "subgoal_id": subgoal.subgoal_id,
                    "text": subgoal.text,
                    "order": subgoal.order,
                    "status": subgoal.status,
                }
                for subgoal in task_plan.subgoals
            ]
            self._run(
                """
                MATCH (t:Task {task_id: $task_id})-[:SEEKS]->(i:Intent {intent_id: $intent_id})
                UNWIND $subgoals AS sg
                MERGE (s:Subgoal {subgoal_id: sg.subgoal_id})
                SET s.text = sg.text,
                    s.order = sg.order,
                    s.status = sg.status
                MERGE (t)-[:DECOMPOSED_INTO]->(s)
                MERGE (s)-[:SUPPORTS_INTENT]->(i)
                """,
                task_id=task_plan.task_id,
                intent_id=task_plan.intent_id,
                subgoals=subgoals_payload,
            )
            self._run(
                """
                UNWIND range(0, size($subgoals) - 2) AS idx
                MATCH (first:Subgoal {subgoal_id: $subgoals[idx].subgoal_id})
                MATCH (second:Subgoal {subgoal_id: $subgoals[idx + 1].subgoal_id})
                MERGE (first)-[:NEXT]->(second)
                """,
                subgoals=subgoals_payload,
            )
        return GraphRunHandle(run_id=run_id, task_plan=task_plan)

    def record_state(
        self,
        run_handle: GraphRunHandle,
        *,
        step: int,
        text: str,
        state_type: str,
        path_context: PathRAGContext | None = None,
        observed_features: Sequence[Any] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        state_id = f"state-{uuid4().hex}"
        payload = _coerce_metadata(metadata)
        self._run(
            """
            MATCH (r:Run {run_id: $run_id})
            MERGE (s:State {state_id: $state_id})
            SET s.step = $step,
                s.text = $text,
                s.timestamp = datetime(),
                s.state_type = $state_type
            MERGE (r)-[:HAS_STATE]->(s)
            """,
            run_id=run_handle.run_id,
            state_id=state_id,
            step=int(step),
            text=_require_text(text, "text"),
            state_type=_require_text(state_type, "state_type"),
        )
        if observed_features:
            self._run(
                """
                MATCH (s:State {state_id: $state_id})
                UNWIND $features AS feature
                MERGE (m:Model {model_name: feature.model_name})
                MERGE (f:InteractionFeature {feature_id: feature.feature_id})
                SET f.category = feature.category,
                    f.summary = feature.summary,
                    f.input_example = feature.input_example,
                    f.output_example = feature.output_example,
                    f.observation_count = feature.observation_count,
                    f.metadata_json = feature.metadata_json
                MERGE (m)-[:EXHIBITS]->(f)
                MERGE (s)-[:OBSERVED_FEATURE]->(f)
                """,
                state_id=state_id,
                features=[
                    {
                        "feature_id": _require_text(feature.feature_id, "feature_id"),
                        "model_name": _require_text(feature.model_name, "model_name"),
                        "category": _require_text(feature.category, "category"),
                        "summary": _require_text(feature.summary, "summary"),
                        "input_example": str(getattr(feature, "input_example", "")),
                        "output_example": str(getattr(feature, "output_example", "")),
                        "observation_count": int(getattr(feature, "observation_count", 1)),
                        "metadata_json": json.dumps(getattr(feature, "metadata", {}), sort_keys=True),
                    }
                    for feature in observed_features
                ],
            )
        if path_context is None:
            return state_id
        all_paths = (
            list(path_context.evidence_paths)
            + list(path_context.analogous_prior_paths)
            + list(path_context.correction_paths)
        )
        if not all_paths:
            return state_id
        self._run(
            """
            MATCH (s:State {state_id: $state_id})
            MATCH (i:Intent {intent_id: $intent_id})
            MATCH (subgoal:Subgoal {subgoal_id: $subgoal_id})
            UNWIND $paths AS path
            MERGE (p:RetrievedPath {path_id: path.path_id})
            SET p.path_text = path.path_text,
                p.path_kind = path.path_kind,
                p.score = path.score,
                p.redundancy_penalty = coalesce(path.redundancy_penalty, 0.0)
            MERGE (s)-[:RETRIEVED]->(p)
            MERGE (p)-[:SERVES]->(i)
            MERGE (p)-[:ANCHORS]->(subgoal)
            """,
            state_id=state_id,
            intent_id=run_handle.task_plan.intent_id,
            subgoal_id=run_handle.task_plan.active_subgoal.subgoal_id,
            paths=[
                {
                    "path_id": path.path_id,
                    "path_text": path.path_text,
                    "path_kind": path.path_kind,
                    "score": path.score,
                    "redundancy_penalty": path.redundancy_penalty,
                }
                for path in all_paths
            ],
        )
        derived_from = []
        for path in all_paths:
            for chunk_id in path.metadata.get("chunk_ids", ()):
                derived_from.append({"path_id": path.path_id, "chunk_id": str(chunk_id)})
        if derived_from:
            self._run(
                """
                UNWIND $derived_from AS row
                MATCH (p:RetrievedPath {path_id: row.path_id})
                MATCH (c:Chunk {chunk_id: row.chunk_id})
                MERGE (p)-[:DERIVED_FROM]->(c)
                """,
                derived_from=derived_from,
            )
        if payload:
            self._run(
                """
                MATCH (s:State {state_id: $state_id})
                SET s.metadata_json = $metadata_json
                """,
                state_id=state_id,
                metadata_json=json.dumps(payload, sort_keys=True),
            )
        return state_id

    def record_verifier_result(
        self,
        run_handle: GraphRunHandle,
        *,
        state_id: str,
        verdict: Any,
    ) -> str:
        result_id = f"verifier-{uuid4().hex}"
        issues = list(getattr(verdict, "issues", ()) or ())
        self._run(
            """
            MATCH (s:State {state_id: $state_id})
            MERGE (v:VerifierResult {result_id: $result_id})
            SET v.groundedness = $groundedness,
                v.consistency = $consistency,
                v.intent_alignment = $intent_alignment,
                v.completeness = $completeness,
                v.timestamp = datetime(),
                v.issues = $issues
            MERGE (s)-[:EVALUATED_AS]->(v)
            """,
            state_id=_require_text(state_id, "state_id"),
            result_id=result_id,
            groundedness=float(getattr(verdict, "confidence", 0.0)),
            consistency=1.0 if bool(getattr(verdict, "passed", False)) else 0.0,
            intent_alignment=float(getattr(verdict, "confidence", 0.0)),
            completeness=_verifier_completeness_from_issues(issues),
            issues=issues,
        )
        return result_id

    def record_drift_and_correction(
        self,
        run_handle: GraphRunHandle,
        *,
        state_id: str,
        step: int,
        drift_kind: str,
        score: float,
        description: str,
        correction_kind: str,
        action: str,
        outcome: str,
    ) -> tuple[str, str]:
        event_id = f"drift-{uuid4().hex}"
        correction_id = f"correction-{uuid4().hex}"
        self._run(
            """
            MATCH (s:State {state_id: $state_id})
            MATCH (subgoal:Subgoal {subgoal_id: $subgoal_id})
            MERGE (d:DriftEvent {event_id: $event_id})
            SET d.kind = $kind,
                d.score = $score,
                d.step = $step,
                d.description = $description,
                d.timestamp = datetime()
            MERGE (s)-[:TRIGGERED]->(d)

            MERGE (c:Correction {correction_id: $correction_id})
            SET c.kind = $correction_kind,
                c.action = $action,
                c.outcome = $outcome,
                c.timestamp = datetime()
            MERGE (d)-[:CORRECTED_BY]->(c)
            MERGE (c)-[:APPLIED_TO]->(s)
            MERGE (c)-[:RESTORED]->(subgoal)
            """,
            state_id=_require_text(state_id, "state_id"),
            subgoal_id=run_handle.task_plan.active_subgoal.subgoal_id,
            event_id=event_id,
            kind=_require_text(drift_kind, "drift_kind"),
            score=float(score),
            step=int(step),
            description=_require_text(description, "description"),
            correction_id=correction_id,
            correction_kind=_require_text(correction_kind, "correction_kind"),
            action=_require_text(action, "action"),
            outcome=_require_text(outcome, "outcome"),
        )
        return event_id, correction_id

    def record_outcome(
        self,
        run_handle: GraphRunHandle,
        *,
        verdict: Any,
        fallback_used: bool,
        notes: str | None = None,
    ) -> str:
        outcome_id = f"outcome-{uuid4().hex}"
        self._run(
            """
            MATCH (r:Run {run_id: $run_id})
            MERGE (o:Outcome {outcome_id: $outcome_id})
            SET o.success = $success,
                o.user_acceptance = $user_acceptance,
                o.groundedness = $groundedness,
                o.notes = $notes
            MERGE (r)-[:ENDED_WITH]->(o)
            SET r.ended_at = datetime(),
                r.status = $status
            """,
            run_id=run_handle.run_id,
            outcome_id=outcome_id,
            success=bool(getattr(verdict, "passed", False)),
            user_acceptance=bool(getattr(verdict, "passed", False)) and not fallback_used,
            groundedness=float(getattr(verdict, "confidence", 0.0)),
            notes=notes or "; ".join(getattr(verdict, "issues", ()) or ()) or None,
            status="completed" if bool(getattr(verdict, "passed", False)) else "failed",
        )
        return outcome_id

    def retrieve_correction_patterns(
        self,
        *,
        intent_id: str,
        drift_kind: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self._run(
            """
            MATCH (i:Intent {intent_id: $intent_id})<-[:SEEKS]-(:Task)<-[:HAS_TASK]-(r:Run)
            MATCH (r)-[:HAS_STATE]->(:State)-[:TRIGGERED]->(d:DriftEvent)-[:CORRECTED_BY]->(:Correction)-[:INSTANCE_OF]->(p:CorrectionPattern)
            MATCH (r)-[:ENDED_WITH]->(o:Outcome)
            WHERE d.kind = $drift_kind AND o.success = true
            RETURN p.name AS name,
                   p.success_rate AS success_rate,
                   p.avg_recovery_gain AS avg_recovery_gain
            ORDER BY success_rate DESC, avg_recovery_gain DESC
            LIMIT $limit
            """,
            intent_id=_require_text(intent_id, "intent_id"),
            drift_kind=_require_text(drift_kind, "drift_kind"),
            limit=max(int(limit), 1),
        )


class Neo4jPathRAGRetriever:
    """Thin adapter that combines Neo4j candidate retrieval with path-oriented Cypher expansion."""

    TOP_K_CONFIG_KEY = "top_k"

    DEFAULT_EVIDENCE_QUERY = """
    MATCH (subgoal:Subgoal {subgoal_id: $subgoal_id})
    MATCH (chunk:Chunk)-[:SUPPORTS]->(claim:Claim)
    WHERE $candidate_chunk_ids = [] OR chunk.chunk_id IN $candidate_chunk_ids
    OPTIONAL MATCH (claim)-[:ABOUT]->(entity:Entity)
    WITH subgoal, chunk, claim, collect(DISTINCT entity.entity_id) AS entity_ids
    RETURN chunk.chunk_id + '::' + claim.claim_id AS path_id,
           'evidence' AS path_kind,
           chunk.text + ' -> ' + claim.text AS path_text,
           coalesce(claim.confidence, 0.0) AS score,
           subgoal.text AS root_anchor,
           [chunk.chunk_id, claim.claim_id] + entity_ids AS supporting_node_ids
    ORDER BY score DESC
    LIMIT $top_k
    """

    DEFAULT_ANALOGOUS_QUERY = """
    MATCH (intent:Intent {intent_id: $intent_id})<-[:SEEKS]-(:Task)<-[:HAS_TASK]-(run:Run)-[:ENDED_WITH]->(outcome:Outcome)
    WHERE outcome.success = true
    OPTIONAL MATCH (run)-[:HAS_STATE]->(:State)-[:RETRIEVED]->(path:RetrievedPath)
    WITH intent, run, outcome, collect(DISTINCT path.path_id) AS supporting_node_ids
    RETURN run.run_id AS path_id,
           'analogous_prior' AS path_kind,
           intent.text + ' -> prior successful run ' + run.run_id AS path_text,
           coalesce(outcome.groundedness, 0.0) AS score,
           intent.text AS root_anchor,
           supporting_node_ids
    ORDER BY score DESC
    LIMIT $top_k
    """

    DEFAULT_CORRECTION_QUERY = """
    MATCH (intent:Intent {intent_id: $intent_id})
    OPTIONAL MATCH (pattern:CorrectionPattern)-[:WORKS_FOR]->(intent)
    RETURN coalesce(pattern.pattern_id, 'no-pattern') AS path_id,
           'correction' AS path_kind,
           coalesce(pattern.name, 'No prior correction pattern') + ' -> ' + coalesce(pattern.drift_kind, 'unknown') AS path_text,
           coalesce(pattern.success_rate, 0.0) AS score,
           intent.text AS root_anchor,
           CASE WHEN pattern IS NULL THEN [] ELSE [pattern.pattern_id] END AS supporting_node_ids
    ORDER BY score DESC
    LIMIT $top_k
    """

    def __init__(
        self,
        driver: Any,
        *,
        database: str = "neo4j",
        candidate_retriever: Any | None = None,
        top_k: int = 3,
        evidence_query: str | None = None,
        analogous_query: str | None = None,
        correction_query: str | None = None,
    ) -> None:
        self.driver = _validate_neo4j_driver(driver)
        self.database = _require_text(database, "database")
        self.candidate_retriever = candidate_retriever
        self.top_k = max(int(top_k), 1)
        self.evidence_query = evidence_query or self.DEFAULT_EVIDENCE_QUERY
        self.analogous_query = analogous_query or self.DEFAULT_ANALOGOUS_QUERY
        self.correction_query = correction_query or self.DEFAULT_CORRECTION_QUERY

    def _run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **parameters)
            rows = []
            for record in result:
                if hasattr(record, "data"):
                    rows.append(record.data())
                else:
                    rows.append(dict(record))
            return rows

    def _retrieve_candidates(self, query_text: str) -> list[str]:
        if self.candidate_retriever is None:
            return []
        search = getattr(self.candidate_retriever, "search", None)
        retrieve = getattr(self.candidate_retriever, "retrieve", None)
        raw_results: Any
        if callable(search):
            raw_results = search(
                query_text=query_text,
                retriever_config={self.TOP_K_CONFIG_KEY: self.top_k},
            )
        elif callable(retrieve):
            raw_results = retrieve(query_text, top_k=self.top_k)
        elif callable(self.candidate_retriever):
            raw_results = self.candidate_retriever(query_text)
        else:
            return []

        if isinstance(raw_results, Mapping):
            records = raw_results.get("items")
            if records is None:
                records = raw_results.get("records")
            if records is None:
                records = raw_results
        else:
            records = getattr(raw_results, "items", None)
            if callable(records):
                records = None
            if records is None:
                records = getattr(raw_results, "records", None)
                if callable(records):
                    records = None
            if records is None:
                records = raw_results

        candidate_chunk_ids: list[str] = []
        for item in records or ():
            metadata = item
            if not isinstance(item, Mapping):
                metadata = getattr(item, "metadata", None) or {}
            else:
                metadata = item.get("metadata", item)
            if not isinstance(metadata, Mapping):
                continue
            chunk_id = metadata.get("chunk_id") or metadata.get("node_id") or metadata.get("id")
            if chunk_id is not None:
                candidate_chunk_ids.append(str(chunk_id))
        return candidate_chunk_ids

    def retrieve(self, task: str, plan: Any) -> PathRAGContext:
        task_plan = GraphTaskPlan.from_task_and_plan(task, plan)
        candidate_chunk_ids = self._retrieve_candidates(task_plan.intent_text)
        query_params = {
            "intent_id": task_plan.intent_id,
            "subgoal_id": task_plan.active_subgoal.subgoal_id,
            "candidate_chunk_ids": candidate_chunk_ids,
            "top_k": self.top_k,
        }
        evidence_paths = [
            RetrievedPath.from_record(record)
            for record in self._run(self.evidence_query, **query_params)
        ]
        analogous_paths = [
            RetrievedPath.from_record(record)
            for record in self._run(self.analogous_query, **query_params)
        ]
        correction_paths = [
            RetrievedPath.from_record(record)
            for record in self._run(self.correction_query, **query_params)
        ]
        return PathRAGContext(
            current_intent=task_plan.intent_text,
            active_subgoal=task_plan.active_subgoal.text,
            evidence_paths=evidence_paths,
            analogous_prior_paths=analogous_paths,
            correction_paths=correction_paths,
            metadata={"candidate_chunk_ids": candidate_chunk_ids},
        )
