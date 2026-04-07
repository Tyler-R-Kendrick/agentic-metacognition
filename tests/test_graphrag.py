from __future__ import annotations

import torch

import activation_steering as steering


def test_path_rag_context_serializes_distinct_sections():
    context = steering.PathRAGContext(
        current_intent="Answer the capital question with evidence.",
        active_subgoal="Use a supported claim about France's capital.",
        evidence_paths=[
            steering.RetrievedPath(
                path_id="evidence-1",
                path_kind="evidence",
                path_text="Intent <- Task -> Subgoal <- RetrievedPath -> Chunk -> Claim -> Entity",
                score=0.95,
                root_anchor="Use a supported claim about France's capital.",
                supporting_node_ids=("chunk-1", "claim-1", "entity-paris"),
                metadata={"support": "Paris is the capital of France."},
            )
        ],
        analogous_prior_paths=[
            steering.RetrievedPath(
                path_id="prior-1",
                path_kind="analogous_prior",
                path_text="Intent -> prior successful Run -> RetrievedPath",
                score=0.6,
                supporting_node_ids=("run-1", "path-1"),
            )
        ],
        correction_paths=[
            steering.RetrievedPath(
                path_id="correction-1",
                path_kind="correction",
                path_text="DriftEvent -> CorrectionPattern -> prior successful correction",
                score=0.4,
                supporting_node_ids=("drift-1", "pattern-1"),
            )
        ],
    )

    sections = context.to_prompt_sections()
    rendered = "\n\n".join(sections)

    assert "CURRENT_INTENT" in rendered
    assert "ACTIVE_SUBGOAL" in rendered
    assert "EVIDENCE_PATHS" in rendered
    assert "ANALOGOUS_PRIOR_PATHS" in rendered
    assert "CORRECTION_PATHS" in rendered
    assert "nodes: chunk-1, claim-1, entity-paris" in rendered
    assert "support: Paris is the capital of France." in rendered


def test_hybrid_agent_records_graph_state_for_path_rag_fallback():
    controller = steering.SteeringController(
        controller_id="retrieval_augmented_context_v1",
        feature_name="retrieval_augmented_context",
        layer_idx=0,
        vector=torch.tensor([1.0, 0.0]),
        task_types=("qa",),
    )
    memory = steering.InMemorySteeringMemory([controller])

    class StubExecutor:
        def __init__(self):
            self.calls = []

        def execute(self, task, plan, context, controller=None, controllers=(), max_new_tokens=80):
            self.calls.append(controller.controller_id if controller else None)
            return steering.ExecutorResult(
                prompt=f"task={task}\ncontext={' | '.join(context)}",
                output_text="steered answer" if controller else "fallback answer",
                controller_id=controller.controller_id if controller else None,
            )

    class StubGraphStore:
        def __init__(self):
            self.started = []
            self.states = []
            self.verifier_results = []
            self.corrections = []
            self.outcomes = []

        def start_run(self, task, plan):
            self.started.append((task, plan.task_type))
            return steering.GraphRunHandle(
                run_id="run-1",
                task_plan=steering.GraphTaskPlan.from_task_and_plan(task, plan),
            )

        def record_state(self, run_handle, **kwargs):
            self.states.append((run_handle.run_id, kwargs))
            return f"state-{len(self.states)}"

        def record_verifier_result(self, run_handle, **kwargs):
            self.verifier_results.append((run_handle.run_id, kwargs))

        def record_drift_and_correction(self, run_handle, **kwargs):
            self.corrections.append((run_handle.run_id, kwargs))

        def record_outcome(self, run_handle, **kwargs):
            self.outcomes.append((run_handle.run_id, kwargs))

    executor = StubExecutor()
    graph_store = StubGraphStore()

    def planner(task, memory_store):
        return steering.PlannerDecision(
            task_type="qa",
            needs_retrieval=True,
            use_steering=True,
            allow_fallback=True,
            metadata={
                "intent_text": "Answer the capital question with evidence.",
                "subgoals": [
                    {"subgoal_id": "sg-1", "text": "Use supported France evidence.", "order": 1}
                ],
                "active_subgoal_id": "sg-1",
            },
        )

    def retriever(task, plan):
        return steering.PathRAGContext(
            current_intent="Answer the capital question with evidence.",
            active_subgoal="Use supported France evidence.",
            evidence_paths=[
                steering.RetrievedPath(
                    path_id="path-1",
                    path_kind="evidence",
                    path_text="Chunk -> Claim <- Evidence",
                    score=0.9,
                    supporting_node_ids=("chunk-1", "claim-1"),
                )
            ],
        )

    def verifier(task, draft, context, plan):
        return steering.VerifierResult(
            passed=draft.controller_id is None,
            confidence=0.95 if draft.controller_id is None else 0.2,
            issues=[] if draft.controller_id is None else ["needs fallback"],
        )

    agent = steering.HybridMetaCognitionAgent(
        planner=planner,
        retriever=retriever,
        executor=executor,
        verifier=verifier,
        memory=memory,
        graph_store=graph_store,
    )

    run = agent.run("What is the capital of France?")

    assert executor.calls == ["retrieval_augmented_context_v1", None]
    assert run.fallback_used is True
    assert run.context[0].startswith("CURRENT_INTENT")
    assert run.metadata["graph_run_id"] == "run-1"
    assert isinstance(run.path_context, steering.PathRAGContext)
    assert len(graph_store.states) == 3
    assert graph_store.corrections[0][1]["action"] == "fallback_to_unsteered_execution"
    assert graph_store.outcomes[0][1]["verdict"].passed is True


def test_neo4j_path_rag_retriever_builds_context_from_candidate_ids():
    class StubCandidateRetriever:
        def search(self, query_text, retriever_config):
            assert query_text == "Answer the capital question with evidence."
            assert retriever_config["top_k"] == 2
            return [{"metadata": {"chunk_id": "chunk-1"}}]

    class StubNeo4jPathRAGRetriever(steering.Neo4jPathRAGRetriever):
        def __init__(self):
            super().__init__(driver=None, candidate_retriever=StubCandidateRetriever(), top_k=2)
            self.calls = []

        def _run(self, query: str, **parameters):
            self.calls.append(parameters)
            if query == self.evidence_query:
                return [
                    {
                        "path_id": "evidence-1",
                        "path_kind": "evidence",
                        "path_text": "Chunk -> Claim <- Evidence",
                        "score": 0.8,
                        "root_anchor": "Use supported France evidence.",
                        "supporting_node_ids": ["chunk-1", "claim-1"],
                    }
                ]
            if query == self.analogous_query:
                return []
            if query == self.correction_query:
                return []
            raise AssertionError("Unexpected query")

    plan = steering.PlannerDecision(
        task_type="qa",
        metadata={
            "intent_id": "intent-1",
            "intent_text": "Answer the capital question with evidence.",
            "subgoals": [{"subgoal_id": "sg-1", "text": "Use supported France evidence.", "order": 1}],
            "active_subgoal_id": "sg-1",
        },
    )
    retriever = StubNeo4jPathRAGRetriever()

    context = retriever.retrieve("What is the capital of France?", plan)

    assert context.current_intent == "Answer the capital question with evidence."
    assert context.active_subgoal == "Use supported France evidence."
    assert [path.path_id for path in context.evidence_paths] == ["evidence-1"]
    assert context.metadata["candidate_chunk_ids"] == ["chunk-1"]
    assert all(call["candidate_chunk_ids"] == ["chunk-1"] for call in retriever.calls)
