from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_CREATOR_ROOT = REPO_ROOT / "skills" / "skill-creator"
sys.path.insert(0, str(SKILL_CREATOR_ROOT))

spec = importlib.util.spec_from_file_location(
    "skill_creator_run_loop",
    SKILL_CREATOR_ROOT / "scripts" / "run_loop.py",
)
assert spec is not None
assert spec.loader is not None
run_loop_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_loop_module)


def test_run_loop_honors_min_iterations_before_early_exit(tmp_path, monkeypatch) -> None:
    skill_path = tmp_path / "demo-skill"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo description\n---\n\n# Demo skill\n",
        encoding="utf-8",
    )

    call_count = 0

    def fake_run_eval(**kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "results": [
                {
                    "query": "demo query",
                    "should_trigger": True,
                    "trigger_rate": 1.0,
                    "triggers": 1,
                    "runs": 1,
                    "pass": True,
                }
            ],
            "summary": {"passed": 1, "failed": 0, "total": 1},
        }

    def fail_improve_description(**kwargs):
        raise AssertionError("improve_description should not be called when all evals already pass")

    monkeypatch.setattr(run_loop_module, "run_eval", fake_run_eval)
    monkeypatch.setattr(run_loop_module, "improve_description", fail_improve_description)
    monkeypatch.setattr(run_loop_module, "find_project_root", lambda: tmp_path)

    result = run_loop_module.run_loop(
        eval_set=[{"query": "demo query", "should_trigger": True}],
        skill_path=skill_path,
        description_override=None,
        num_workers=1,
        timeout=1,
        max_iterations=5,
        min_iterations=3,
        runs_per_query=1,
        trigger_threshold=0.5,
        holdout=0,
        model="demo-model",
        verbose=False,
    )

    assert call_count == 3
    assert result["iterations_run"] == 3
    assert result["best_description"] == "demo description"


def test_load_eval_set_accepts_evals_json_schema(tmp_path) -> None:
    eval_file = tmp_path / "evals.json"
    eval_file.write_text(
        """
        {
          "skill_name": "demo-skill",
          "evals": [
            {
              "id": 1,
              "prompt": "run the demo skill",
              "should_trigger": true,
              "expected_output": "Trigger the skill.",
              "files": []
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    assert run_loop_module.load_eval_set(eval_file) == [
        {
            "query": "run the demo skill",
            "should_trigger": True,
        }
    ]


def test_load_eval_set_rejects_entries_without_trigger_expectation(tmp_path) -> None:
    eval_file = tmp_path / "evals.json"
    eval_file.write_text(
        """
        {
          "skill_name": "demo-skill",
          "evals": [
            {
              "id": 1,
              "prompt": "run the demo skill",
              "expected_output": "Trigger the skill.",
              "files": []
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    try:
        run_loop_module.load_eval_set(eval_file)
    except ValueError as exc:
        assert "Missing should_trigger" in str(exc)
    else:
        raise AssertionError("Expected load_eval_set to reject eval entries without should_trigger")
