from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.lazy_assignment.experiment3 import run_experiment3_pipeline as pipeline


def test_stage_plan_is_exactly_three_validation_families(tmp_path: Path) -> None:
    stages = pipeline._build_stages(
        python="/env/bin/python",
        experiment2_root=tmp_path / "experiment2",
        run_root=tmp_path / "experiment3",
        device="cuda:0",
        num_workers=4,
    )
    names = [stage.name for stage in stages]

    assert names[0:3] == ["input_audit", "test_experiment3", "ruff_experiment3"]
    assert names[-3:] == [
        "final_source_immutability",
        "rule_selected_examples",
        "final_reports",
    ]
    assert any(name == "validation_a_analysis" for name in names)
    assert any(name == "validation_b_analysis" for name in names)
    assert any(name == "validation_c_analysis" for name in names)
    assert not any(
        "train" in token.lower() for stage in stages for token in stage.command
    )

    full_inference = [
        stage
        for stage in stages
        if stage.name.startswith("validation_") and "_full_" in stage.name
    ]
    assert {stage.name for stage in full_inference} == {
        "validation_a_full_mctformer_plus",
        "validation_a_full_mctformer",
        "validation_b_full_mctformer_plus",
        "validation_b_full_mctformer",
        "validation_c_full_mctformer_plus",
    }
    for stage in full_inference:
        assert stage.command[stage.command.index("--batch-size") + 1] == "8"
        assert "--limit" not in stage.command


def test_initialize_writes_immutable_command_and_pipeline_controls(
    tmp_path: Path, monkeypatch
) -> None:
    experiment2 = tmp_path / "experiment2"
    experiment2.mkdir()
    context = tmp_path / "plan.md"
    context.write_text("frozen plan\n", encoding="utf-8")
    head = "a" * 40
    monkeypatch.setattr(pipeline, "require_tgca_repro", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "git_state",
        lambda _root: {
            "commit": head,
            "branch": "",
            "status_short": [],
            "repository_url": "example.invalid/repo",
            "host": "test",
        },
    )
    monkeypatch.setattr(pipeline, "_git_tag_commit", lambda _tag: head)

    run_root, stages = pipeline.initialize(
        run_root=tmp_path / "run",
        experiment2_root=experiment2,
        implementation_tag="experiment3-production-test",
        device="cpu",
        num_workers=0,
        context_documents=[context],
        invocation=["python", "run_experiment3_pipeline.py"],
    )

    commands = run_root / "exact_commands.sh"
    before = commands.read_bytes()
    metadata = json.loads((run_root / "pipeline_metadata.json").read_text())
    status = json.loads((run_root / "pipeline_status.json").read_text())
    assert commands.stat().st_mode & 0o111
    assert metadata["implementation_commit"] == head
    assert metadata["implementation_tag"] == "experiment3-production-test"
    assert metadata["evaluation"]["bootstrap_repeats"] == 5000
    assert set(status["stages"]) == {stage.name for stage in stages}
    assert all(record["status"] == "pending" for record in status["stages"].values())
    assert commands.read_bytes() == before


def test_record_stage_preserves_the_rest_of_the_ledger(tmp_path: Path) -> None:
    status = {
        "status": "running",
        "stages": {
            "one": {"status": "pending", "command": "one"},
            "two": {"status": "pending", "command": "two"},
        },
    }
    pipeline.json_dump(tmp_path / "pipeline_status.json", status)
    pipeline._record_stage(
        tmp_path,
        pipeline.Stage("one", ("python", "one.py")),
        "complete",
        returncode=0,
    )
    result = pipeline.read_json(tmp_path / "pipeline_status.json")
    assert result["stages"]["one"]["status"] == "complete"
    assert result["stages"]["one"]["returncode"] == 0
    assert result["stages"]["two"] == {"status": "pending", "command": "two"}


def test_initialize_failure_does_not_publish_partial_run_root(
    tmp_path: Path, monkeypatch
) -> None:
    experiment2 = tmp_path / "experiment2"
    experiment2.mkdir()
    head = "b" * 40
    monkeypatch.setattr(pipeline, "require_tgca_repro", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "git_state",
        lambda _root: {"commit": head, "status_short": []},
    )
    monkeypatch.setattr(pipeline, "_git_tag_commit", lambda _tag: head)
    run_root = tmp_path / "run"

    with pytest.raises(FileNotFoundError):
        pipeline.initialize(
            run_root=run_root,
            experiment2_root=experiment2,
            implementation_tag="experiment3-production-test",
            device="cpu",
            num_workers=0,
            context_documents=[tmp_path / "missing-plan.md"],
            invocation=["python", "run_experiment3_pipeline.py"],
        )

    assert not run_root.exists()
    assert not list(tmp_path.glob(".run.initialize-*"))
