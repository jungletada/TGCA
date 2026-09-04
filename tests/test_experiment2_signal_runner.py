from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from analysis.lazy_assignment.experiment2.run_experiment2_signals import (
    SIGNAL_KEYS,
    diagnostic_propagated_cam,
    parse_args,
    qk_head_region_means,
    resolve_inputs,
    save_signal,
    validate_signal_payload,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_metadata(tmp_path, model="mctformer"):
    result = tmp_path / "experiment1"
    result.mkdir()
    (result / "scores").mkdir()
    for name in ("metadata.json", "completion.json"):
        _write_json(result / name, {})
    (result / "manifest.jsonl").write_text("", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"frozen checkpoint fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    voc = tmp_path / "VOC2012"
    voc.mkdir()
    val_list = tmp_path / "val_id.txt"
    val_list.write_text("2007_000001\n", encoding="utf-8")
    factory = "mctformerv2" if model == "mctformer" else "mctformerplus"
    metadata = {
        "status": "complete",
        "integrity_passed": True,
        "sources": {
            model: {
                "result_root": str(result),
                "model_cli_name": factory,
                "checkpoint": {"path": str(checkpoint), "sha256": digest},
            }
        },
        "dataset": {
            "voc_root": str(voc),
            "list_path": str(val_list),
            "input_size": 448,
            "patch_size": 16,
            "num_images": 1,
        },
        "experiment1_metadata": {
            model: {
                "model": {"name": factory},
                "input": {"size": 448},
                "representation": "post_block_pre_final_norm",
            }
        },
    }
    source = tmp_path / "source_metadata.json"
    _write_json(source, metadata)
    return source, result, checkpoint


def test_resolve_inputs_reads_only_stable_audit_schema(tmp_path):
    source, result, checkpoint = _source_metadata(tmp_path)
    resolved = resolve_inputs(source, "mctformer")
    assert resolved.result_root == result.resolve()
    assert resolved.checkpoint == checkpoint.resolve()
    assert resolved.model_factory_name == "mctformerv2"
    assert resolved.expected_images == 1


def test_resolve_inputs_rejects_failed_audit_and_wrong_factory(tmp_path):
    source, _, _ = _source_metadata(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["integrity_passed"] = False
    _write_json(source, payload)
    with pytest.raises(RuntimeError, match="not a completed passing"):
        resolve_inputs(source, "mctformer")

    payload["integrity_passed"] = True
    payload["sources"]["mctformer"]["model_cli_name"] = "mctformerplus"
    _write_json(source, payload)
    with pytest.raises(ValueError, match="model_cli_name"):
        resolve_inputs(source, "mctformer")


def test_cli_fixes_resolution_and_dirty_override_to_smoke_only(tmp_path):
    base = [
        "--model",
        "mctformer",
        "--source-metadata",
        str(tmp_path / "source.json"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    assert parse_args(base).input_size == 448
    assert parse_args(base).batch_size == 8
    with pytest.raises(SystemExit):
        parse_args(base + ["--input-size", "320"])
    with pytest.raises(SystemExit):
        parse_args(base + ["--allow-uncommitted-source"])
    with pytest.raises(SystemExit):
        parse_args(base + ["--batch-size", "1"])
    args = parse_args(base + ["--limit", "2", "--allow-uncommitted-source"])
    assert args.limit == 2 and args.allow_uncommitted_source


def test_qk_region_summary_uses_target_other_background_and_nan_for_empty():
    qk = torch.arange(2 * 3 * 20 * 4, dtype=torch.float32).reshape(2, 3, 20, 4)
    positive = np.asarray([2], dtype=np.int64)
    regions = np.asarray([[0, 1, 2, 3]], dtype=np.int8)
    summary = qk_head_region_means(qk, positive, regions)
    assert summary.shape == (2, 3, 1, 3)
    np.testing.assert_allclose(summary[:, :, 0, 0], qk[:, :, 2, 0].numpy())
    np.testing.assert_allclose(summary[:, :, 0, 1], qk[:, :, 2, 1].numpy())
    np.testing.assert_allclose(summary[:, :, 0, 2], qk[:, :, 2, 2].numpy())

    empty_target = qk_head_region_means(
        qk, positive, np.asarray([[3, 1, 2, 3]], dtype=np.int8)
    )
    assert np.isnan(empty_target[:, :, 0, 0]).all()


def test_diagnostic_cam_preserves_host_fusion_and_p2p_direction():
    patch_cam = torch.tensor([[[[1.0, 4.0]]]])
    c2p = torch.tensor(
        [
            [[[[0.2, 0.4]]]],
            [[[[0.6, 0.8]]]],
            [[[[1.0, 1.2]]]],
        ]
    ).reshape(3, 1, 1, 2)
    p2p = torch.tensor([[[1.0, 0.0], [0.25, 0.75]]])

    mct = diagnostic_propagated_cam("mctformer", patch_cam, c2p, p2p)
    plus = diagnostic_propagated_cam("mctformer_plus", patch_cam, c2p, p2p)
    mct_c1 = c2p.sum(0) * patch_cam.flatten(2)
    plus_c1 = torch.sqrt(c2p.mean(0) * patch_cam.flatten(2))
    expected_mct = (p2p @ mct_c1.transpose(1, 2)).transpose(1, 2).reshape_as(mct)
    expected_plus = (p2p @ plus_c1.transpose(1, 2)).transpose(1, 2).reshape_as(plus)
    torch.testing.assert_close(mct, expected_mct)
    torch.testing.assert_close(plus, expected_plus)


def _valid_payload():
    classes, patches, heads = 2, 4, 3
    payload = {
        "image_id": np.asarray("2007_000001"),
        "positive_class_ids": np.asarray([1, 7], dtype=np.int64),
        "grid_h": np.asarray(2, dtype=np.int32),
        "grid_w": np.asarray(2, dtype=np.int32),
        "patch_label_counts": np.pad(
            np.full((patches, 1), 256, dtype=np.uint16), ((0, 0), (0, 21))
        ),
        "region_masks_rho05": np.zeros((classes, patches), dtype=np.int8),
        "region_masks_rho07": np.zeros((classes, patches), dtype=np.int8),
        "feature_post_scores": np.zeros((12, classes, patches), dtype=np.float32),
        "feature_norm_scores": np.zeros((12, classes, patches), dtype=np.float32),
        "feature_final_norm_scores": np.zeros((classes, patches), dtype=np.float32),
        "qk_mean_scores": np.zeros((12, classes, patches), dtype=np.float32),
        "qk_head_std": np.zeros((12, classes, patches), dtype=np.float32),
        "attn_c2p_raw": np.zeros((12, classes, patches), dtype=np.float32),
        "attn_c2p_conditional": np.zeros((12, classes, patches), dtype=np.float32),
        "attn_patch_mass": np.zeros((12, classes), dtype=np.float32),
        "class_logits": np.zeros(classes, dtype=np.float32),
        "patch_class_logits": np.zeros(classes, dtype=np.float32),
        "class_logits_all": np.zeros(20, dtype=np.float32),
        "patch_class_logits_all": np.zeros(20, dtype=np.float32),
        "raw_final_cam_confusion_t045": np.pad(
            np.asarray([[patches * 256]], dtype=np.int64), ((0, 20), (0, 20))
        ),
        "class_token_pairwise_cosine": np.zeros(
            (12, classes, classes), dtype=np.float32
        ),
        "patch_norms": np.zeros((12, patches), dtype=np.float32),
        "qk_head_region_mean_rho05": np.full(
            (12, heads, classes, 3), np.nan, dtype=np.float32
        ),
        "qk_head_region_mean_rho07": np.full(
            (12, heads, classes, 3), np.nan, dtype=np.float32
        ),
    }
    for name in (
        "patch_logits",
        "patch_cam",
        "attn_official_raw",
        "attn_official_conditional",
        "attn_mid3_raw",
        "attn_mid3_conditional",
        "c2p_cam",
        "final_cam",
        "diagnostic_c2p_cam_l10",
        "diagnostic_c2p_cam_l11",
        "diagnostic_c2p_cam_l12",
        "diagnostic_c2p_cam_mid3",
    ):
        payload[name] = np.zeros((classes, patches), dtype=np.float32)
    assert set(payload) == SIGNAL_KEYS
    return payload


def test_exact_npz_schema_round_trip_and_no_overwrite(tmp_path):
    payload = _valid_payload()
    validate_signal_payload(payload)
    path = tmp_path / "signal.npz"
    digest = save_signal(path, payload)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as artifact:
        assert set(artifact.files) == SIGNAL_KEYS
        assert artifact["positive_class_ids"].dtype == np.int64
        assert artifact["patch_label_counts"].dtype == np.uint16
        assert artifact["feature_post_scores"].dtype == np.float32
    with pytest.raises(FileExistsError):
        save_signal(path, payload)
