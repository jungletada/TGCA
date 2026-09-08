"""Compact report-table contracts independent of model/data execution."""

from __future__ import annotations

import pandas as pd

from analysis.relational_selector.finalize_report import _shared_table, _selector_main


def test_compact_table_column_contracts(tmp_path) -> None:
    pd.DataFrame({
        "layer": [1], "raw_pair_corr": [0.1], "residual_pair_corr": [0.2],
        "raw_pair_jaccard_top05": [0.3], "residual_pair_jaccard_top05": [0.4],
        "common_r2": [0.5], "common_fg_top05": [0.6], "common_bg_top05": [0.7],
    }).to_csv(tmp_path / "task_a_layer_summary.csv", index=False)
    pd.DataFrame({"layer": [1], "pc1_fraction": [0.8], "r95": [6], "pc4_cumulative": [0.9], "effective_rank": [2.0]}).to_csv(tmp_path / "task_a_pca.csv", index=False)
    shared = _shared_table(tmp_path)
    assert list(shared.columns)[:9] == ["layer", "raw_corr", "residual_corr", "raw_collision_at5", "residual_collision_at5", "common_r2", "common_fg_at5", "common_bg_at5", "pca_pc1"]
    pd.DataFrame({
        "selector": ["S0"], "num_images": [1], "num_image_class": [1],
        "auc_target_other": [0.8], "ap_target_other": [0.7], "auc_target_bg": [0.9],
        "target_top05_fraction": [0.5], "other_fg_top05_fraction": [0.2], "bg_top05_fraction": [0.3], "pair_jaccard_top05": [0.1],
    }).to_csv(tmp_path / "selector_summary.csv", index=False)
    selector = _selector_main(tmp_path)
    assert list(selector.columns) == ["selector", "num_images", "num_image_class", "target_vs_other_auroc", "target_vs_other_ap", "target_vs_bg_auroc", "target_at5", "other_fg_at5", "bg_at5", "collision_at5"]
