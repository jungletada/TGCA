#!/usr/bin/env python3
"""Deterministically select Experiment 2 examples from canonical metrics.

Selection is rule-based. This stage writes only a CSV manifest and metadata;
the separate model-free renderer consumes that manifest and existing NPZ maps.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from analysis.lazy_assignment.experiment2.common import sha256_file  # noqa: E402


VOC_CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
CANONICAL_FILES = {
    "layer": "per_image_class_layer_signal.parquet",
    "cam": "per_image_class_cam_stage.parquet",
    "transition": "per_image_class_stage_transition.parquet",
    "shared": "per_shared_patch_ownership.parquet",
    "token": "per_class_token_pair_layer.parquet",
}
OUTPUT_FIELDS = (
    "case_id",
    "category",
    "selection_rank",
    "model",
    "image_id",
    "class_id",
    "class_name",
    "companion_class_id",
    "companion_class_name",
    "class_a",
    "class_a_name",
    "class_b",
    "class_b_name",
    "positive_class_ids_json",
    "num_positive_classes",
    "layer_or_stage",
    "selection_metric",
    "selection_value",
    "selection_direction",
    "selection_rule",
    "source_table",
    "source_row_index",
    "experiment1_case_id",
    "experiment1_raw_figure",
    "experiment1_minmax_figure",
)
NEW_CATEGORIES = (
    "shared_support_mostly_background",
    "shared_support_mostly_target_a",
    "shared_support_mostly_target_b",
    "feature_bg_high_attention_filters",
    "attention_introduces_background",
    "p2p_propagation_introduces_background",
    "raw_cosine_fails_attention_cam_succeeds",
    "all_three_stages_fail",
    "train_representative",
    "bird_negative_cosine_control",
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment1-analysis-root", type=Path)
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--per-category", type=int, default=10)
    args = parser.parse_args(argv)
    if args.per_category < 1:
        parser.error("--per-category must be positive")
    return args


def _load_canonical(directory: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for key, filename in CANONICAL_FILES.items():
        path = directory / filename
        result[key] = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
    return result


def _preferred(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column, value in (
        ("model", "mctformer_plus"),
        ("rho", 0.5),
        ("topk_ratio", 0.1),
    ):
        if column not in result:
            continue
        if isinstance(value, float):
            numeric = pd.to_numeric(result[column], errors="coerce")
            result = result[np.isclose(numeric, value, atol=1e-9)]
        else:
            result = result[result[column].astype(str) == value]
    return result


def _layer_number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"L?(\d+)", str(value))
        return float(match.group(1)) if match else float("nan")


def _at_layer(frame: pd.DataFrame, layer: int) -> pd.DataFrame:
    for column in ("layer", "layer_or_stage"):
        if column in frame:
            return frame[np.isclose(frame[column].map(_layer_number), float(layer))]
    return frame.iloc[0:0]


def _class_name(value: object) -> str:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return ""
    return VOC_CLASS_NAMES[index] if 0 <= index < len(VOC_CLASS_NAMES) else ""


def _first_value(row: pd.Series, *names: str) -> object:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return ""


def _optional_class_id(value: object) -> Optional[int]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"invalid class ID in example selection: {value!r}")
    class_id = int(numeric)
    if not 0 <= class_id < len(VOC_CLASS_NAMES):
        raise ValueError(f"class ID outside [0,19]: {class_id}")
    return class_id


def _positive_classes_by_image(
    layer: pd.DataFrame,
) -> dict[tuple[str, str], tuple[int, ...]]:
    required = {"model", "image_id", "class_id"}
    if not required.issubset(layer.columns):
        missing = sorted(required.difference(layer.columns))
        raise ValueError(
            f"canonical layer table lacks positive-class identity columns: {missing}"
        )
    result: dict[tuple[str, str], tuple[int, ...]] = {}
    identities = layer[["model", "image_id", "class_id"]].drop_duplicates()
    for (model, image_id), rows in identities.groupby(["model", "image_id"], sort=True):
        parsed = [_optional_class_id(value) for value in rows["class_id"]]
        if not parsed or any(value is None for value in parsed):
            raise ValueError(f"missing positive class IDs for {(model, image_id)}")
        result[(str(model), str(image_id))] = tuple(
            sorted({int(value) for value in parsed if value is not None})
        )
    return result


def _attach_positive_class_context(
    selected: list[dict[str, object]], layer: pd.DataFrame
) -> None:
    """Bind every new case to all GT-positive classes used by its panel.

    A focal single-class selection on a multi-label image receives the smallest
    other positive class as a deterministic companion. Pair selections retain
    their selected pair. Fixed Experiment 1 examples remain immutable links.
    """

    positive_by_image = _positive_classes_by_image(layer)
    for row in selected:
        if str(row.get("category", "")).startswith("experiment1_fixed::"):
            row.update(
                {
                    "companion_class_id": "",
                    "companion_class_name": "",
                    "positive_class_ids_json": "",
                    "num_positive_classes": "",
                }
            )
            continue
        key = (str(row.get("model", "")), str(row.get("image_id", "")))
        if key not in positive_by_image:
            raise ValueError(
                f"selected Experiment 2 case is absent from layer table: {key}"
            )
        positive = positive_by_image[key]
        class_id = _optional_class_id(row.get("class_id"))
        class_a = _optional_class_id(row.get("class_a"))
        class_b = _optional_class_id(row.get("class_b"))
        selected_ids = (
            (class_a, class_b)
            if class_a is not None and class_b is not None
            else (class_id,)
        )
        if any(value is None or value not in positive for value in selected_ids):
            raise ValueError(
                f"selected classes {selected_ids} are not all GT-positive for "
                f"{key}: {positive}"
            )
        companion = None
        if class_id is not None and len(positive) >= 2:
            companion = next(value for value in positive if value != class_id)
        row.update(
            {
                "companion_class_id": companion if companion is not None else "",
                "companion_class_name": _class_name(companion),
                "positive_class_ids_json": json.dumps(list(positive)),
                "num_positive_classes": len(positive),
            }
        )


def _selection_rows(
    frame: pd.DataFrame,
    *,
    category: str,
    metric: str,
    direction: str,
    rule: str,
    source_table: str,
    count: int,
) -> list[dict[str, object]]:
    if frame.empty or metric not in frame:
        return []
    candidates = frame.copy()
    candidates[metric] = pd.to_numeric(candidates[metric], errors="coerce")
    candidates = candidates[np.isfinite(candidates[metric])]
    if candidates.empty:
        return []
    tie_columns = [
        column
        for column in (
            "image_id",
            "class_id",
            "class_a",
            "class_a_id",
            "class_b",
            "class_b_id",
        )
        if column in candidates
    ]
    candidates = candidates.sort_values(
        [metric, *tie_columns],
        ascending=[direction == "smallest", *([True] * len(tie_columns))],
        kind="stable",
    ).head(count)
    rows: list[dict[str, object]] = []
    for rank, (source_index, row) in enumerate(candidates.iterrows(), start=1):
        class_id = _first_value(row, "class_id")
        class_a = _first_value(row, "class_a", "class_a_id")
        class_b = _first_value(row, "class_b", "class_b_id")
        rows.append(
            {
                "category": category,
                "selection_rank": rank,
                "model": _first_value(row, "model"),
                "image_id": _first_value(row, "image_id"),
                "class_id": class_id,
                "class_name": _class_name(class_id),
                "class_a": class_a,
                "class_a_name": _class_name(class_a),
                "class_b": class_b,
                "class_b_name": _class_name(class_b),
                "layer_or_stage": _first_value(
                    row, "layer", "layer_or_stage", "stage", "transition"
                ),
                "selection_metric": metric,
                "selection_value": float(row[metric]),
                "selection_direction": direction,
                "selection_rule": rule,
                "source_table": source_table,
                "source_row_index": source_index,
                "experiment1_case_id": "",
                "experiment1_raw_figure": "",
                "experiment1_minmax_figure": "",
            }
        )
    return rows


def _contains_all(series: pd.Series, terms: tuple[str, ...]) -> pd.Series:
    text = series.astype(str).str.lower()
    selected = pd.Series(True, index=series.index)
    for term in terms:
        selected &= text.str.contains(term, regex=False)
    return selected


def _transition_candidates(
    frame: pd.DataFrame, terms: tuple[str, ...], score_kind: str
) -> tuple[pd.DataFrame, str]:
    if frame.empty or "transition" not in frame:
        return frame.iloc[0:0], "_selection_score"
    selected = frame[_contains_all(frame["transition"], terms)].copy()
    if score_kind == "filter":
        required = {"removed_background_fraction", "survive_background"}
        if not required.issubset(selected.columns):
            return selected.iloc[0:0], "_selection_score"
        selected["_selection_score"] = pd.to_numeric(
            selected["removed_background_fraction"], errors="coerce"
        ) - pd.to_numeric(selected["survive_background"], errors="coerce")
    else:
        if "introduced_background_fraction" not in selected:
            return selected.iloc[0:0], "_selection_score"
        selected["_selection_score"] = pd.to_numeric(
            selected["introduced_background_fraction"], errors="coerce"
        )
    return selected, "_selection_score"


def _signal_wide(layer: pd.DataFrame, cam: pd.DataFrame) -> pd.DataFrame:
    layer = _at_layer(_preferred(layer), 12)
    required = {"image_id", "class_id", "signal"}
    metrics = [
        name
        for name in ("auc_target_bg", "bg_tail_enrich_10", "score_q95")
        if name in layer
    ]
    if not required.issubset(layer.columns) or not metrics:
        return pd.DataFrame()
    pieces = []
    keys = [column for column in ("model", "image_id", "class_id") if column in layer]
    signal_aliases = (
        ("feature_post", ("feature_post",)),
        ("attn", ("attn_c2p_conditional", "attn")),
    )
    for logical_name, aliases in signal_aliases:
        available = layer["signal"].astype(str)
        selected_alias = next(
            (name for name in aliases if (available == name).any()), None
        )
        if selected_alias is None:
            pieces.append(pd.DataFrame())
            continue
        subset = layer[available == selected_alias][[*keys, *metrics]].copy()
        subset = subset.rename(
            columns={metric: f"{logical_name}_{metric}" for metric in metrics}
        )
        pieces.append(subset)
    if any(piece.empty for piece in pieces):
        return pd.DataFrame()
    result = pieces[0].merge(pieces[1], on=keys, how="inner", validate="one_to_one")
    cam = _preferred(cam)
    if not cam.empty and {"image_id", "class_id", "stage"}.issubset(cam.columns):
        final = cam[cam["stage"].astype(str) == "final_cam"]
        cam_metrics = [
            name for name in ("auc_target_bg", "bg_tail_enrich_10") if name in final
        ]
        if cam_metrics:
            final = final[[*keys, *cam_metrics]].rename(
                columns={metric: f"final_cam_{metric}" for metric in cam_metrics}
            )
            result = result.merge(final, on=keys, how="inner", validate="one_to_one")
    result["layer"] = 12
    return result


def _find_experiment1_selection(
    canonical_dir: Path,
    explicit_root: Optional[Path],
    source_metadata: Optional[Path],
) -> Optional[Path]:
    roots: list[Path] = []
    if explicit_root is not None:
        roots.append(explicit_root.expanduser().resolve())
    metadata_candidates = []
    if source_metadata is not None:
        metadata_candidates.append(source_metadata)
    metadata_candidates.append(canonical_dir.parent / "audit/source_metadata.json")
    for path in metadata_candidates:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload.get("paired_analysis_root")
            if isinstance(value, str):
                roots.append(Path(value).expanduser().resolve())
    for root in roots:
        candidate = root / "examples/example_selection.csv"
        if candidate.is_file():
            return candidate
    return None


def select_examples(
    canonical_dir: Path,
    output_dir: Path,
    *,
    per_category: int = 10,
    experiment1_analysis_root: Optional[Path] = None,
    source_metadata: Optional[Path] = None,
    command: Optional[str] = None,
) -> dict[str, object]:
    canonical_dir = canonical_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite example directory: {output_dir}")
    frames = _load_canonical(canonical_dir)
    layer = frames["layer"]
    cam = frames["cam"]
    transition = _preferred(frames["transition"])
    shared = _at_layer(_preferred(frames["shared"]), 12)
    if "signal" in shared:
        shared = shared[shared["signal"].astype(str) == "feature_post"]
    wide = _signal_wide(layer, cam)
    selected: list[dict[str, object]] = []

    for category, metric, rule in (
        (
            "shared_support_mostly_background",
            "shared_background_fraction",
            "largest L12 feature shared-top10 background fraction",
        ),
        (
            "shared_support_mostly_target_a",
            "shared_target_a_fraction",
            "largest L12 feature shared-top10 target-A fraction",
        ),
        (
            "shared_support_mostly_target_b",
            "shared_target_b_fraction",
            "largest L12 feature shared-top10 target-B fraction",
        ),
    ):
        selected.extend(
            _selection_rows(
                shared,
                category=category,
                metric=metric,
                direction="largest",
                rule=rule,
                source_table=CANONICAL_FILES["shared"],
                count=per_category,
            )
        )

    feature_filter, metric = _transition_candidates(
        transition, ("feature", "attn"), "filter"
    )
    feature_filter = _at_layer(feature_filter, 12)
    feature_enrichment = "feature_post_bg_tail_enrich_10"
    join_keys = [
        name
        for name in ("model", "image_id", "class_id")
        if name in feature_filter and name in wide
    ]
    if feature_enrichment in wide and join_keys:
        feature_filter = feature_filter.merge(
            wide[[*join_keys, feature_enrichment]],
            on=join_keys,
            how="inner",
            validate="many_to_one",
        )
        feature_filter = feature_filter[feature_filter[feature_enrichment] > 1.0]
    else:
        # Without a feature-stage enrichment measurement the named category
        # cannot be established, so emit no case rather than infer one.
        feature_filter = feature_filter.iloc[0:0]
    selected.extend(
        _selection_rows(
            feature_filter,
            category="feature_bg_high_attention_filters",
            metric=metric,
            direction="largest",
            rule="largest removed-BG minus BG-survival for feature-to-attention top10 transition",
            source_table=CANONICAL_FILES["transition"],
            count=per_category,
        )
    )
    attention_intro, metric = _transition_candidates(
        transition, ("feature", "attn"), "introduce"
    )
    selected.extend(
        _selection_rows(
            attention_intro,
            category="attention_introduces_background",
            metric=metric,
            direction="largest",
            rule="largest introduced-background fraction at feature-to-attention transition",
            source_table=CANONICAL_FILES["transition"],
            count=per_category,
        )
    )
    propagation = transition.iloc[0:0]
    if "transition" in transition:
        text = transition["transition"].astype(str).str.lower()
        propagation = transition[
            text.str.contains("p2p", regex=False)
            | (
                text.str.contains("c2p", regex=False)
                & text.str.contains("final", regex=False)
            )
        ].copy()
    if "introduced_background_fraction" in propagation:
        propagation["_selection_score"] = pd.to_numeric(
            propagation["introduced_background_fraction"], errors="coerce"
        )
    selected.extend(
        _selection_rows(
            propagation,
            category="p2p_propagation_introduces_background",
            metric="_selection_score",
            direction="largest",
            rule="largest introduced-background fraction at c2p-CAM to final/P2P transition",
            source_table=CANONICAL_FILES["transition"],
            count=per_category,
        )
    )

    reversal_required = {
        "feature_post_auc_target_bg",
        "attn_auc_target_bg",
        "final_cam_auc_target_bg",
    }
    reversal = wide.iloc[0:0]
    if reversal_required.issubset(wide.columns):
        reversal = wide[
            (wide["feature_post_auc_target_bg"] < 0.5)
            & (wide["attn_auc_target_bg"] > 0.5)
            & (wide["final_cam_auc_target_bg"] > 0.5)
        ].copy()
        reversal["_selection_score"] = (
            wide.loc[reversal.index, "attn_auc_target_bg"]
            + wide.loc[reversal.index, "final_cam_auc_target_bg"]
            - 2 * wide.loc[reversal.index, "feature_post_auc_target_bg"]
        )
    selected.extend(
        _selection_rows(
            reversal,
            category="raw_cosine_fails_attention_cam_succeeds",
            metric="_selection_score",
            direction="largest",
            rule="raw feature AUROC<0.5 while attention and final-CAM AUROC>0.5; rank by reversal margin",
            source_table=f"{CANONICAL_FILES['layer']} + {CANONICAL_FILES['cam']}",
            count=per_category,
        )
    )
    failure_required = {
        "feature_post_bg_tail_enrich_10",
        "attn_bg_tail_enrich_10",
        "final_cam_bg_tail_enrich_10",
    }
    failures = wide.iloc[0:0]
    if failure_required.issubset(wide.columns):
        failures = wide[
            (wide["feature_post_bg_tail_enrich_10"] > 1)
            & (wide["attn_bg_tail_enrich_10"] > 1)
            & (wide["final_cam_bg_tail_enrich_10"] > 1)
        ].copy()
        failures["_selection_score"] = failures[list(failure_required)].sum(axis=1)
    selected.extend(
        _selection_rows(
            failures,
            category="all_three_stages_fail",
            metric="_selection_score",
            direction="largest",
            rule="feature, attention, and final CAM all have BG-tail enrichment >1; rank by sum",
            source_table=f"{CANONICAL_FILES['layer']} + {CANONICAL_FILES['cam']}",
            count=per_category,
        )
    )

    late_feature = _at_layer(_preferred(layer), 12)
    if "signal" in late_feature:
        late_feature = late_feature[
            late_feature["signal"].astype(str) == "feature_post"
        ]
    train = (
        late_feature[late_feature["class_id"] == 18].copy()
        if "class_id" in late_feature
        else pd.DataFrame()
    )
    if not train.empty and "bg_tail_enrich_10" in train:
        values = pd.to_numeric(train["bg_tail_enrich_10"], errors="coerce")
        train["_selection_score"] = np.abs(values - values.median())
    selected.extend(
        _selection_rows(
            train,
            category="train_representative",
            metric="_selection_score",
            direction="smallest",
            rule="L12 train feature BG enrichment closest to the train-sample median",
            source_table=CANONICAL_FILES["layer"],
            count=per_category,
        )
    )
    bird = (
        late_feature[late_feature["class_id"] == 2].copy()
        if "class_id" in late_feature
        else pd.DataFrame()
    )
    if "score_q95" in bird:
        bird = bird[pd.to_numeric(bird["score_q95"], errors="coerce") < 0]
    selected.extend(
        _selection_rows(
            bird,
            category="bird_negative_cosine_control",
            metric="score_q95",
            direction="smallest",
            rule="most negative L12 bird feature q95; negative-cosine probe control",
            source_table=CANONICAL_FILES["layer"],
            count=per_category,
        )
    )

    exp1_selection = _find_experiment1_selection(
        canonical_dir, experiment1_analysis_root, source_metadata
    )
    fixed_count = 0
    if exp1_selection is not None:
        fixed = pd.read_csv(exp1_selection)
        for _, row in fixed.sort_values("case_id", kind="stable").iterrows():
            class_id = _first_value(row, "class_id")
            selected.append(
                {
                    "category": f"experiment1_fixed::{_first_value(row, 'category')}",
                    "selection_rank": int(_first_value(row, "selection_rank") or 0),
                    "model": _first_value(row, "selection_scope_model"),
                    "image_id": _first_value(row, "image_id"),
                    "class_id": class_id,
                    "class_name": _class_name(class_id),
                    "class_a": "",
                    "class_a_name": "",
                    "class_b": _first_value(row, "comparison_class_id"),
                    "class_b_name": _first_value(row, "comparison_class_name"),
                    "layer_or_stage": _first_value(row, "selection_layer"),
                    "selection_metric": _first_value(row, "selection_metric"),
                    "selection_value": _first_value(row, "selection_value"),
                    "selection_direction": _first_value(row, "selection_direction"),
                    "selection_rule": "fixed Experiment 1 rule-selected case; retained without reselection",
                    "source_table": str(exp1_selection),
                    "source_row_index": "",
                    "experiment1_case_id": _first_value(row, "case_id"),
                    "experiment1_raw_figure": _first_value(row, "raw_cosine_figure"),
                    "experiment1_minmax_figure": _first_value(row, "minmax_figure"),
                }
            )
            fixed_count += 1

    _attach_positive_class_context(selected, layer)
    for index, row in enumerate(selected, start=1):
        row["case_id"] = f"exp2_case_{index:04d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    output_frame = pd.DataFrame(selected, columns=OUTPUT_FIELDS)
    selection_path = output_dir / "example_selection.csv"
    output_frame.to_csv(selection_path, index=False, float_format="%.10g")
    category_counts = output_frame["category"].value_counts().sort_index().to_dict()
    metadata = {
        "canonical_dir": str(canonical_dir),
        "selection_is_deterministic": True,
        "manual_cherry_picking": False,
        "per_new_category_limit": per_category,
        "new_categories": list(NEW_CATEGORIES),
        "missing_new_categories": [
            name for name in NEW_CATEGORIES if name not in category_counts
        ],
        "category_counts": {
            str(key): int(value) for key, value in category_counts.items()
        },
        "experiment1_selection_path": str(exp1_selection) if exp1_selection else None,
        "experiment1_fixed_rows_retained": fixed_count,
        "multilabel_new_cases": sum(
            int(row.get("num_positive_classes") or 0) >= 2
            for row in selected
            if not str(row.get("category", "")).startswith("experiment1_fixed::")
        ),
        "multilabel_panel_contract": (
            "Each selected Experiment 2 case from a multi-label image explicitly "
            "records at least two GT-positive display classes."
        ),
        "visual_panels_generated": False,
        "visual_panel_note": (
            "Selection does not render. render_experiment2_examples.py reads the "
            "manifested immutable signal NPZ files and fails on missing maps."
        ),
        "total_rows": len(output_frame),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "command": command,
    }
    (output_dir / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    metadata = select_examples(
        args.canonical_dir,
        args.output_dir,
        per_category=args.per_category,
        experiment1_analysis_root=args.experiment1_analysis_root,
        source_metadata=args.source_metadata,
        command=shlex.join([sys.executable, *sys.argv]),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
