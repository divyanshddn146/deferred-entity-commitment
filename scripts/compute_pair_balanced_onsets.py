"""
compute_pair_balanced_onsets.py

Recomputes relation/entity onset from existing CSVs only.
No model loading. No rerun.

Relation onset:
  average relation_transfer_pct across relation_pair_id at each layer

Entity onset:
  average entity_score across family at each layer

This checks whether relation-before-entity still holds when relation
is pair-balanced instead of raw-record-weighted.

Also saves a full aggregate curve table showing relation/entity scores
side-by-side for each model and layer.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = ROOT / "results/exp1_relation_entity_transfer"

REL_BY_PAIR_CSV = RESULT_DIR / "relation_transfer_by_pair.csv"
ENTITY_SUMMARY_CSV = RESULT_DIR / "entity_transfer_summary.csv"


# ============================================================
# CONFIG
# ============================================================

THRESHOLDS = [0.2, 0.3, 0.4, 0.5]
STABLE_STEPS = 2


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def first_stable_onset(df, score_col, threshold, stable_steps=2):
    """
    First layer where score stays above threshold for stable_steps.
    If no stable run exists, fall back to first layer above threshold.
    """
    df = df.sort_values("layer_idx").reset_index(drop=True)
    vals = df[score_col].tolist()
    layers = df["layer_idx"].tolist()

    for i in range(len(vals) - stable_steps + 1):
        if all(v >= threshold for v in vals[i:i + stable_steps]):
            return int(layers[i])

    for i, v in enumerate(vals):
        if v >= threshold:
            return int(layers[i])

    return None


# ============================================================
# MAIN
# ============================================================

def main():
    rel_by_pair = pd.read_csv(require_file(REL_BY_PAIR_CSV))
    ent_summary = pd.read_csv(require_file(ENTITY_SUMMARY_CSV))

    # ------------------------------------------------------------
    # Relation: pair-balanced average
    #
    # relation_transfer_by_pair.csv should have:
    #   model_name, relation_pair_id, layer_idx, relation_transfer_pct
    #
    # We average relation_pair_id equally at each layer.
    # ------------------------------------------------------------
    relation_pair_balanced = (
        rel_by_pair
        .groupby(["model_name", "layer_idx"], as_index=False)
        .agg(
            relation_score_pair_balanced=("relation_transfer_pct", "mean"),
            n_relation_pairs=("relation_pair_id", "nunique"),
        )
        .sort_values(["model_name", "layer_idx"])
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Entity: family-balanced average
    #
    # entity_transfer_summary.csv should have:
    #   model_name, family, layer_idx, entity_score
    #
    # We average families equally at each layer.
    # ------------------------------------------------------------
    entity_family_balanced = (
        ent_summary
        .groupby(["model_name", "layer_idx"], as_index=False)
        .agg(
            entity_score_family_balanced=("entity_score", "mean"),
            n_entity_families=("family", "nunique"),
        )
        .sort_values(["model_name", "layer_idx"])
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Full aggregate curve table
    #
    # Side-by-side relation/entity scores for each model/layer.
    # This helps inspect the full curves directly, not only onsets.
    # ------------------------------------------------------------
    full_aggregate_curve = (
        relation_pair_balanced
        .merge(
            entity_family_balanced,
            on=["model_name", "layer_idx"],
            how="inner",
        )
        .sort_values(["model_name", "layer_idx"])
        .reset_index(drop=True)
    )

    full_aggregate_curve["relation_minus_entity"] = (
        full_aggregate_curve["relation_score_pair_balanced"]
        - full_aggregate_curve["entity_score_family_balanced"]
    )

    full_aggregate_curve["relation_gt_entity"] = (
        full_aggregate_curve["relation_score_pair_balanced"]
        > full_aggregate_curve["entity_score_family_balanced"]
    )

    # ------------------------------------------------------------
    # Onset table under multiple thresholds
    # ------------------------------------------------------------
    rows = []

    models = sorted(
        set(relation_pair_balanced["model_name"])
        & set(entity_family_balanced["model_name"])
    )

    for model_name in models:
        rel_m = relation_pair_balanced[
            relation_pair_balanced["model_name"] == model_name
        ].sort_values("layer_idx")

        ent_m = entity_family_balanced[
            entity_family_balanced["model_name"] == model_name
        ].sort_values("layer_idx")

        for threshold in THRESHOLDS:
            rel_on = first_stable_onset(
                rel_m,
                "relation_score_pair_balanced",
                threshold,
                stable_steps=STABLE_STEPS,
            )

            ent_on = first_stable_onset(
                ent_m,
                "entity_score_family_balanced",
                threshold,
                stable_steps=STABLE_STEPS,
            )

            rows.append({
                "model_name": model_name,
                "threshold": threshold,
                "relation_onset_pair_balanced": rel_on,
                "entity_onset_family_balanced": ent_on,
                "gap_entity_minus_relation": (
                    ent_on - rel_on
                    if rel_on is not None and ent_on is not None
                    else None
                ),
                "relation_before_entity": (
                    rel_on is not None
                    and ent_on is not None
                    and rel_on < ent_on
                ),
            })

    onset_df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Optional compact summary: threshold=0.4 only
    # ------------------------------------------------------------
    onset_04 = onset_df[onset_df["threshold"] == 0.4].copy()

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    out1 = RESULT_DIR / "onset_pair_balanced_check.csv"
    out2 = RESULT_DIR / "relation_pair_balanced_curve.csv"
    out3 = RESULT_DIR / "entity_family_balanced_curve.csv"
    out4 = RESULT_DIR / "full_pair_family_balanced_aggregate_curve.csv"
    out5 = RESULT_DIR / "onset_pair_balanced_threshold_04.csv"

    onset_df.to_csv(out1, index=False)
    relation_pair_balanced.to_csv(out2, index=False)
    entity_family_balanced.to_csv(out3, index=False)
    full_aggregate_curve.to_csv(out4, index=False)
    onset_04.to_csv(out5, index=False)

    # ------------------------------------------------------------
    # Print outputs
    # ------------------------------------------------------------
    print("\nPAIR-BALANCED ONSET CHECK")
    print("=" * 100)
    print(onset_df.to_string(index=False))

    print("\nPAIR-BALANCED ONSET CHECK — THRESHOLD 0.4 ONLY")
    print("=" * 100)
    print(onset_04.to_string(index=False))

    print("\nFULL PAIR/FAMILY-BALANCED AGGREGATE CURVE")
    print("=" * 100)
    print(full_aggregate_curve.to_string(index=False))

    print(f"\n[saved] {out1}")
    print(f"[saved] {out2}")
    print(f"[saved] {out3}")
    print(f"[saved] {out4}")
    print(f"[saved] {out5}")


if __name__ == "__main__":
    main()