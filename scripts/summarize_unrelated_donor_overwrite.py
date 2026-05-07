"""
summarize_unrelated_donor_overwrite.py

Summarize the unrelated-donor overwrite diagnostic for Experiment 2.

This script uses existing CSV results only. It does not load models or rerun
activation patching.

Interpretation:
Unrelated-donor patches test whether late final-token states become broadly
overwrite-sensitive. The key paper-relevant check is that unrelated relation-wins
remain near zero, so mid-layer relation dominance in the both-change experiment
is not explained by generic patching.
"""


from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = ROOT / "results/exp2_both_change"

df = pd.read_csv(RESULT_DIR / "unrelated_ctrl_all_models_raw.csv")

summary = (
    df.groupby(["model_name", "layer_idx"], as_index=False)
    .agg(
        unrelated_entity_wins=("entity_wins", "mean"),
        unrelated_relation_wins=("relation_wins", "mean"),
        unrelated_original_retained=("original_retained", "mean"),
        unrelated_mixed_or_other=("mixed_or_other", "mean"),
        n=("entity_wins", "count"),
    )
)

peak_entity = summary.loc[
    summary.groupby("model_name")["unrelated_entity_wins"].idxmax()
].sort_values("model_name")

peak_relation = summary.loc[
    summary.groupby("model_name")["unrelated_relation_wins"].idxmax()
].sort_values("model_name")

print("\nUNRELATED DONOR — PEAK ENTITY-LIKE OVERWRITE")
print(peak_entity[[
    "model_name",
    "layer_idx",
    "unrelated_entity_wins",
    "unrelated_relation_wins",
    "unrelated_original_retained",
    "unrelated_mixed_or_other",
    "n",
]].to_string(index=False))

print("\nUNRELATED DONOR — PEAK RELATION-WINS")
print(peak_relation[[
    "model_name",
    "layer_idx",
    "unrelated_entity_wins",
    "unrelated_relation_wins",
    "unrelated_original_retained",
    "unrelated_mixed_or_other",
    "n",
]].to_string(index=False))

summary.to_csv(RESULT_DIR / "unrelated_donor_by_layer_summary.csv", index=False)
peak_entity.to_csv(RESULT_DIR / "unrelated_donor_peak_entity_overwrite.csv", index=False)
peak_relation.to_csv(RESULT_DIR / "unrelated_donor_peak_relation_wins.csv", index=False)