"""
make_paper_figures.py

Generate the final paper figures from precomputed experiment CSVs.

This script does not run model inference or activation patching. It only loads
saved result CSVs from the four main experiments and writes the figures used in
the paper/review package.

Inputs:
  results/exp1_relation_entity_transfer/
  results/exp2_both_change/
  results/exp3_subject_token_patching/
  results/exp4_steering/

Outputs:
  figures/
"""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MAIN_DIR  = ROOT / "results/exp1_relation_entity_transfer"
BOTH_DIR  = ROOT / "results/exp2_both_change"
SUBJ_DIR  = ROOT / "results/exp3_subject_token_patching"
STEER_DIR = ROOT / "results/exp4_steering"

OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path


# ============================================================
# EMNLP / ACL STYLE
# ============================================================

DPI = 600

FULL_WIDTH   = 6.30
SINGLE_WIDTH = 3.03

mpl.rcParams.update({
    "figure.dpi":    160,
    "savefig.dpi":   DPI,
    "pdf.fonttype":  42,
    "ps.fonttype":   42,

    "font.family": "serif",
    "font.serif":  ["Times", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],

    "font.size":        9.5,
    "axes.titlesize":   9.5,
    "axes.labelsize":   9,
    "legend.fontsize":  7.5,
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,

    "axes.linewidth":   0.75,
    "lines.linewidth":  1.65,
    "lines.markersize": 3.8,
    "legend.frameon":   True,
})


# ============================================================
# MODELS
# ============================================================

MODEL_ORDER = [
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen2.5-3B",
    "microsoft/phi-2",
]

SHORT = {
    "meta-llama/Llama-3.2-3B":     "Llama-3.2-3B",
    "meta-llama/Meta-Llama-3-8B":  "Llama-3-8B",
    "Qwen/Qwen2.5-3B":             "Qwen2.5-3B",
    "microsoft/phi-2":             "Phi-2",
}

NUM_LAYERS = {
    "meta-llama/Llama-3.2-3B":    28,
    "meta-llama/Meta-Llama-3-8B": 32,
    "Qwen/Qwen2.5-3B":            36,
    "microsoft/phi-2":            32,
}


# ============================================================
# COLORS + STYLES
# ============================================================

REL_COLOR  = "#1f77b4"
ENT_COLOR  = "#d62728"
SUBJ_COLOR = "#7b3294"
LAST_COLOR = "#2ca25f"
GRAY       = "#6f6f6f"
LIGHT_GRAY = "#e7e7e7"
DARK       = "#202020"

LIGHT_REL = "#eaf3fb"
LIGHT_ENT = "#fdecec"

GRID_ALPHA = 0.18
BAND_ALPHA = 0.14

MAIN_ONSET_THRESHOLD = 0.40


# ============================================================
# BASIC HELPERS
# ============================================================

def savefig(name: str):
    path_pdf = OUT_DIR / f"{name}.pdf"
    path_png = OUT_DIR / f"{name}.png"
    plt.savefig(path_pdf, bbox_inches="tight", pad_inches=0.025)
    plt.savefig(path_png, dpi=DPI, bbox_inches="tight", pad_inches=0.025)
    plt.close()
    print(f"[saved] {path_pdf}")
    print(f"[saved] {path_png}")


def clean_axis(ax, grid_axis="both"):
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=GRID_ALPHA, linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax, label):
    ax.text(
        -0.08, 1.06, label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
        ha="right",
    )


def sem(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def first_stable_onset(df, score_col, threshold, stable_steps=2):
    df   = df.sort_values("layer_idx").reset_index(drop=True)
    vals = df[score_col].tolist()
    layers = df["layer_idx"].tolist()
    for i in range(len(vals) - stable_steps + 1):
        if all(v >= threshold for v in vals[i:i + stable_steps]):
            return int(layers[i])
    for i, v in enumerate(vals):
        if v >= threshold:
            return int(layers[i])
    return None


def find_crossover(sub):
    """
    Discrete crossover: first tested layer where entity_wins >= relation_wins
    after relation was previously dominant.
    """
    sub = sub.sort_values("layer_idx").reset_index(drop=True)
    prev_rel_higher = None
    for _, row in sub.iterrows():
        rel_higher = row["relation_wins"] > row["entity_wins"]
        if prev_rel_higher is not None and prev_rel_higher and not rel_higher:
            return int(row["layer_idx"])
        prev_rel_higher = rel_higher
    return None


def find_crossover_interpolated(sub):
    """
    Visual crossover: linearly interpolated layer where entity_wins crosses
    relation_wins between adjacent tested layers.
    """
    sub    = sub.sort_values("layer_idx").reset_index(drop=True)
    layers = sub["layer_idx"].to_numpy(dtype=float)
    diff   = (sub["entity_wins"] - sub["relation_wins"]).to_numpy(dtype=float)
    for i in range(1, len(diff)):
        if diff[i - 1] < 0 and diff[i] >= 0:
            x0, x1 = layers[i - 1], layers[i]
            y0, y1 = diff[i - 1], diff[i]
            if y1 == y0:
                return float(x1)
            return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
    return None

def find_subject_last_crossover(sub):
    """
    Find the visual crossover where last-token patching catches up to
    subject-token patching.
    """
    sub = sub.sort_values("layer_idx").reset_index(drop=True)

    layers = sub["layer_idx"].to_numpy(dtype=float)
    diff = (
        sub["last_token_pct"].to_numpy(dtype=float)
        - sub["subj_token_pct"].to_numpy(dtype=float)
    )

    for i in range(1, len(diff)):
        if diff[i - 1] < 0 and diff[i] >= 0:
            x0, x1 = layers[i - 1], layers[i]
            y0, y1 = diff[i - 1], diff[i]

            if y1 == y0:
                return float(x1)

            return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))

    return None


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.7,
          style="-|>", ms=11, zorder=3):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arr)


def rounded_box(ax, x, y, w, h, title, body, facecolor, title_color=DARK):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0,
        edgecolor=DARK,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.69, title,
            ha="center", va="center", fontsize=8.9,
            fontweight="bold", color=title_color, zorder=3)
    ax.text(x + w / 2, y + h * 0.35, body,
            ha="center", va="center", fontsize=7.25,
            color=DARK, linespacing=1.12, zorder=3)


def add_boxed_legend(fig, handles, labels,
                     y=-0.005, ncol=2, handlelength=2.4):
    leg = fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        fontsize=7.4,
        handlelength=handlelength,
        columnspacing=1.6,
        borderpad=0.35,
        labelspacing=0.35,
        markerscale=0.9,
    )
    frame = leg.get_frame()
    frame.set_edgecolor("#bdbdbd")
    frame.set_linewidth(0.6)
    frame.set_facecolor("#f8f8f8")
    return leg


def make_line_legend(color, marker, linestyle, label):
    return Line2D(
        [0], [0],
        color=color,
        linestyle=linestyle,
        linewidth=1.8,
        marker=marker,
        markersize=4.2,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.55,
        label=label,
    )


def make_marker_legend(color, marker, label):
    return Line2D(
        [0], [0],
        color="none",
        marker=marker,
        markersize=5.2,
        markerfacecolor=color,
        markeredgecolor=DARK,
        markeredgewidth=0.45,
        label=label,
    )


# ============================================================
# LOAD DATA
# ============================================================

entity_summary   = pd.read_csv(require_file(MAIN_DIR / "entity_transfer_summary.csv"))
relation_summary = pd.read_csv(require_file(MAIN_DIR / "relation_transfer_by_pair.csv"))
both_summary     = pd.read_csv(require_file(BOTH_DIR  / "both_change_summary.csv"))
steer_raw        = pd.read_csv(require_file(STEER_DIR / "steering_all_models_raw.csv"))
subject_summary  = pd.read_csv(require_file(SUBJ_DIR  / "subject_patch_model_summary.csv"))

# Optional raw files — used for SEM bands if available
both_raw_path    = BOTH_DIR / "both_change_all_models_raw.csv"
both_raw         = pd.read_csv(both_raw_path) if both_raw_path.exists() else None

subject_raw_path = SUBJ_DIR / "subject_patch_all_models_raw.csv"
subject_raw      = pd.read_csv(subject_raw_path) if subject_raw_path.exists() else None


# ============================================================
# NORMALIZE SUMMARY COLUMNS
# ============================================================

def normalize_relation_summary(df):
    df = df.copy()
    if "relation_score" not in df.columns:
        for c in ["relation_transfer_pct", "target_hit_pct",
                  "relation_wins", "score"]:
            if c in df.columns:
                df = df.rename(columns={c: "relation_score"})
                break
    if "relation_score" not in df.columns:
        raise ValueError(
            f"Could not find relation score column in relation_summary. "
            f"Columns={list(df.columns)}")
    return df


def normalize_entity_summary(df):
    df = df.copy()
    if "entity_score" not in df.columns:
        for c in ["entity_transfer_pct", "target_hit_pct",
                  "entity_wins", "score"]:
            if c in df.columns:
                df = df.rename(columns={c: "entity_score"})
                break
    if "entity_score" not in df.columns:
        raise ValueError(
            f"Could not find entity score column in entity_summary. "
            f"Columns={list(df.columns)}")
    return df


entity_summary   = normalize_entity_summary(entity_summary)
relation_summary = normalize_relation_summary(relation_summary)


# ============================================================
# CURVE HELPERS
# ============================================================

def get_model_curve_entity(model_name):
    sub = entity_summary[entity_summary["model_name"] == model_name].copy()
    if "family" in sub.columns:
        fam_layer = (
            sub.groupby(["family", "layer_idx"], as_index=False)
            .agg(entity_score=("entity_score", "mean"))
        )
        out = (
            fam_layer.groupby("layer_idx", as_index=False)
            .agg(entity_score=("entity_score", "mean"),
                 entity_sem=("entity_score", sem),
                 n=("entity_score", "count"))
        )
    else:
        out = (
            sub.groupby("layer_idx", as_index=False)
            .agg(entity_score=("entity_score", "mean"),
                 entity_sem=("entity_score", sem),
                 n=("entity_score", "count"))
        )
    return out.sort_values("layer_idx")


def get_model_curve_relation(model_name):
    sub = relation_summary[
        relation_summary["model_name"] == model_name].copy()
    pair_col = None
    for c in ["relation_pair_id", "family_pair", "source_family"]:
        if c in sub.columns:
            pair_col = c
            break
    if pair_col is not None:
        pair_layer = (
            sub.groupby([pair_col, "layer_idx"], as_index=False)
            .agg(relation_score=("relation_score", "mean"))
        )
        out = (
            pair_layer.groupby("layer_idx", as_index=False)
            .agg(relation_score=("relation_score", "mean"),
                 relation_sem=("relation_score", sem),
                 n=("relation_score", "count"))
        )
    else:
        out = (
            sub.groupby("layer_idx", as_index=False)
            .agg(relation_score=("relation_score", "mean"),
                 relation_sem=("relation_score", sem),
                 n=("relation_score", "count"))
        )
    return out.sort_values("layer_idx")


def build_shared_threshold_onset_table(threshold=MAIN_ONSET_THRESHOLD):
    rows = []
    for model_name in MODEL_ORDER:
        rel    = get_model_curve_relation(model_name)
        ent    = get_model_curve_entity(model_name)
        rel_on = first_stable_onset(rel, "relation_score", threshold)
        ent_on = first_stable_onset(ent, "entity_score",   threshold)
        rows.append({
            "model_name":     model_name,
            "short":          SHORT[model_name],
            "num_layers":     NUM_LAYERS[model_name],
            "relation_onset": rel_on,
            "entity_onset":   ent_on,
            "gap": None if rel_on is None or ent_on is None
                   else ent_on - rel_on,
        })
    return pd.DataFrame(rows)


# ============================================================
# FIGURE 1A: CONCEPTUAL SCHEMATIC
# ============================================================

def draw_conceptual_schematic(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Two-stage factual recall",
            ha="center", va="center", fontsize=12.2,
            fontweight="bold", color=DARK)

    y_box = 0.705
    w_box = 0.245
    h_box = 0.155
    x1, x2, x3 = 0.065, 0.3775, 0.690

    rounded_box(ax, x1, y_box, w_box, h_box,
                "Early layers",
                "Entity information\navailable at subject token",
                LIGHT_ENT, title_color=ENT_COLOR)
    rounded_box(ax, x2, y_box, w_box, h_box,
                "Mid layers",
                "Relation selected\nat final token",
                LIGHT_REL, title_color=REL_COLOR)
    rounded_box(ax, x3, y_box, w_box, h_box,
                "Late layers",
                "Entity committed\nat final token",
                LIGHT_ENT, title_color=ENT_COLOR)

    arrow(ax, x1 + w_box + 0.025, y_box + h_box / 2,
          x2 - 0.025,             y_box + h_box / 2, lw=1.55, ms=10)
    arrow(ax, x2 + w_box + 0.025, y_box + h_box / 2,
          x3 - 0.025,             y_box + h_box / 2, lw=1.55, ms=10)

    ax.text(0.5, 0.545,
            'Prompt:  "The capital of France is ___"',
            ha="center", va="center", fontsize=9.8,
            fontweight="bold", color=DARK)

    subj_x, final_x = 0.25, 0.75
    token_y = 0.355
    card_w, card_h = 0.22, 0.11

    ax.add_patch(FancyBboxPatch(
        (subj_x - card_w / 2, token_y - card_h / 2), card_w, card_h,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=0.8, edgecolor="#d8b5b5", facecolor="#fff5f5", zorder=2))
    ax.add_patch(FancyBboxPatch(
        (final_x - card_w / 2, token_y - card_h / 2), card_w, card_h,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=0.8, edgecolor="#b8d1e8", facecolor="#f3f8fd", zorder=2))

    ax.text(subj_x,  token_y + 0.025, "Subject token",
            ha="center", va="center", fontsize=7.8,
            fontweight="bold", color=DARK, zorder=3)
    ax.text(subj_x,  token_y - 0.023, "France",
            ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=ENT_COLOR, zorder=3)
    ax.text(final_x, token_y + 0.025, "Final token",
            ha="center", va="center", fontsize=7.8,
            fontweight="bold", color=DARK, zorder=3)
    ax.text(final_x, token_y - 0.023, "is",
            ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=REL_COLOR, zorder=3)

    arrow(ax, subj_x + card_w / 2 + 0.035, token_y,
          final_x - card_w / 2 - 0.035,    token_y, lw=2.0, ms=12)
    ax.text(0.5, token_y + 0.055, "routing / migration",
            ha="center", va="center", fontsize=7.7,
            color=DARK, style="italic")
    ax.text(subj_x,  0.21, "entity signal\npresent early",
            ha="center", va="center", fontsize=7.5,
            color=ENT_COLOR, linespacing=1.08)
    ax.text(final_x, 0.21, "generation-\ncontrolling state",
            ha="center", va="center", fontsize=7.5,
            color=DARK, linespacing=1.08)

    ax.add_patch(FancyBboxPatch(
        (0.075, 0.055), 0.85, 0.075,
        boxstyle="round,pad=0.010,rounding_size=0.010",
        linewidth=0.55, edgecolor="#d9d9d9",
        facecolor="#fbfbfb", zorder=1))
    ax.text(0.5, 0.092,
            "Interpretation: relation selection becomes active before entity "
            "commitment;\n"
            "entity information becomes generation-controlling only after "
            "reaching the final token.",
            ha="center", va="center", fontsize=6.9,
            color=DARK, linespacing=1.22, zorder=2)


def fig1a_conceptual_schematic():
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.15))
    draw_conceptual_schematic(ax)
    plt.tight_layout()
    savefig("fig1a_conceptual_schematic")


# ============================================================
# FIGURE 1B: ONSET TIMELINE
# ============================================================

def draw_onset_timeline(ax):
    df = build_shared_threshold_onset_table(threshold=MAIN_ONSET_THRESHOLD)

    print("\n[Figure 1B shared-threshold onset table]")
    print(df[["short", "relation_onset", "entity_onset", "gap"]]
          .to_string(index=False))

    df["order"] = df["model_name"].apply(lambda x: MODEL_ORDER.index(x))
    df = df.sort_values("order")
    y_positions = np.arange(len(df))[::-1]

    for y, (_, row) in zip(y_positions, df.iterrows()):
        rel   = row["relation_onset"]
        ent   = row["entity_onset"]
        total = int(row["num_layers"])

        ax.hlines(y, 0, total - 1, color=LIGHT_GRAY, linewidth=6.5, zorder=1)

        if rel is not None and ent is not None:
            ax.hlines(y, rel, ent, color=DARK, linewidth=1.7, zorder=2)
            ax.scatter(rel, y, s=78, marker="o", color=REL_COLOR,
                       edgecolor=DARK, linewidth=0.45, zorder=3)
            ax.scatter(ent, y, s=86, marker="s", color=ENT_COLOR,
                       edgecolor=DARK, linewidth=0.45, zorder=3)
            ax.text((rel + ent) / 2, y + 0.12,
                    f"+{int(ent - rel)}",
                    ha="center", va="bottom", fontsize=7.2)

        ax.text(-1.0, y, row["short"],
                ha="right", va="center", fontsize=8.4)

    ax.set_yticks([])
    ax.set_xlabel("Layer")
    ax.set_title(
        f"Relation reaches majority transfer before entity "
        f"(threshold={MAIN_ONSET_THRESHOLD:.2f})",
        fontsize=9.5, fontweight="bold", pad=7)
    ax.set_xlim(-2, max(NUM_LAYERS.values()) + 1)
    ax.set_ylim(-0.55, len(df) - 0.45)
    ax.grid(axis="x", alpha=0.16, linewidth=0.65)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    handles = [make_marker_legend(REL_COLOR, "o", "Relation onset"),
               make_marker_legend(ENT_COLOR, "s", "Entity onset")]
    labels  = ["Relation onset", "Entity onset"]
    return handles, labels


def fig1b_onset_timeline():
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.55))
    handles, labels = draw_onset_timeline(ax)
    add_boxed_legend(fig, handles, labels, y=-0.01, ncol=2, handlelength=1.6)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    savefig("fig1b_onset_timeline")


# ============================================================
# FIGURE 1 COMBINED
# ============================================================

def fig1_combined_concept_onset():
    fig = plt.figure(figsize=(FULL_WIDTH, 5.75))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    draw_conceptual_schematic(ax1)
    handles, labels = draw_onset_timeline(ax2)
    panel_label(ax1, "A")
    panel_label(ax2, "B")
    add_boxed_legend(fig, handles, labels, y=-0.005, ncol=2, handlelength=1.6)
    plt.tight_layout(rect=[0, 0.055, 1, 1])
    savefig("fig1_combined_concept_onset")


# ============================================================
# FIGURE 2: TRANSFER CURVES
# ============================================================

def fig2_transfer_curves():
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.05), sharey=True)
    axes = axes.flatten()

    legend_handles = [
        make_line_legend(REL_COLOR, "o", "-",  "Relation transfer"),
        make_line_legend(ENT_COLOR, "s", "--", "Entity transfer"),
    ]
    legend_labels = ["Relation transfer", "Entity transfer"]

    for ax, model_name in zip(axes, MODEL_ORDER):
        ent    = get_model_curve_entity(model_name)
        rel    = get_model_curve_relation(model_name)
        rel_on = first_stable_onset(rel, "relation_score", MAIN_ONSET_THRESHOLD)
        ent_on = first_stable_onset(ent, "entity_score",   MAIN_ONSET_THRESHOLD)

        ax.plot(rel["layer_idx"], rel["relation_score"],
                marker="o", linestyle="-", linewidth=1.6, color=REL_COLOR,
                markerfacecolor=REL_COLOR, markeredgecolor=REL_COLOR,
                markeredgewidth=0.5)
        ax.fill_between(
            rel["layer_idx"],
            np.clip(rel["relation_score"] - rel["relation_sem"], 0, 1),
            np.clip(rel["relation_score"] + rel["relation_sem"], 0, 1),
            color=REL_COLOR, alpha=BAND_ALPHA, linewidth=0)

        ax.plot(ent["layer_idx"], ent["entity_score"],
                marker="s", linestyle="--", linewidth=1.6, color=ENT_COLOR,
                markerfacecolor=ENT_COLOR, markeredgecolor=ENT_COLOR,
                markeredgewidth=0.5)
        ax.fill_between(
            ent["layer_idx"],
            np.clip(ent["entity_score"] - ent["entity_sem"], 0, 1),
            np.clip(ent["entity_score"] + ent["entity_sem"], 0, 1),
            color=ENT_COLOR, alpha=BAND_ALPHA, linewidth=0)

        if rel_on is not None:
            ax.axvline(rel_on, linestyle=":", color=REL_COLOR, linewidth=1.1)
        if ent_on is not None:
            ax.axvline(ent_on, linestyle=":", color=ENT_COLOR, linewidth=1.1)

        ax.set_title(SHORT[model_name], fontsize=8.8, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, linestyle="--", color="gray",
                   linewidth=0.75, alpha=0.35)
        clean_axis(ax)

    axes[0].set_ylabel("Transfer rate")
    axes[2].set_ylabel("Transfer rate")
    add_boxed_legend(fig, legend_handles, legend_labels, y=-0.005, ncol=2)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    savefig("fig2_transfer_curves")


# ============================================================
# FIGURE 3: BOTH-CHANGE COMPETITION
# ============================================================

def get_both_curve(model_name):
    if (both_raw is not None
            and {"model_name", "layer_idx",
                 "entity_wins", "relation_wins"}.issubset(both_raw.columns)):
        sub = both_raw[both_raw["model_name"] == model_name].copy()
        out = (
            sub.groupby("layer_idx", as_index=False)
            .agg(entity_wins=("entity_wins",     "mean"),
                 relation_wins=("relation_wins",  "mean"),
                 entity_sem=("entity_wins",       sem),
                 relation_sem=("relation_wins",   sem),
                 n=("entity_wins",                "count"))
        )
    else:
        sub = both_summary[both_summary["model_name"] == model_name].copy()
        out = sub.copy()
        out["entity_sem"]   = 0.0
        out["relation_sem"] = 0.0
    return out.sort_values("layer_idx")


def fig3_both_change_competition():
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.05), sharey=True)
    axes = axes.flatten()

    legend_handles = [
        make_line_legend(REL_COLOR, "o", "-",  "Relation wins"),
        make_line_legend(ENT_COLOR, "s", "--", "Entity wins"),
    ]
    legend_labels = ["Relation wins", "Entity wins"]

    for ax, model_name in zip(axes, MODEL_ORDER):
        sub = get_both_curve(model_name)

        ax.plot(sub["layer_idx"], sub["relation_wins"],
                marker="o", linestyle="-", linewidth=1.6, color=REL_COLOR,
                markerfacecolor=REL_COLOR, markeredgecolor=REL_COLOR,
                markeredgewidth=0.5)
        ax.fill_between(
            sub["layer_idx"],
            np.clip(sub["relation_wins"] - sub["relation_sem"], 0, 1),
            np.clip(sub["relation_wins"] + sub["relation_sem"], 0, 1),
            color=REL_COLOR, alpha=BAND_ALPHA, linewidth=0)

        ax.plot(sub["layer_idx"], sub["entity_wins"],
                marker="s", linestyle="--", linewidth=1.6, color=ENT_COLOR,
                markerfacecolor=ENT_COLOR, markeredgecolor=ENT_COLOR,
                markeredgewidth=0.5)
        ax.fill_between(
            sub["layer_idx"],
            np.clip(sub["entity_wins"] - sub["entity_sem"], 0, 1),
            np.clip(sub["entity_wins"] + sub["entity_sem"], 0, 1),
            color=ENT_COLOR, alpha=BAND_ALPHA, linewidth=0)

        sampled_crossover = find_crossover(sub)

        if sampled_crossover is not None:
            ax.axvline(sampled_crossover, linestyle=":",
                       color=DARK, linewidth=1.0)
            ax.text(sampled_crossover + 0.25, 0.91,
                    f"L{sampled_crossover}", fontsize=7.0)

        ax.set_title(SHORT[model_name], fontsize=8.8, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, linestyle="--", color="gray",
                   linewidth=0.75, alpha=0.35)
        clean_axis(ax)

    axes[0].set_ylabel("Fraction")
    axes[2].set_ylabel("Fraction")
    add_boxed_legend(fig, legend_handles, legend_labels, y=-0.005, ncol=2)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    savefig("fig3_both_change_competition")


# ============================================================
# FIGURE 4: SUBJECT VS LAST TOKEN PATCHING
# ============================================================

def get_subject_curve(model_name):
    if (subject_raw is not None
            and {"model_name", "layer_idx",
                 "last_token_hit", "subj_token_hit"}.issubset(
                     subject_raw.columns)):
        sub = subject_raw[subject_raw["model_name"] == model_name].copy()
        out = (
            sub.groupby("layer_idx", as_index=False)
            .agg(last_token_pct=("last_token_hit", "mean"),
                 subj_token_pct=("subj_token_hit", "mean"),
                 last_sem=("last_token_hit",        sem),
                 subj_sem=("subj_token_hit",        sem),
                 n=("last_token_hit",               "count"))
        )
    else:
        sub = subject_summary[
            subject_summary["model_name"] == model_name].copy()
        out = sub.copy()
        out["last_sem"] = 0.0
        out["subj_sem"] = 0.0
    return out.sort_values("layer_idx")


def fig4_subject_vs_last_patching():
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.65), sharey=True)
    axes = axes.flatten()

    legend_handles = [
        make_line_legend(SUBJ_COLOR, "o", "-",  "Entity-token patch"),
        make_line_legend(LAST_COLOR, "s", "--", "Final-token patch"),
    ]
    legend_labels = ["Entity-token patch", "Final-token patch"]

    for ax, model_name in zip(axes, MODEL_ORDER):
        sub = get_subject_curve(model_name)

        ax.plot(sub["layer_idx"], sub["subj_token_pct"],
                marker="o", linestyle="-", linewidth=1.6, color=SUBJ_COLOR,
                markerfacecolor=SUBJ_COLOR, markeredgecolor=SUBJ_COLOR,
                markeredgewidth=0.5)
        ax.fill_between(
            sub["layer_idx"],
            np.clip(sub["subj_token_pct"] - sub["subj_sem"], 0, 1),
            np.clip(sub["subj_token_pct"] + sub["subj_sem"], 0, 1),
            color=SUBJ_COLOR, alpha=BAND_ALPHA, linewidth=0)

        ax.plot(sub["layer_idx"], sub["last_token_pct"],
                marker="s", linestyle="--", linewidth=1.6, color=LAST_COLOR,
                markerfacecolor=LAST_COLOR, markeredgecolor=LAST_COLOR,
                markeredgewidth=0.5)
        ax.fill_between(
            sub["layer_idx"],
            np.clip(sub["last_token_pct"] - sub["last_sem"], 0, 1),
            np.clip(sub["last_token_pct"] + sub["last_sem"], 0, 1),
            color=LAST_COLOR, alpha=BAND_ALPHA, linewidth=0)

        # Crossover line: where last-token patching catches up to
        # subject-token patching.
        crossover = find_subject_last_crossover(sub)
        if crossover is not None:
            ax.axvline(
                crossover,
                linestyle=":",
                color=DARK,
                linewidth=1.0,
            )

        ax.set_title(SHORT[model_name], fontsize=8.8, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, linestyle="--", color="gray",
                   linewidth=0.75, alpha=0.35)
        clean_axis(ax)

    axes[0].set_ylabel("Entity transfer rate")
    axes[2].set_ylabel("Entity transfer rate")
    add_boxed_legend(fig, legend_handles, legend_labels, y=-0.005, ncol=2)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    savefig("fig4_subject_vs_last_patching")


# ============================================================
# FIGURE 5: STEERING ASYMMETRY
# ============================================================

def fig5_steering_asymmetry():
    best = steer_raw[steer_raw["alpha"] == 1.0].copy()

    conditions = ["relation_mid", "entity_mid", "entity_late", "relation_late"]
    labels_x   = ["Rel.\nmid", "Ent.\nmid", "Ent.\nlate", "Rel.\nlate"]

    condition_colors = {
        "relation_mid":  REL_COLOR,
        "entity_mid":    "#fdae61",
        "entity_late":   ENT_COLOR,
        "relation_late": GRAY,
    }

    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 1.85), sharey=True)

    random_handle = Line2D(
        [0], [0], color="black", marker="x", linestyle="None",
        markersize=4.6, markeredgewidth=1.2, label="Random direction")

    for ax, model_name in zip(axes, MODEL_ORDER):
        m = best[best["model_name"] == model_name]

        real_vals, rand_vals, err_vals = [], [], []
        for cond in conditions:
            sub = m[m["steer_type"] == cond]
            real_vals.append(sub["target_hit"].mean())
            rand_vals.append(sub["random_target_hit"].mean())
            err_vals.append(sem(sub["target_hit"]))

        x      = np.arange(len(conditions))
        colors = [condition_colors[c] for c in conditions]

        ax.bar(x, real_vals, yerr=err_vals, capsize=1.8,
               color=colors, width=0.70, edgecolor=DARK, linewidth=0.42,
               error_kw=dict(linewidth=0.75))
        ax.scatter(x, rand_vals, marker="x", s=35,
                   color="black", linewidth=1.25, zorder=4)

        for xi, val in zip(x, real_vals):
            ax.text(xi, min(val + 0.045, 1.04), f"{val:.2f}",
                    ha="center", va="bottom", fontsize=6.2)

        ax.set_title(SHORT[model_name], fontsize=8.0, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=6.8)
        ax.set_ylim(0, 1.10)
        ax.axhline(0.5, linestyle="--", color="gray",
                   alpha=0.35, linewidth=0.75)
        ax.grid(axis="y", alpha=0.18, linewidth=0.65)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Target hit rate")
    add_boxed_legend(fig, [random_handle], ["Random direction"],
                     y=-0.015, ncol=1, handlelength=1.2)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    savefig("fig5_steering_asymmetry")


# ============================================================
# APPENDIX TABLES
# ============================================================

def appendix_threshold_sensitivity():
    thresholds = [0.2, 0.3, 0.4, 0.5]
    rows = []
    for model_name in MODEL_ORDER:
        ent = get_model_curve_entity(model_name)
        rel = get_model_curve_relation(model_name)
        for t in thresholds:
            rel_on = first_stable_onset(rel, "relation_score", t)
            ent_on = first_stable_onset(ent, "entity_score",   t)
            rows.append({
                "model_name": model_name,
                "model":      SHORT[model_name],
                "threshold":  t,
                "relation_onset": rel_on,
                "entity_onset":   ent_on,
                "relation_before_entity": (
                    rel_on is not None and ent_on is not None
                    and rel_on < ent_on),
                "gap": (ent_on - rel_on
                        if rel_on is not None and ent_on is not None
                        else None),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "appendix_threshold_sensitivity.csv", index=False)

    latex = out[["model", "threshold", "relation_onset",
                 "entity_onset", "gap",
                 "relation_before_entity"]].to_latex(
                     index=False, escape=False)
    (OUT_DIR / "appendix_threshold_sensitivity.tex").write_text(
        latex, encoding="utf-8")

    print("[saved] appendix_threshold_sensitivity.csv")
    print("[saved] appendix_threshold_sensitivity.tex")
    print(out[["model", "threshold", "relation_onset",
               "entity_onset", "gap",
               "relation_before_entity"]].to_string(index=False))
    return out


def appendix_peak_layer_check():
    rows = []
    for model_name in MODEL_ORDER:
        ent = get_model_curve_entity(model_name)
        rel = get_model_curve_relation(model_name)
        rel_peak_row = rel.loc[rel["relation_score"].idxmax()]
        ent_peak_row = ent.loc[ent["entity_score"].idxmax()]
        rows.append({
            "model_name":   model_name,
            "model":        SHORT[model_name],
            "relation_peak_layer": int(rel_peak_row["layer_idx"]),
            "relation_peak_score": float(rel_peak_row["relation_score"]),
            "entity_peak_layer":   int(ent_peak_row["layer_idx"]),
            "entity_peak_score":   float(ent_peak_row["entity_score"]),
            "relation_peak_before_entity_peak": (
                int(rel_peak_row["layer_idx"]) < int(ent_peak_row["layer_idx"])),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "appendix_peak_layer_threshold_free.csv", index=False)

    latex = out[["model", "relation_peak_layer", "relation_peak_score",
                 "entity_peak_layer", "entity_peak_score",
                 "relation_peak_before_entity_peak"]].to_latex(
                     index=False, float_format="%.3f", escape=False)
    (OUT_DIR / "appendix_peak_layer_threshold_free.tex").write_text(
        latex, encoding="utf-8")

    print("[saved] appendix_peak_layer_threshold_free.csv")
    print("[saved] appendix_peak_layer_threshold_free.tex")
    print(out[["model", "relation_peak_layer", "relation_peak_score",
               "entity_peak_layer", "entity_peak_score",
               "relation_peak_before_entity_peak"]].to_string(index=False))
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n[1/5] Combined Figure 1")
    fig1_combined_concept_onset()

    print("\n[2/5] Transfer curves")
    fig2_transfer_curves()

    print("\n[3/5] Both-change competition")
    fig3_both_change_competition()

    print("\n[4/5] Subject vs final-token patching")
    fig4_subject_vs_last_patching()

    print("\n[5/5] Steering asymmetry")
    fig5_steering_asymmetry()

    metadata = {
        "script": "make_paper_figures.py",
        "purpose": "Generate final figures from precomputed result CSVs.",
        "output_dir": str(OUT_DIR),
        "models": MODEL_ORDER,
        "main_onset_threshold": MAIN_ONSET_THRESHOLD,
        "figures": [
            "fig1_combined_concept_onset",
            "fig2_transfer_curves",
            "fig3_both_change_competition",
            "fig4_subject_vs_last_patching",
            "fig5_steering_asymmetry",
        ],
        "note": (
            "This script does not run interventions; it only visualizes saved "
            "experiment results. Standalone schematic/onset figures and appendix "
            "tables are defined in the file but not generated by default."
        ),
    }
    with open(OUT_DIR / "figure_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[done] Final figures saved to: {OUT_DIR}/")
    print("Use the .pdf files in LaTeX, not the .png files.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()