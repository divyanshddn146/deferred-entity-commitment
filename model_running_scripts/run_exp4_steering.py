"""
run_exp4_steering.py

Full model-running script for Experiment 4:
Layer-zone steering for relation and entity directions.

This experiment tests whether relation/entity directions are effective in
different layer zones.

Conditions:
1. relation_mid: relation direction applied in mid layers.
2. entity_mid: entity direction applied in mid layers.
3. entity_late: entity direction applied in late layers.
4. relation_late: relation direction applied in late layers as a persistence check.

Random-direction control:
For each real steering vector, a random vector with matched norm is also applied.
This tests whether effects are direction-specific rather than caused by generic
large perturbations.

This is experimental research code used for the submitted paper. It requires
GPU access, HuggingFace model downloads/access, and substantial runtime.
For lightweight reproduction of paper figures/tables, use the processed CSVs
and analysis scripts.

Models: Llama-3.2-3B, Llama-3-8B, Qwen2.5-3B, Phi-2.
"""

import re
import gc
import random
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================
MODELS = [
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen2.5-3B",
    "microsoft/phi-2",
]

LOCAL_FILES_ONLY = False
USE_FP16_ON_CUDA = True
MAX_NEW_TOKENS = 12
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "exp4_steering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Alpha values to test — sweet spot is 1.0-2.0
ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0]

# Layer zones per model (mid=relation zone, late=entity/answer zone)
# Based on your main experiment onset results
MODEL_LAYER_ZONES = {
    "meta-llama/Llama-3.2-3B": {
        "mid":  [6, 8, 10, 12, 14],
        "late": [18, 20, 22, 24, 26],
        "total": 28,
    },
    "meta-llama/Meta-Llama-3-8B": {
    "mid":  [8, 10, 12, 14, 16],
    "late": [22, 24, 26, 28, 30],
    "total": 32,
    },
    "Qwen/Qwen2.5-3B": {
        "mid":  [14, 16, 18, 20, 22],
        "late": [28, 30, 32, 34],
        "total": 36,
    },
    "microsoft/phi-2": {
        "mid":  [12, 14, 16, 18, 20],
        "late": [22, 24, 26, 28, 30],
        "total": 32,
    },
}

# ============================================================
# FULL DATASET (from unified_law_final main experiment)
# ============================================================
COUNTRY_BANK = [
    ("France",       "paris",        "french"),
    ("Germany",      "berlin",       "german"),
    ("Italy",        "rome",         "italian"),
    ("Spain",        "madrid",       "spanish"),
    ("Russia",       "moscow",       "russian"),
    ("Japan",        "tokyo",        "japanese"),
    ("South Korea",  "seoul",        "korean"),
    ("Greece",       "athens",       "greek"),
    ("Turkey",       "ankara",       "turkish"),
    ("Poland",       "warsaw",       "polish"),
    ("Netherlands",  "amsterdam",    "dutch"),
    ("Sweden",       "stockholm",    "swedish"),
    ("Norway",       "oslo",         "norwegian"),
    ("Denmark",      "copenhagen",   "danish"),
    ("Finland",      "helsinki",     "finnish"),
    ("Hungary",      "budapest",     "hungarian"),
    ("Thailand",     "bangkok",      "thai"),
    ("Vietnam",      "hanoi",        "vietnamese"),
    ("Egypt",        "cairo",        "arabic"),
    ("Portugal",     "lisbon",       "portuguese"),
    ("Romania",      "bucharest",    "romanian"),
    ("Austria",      "vienna",       "german"),
    ("Belgium",      "brussels",     "dutch"),
    ("Switzerland",  "bern",         "german"),
    ("Czechia",      "prague",       "czech"),
    ("Peru",         "lima",         "spanish"),
    ("Argentina",    "buenos aires", "spanish"),
    ("Colombia",     "bogota",       "spanish"),
    ("Saudi Arabia", "riyadh",       "arabic"),
    ("Qatar",        "doha",         "arabic"),
    ("UAE",          "abu dhabi",    "arabic"),
    ("Nigeria",      "abuja",        "english"),
    ("Kenya",        "nairobi",      "english"),
]

TENSE_BANK = [
    ("walk",  "walked",  "walking"),
    ("play",  "played",  "playing"),
    ("talk",  "talked",  "talking"),
    ("work",  "worked",  "working"),
    ("jump",  "jumped",  "jumping"),
    ("look",  "looked",  "looking"),
    ("live",  "lived",   "living"),
    ("love",  "loved",   "loving"),
    ("wash",  "washed",  "washing"),
    ("watch", "watched", "watching"),
    ("cook",  "cooked",  "cooking"),
    ("start", "started", "starting"),
    ("stop",  "stopped", "stopping"),
    ("drop",  "dropped", "dropping"),
    ("study", "studied", "studying"),
    ("try",   "tried",   "trying"),
    ("learn", "learned", "learning"),
    ("paint", "painted", "painting"),
    ("send",  "sent",    "sending"),
    ("make",  "made",    "making"),
    ("keep",  "kept",    "keeping"),
]

ADJECTIVE_BANK = [
    ("tall",   "short",  "taller"),
    ("fast",   "slow",   "faster"),
    ("hard",   "soft",   "harder"),
    ("wide",   "narrow", "wider"),
    ("clean",  "dirty",  "cleaner"),
    ("strong", "weak",   "stronger"),
    ("dark",   "light",  "darker"),
    ("large",  "small",  "larger"),
    ("big",    "small",  "bigger"),
    ("hot",    "cold",   "hotter"),
    ("loud",   "quiet",  "louder"),
    ("easy",   "hard",   "easier"),
]

# ============================================================
# PROMPTS
# ============================================================
def make_capital_prompt(x):      return f"The capital of {x} is"
def make_language_prompt(x):     return f"The official language of {x} is"
def make_past_prompt(x):         return f"The past tense of {x} is"
def make_participle_prompt(x):   return f"The present participle of {x} is"
def make_opposite_prompt(x):     return f"The opposite of {x} is"
def make_comparative_prompt(x):  return f"The comparative form of {x} is"

# ============================================================
# TEXT HELPERS
# ============================================================
def normalize(x):
    return " ".join(x.lower().strip().replace("\n", " ").split())

def extract_continuation(prompt, full_text):
    return full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()

def extract_first_phrase(prompt, full_text, max_words=2):
    cont = extract_continuation(prompt, full_text)
    piece = re.split(r"[,\.\n;:!?]", cont, maxsplit=1)[0].strip()
    return " ".join(piece.split()[:max_words]).lower().strip()

def extract_slot(family, prompt, full_text):
    if family == "capital":
        return extract_first_phrase(prompt, full_text, max_words=2)
    return extract_first_phrase(prompt, full_text, max_words=1)

def canonical_match(family, pred, expected):
    pred, exp = normalize(pred), normalize(expected)
    return pred.split()[0] == exp.split()[0] if pred and exp else False

def label_steering(family, prompt, text, original_answer, target_answer):
    pred = extract_slot(family, prompt, text)
    if canonical_match(family, pred, target_answer):   return "target_hit"
    if canonical_match(family, pred, original_answer): return "original_retained"
    return "mixed_or_other"

# ============================================================
# BUILD RECORDS
# ============================================================
def build_relation_records():
    """
    TYPE 2 (relation donor): Same entity, different relation.
    For steering: compute mean-difference direction across entities.
    """
    records = []

    # capital ↔ language (same country)
    for country, cap, lang in COUNTRY_BANK:
        records += [
            {"group": "country", "entity": country,
             "family_a": "capital",  "family_b": "language",
             "prompt_a": make_capital_prompt(country),
             "prompt_b": make_language_prompt(country),
             "answer_a": cap,  "answer_b": lang},
            {"group": "country", "entity": country,
             "family_a": "language", "family_b": "capital",
             "prompt_a": make_language_prompt(country),
             "prompt_b": make_capital_prompt(country),
             "answer_a": lang, "answer_b": cap},
        ]

    # past_tense ↔ present_participle (same verb)
    for verb, past, part in TENSE_BANK:
        records += [
            {"group": "tense", "entity": verb,
             "family_a": "past_tense",        "family_b": "present_participle",
             "prompt_a": make_past_prompt(verb),
             "prompt_b": make_participle_prompt(verb),
             "answer_a": past, "answer_b": part},
            {"group": "tense", "entity": verb,
             "family_a": "present_participle", "family_b": "past_tense",
             "prompt_a": make_participle_prompt(verb),
             "prompt_b": make_past_prompt(verb),
             "answer_a": part, "answer_b": past},
        ]

    # opposite ↔ comparative (same adjective)
    for adj, opp, comp in ADJECTIVE_BANK:
        records += [
            {"group": "adjective", "entity": adj,
             "family_a": "opposite",    "family_b": "comparative",
             "prompt_a": make_opposite_prompt(adj),
             "prompt_b": make_comparative_prompt(adj),
             "answer_a": opp,  "answer_b": comp},
            {"group": "adjective", "entity": adj,
             "family_a": "comparative", "family_b": "opposite",
             "prompt_a": make_comparative_prompt(adj),
             "prompt_b": make_opposite_prompt(adj),
             "answer_a": comp, "answer_b": opp},
        ]

    print(f"[records] relation_records = {len(records)}")
    for g, n in Counter(r["group"] for r in records).items():
        print(f"  {g}: {n}")
    return records


def build_entity_records():
    """
    TYPE 1 (full donor): Same relation, different entity.
    For steering: pair-specific direction vector.
    """
    records = []
    rng = random.Random(SEED + 100)

    # Capital — pair each country with a random different country
    for i, (ca, capa, langa) in enumerate(COUNTRY_BANK):
        candidates = [(c, cap, lang) for j, (c, cap, lang)
                      in enumerate(COUNTRY_BANK) if j != i]
        cb, capb, langb = rng.choice(candidates)
        records += [
            {"group": "capital_entity", "family": "capital",
             "entity_a": ca, "entity_b": cb,
             "prompt_a": make_capital_prompt(ca),
             "prompt_b": make_capital_prompt(cb),
             "answer_a": capa, "answer_b": capb},
            {"group": "language_entity", "family": "language",
             "entity_a": ca, "entity_b": cb,
             "prompt_a": make_language_prompt(ca),
             "prompt_b": make_language_prompt(cb),
             "answer_a": langa, "answer_b": langb},
        ]

    # Tense — pair each verb with a random different verb
    for i, (va, pasta, parta) in enumerate(TENSE_BANK):
        candidates = [(v, p, pt) for j, (v, p, pt)
                      in enumerate(TENSE_BANK) if j != i]
        vb, pastb, partb = rng.choice(candidates)
        records += [
            {"group": "past_entity", "family": "past_tense",
             "entity_a": va, "entity_b": vb,
             "prompt_a": make_past_prompt(va),
             "prompt_b": make_past_prompt(vb),
             "answer_a": pasta, "answer_b": pastb},
            {"group": "participle_entity", "family": "present_participle",
             "entity_a": va, "entity_b": vb,
             "prompt_a": make_participle_prompt(va),
             "prompt_b": make_participle_prompt(vb),
             "answer_a": parta, "answer_b": partb},
        ]

    # Adjective — pair each adjective with a random different adjective
    for i, (aa, oppa, compa) in enumerate(ADJECTIVE_BANK):
        candidates = [(a, o, c) for j, (a, o, c)
                      in enumerate(ADJECTIVE_BANK) if j != i]
        ab, oppb, compb = rng.choice(candidates)
        records += [
            {"group": "opposite_entity", "family": "opposite",
             "entity_a": aa, "entity_b": ab,
             "prompt_a": make_opposite_prompt(aa),
             "prompt_b": make_opposite_prompt(ab),
             "answer_a": oppa, "answer_b": oppb},
            {"group": "comparative_entity", "family": "comparative",
             "entity_a": aa, "entity_b": ab,
             "prompt_a": make_comparative_prompt(aa),
             "prompt_b": make_comparative_prompt(ab),
             "answer_a": compa, "answer_b": compb},
        ]

    print(f"[records] entity_records = {len(records)}")
    for g, n in Counter(r["group"] for r in records).items():
        print(f"  {g}: {n}")
    return records


RELATION_RECORDS = build_relation_records()
ENTITY_RECORDS   = build_entity_records()

# ============================================================
# MODEL WRAPPER
# ============================================================
class SteeringRunner:
    def __init__(self, model_name):
        print(f"\n[load] {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=LOCAL_FILES_ONLY,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() and USE_FP16_ON_CUDA
                           else torch.float32,
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=LOCAL_FILES_ONLY,
            **kwargs,
        )
        self.model.eval()
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(self.model, "generation_config") and \
           getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.layers = self._get_layers()
        self.num_layers = len(self.layers)
        print(f"[info] num_layers={self.num_layers}")

    def _get_layers(self):
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"): return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"): return m.transformer.h
        raise ValueError(f"Unsupported: {self.model_name}")

    def tokenize(self, prompt):
        return {k: v.to(self.model.device)
                for k, v in self.tokenizer(prompt, return_tensors="pt").items()}

    def forward(self, prompt):
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            out = self.model(**inputs, return_dict=True, use_cache=False,
                             output_hidden_states=True)
        pos = inputs["input_ids"].shape[1] - 1
        return pos, out

    def generate_text(self, prompt):
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False, use_cache=True,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def generate_with_steering(self, prompt, layer_idx, steer_vec, alpha):
        inputs = self.tokenize(prompt)
        pos_idx = inputs["input_ids"].shape[1] - 1
        done = {"v": False}

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if not done["v"] and h.shape[1] > pos_idx:
                h = h.clone()
                h[:, pos_idx, :] = h[:, pos_idx, :] + alpha * steer_vec.to(h.device, h.dtype)
                done["v"] = True
            return (h,) + out[1:] if isinstance(out, tuple) else h

        handle = self.layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False, use_cache=True,
                                      pad_token_id=self.tokenizer.pad_token_id)
        handle.remove()
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

# ============================================================
# CACHE HIDDEN STATES
# ============================================================
def cache_hidden_states(runner, prompts):
    cache = {}
    for prompt in tqdm(sorted(set(prompts)), desc="Caching hidden states"):
        pos, out = runner.forward(prompt)
        cache[prompt] = {
            "hidden_states": out.hidden_states,
            "pos": pos,
            "normal_text": runner.generate_text(prompt),
        }
    return cache

def get_vec(cache, prompt, layer_idx):
    pos = cache[prompt]["pos"]
    return cache[prompt]["hidden_states"][layer_idx + 1][:, pos, :].detach().clone()

# ============================================================
# DIRECTION VECTORS
# ============================================================
def compute_relation_direction(cache, records, layer_idx, target_rec):
    """
    Mean-difference direction for relation steering.
    h(prompt_b) - h(prompt_a) averaged across all entities of same direction.
    Leave-one-out: exclude the target entity being tested.
    """
    vecs = []
    for r in records:
        same_dir = (r["family_a"] == target_rec["family_a"] and
                    r["family_b"] == target_rec["family_b"] and
                    r["group"]    == target_rec["group"])
        if same_dir and r["entity"] != target_rec["entity"]:
            va = get_vec(cache, r["prompt_a"], layer_idx)
            vb = get_vec(cache, r["prompt_b"], layer_idx)
            vecs.append(vb - va)
    return torch.stack(vecs).mean(0) if vecs else None

def compute_entity_direction(cache, record, layer_idx):
    """Pair-specific direction for entity steering."""
    va = get_vec(cache, record["prompt_a"], layer_idx)
    vb = get_vec(cache, record["prompt_b"], layer_idx)
    return vb - va

# ============================================================
# RUN STEERING EXPERIMENTS
# ============================================================
def run_steering(runner, cache, records, steer_type, layers, model_name):
    """
    Generic steering runner for both relation and entity experiments.
    steer_type: "relation_mid", "relation_late", "entity_mid", "entity_late"

    For each record × layer × alpha, runs TWO steerings:
      1. Real direction vector (mean-diff or pair-specific)
      2. Random unit-norm vector of same shape (negative control)

    real >> random  → effect is direction-specific
    real ≈ random   → effect is just large perturbation, not informative
    """
    rows     = []
    zone     = "mid"      if "mid"      in steer_type else "late"
    exp_type = "relation" if "relation" in steer_type else "entity"

    for rec in tqdm(records, desc=f"{steer_type}"):
        for layer_idx in layers:
            if layer_idx >= runner.num_layers - 1:
                continue

            # ── Compute direction vector ──────────────────────
            if exp_type == "relation":
                steer_vec = compute_relation_direction(
                    cache, records, layer_idx, rec)
                family          = rec["family_a"]
                original_answer = rec["answer_a"]
                target_answer   = rec["answer_b"]
                prompt          = rec["prompt_a"]
                entity          = rec["entity"]
                source_fam      = rec["family_a"]
                target_fam      = rec["family_b"]
                group           = rec["group"]
            else:
                steer_vec = compute_entity_direction(cache, rec, layer_idx)
                family          = rec["family"]
                original_answer = rec["answer_a"]
                target_answer   = rec["answer_b"]
                prompt          = rec["prompt_a"]
                entity          = rec["entity_a"]
                source_fam      = rec["family"]
                target_fam      = rec["family"]
                group           = rec["group"]

            if steer_vec is None:
                continue

            # ── Normalise real vector to unit norm ────────────
            # Ensures alpha directly controls magnitude and makes
            # real vs random comparison fair (same scale)
            # Scale steering vector to hidden-state norm
            base_vec = get_vec(cache, prompt, layer_idx)
            base_norm = base_vec.norm()

            steer_vec = steer_vec / (steer_vec.norm() + 1e-6)
            steer_vec = steer_vec * base_norm

            # ── Random control vector (unit norm, same shape) ─
            rand_vec = torch.randn_like(steer_vec)
            rand_vec = rand_vec / (rand_vec.norm() + 1e-6)
            rand_vec = rand_vec * base_norm

            normal_text = cache[prompt]["normal_text"]

            for alpha in ALPHAS:
                # Real steering
                steered_text = runner.generate_with_steering(
                    prompt, layer_idx, steer_vec, alpha)
                effect = label_steering(
                    family, prompt, steered_text, original_answer, target_answer)

                # Random direction control
                rand_text = runner.generate_with_steering(
                    prompt, layer_idx, rand_vec, alpha)
                rand_effect = label_steering(
                    family, prompt, rand_text, original_answer, target_answer)

                rows.append({
                    "model_name":            model_name,
                    "steer_type":            steer_type,
                    "experiment_type":       exp_type,
                    "zone":                  zone,
                    "group":                 group,
                    "source_family":         source_fam,
                    "target_family":         target_fam,
                    "entity":                entity,
                    "layer_idx":             layer_idx,
                    "alpha":                 alpha,
                    "prompt":                prompt,
                    "original_answer":       original_answer,
                    "target_answer":         target_answer,
                    "normal_text":           normal_text,
                    "steered_text":          steered_text,
                    "effect":                effect,
                    "target_hit":            int(effect == "target_hit"),
                    "original_retained":     int(effect == "original_retained"),
                    "mixed_or_other":        int(effect == "mixed_or_other"),
                    # Random direction control
                    "random_text":           rand_text,
                    "random_effect":         rand_effect,
                    "random_target_hit":     int(rand_effect == "target_hit"),
                    "random_original_retained": int(rand_effect == "original_retained"),
                    "random_mixed_or_other": int(rand_effect == "mixed_or_other"),
                })

    return pd.DataFrame(rows)

# ============================================================
# SUMMARIES + PLOTS
# ============================================================
def summarize_by_zone(df):
    """
    Key summary: target_hit rate vs random baseline by steer_type × layer × alpha.
    real >> random  → direction-specific effect
    real ≈ random   → effect is just perturbation magnitude, not informative
    """
    return (
        df.groupby(["model_name", "steer_type", "zone", "source_family",
                    "layer_idx", "alpha"])
        .agg(
            target_hit_pct=("target_hit",                "mean"),
            random_target_hit_pct=("random_target_hit",  "mean"),
            original_pct=("original_retained",           "mean"),
            random_original_pct=("random_original_retained", "mean"),
            mixed_pct=("mixed_or_other",                 "mean"),
            random_mixed_pct=("random_mixed_or_other",   "mean"),
            n=("target_hit",                             "count"),
        )
        .reset_index()
    )


def print_temporal_asymmetry_table(df):
    """
    Print the key table showing temporal asymmetry at alpha=1.0.
    Shows real vs random to prove direction-specificity.
    Main result table for the paper.
    """
    best = df[df["alpha"] == 1.0]

    print("\n" + "="*90)
    print("TEMPORAL ASYMMETRY — STEERING RESULTS (alpha=1.0)")
    print("="*90)
    print(f"{'Experiment':<30} {'Zone':<8} {'Real':>8} {'Random':>8} "
          f"{'Real-Rand':>10}  Interpretation")
    print("-"*90)

    for model_name in df["model_name"].unique():
        short = model_name.split("/")[-1]
        print(f"\n  Model: {short}")

        m = best[best["model_name"] == model_name]

        for steer_type, zone, interp in [
            ("relation_mid",  "mid",  "SHOULD SUCCEED — relation active at mid"),
            ("relation_late", "late", "cross-check — relation persists?"),
            ("entity_mid",    "mid",  "SHOULD FAIL   — answer not yet committed"),
            ("entity_late",   "late", "SHOULD SUCCEED — answer committed at late"),
        ]:
            sub = m[m["steer_type"] == steer_type]
            if sub.empty:
                continue
            real   = sub["target_hit"].mean()
            rand   = sub["random_target_hit"].mean()
            diff   = real - rand
            marker = "OK" if (
                ("SUCCEED" in interp and real > 0.5 and diff > 0.1)
                or ("FAIL"    in interp and real < 0.1)
            ) else "--"
            print(f"    {steer_type:<28} {zone:<8} "
                  f"{real:>8.3f} {rand:>8.3f} {diff:>10.3f}  "
                  f"[{marker}] {interp}")


def plot_temporal_asymmetry(df):
    """
    Main figure: 4-panel showing all 4 steering conditions per model.
    Each panel shows target_hit rate by layer at alpha=1.0.
    """
    best = df[df["alpha"] == 1.0]
    models = df["model_name"].unique()

    fig, axes = plt.subplots(len(models), 4,
                             figsize=(20, 4 * len(models)),
                             sharey=True)
    if len(models) == 1:
        axes = axes.reshape(1, -1)

    conditions = [
        ("relation_mid",  "Relation direction\n@ MID layers\n(should SUCCEED)",  "tab:blue"),
        ("entity_mid",    "Entity direction\n@ MID layers\n(should FAIL)",        "tab:orange"),
        ("entity_late",   "Entity direction\n@ LATE layers\n(should SUCCEED)",    "tab:green"),
        ("relation_late", "Relation direction\n@ LATE layers\n(cross-check)",     "tab:red"),
    ]

    for row_idx, model_name in enumerate(models):
        short = model_name.split("/")[-1]
        m = best[best["model_name"] == model_name]

        for col_idx, (steer_type, title, color) in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            sub = m[m["steer_type"] == steer_type]

            if sub.empty:
                ax.set_title(f"{short}\n{title}\n(no data)", fontsize=8)
                continue

            agg = (sub.groupby(["source_family", "layer_idx"])["target_hit"]
                   .mean().reset_index())

            for fam, fdf in agg.groupby("source_family"):
                fdf = fdf.sort_values("layer_idx")
                ax.plot(fdf["layer_idx"], fdf["target_hit"],
                        marker="o", linewidth=1.5, label=fam, alpha=0.85)

            overall = sub["target_hit"].mean()
            ax.axhline(0.5, linestyle="--", color="gray", alpha=0.4)
            ax.axhline(overall, linestyle=":", color=color, alpha=0.6,
                       label=f"avg={overall:.2f}")

            if row_idx == 0:
                ax.set_title(title, fontsize=9)
            ax.set_xlabel("Layer", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{short}\nTarget hit rate", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=6, ncol=2)

    plt.suptitle(
        "Temporal Asymmetry in Steering Effectiveness\n"
        "Relation direction works at mid layers; Entity direction only works at late layers",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    fname = OUTPUT_DIR / "temporal_asymmetry_full.png"
    plt.savefig(fname, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"\n[saved] {fname}")


def plot_summary_bars(df):
    """
    Summary bar chart: 4 conditions × 3 models at alpha=1.0.
    The key figure for the paper showing temporal asymmetry across all models.
    """
    best = df[df["alpha"] == 1.0]
    models = sorted(df["model_name"].unique())
    shorts = [m.split("/")[-1] for m in models]

    conditions = ["relation_mid", "entity_mid", "entity_late", "relation_late"]
    labels = [
        "Relation@mid\n(main)",
        "Entity@mid\n(cross-check\nshould FAIL)",
        "Entity@late\n(main)",
        "Relation@late\n(cross-check)",
    ]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    hatches = ["", "///", "", "///"]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (model_name, short) in enumerate(zip(models, shorts)):
        m = best[best["model_name"] == model_name]
        vals = []
        for cond in conditions:
            sub = m[m["steer_type"] == cond]
            vals.append(sub["target_hit"].mean() if not sub.empty else 0.0)

        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width,
                      label=short, alpha=0.8,
                      color=[colors[j] for j in range(len(conditions))])
        for bar, val, hatch in zip(bars, vals, hatches):
            bar.set_hatch(hatch)
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Target hit rate (alpha=1.0)", fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.4)
    ax.legend(title="Model", fontsize=9)
    ax.set_title(
        "Temporal Asymmetry: Steering succeeds only in the correct layer zone\n"
        "Solid = main experiment | Hatched = cross-check\n"
        "Entity@mid should be near zero (✅ if confirmed)",
        fontsize=10
    )
    plt.tight_layout()
    fname = OUTPUT_DIR / "temporal_asymmetry_bars.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[saved] {fname}")


def plot_alpha_sensitivity(df, model_name):
    """Show how target_hit rate varies with alpha for each condition."""
    short = model_name.split("/")[-1]
    m = df[df["model_name"] == model_name]

    conditions = [
        ("relation_mid",  "Relation@mid",  "tab:blue"),
        ("entity_late",   "Entity@late",   "tab:green"),
        ("entity_mid",    "Entity@mid",    "tab:orange"),
        ("relation_late", "Relation@late", "tab:red"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    for steer_type, label, color in conditions:
        sub = m[m["steer_type"] == steer_type]
        if sub.empty:
            continue
        agg = sub.groupby("alpha")["target_hit"].mean().reset_index()
        ax.plot(agg["alpha"], agg["target_hit"],
                marker="o", color=color, linewidth=2, label=label)

    ax.set_xlabel("Alpha (steering strength)")
    ax.set_ylabel("Target hit rate")
    ax.set_title(f"Alpha sensitivity — {short}\nOptimal alpha is typically 1.0-2.0")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = OUTPUT_DIR / f"{short}_alpha_sensitivity.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[saved] {fname}")


# ============================================================
# MAIN
# ============================================================
all_rows = []

for model_name in MODELS:
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    zones = MODEL_LAYER_ZONES[model_name]
    mid_layers  = zones["mid"]
    late_layers = zones["late"]

    # Collect all prompts to cache
    all_prompts = []
    for r in RELATION_RECORDS:
        all_prompts += [r["prompt_a"], r["prompt_b"]]
    for r in ENTITY_RECORDS:
        all_prompts += [r["prompt_a"], r["prompt_b"]]

    runner = SteeringRunner(model_name)
    cache  = cache_hidden_states(runner, all_prompts)

    # ── 4 steering conditions ────────────────────────────────
    # Main: relation at mid (should succeed)
    print(f"\n[1/4] Relation steering @ MID layers {mid_layers}")
    rel_mid_df = run_steering(
        runner, cache, RELATION_RECORDS,
        "relation_mid", mid_layers, model_name)

    # Cross-check: entity at mid (should FAIL)
    print(f"\n[2/4] Entity steering @ MID layers {mid_layers}")
    ent_mid_df = run_steering(
        runner, cache, ENTITY_RECORDS,
        "entity_mid", mid_layers, model_name)

    # Main: entity at late (should succeed)
    print(f"\n[3/4] Entity steering @ LATE layers {late_layers}")
    ent_late_df = run_steering(
        runner, cache, ENTITY_RECORDS,
        "entity_late", late_layers, model_name)

    # Cross-check: relation at late (persistence check)
    print(f"\n[4/4] Relation steering @ LATE layers {late_layers}")
    rel_late_df = run_steering(
        runner, cache, RELATION_RECORDS,
        "relation_late", late_layers, model_name)

    model_df = pd.concat(
        [rel_mid_df, ent_mid_df, ent_late_df, rel_late_df],
        ignore_index=True)
    all_rows.append(model_df)

    # Per-model CSV
    short = model_name.replace("/", "_")
    model_df.to_csv(OUTPUT_DIR / f"{short}_steering_raw.csv", index=False)

    # Per-model alpha sensitivity plot
    plot_alpha_sensitivity(model_df, model_name)

    del runner.model, runner.tokenizer, runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Aggregate all models ─────────────────────────────────────
all_df = pd.concat(all_rows, ignore_index=True)
all_df.to_csv(OUTPUT_DIR / "steering_all_models_raw.csv", index=False)

summary_df = summarize_by_zone(all_df)
summary_df.to_csv(OUTPUT_DIR / "steering_all_models_summary.csv", index=False)

# ── Print temporal asymmetry table ───────────────────────────
print_temporal_asymmetry_table(all_df)

# ── Plots ─────────────────────────────────────────────────────
plot_temporal_asymmetry(all_df)
plot_summary_bars(all_df)

# ── Final paper numbers ───────────────────────────────────────
print("\n" + "="*90)
print("PAPER NUMBERS (alpha=1.0, averaged across families and layers)")
print("real >> random  → direction-specific effect")
print("real ≈ random   → just perturbation, not informative")
print("="*90)
best = all_df[all_df["alpha"] == 1.0]
for model_name in MODELS:
    short = model_name.split("/")[-1]
    m = best[best["model_name"] == model_name]
    print(f"\n{short}:")
    print(f"  {'Condition':<22} {'Real':>8} {'Random':>8} {'Real-Rand':>10}")
    print(f"  {'─'*52}")
    for steer_type in ["relation_mid", "entity_mid",
                       "entity_late", "relation_late"]:
        sub = m[m["steer_type"] == steer_type]
        if not sub.empty:
            real = sub["target_hit"].mean()
            rand = sub["random_target_hit"].mean()
            diff = real - rand
            print(f"  {steer_type:<22}: {real:>8.3f} {rand:>8.3f} {diff:>10.3f}")

print(f"\n[done] All results saved to {OUTPUT_DIR}/")