"""
run_exp3_subject_token_patching.py

Full model-running script for Experiment 3:
Subject-token vs. final-token entity patching.

This experiment tests whether entity information can be causally effective at
the subject/entity token earlier than it becomes effective at the final
prediction token.

For each model, family, record, and layer, the script compares:
1. Final-token patch:
   donor final-token hidden state -> recipient final-token position.
2. Subject-token patch:
   donor subject/entity-token hidden state -> recipient subject/entity positions.
3. Subject-average patch:
   average donor subject/entity-token hidden state -> recipient subject/entity positions.

This is experimental research code used for the submitted paper. It requires
GPU access, HuggingFace model downloads/access, and substantial runtime.
For lightweight reproduction of paper figures/tables, use the processed CSVs
and analysis scripts.

Models: Llama-3.2-3B, Llama-3-8B, Qwen2.5-3B, Phi-2.
"""

import gc
import re
from pathlib import Path
from collections import Counter

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
MODELS = [
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen2.5-3B",
    "microsoft/phi-2",
]

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "exp3_subject_token_patching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_FILES_ONLY = False
USE_FP16_ON_CUDA = True
MAX_NEW_TOKENS = 8
LAYERS_TO_TEST_STEP = 2

# Optional fixed layers per model. If None, use every 2 layers + final-1.
CUSTOM_LAYERS = None

# ============================================================
# FULL DATASET
# ============================================================
COUNTRY_BANK = [
    ("France", "paris", "french"),
    ("Germany", "berlin", "german"),
    ("Italy", "rome", "italian"),
    ("Spain", "madrid", "spanish"),
    ("Russia", "moscow", "russian"),
    ("Japan", "tokyo", "japanese"),
    ("South Korea", "seoul", "korean"),
    ("Greece", "athens", "greek"),
    ("Turkey", "ankara", "turkish"),
    ("Poland", "warsaw", "polish"),
    ("Netherlands", "amsterdam", "dutch"),
    ("Sweden", "stockholm", "swedish"),
    ("Norway", "oslo", "norwegian"),
    ("Denmark", "copenhagen", "danish"),
    ("Finland", "helsinki", "finnish"),
    ("Hungary", "budapest", "hungarian"),
    ("Thailand", "bangkok", "thai"),
    ("Vietnam", "hanoi", "vietnamese"),
    ("Egypt", "cairo", "arabic"),
    ("Portugal", "lisbon", "portuguese"),
    ("Romania", "bucharest", "romanian"),
    ("Austria", "vienna", "german"),
    ("Belgium", "brussels", "dutch"),
    ("Switzerland", "bern", "german"),
    ("Czechia", "prague", "czech"),
    ("Peru", "lima", "spanish"),
    ("Argentina", "buenos aires", "spanish"),
    ("Colombia", "bogota", "spanish"),
    ("Saudi Arabia", "riyadh", "arabic"),
    ("Qatar", "doha", "arabic"),
    ("UAE", "abu dhabi", "arabic"),
    ("Nigeria", "abuja", "english"),
    ("Kenya", "nairobi", "english"),
]

TENSE_BANK = [
    ("walk", "walked", "walking"),
    ("play", "played", "playing"),
    ("talk", "talked", "talking"),
    ("work", "worked", "working"),
    ("jump", "jumped", "jumping"),
    ("look", "looked", "looking"),
    ("live", "lived", "living"),
    ("love", "loved", "loving"),
    ("wash", "washed", "washing"),
    ("watch", "watched", "watching"),
    ("cook", "cooked", "cooking"),
    ("start", "started", "starting"),
    ("stop", "stopped", "stopping"),
    ("drop", "dropped", "dropping"),
    ("study", "studied", "studying"),
    ("try", "tried", "trying"),
    ("learn", "learned", "learning"),
    ("paint", "painted", "painting"),
    ("send", "sent", "sending"),
    ("make", "made", "making"),
    ("keep", "kept", "keeping"),
]

PLURAL_BANK = [
    ("cat", "cats"),
    ("dog", "dogs"),
    ("car", "cars"),
    ("tree", "trees"),
    ("river", "rivers"),
    ("mountain", "mountains"),
    ("teacher", "teachers"),
    ("table", "tables"),
    ("chair", "chairs"),
    ("window", "windows"),
    ("apple", "apples"),
    ("flower", "flowers"),
    ("bird", "birds"),
    ("cloud", "clouds"),
    ("road", "roads"),
    ("city", "cities"),
    ("baby", "babies"),
    ("story", "stories"),
    ("bus", "buses"),
    ("box", "boxes"),
    ("match", "matches"),
    ("dish", "dishes"),
    ("watch", "watches"),
    ("bottle", "bottles"),
    ("camera", "cameras"),
    ("pencil", "pencils"),
    ("paper", "papers"),
]

ADJECTIVE_BANK = [
    ("tall", "short", "taller"),
    ("fast", "slow", "faster"),
    ("hard", "soft", "harder"),
    ("wide", "narrow", "wider"),
    ("clean", "dirty", "cleaner"),
    ("strong", "weak", "stronger"),
    ("dark", "light", "darker"),
    ("large", "small", "larger"),
    ("big", "small", "bigger"),
    ("hot", "cold", "hotter"),
    ("loud", "quiet", "louder"),
    ("easy", "hard", "easier"),
]

ELEMENT_BANK = [
    ("gold", "au"),
    ("silver", "ag"),
    ("iron", "fe"),
    ("copper", "cu"),
    ("calcium", "ca"),
    ("carbon", "c"),
    ("hydrogen", "h"),
    ("potassium", "k"),
    ("chlorine", "cl"),
    ("fluorine", "f"),
    ("platinum", "pt"),
    ("uranium", "u"),
]

# ============================================================
# PROMPTS
# ============================================================
def make_capital_prompt(x): return f"The capital of {x} is"
def make_language_prompt(x): return f"The official language of {x} is"
def make_opposite_prompt(x): return f"The opposite of {x} is"
def make_comparative_prompt(x): return f"The comparative form of {x} is"
def make_symbol_prompt(x): return f"The chemical symbol for {x} is"
def make_plural_prompt(x): return f"The plural of {x} is"
def make_past_tense_prompt(x): return f"The past tense of {x} is"
def make_present_participle_prompt(x): return f"The present participle of {x} is"

# ============================================================
# TEXT HELPERS
# ============================================================
def normalize(x):
    return " ".join(str(x).lower().strip().replace("\n", " ").split())

def extract_continuation(prompt, full_text):
    return full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()

def extract_first_phrase(prompt, full_text, max_words=3):
    cont = extract_continuation(prompt, full_text)
    piece = re.split(r"[,\.\n;:!?]", cont, maxsplit=1)[0].strip()
    return " ".join(piece.split()[:max_words]).lower().strip()

def extract_slot(family, prompt, full_text):
    if family == "capital":
        return extract_first_phrase(prompt, full_text, max_words=2)
    return extract_first_phrase(prompt, full_text, max_words=1)

def canonical_match(family, pred, expected):
    pred = normalize(pred)
    exp = normalize(expected)
    return pred.split()[0] == exp.split()[0] if pred and exp else False

# ============================================================
# RECORD BUILDING
# ============================================================
def make_even(items):
    return items if len(items) % 2 == 0 else items[:-1]

def pair_up(items):
    items = make_even(items)
    return [(items[i], items[i + 1]) for i in range(0, len(items), 2)]

def build_entity_records():
    """
    Same relation, different entity.
    This mirrors full_donor/entity transfer records from your main experiment.
    """
    records = []

    for (a, a_cap, _), (b, b_cap, _) in pair_up(COUNTRY_BANK):
        records.append({
            "family": "capital",
            "entity_a": a,
            "entity_b": b,
            "prompt_a": make_capital_prompt(a),
            "prompt_b": make_capital_prompt(b),
            "answer_a": a_cap,
            "answer_b": b_cap,
            "subject_a": a,
            "subject_b": b,
        })

    for (a, _, a_lang), (b, _, b_lang) in pair_up(COUNTRY_BANK):
        records.append({
            "family": "language",
            "entity_a": a,
            "entity_b": b,
            "prompt_a": make_language_prompt(a),
            "prompt_b": make_language_prompt(b),
            "answer_a": a_lang,
            "answer_b": b_lang,
            "subject_a": a,
            "subject_b": b,
        })

    for (a_v, a_past, _), (b_v, b_past, _) in pair_up(TENSE_BANK):
        records.append({
            "family": "past_tense",
            "entity_a": a_v,
            "entity_b": b_v,
            "prompt_a": make_past_tense_prompt(a_v),
            "prompt_b": make_past_tense_prompt(b_v),
            "answer_a": a_past,
            "answer_b": b_past,
            "subject_a": a_v,
            "subject_b": b_v,
        })

    for (a_v, _, a_part), (b_v, _, b_part) in pair_up(TENSE_BANK):
        records.append({
            "family": "present_participle",
            "entity_a": a_v,
            "entity_b": b_v,
            "prompt_a": make_present_participle_prompt(a_v),
            "prompt_b": make_present_participle_prompt(b_v),
            "answer_a": a_part,
            "answer_b": b_part,
            "subject_a": a_v,
            "subject_b": b_v,
        })

    for (a_w, a_pl), (b_w, b_pl) in pair_up(PLURAL_BANK):
        records.append({
            "family": "plural",
            "entity_a": a_w,
            "entity_b": b_w,
            "prompt_a": make_plural_prompt(a_w),
            "prompt_b": make_plural_prompt(b_w),
            "answer_a": a_pl,
            "answer_b": b_pl,
            "subject_a": a_w,
            "subject_b": b_w,
        })

    for (a_adj, a_opp, _), (b_adj, b_opp, _) in pair_up(ADJECTIVE_BANK):
        records.append({
            "family": "opposite",
            "entity_a": a_adj,
            "entity_b": b_adj,
            "prompt_a": make_opposite_prompt(a_adj),
            "prompt_b": make_opposite_prompt(b_adj),
            "answer_a": a_opp,
            "answer_b": b_opp,
            "subject_a": a_adj,
            "subject_b": b_adj,
        })

    for (a_adj, _, a_comp), (b_adj, _, b_comp) in pair_up(ADJECTIVE_BANK):
        records.append({
            "family": "comparative",
            "entity_a": a_adj,
            "entity_b": b_adj,
            "prompt_a": make_comparative_prompt(a_adj),
            "prompt_b": make_comparative_prompt(b_adj),
            "answer_a": a_comp,
            "answer_b": b_comp,
            "subject_a": a_adj,
            "subject_b": b_adj,
        })

    for (a_el, a_sym), (b_el, b_sym) in pair_up(ELEMENT_BANK):
        records.append({
            "family": "symbol",
            "entity_a": a_el,
            "entity_b": b_el,
            "prompt_a": make_symbol_prompt(a_el),
            "prompt_b": make_symbol_prompt(b_el),
            "answer_a": a_sym,
            "answer_b": b_sym,
            "subject_a": a_el,
            "subject_b": b_el,
        })

    for i, r in enumerate(records, 1):
        r["record_id"] = i
    return records

RECORDS = build_entity_records()
print(f"[records] total={len(RECORDS)}")
for fam, cnt in Counter(r["family"] for r in RECORDS).items():
    print(f"  {fam}: {cnt}")

# ============================================================
# MODEL WRAPPER
# ============================================================
class Runner:
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
            "torch_dtype": torch.float16 if torch.cuda.is_available() and USE_FP16_ON_CUDA else torch.float32,
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
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.layers = self._get_layers()
        self.num_layers = len(self.layers)
        self.layers_to_test = self._make_layers_to_test()
        print(f"[info] num_layers={self.num_layers}")
        print(f"[info] layers_to_test={self.layers_to_test}")

    def _get_layers(self):
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers
        raise ValueError(f"Unsupported architecture: {self.model_name}")

    def _make_layers_to_test(self):
        if CUSTOM_LAYERS is not None:
            return [l for l in CUSTOM_LAYERS if 0 <= l < self.num_layers - 1]
        layers = list(range(0, self.num_layers - 1, LAYERS_TO_TEST_STEP))
        if self.num_layers - 2 not in layers:
            layers.append(self.num_layers - 2)
        return sorted(set(layers))

    def tokenize(self, prompt):
        return {k: v.to(self.model.device) for k, v in self.tokenizer(prompt, return_tensors="pt").items()}

    def forward(self, prompt):
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            out = self.model(
                **inputs,
                return_dict=True,
                use_cache=False,
                output_hidden_states=True,
            )
        return inputs, out

    def generate(self, prompt):
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def find_subject_positions(self, ids_list, subject):
        """
        Find token positions for the entity/subject string in tokenized prompt.
        Handles leading-space tokenization and multi-token subjects like South Korea.
        """
        attempts = [" " + subject, subject]
        for text in attempts:
            subj_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not subj_ids:
                continue
            for i in range(len(ids_list) - len(subj_ids) + 1):
                if ids_list[i:i + len(subj_ids)] == subj_ids:
                    return list(range(i, i + len(subj_ids)))
        return []

    def generate_with_patch(self, prompt, patch_vec, layer_idx, patch_positions):
        """
        Patches layer output hidden state on the prompt forward pass only.
        patch_vec shape: [1, d_model]
        patch_positions: list[int]
        """
        inputs = self.tokenize(prompt)
        done = {"v": False}

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if not done["v"]:
                h = h.clone()
                for pos in patch_positions:
                    if 0 <= pos < h.shape[1]:
                        h[:, pos, :] = patch_vec.to(device=h.device, dtype=h.dtype)
                done["v"] = True
            return (h,) + out[1:] if isinstance(out, tuple) else h

        handle = self.layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        handle.remove()
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

# ============================================================
# CACHE
# ============================================================
def build_cache(runner, records):
    cached = []
    skipped = []

    for rec in tqdm(records, desc="Caching"):
        inputs_a, out_a = runner.forward(rec["prompt_a"])
        inputs_b, out_b = runner.forward(rec["prompt_b"])

        ids_a = inputs_a["input_ids"][0].tolist()
        ids_b = inputs_b["input_ids"][0].tolist()

        last_pos_a = len(ids_a) - 1
        last_pos_b = len(ids_b) - 1

        subj_pos_a = runner.find_subject_positions(ids_a, rec["subject_a"])
        subj_pos_b = runner.find_subject_positions(ids_b, rec["subject_b"])

        if not subj_pos_a or not subj_pos_b:
            skipped.append({
                "record_id": rec["record_id"],
                "family": rec["family"],
                "entity_a": rec["entity_a"],
                "entity_b": rec["entity_b"],
                "subj_pos_a": str(subj_pos_a),
                "subj_pos_b": str(subj_pos_b),
            })
            continue

        cached.append({
            **rec,
            "out_a": out_a,
            "out_b": out_b,
            "last_pos_a": last_pos_a,
            "last_pos_b": last_pos_b,
            "subj_pos_a": subj_pos_a,
            "subj_pos_b": subj_pos_b,
            "normal_a": runner.generate(rec["prompt_a"]),
            "normal_b": runner.generate(rec["prompt_b"]),
        })

    if skipped:
        skipped_df = pd.DataFrame(skipped)
        safe = runner.model_name.replace("/", "_")
        skipped_df.to_csv(OUTPUT_DIR / f"{safe}_skipped_subject_not_found.csv", index=False)
        print(f"[warn] skipped {len(skipped)} records where subject position was not found")

    return cached

# ============================================================
# EXPERIMENT
# ============================================================
def zone_for_layer(layer_idx, num_layers):
    pct = layer_idx / max(num_layers - 1, 1)
    if pct < 0.35:
        return "early"
    if pct < 0.65:
        return "mid"
    return "late"

def run_patch_experiment(runner, cached):
    rows = []

    for rec in tqdm(cached, desc="Patch experiment"):
        for layer_idx in runner.layers_to_test:
            if layer_idx >= runner.num_layers - 1:
                continue

            h_b = rec["out_b"].hidden_states[layer_idx + 1]

            donor_last_vec = h_b[:, rec["last_pos_b"], :].clone()
            donor_subj_last_vec = h_b[:, rec["subj_pos_b"][-1], :].clone()
            donor_subj_avg_vec = h_b[:, rec["subj_pos_b"], :].mean(dim=1).clone()

            # 1. last-token patch
            text_last = runner.generate_with_patch(
                rec["prompt_a"], donor_last_vec, layer_idx, [rec["last_pos_a"]]
            )
            pred_last = extract_slot(rec["family"], rec["prompt_a"], text_last)
            hit_last = canonical_match(rec["family"], pred_last, rec["answer_b"])

            # 2. subject patch using last subject token vector
            text_subj = runner.generate_with_patch(
                rec["prompt_a"], donor_subj_last_vec, layer_idx, rec["subj_pos_a"]
            )
            pred_subj = extract_slot(rec["family"], rec["prompt_a"], text_subj)
            hit_subj = canonical_match(rec["family"], pred_subj, rec["answer_b"])

            # 3. subject patch using average subject vector
            text_subj_avg = runner.generate_with_patch(
                rec["prompt_a"], donor_subj_avg_vec, layer_idx, rec["subj_pos_a"]
            )
            pred_subj_avg = extract_slot(rec["family"], rec["prompt_a"], text_subj_avg)
            hit_subj_avg = canonical_match(rec["family"], pred_subj_avg, rec["answer_b"])

            rows.append({
                "model_name": runner.model_name,
                "record_id": rec["record_id"],
                "family": rec["family"],
                "entity_a": rec["entity_a"],
                "entity_b": rec["entity_b"],
                "layer_idx": layer_idx,
                "zone": zone_for_layer(layer_idx, runner.num_layers),
                "answer_b": rec["answer_b"],
                "last_token_hit": int(hit_last),
                "last_token_pred": pred_last,
                "subj_token_hit": int(hit_subj),
                "subj_token_pred": pred_subj,
                "subj_avg_hit": int(hit_subj_avg),
                "subj_avg_pred": pred_subj_avg,
                "subj_pos_a": str(rec["subj_pos_a"]),
                "subj_pos_b": str(rec["subj_pos_b"]),
            })

    return pd.DataFrame(rows)

# ============================================================
# SUMMARIES / PLOTS
# ============================================================
def summarize(df):
    return (
        df.groupby(["model_name", "family", "zone", "layer_idx"], as_index=False)
        .agg(
            n=("last_token_hit", "count"),
            last_token_pct=("last_token_hit", "mean"),
            subj_token_pct=("subj_token_hit", "mean"),
            subj_avg_pct=("subj_avg_hit", "mean"),
        )
    )

def summarize_model(df):
    return (
        df.groupby(["model_name", "zone", "layer_idx"], as_index=False)
        .agg(
            n=("last_token_hit", "count"),
            last_token_pct=("last_token_hit", "mean"),
            subj_token_pct=("subj_token_hit", "mean"),
            subj_avg_pct=("subj_avg_hit", "mean"),
        )
    )

def plot_model(summary_model, model_name):
    sub = summary_model[summary_model["model_name"] == model_name].sort_values("layer_idx")
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sub["layer_idx"], sub["last_token_pct"], marker="o", linewidth=2, label="Last-token patch")
    ax.plot(sub["layer_idx"], sub["subj_token_pct"], marker="s", linewidth=2, label="Subject-token patch")
    ax.plot(sub["layer_idx"], sub["subj_avg_pct"], marker="^", linewidth=1.5, linestyle="--", label="Subject avg patch")
    ax.axhline(0.5, linestyle="--", alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entity transfer rate")
    ax.set_title(f"Subject vs Last Token Patching — {model_name.split('/')[-1]}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = OUTPUT_DIR / f"{model_name.replace('/', '_')}_subject_vs_last_patch.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[saved] {fname}")

def print_summary(summary_model):
    print("\n" + "=" * 100)
    
    print("=" * 100)
    for model_name, sub in summary_model.groupby("model_name"):
        print(f"\nMODEL: {model_name}")
        print(f"{'Layer':<8} {'Zone':<8} {'N':>5} {'Last%':>10} {'Subj%':>10} {'SubjAvg%':>10}")
        print("-" * 60)
        for _, row in sub.sort_values("layer_idx").iterrows():
            print(
                f"L{int(row['layer_idx']):<7} {row['zone']:<8} {int(row['n']):>5} "
                f"{row['last_token_pct']:>10.1%} {row['subj_token_pct']:>10.1%} {row['subj_avg_pct']:>10.1%}"
            )

# ============================================================
# MAIN
# ============================================================
def main():
    all_rows = []

    for model_name in MODELS:
        print(f"\n{'=' * 100}\nMODEL: {model_name}\n{'=' * 100}")
        runner = Runner(model_name)

        cached = build_cache(runner, RECORDS)
        print(f"[cache] kept={len(cached)} / total={len(RECORDS)}")

        results_df = run_patch_experiment(runner, cached)
        safe = model_name.replace("/", "_")
        results_df.to_csv(OUTPUT_DIR / f"{safe}_subject_patch_raw.csv", index=False)
        all_rows.append(results_df)

        del runner.model, runner.tokenizer, runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(OUTPUT_DIR / "subject_patch_all_models_raw.csv", index=False)

    family_summary = summarize(all_df)
    model_summary = summarize_model(all_df)
    family_summary.to_csv(OUTPUT_DIR / "subject_patch_by_family_summary.csv", index=False)
    model_summary.to_csv(OUTPUT_DIR / "subject_patch_model_summary.csv", index=False)

    print_summary(model_summary)

    for model_name in MODELS:
        plot_model(model_summary, model_name)

    print(f"\n[done] All results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
