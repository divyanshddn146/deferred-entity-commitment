"""
run_exp2_both_change_and_controls.py

Full model-running script for Experiment 2:
Both-change relation/entity competition.

When both entity and relation change simultaneously, this experiment asks
which signal dominates at each layer.

Prediction:
  Mid layers:  relation_wins
  Late layers: entity_wins
  Crossover layer approximately tracks late entity commitment.

Controls/diagnostics:
  unrelated_donor  — overwrite diagnostic / control for structured transfer.
                     It tests whether arbitrary donor states produce structured
                     relation/entity wins; late entity-like overwrite may occur
                     and is reported separately.
  alternate_donor  — robustness check, not a negative control.
  noise_patch      — true negative control; structured wins should be near zero.
  self_patch       — hook sanity check; original_retained should be near 100%.

This is experimental research code used for the submitted paper. It requires
GPU access, HuggingFace model downloads/access, and substantial runtime.
For lightweight reproduction of paper figures/tables, use the processed CSVs
and analysis scripts.

Models: Llama-3.2-3B, Llama-3-8B, Qwen2.5-3B, Phi-2.
"""

import gc
import random
import re
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

OUTPUT_DIR = ROOT / "results" / "exp2_both_change"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_FILES_ONLY = False
USE_4BIT         = False
USE_FP16_ON_CUDA = True
MAX_NEW_TOKENS   = 8
LAYERS_TO_TEST_STEP = 2   # test every 2 layers + final layer
VERBOSE          = False   # set True only for debugging

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Bootstrap
BOOTSTRAP_SAMPLES = 300
BOOTSTRAP_ALPHA   = 0.05

# ============================================================
# FULL DATASET — same as unified_law_final.py
# ============================================================

# COUNTRY_BANK (33): capital + language pairs
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

# TENSE_BANK (21): past_tense + present_participle pairs
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

# ADJECTIVE_BANK (12): opposite + comparative pairs
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
# PROMPTS — same as unified_law_final.py
# ============================================================
def make_capital_prompt(x):            return f"The capital of {x} is"
def make_language_prompt(x):           return f"The official language of {x} is"
def make_past_prompt(x):               return f"The past tense of {x} is"
def make_participle_prompt(x):         return f"The present participle of {x} is"
def make_opposite_prompt(x):           return f"The opposite of {x} is"
def make_comparative_prompt(x):        return f"The comparative form of {x} is"

# ============================================================
# TEXT HELPERS
# ============================================================
def normalize(x):
    return " ".join(x.lower().strip().replace("\n", " ").split())

def extract_continuation(prompt, full_text):
    return full_text[len(prompt):].strip() \
        if full_text.startswith(prompt) else full_text.strip()

def extract_first_phrase(prompt, full_text, max_words=2):
    cont  = extract_continuation(prompt, full_text)
    piece = re.split(r"[,\.\n;:!?]", cont, maxsplit=1)[0].strip()
    return " ".join(piece.split()[:max_words]).lower().strip()

def extract_slot(family, prompt, full_text):
    if family == "capital":
        return extract_first_phrase(prompt, full_text, max_words=2)
    return extract_first_phrase(prompt, full_text, max_words=1)

def canonical_match(family, pred, expected):
    pred, exp = normalize(pred), normalize(expected)
    return pred.split()[0] == exp.split()[0] if pred and exp else False

# ============================================================
# BUILD BOTH-CHANGE RECORDS
#
# For each record:
#   prompt_a  = recipient (entity_a, relation_a)
#   prompt_b  = donor     (entity_b, relation_b)
#   Both entity AND relation differ from recipient.
#
#   answer_a        = what model normally outputs for prompt_a
#   answer_b        = donor's specific answer  → ENTITY WINS if output matches
#   answer_relation = relation_b applied to entity_a → RELATION WINS if output matches
#
# Example (capital/language, France/Japan):
#   prompt_a          = "The capital of France is"   → paris
#   prompt_b          = "The official language of Japan is" → japanese
#   answer_a          = paris     (original)
#   answer_b          = japanese  (entity wins: Japan's language)
#   answer_relation   = french    (relation wins: France's language)
# ============================================================
def build_both_change_records():
    rng     = random.Random(SEED + 789)
    records = []

    # ── Country: capital ↔ language ──────────────────────────
    for i, (ca, cap_a, lang_a) in enumerate(COUNTRY_BANK):
        others = [(c, cap, lang) for j, (c, cap, lang)
                  in enumerate(COUNTRY_BANK) if j != i]
        cb, cap_b, lang_b = rng.choice(others)

        # cap_a receives lang_b (different entity)
        records.append({
            "family_pair": "capital__language",
            "recipient_family": "capital",
            "donor_family": "language",
            "entity_a": ca, "entity_b": cb,
            "prompt_a": make_capital_prompt(ca),
            "prompt_b": make_language_prompt(cb),
            "answer_a": cap_a,   # paris      original
            "answer_b": lang_b,  # japanese   entity wins
            "answer_relation": lang_a,  # french  relation wins
        })

        # lang_a receives cap_b (different entity)
        records.append({
            "family_pair": "language__capital",
            "recipient_family": "language",
            "donor_family": "capital",
            "entity_a": ca, "entity_b": cb,
            "prompt_a": make_language_prompt(ca),
            "prompt_b": make_capital_prompt(cb),
            "answer_a": lang_a,  # french  original
            "answer_b": cap_b,   # tokyo   entity wins
            "answer_relation": cap_a,   # paris   relation wins
        })

    # ── Tense: past_tense ↔ present_participle ───────────────
    for i, (va, past_a, part_a) in enumerate(TENSE_BANK):
        others = [(v, past, part) for j, (v, past, part)
                  in enumerate(TENSE_BANK) if j != i]
        vb, past_b, part_b = rng.choice(others)

        # past_a receives part_b
        records.append({
            "family_pair": "past_tense__present_participle",
            "recipient_family": "past_tense",
            "donor_family": "present_participle",
            "entity_a": va, "entity_b": vb,
            "prompt_a": make_past_prompt(va),
            "prompt_b": make_participle_prompt(vb),
            "answer_a": past_a,   # walked   original
            "answer_b": part_b,   # playing  entity wins
            "answer_relation": part_a,  # walking  relation wins
        })

        # part_a receives past_b
        records.append({
            "family_pair": "present_participle__past_tense",
            "recipient_family": "present_participle",
            "donor_family": "past_tense",
            "entity_a": va, "entity_b": vb,
            "prompt_a": make_participle_prompt(va),
            "prompt_b": make_past_prompt(vb),
            "answer_a": part_a,   # walking  original
            "answer_b": past_b,   # played   entity wins
            "answer_relation": past_a,  # walked   relation wins
        })

    # ── Adjective: opposite ↔ comparative ────────────────────
    for i, (adj_a, opp_a, comp_a) in enumerate(ADJECTIVE_BANK):
        others = [(a, opp, comp) for j, (a, opp, comp)
                  in enumerate(ADJECTIVE_BANK) if j != i]
        adj_b, opp_b, comp_b = rng.choice(others)

        # opp_a receives comp_b
        records.append({
            "family_pair": "opposite__comparative",
            "recipient_family": "opposite",
            "donor_family": "comparative",
            "entity_a": adj_a, "entity_b": adj_b,
            "prompt_a": make_opposite_prompt(adj_a),
            "prompt_b": make_comparative_prompt(adj_b),
            "answer_a": opp_a,    # short    original
            "answer_b": comp_b,   # faster   entity wins
            "answer_relation": comp_a,  # taller   relation wins
        })

        # comp_a receives opp_b
        records.append({
            "family_pair": "comparative__opposite",
            "recipient_family": "comparative",
            "donor_family": "opposite",
            "entity_a": adj_a, "entity_b": adj_b,
            "prompt_a": make_comparative_prompt(adj_a),
            "prompt_b": make_opposite_prompt(adj_b),
            "answer_a": comp_a,   # taller  original
            "answer_b": opp_b,    # slow    entity wins
            "answer_relation": opp_a,   # short   relation wins
        })

    for i, r in enumerate(records, 1):
        r["record_id"] = i
    return records


def build_unrelated_controls(records):
    """
    Unrelated-donor overwrite diagnostic.

    Different entity and different family. This tests whether arbitrary donor
    states produce structured relation/entity wins. Late entity-like overwrite can
    occur, so this is not treated as a pure no-effect negative control.
    """
    rng  = random.Random(SEED + 111)
    ctrl = []
    pool = [(r["prompt_b"], r["answer_b"], r["donor_family"], r["entity_b"])
            for r in records]

    for rec in records:
        candidates = [
            (pb, ab, df, eb) for pb, ab, df, eb in pool
            if df != rec["recipient_family"]
            and df != rec["donor_family"]
            and eb != rec["entity_a"]
        ]
        if not candidates:
            continue
        pb, ab, df, eb = rng.choice(candidates)
        ctrl.append({
            "control_type":    "unrelated_donor",
            "family_pair":     rec["family_pair"],
            "recipient_family": rec["recipient_family"],
            "donor_family":    df,
            "entity_a":        rec["entity_a"],
            "entity_b":        eb,
            "prompt_a":        rec["prompt_a"],
            "prompt_b":        pb,
            "answer_a":        rec["answer_a"],
            "answer_b":        ab,
            "answer_relation": rec["answer_relation"],
        })

    for i, r in enumerate(ctrl, 1):
        r["record_id"] = i
    return ctrl


def build_alternate_donor_controls(records):
    """
    ROBUSTNESS CHECK (not a negative control): alternate donor.
    Same family pair, but a different entity_b than the main record.

    NOTE: This is NOT a negative control. Because entity and relation
    both still differ from recipient, this is another valid both-change
    donor. It may legitimately produce entity_wins or relation_wins.
    Use it to check robustness of the crossover pattern across donor
    choice, not to show that transfer should be low.
    """
    rng  = random.Random(SEED + 222)
    ctrl = []

    for rec in records:
        # find donors with same family_pair but different entity_b
        same_pair = [
            r for r in records
            if r["family_pair"] == rec["family_pair"]
            and r["entity_b"] != rec["entity_b"]
            and r["entity_b"] != rec["entity_a"]
        ]
        if not same_pair:
            continue
        alt = rng.choice(same_pair)
        ctrl.append({
            "control_type":     "alternate_donor",
            "family_pair":      rec["family_pair"],
            "recipient_family": rec["recipient_family"],
            "donor_family":     rec["donor_family"],
            "entity_a":         rec["entity_a"],
            "entity_b":         alt["entity_b"],   # different entity
            "prompt_a":         rec["prompt_a"],
            "prompt_b":         alt["prompt_b"],   # different donor prompt
            "answer_a":         rec["answer_a"],
            "answer_b":         alt["answer_b"],   # different entity's answer
            "answer_relation":  rec["answer_relation"],  # still same relation target
        })

    for i, r in enumerate(ctrl, 1):
        r["record_id"] = i
    return ctrl


BOTH_CHANGE_RECORDS       = build_both_change_records()
UNRELATED_CTRL_RECORDS    = build_unrelated_controls(BOTH_CHANGE_RECORDS)
ALTERNATE_DONOR_RECORDS   = build_alternate_donor_controls(BOTH_CHANGE_RECORDS)

print(f"[records] both_change       = {len(BOTH_CHANGE_RECORDS)}")
print(f"[records] unrelated_ctrl    = {len(UNRELATED_CTRL_RECORDS)}")
print(f"[records] alternate_donor   = {len(ALTERNATE_DONOR_RECORDS)} "
      f"(robustness check, not negative control)")
print("[family pairs]")
for fp, cnt in Counter(
        r["family_pair"] for r in BOTH_CHANGE_RECORDS).items():
    print(f"  {fp}: {cnt}")

# ============================================================
# MODEL WRAPPER
# ============================================================
class ModelRunner:
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"\n[load] {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=LOCAL_FILES_ONLY,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}
        if USE_4BIT and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = (
                torch.float16
                if torch.cuda.is_available() and USE_FP16_ON_CUDA
                else torch.float32)
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
            local_files_only=LOCAL_FILES_ONLY,
        )
        self.model.eval()
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(self.model, "generation_config") and \
           getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = \
                self.tokenizer.pad_token_id

        self.layers     = self._get_layers()
        self.num_layers = len(self.layers)
        # Auto layer detection — every 2 layers + final layer
        self.layers_to_test = list(
            range(0, self.num_layers, LAYERS_TO_TEST_STEP))
        if (self.num_layers - 1) not in self.layers_to_test:
            self.layers_to_test.append(self.num_layers - 1)
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
        raise ValueError(f"Unsupported: {self.model_name}")

    def tokenize(self, prompt):
        return {k: v.to(self.model.device)
                for k, v in self.tokenizer(
                    prompt, return_tensors="pt").items()}

    def forward(self, prompt):
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            out = self.model(**inputs, return_dict=True,
                             use_cache=False, output_hidden_states=True)
        return inputs, out

    def generate_text(self, prompt):
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def generate_with_patch(self, prompt, patch_vector,
                             layer_idx, pos_idx):
        inputs = self.tokenize(prompt)
        done   = {"v": False}

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if not done["v"] and h.shape[1] > pos_idx:
                h = h.clone()
                h[:, pos_idx, :] = patch_vector.to(h.dtype)
                done["v"] = True
            return (h,) + out[1:] if isinstance(out, tuple) else h

        handle = self.layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id)
        handle.remove()
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def generate_with_noise_patch(self, prompt, ref_vector,
                               layer_idx, pos_idx, noise_std=1.0):
        noise = torch.randn_like(ref_vector)
        noise = noise / (noise.norm() + 1e-8)
        noise = noise * ref_vector.norm() * noise_std
        return self.generate_with_patch(prompt, noise, layer_idx, pos_idx)

    def generate_with_self_patch(self, prompt, out_a,
                                  layer_idx, pos_idx):
        """
        SELF PATCH CONTROL: patch prompt_a's own hidden state back in.
        Should produce original_retained near 100% (no change expected).
        """
        self_vec = out_a.hidden_states[layer_idx + 1][:, pos_idx, :].clone()
        return self.generate_with_patch(prompt, self_vec, layer_idx, pos_idx)


# ============================================================
# CACHING
# ============================================================
def cache_records(runner, records, desc="Caching"):
    cached = []
    for rec in tqdm(records, desc=desc):
        inputs_a, out_a = runner.forward(rec["prompt_a"])
        inputs_b, out_b = runner.forward(rec["prompt_b"])
        cached.append({
            **rec,
            "out_a":        out_a,
            "out_b":        out_b,
            "pos_a":        inputs_a["input_ids"].shape[1] - 1,
            "pos_b":        inputs_b["input_ids"].shape[1] - 1,
            "normal_a_text": runner.generate_text(rec["prompt_a"]),
            "normal_b_text": runner.generate_text(rec["prompt_b"]),
        })
    return cached


# ============================================================
# CLASSIFY BOTH-CHANGE OUTCOME
# ============================================================
def classify_both_change(rec, patched_text):
    """
    Four possible outcomes:
      original_retained  output = answer_a     (patch had no effect)
      entity_wins        output = answer_b     (donor's specific answer dominated)
      relation_wins      output = answer_relation (relation frame applied to entity_a)
      mixed_or_other     output = something else

    Priority: original > entity > relation > mixed
    At mid layers: predict relation_wins
    At late layers: predict entity_wins
    """
    rf     = rec["recipient_family"]
    p_slot = extract_slot(rf, rec["prompt_a"], patched_text)

    is_original = canonical_match(rf, p_slot, rec["answer_a"])
    is_entity   = canonical_match(rf, p_slot, rec["answer_b"])
    is_relation = canonical_match(rf, p_slot, rec["answer_relation"])

    if is_original: return "original_retained"
    if is_entity:   return "entity_wins"
    if is_relation: return "relation_wins"
    return "mixed_or_other"


# ============================================================
# RUN EXPERIMENT
# ============================================================
def run_experiment(runner, cached_records, desc="Experiment"):
    rows = []

    for rec in tqdm(cached_records, desc=desc):
        for layer_idx in runner.layers_to_test:
            if layer_idx >= runner.num_layers - 1:
                continue

            donor_h = rec["out_b"].hidden_states[
                layer_idx + 1][:, rec["pos_b"], :].clone()
            patched_text = runner.generate_with_patch(
                rec["prompt_a"], donor_h, layer_idx, rec["pos_a"])

            effect = classify_both_change(rec, patched_text)

            if VERBOSE:
                print(f"\nL{layer_idx} | {rec['recipient_family']} "
                      f"{rec['entity_a']} <- {rec['donor_family']} "
                      f"{rec['entity_b']} | {effect}")
                print(f"  patched: {patched_text}")

            rows.append({
                "model_name":       runner.model_name,
                "family_pair":      rec["family_pair"],
                "recipient_family": rec["recipient_family"],
                "donor_family":     rec["donor_family"],
                "entity_a":         rec["entity_a"],
                "entity_b":         rec["entity_b"],
                "layer_idx":        layer_idx,
                "answer_a":         rec["answer_a"],
                "answer_b":         rec["answer_b"],
                "answer_relation":  rec["answer_relation"],
                "patched_text":     patched_text,
                "effect":           effect,
                "entity_wins":      int(effect == "entity_wins"),
                "relation_wins":    int(effect == "relation_wins"),
                "original_retained":int(effect == "original_retained"),
                "mixed_or_other":   int(effect == "mixed_or_other"),
            })

    return pd.DataFrame(rows)


def run_noise_control(runner, cached_records, desc="Noise control"):
    """
    NOISE CONTROL: patch with Gaussian noise scaled to donor magnitude.
    Structured transfer should be near zero — any signal above this
    baseline is genuinely from the donor hidden state, not from the
    mechanics of patching itself.
    """
    rows = []
    for rec in tqdm(cached_records, desc=desc):
        for layer_idx in runner.layers_to_test:
            if layer_idx >= runner.num_layers - 1:
                continue
            ref_vec = rec["out_b"].hidden_states[
                layer_idx + 1][:, rec["pos_b"], :].clone()
            patched_text = runner.generate_with_noise_patch(
                rec["prompt_a"], ref_vec, layer_idx, rec["pos_a"])
            effect = classify_both_change(rec, patched_text)
            rows.append({
                "model_name":       runner.model_name,
                "family_pair":      rec["family_pair"],
                "recipient_family": rec["recipient_family"],
                "layer_idx":        layer_idx,
                "effect":           effect,
                "entity_wins":      int(effect == "entity_wins"),
                "relation_wins":    int(effect == "relation_wins"),
                "original_retained":int(effect == "original_retained"),
                "mixed_or_other":   int(effect == "mixed_or_other"),
            })
    return pd.DataFrame(rows)


def run_self_patch_control(runner, cached_records, desc="Self-patch control"):
    """
    SELF-PATCH CONTROL: patch prompt_a's own hidden state back in.
    original_retained should be near 100%.
    Any deviation indicates numerical instability in the patching hook.
    """
    rows = []
    for rec in tqdm(cached_records, desc=desc):
        for layer_idx in runner.layers_to_test:
            if layer_idx >= runner.num_layers - 1:
                continue
            patched_text = runner.generate_with_self_patch(
                rec["prompt_a"], rec["out_a"], layer_idx, rec["pos_a"])
            effect = classify_both_change(rec, patched_text)
            rows.append({
                "model_name":       runner.model_name,
                "family_pair":      rec["family_pair"],
                "recipient_family": rec["recipient_family"],
                "layer_idx":        layer_idx,
                "effect":           effect,
                "entity_wins":      int(effect == "entity_wins"),
                "relation_wins":    int(effect == "relation_wins"),
                "original_retained":int(effect == "original_retained"),
                "mixed_or_other":   int(effect == "mixed_or_other"),
            })
    return pd.DataFrame(rows)


# ============================================================
# BOOTSTRAP CI
# ============================================================
def bootstrap_ci(df, model_name, n_boot=BOOTSTRAP_SAMPLES):
    """
    Bootstrap CIs over family_pairs.
    Returns per-layer 95% CI for entity_wins and relation_wins.
    """
    sub    = df[df["model_name"] == model_name].copy()
    pairs  = sorted(sub["family_pair"].unique())
    layers = sorted(sub["layer_idx"].unique())

    lut_ent = {
        (fp, l): v
        for fp, l, v in sub.groupby(
            ["family_pair", "layer_idx"])["entity_wins"]
        .mean().reset_index().itertuples(index=False)
    }
    lut_rel = {
        (fp, l): v
        for fp, l, v in sub.groupby(
            ["family_pair", "layer_idx"])["relation_wins"]
        .mean().reset_index().itertuples(index=False)
    }

    rng       = np.random.default_rng(SEED + 999)
    boot_rows = []

    for _ in range(n_boot):
        sample = rng.choice(pairs, size=len(pairs), replace=True)
        for l in layers:
            ev = [lut_ent[(fp, l)] for fp in sample if (fp, l) in lut_ent]
            rv = [lut_rel[(fp, l)] for fp in sample if (fp, l) in lut_rel]
            boot_rows.append({
                "layer_idx":    l,
                "entity_wins":  float(np.mean(ev)) if ev else 0.0,
                "relation_wins":float(np.mean(rv)) if rv else 0.0,
            })

    boot_df = pd.DataFrame(boot_rows)
    lo = 100 * (BOOTSTRAP_ALPHA / 2)
    hi = 100 * (1 - BOOTSTRAP_ALPHA / 2)
    ci_rows = []
    for l, sdf in boot_df.groupby("layer_idx"):
        ci_rows.append({
            "model_name":      model_name,
            "layer_idx":       l,
            "ent_low":         np.percentile(sdf["entity_wins"],   lo),
            "ent_high":        np.percentile(sdf["entity_wins"],   hi),
            "rel_low":         np.percentile(sdf["relation_wins"], lo),
            "rel_high":        np.percentile(sdf["relation_wins"], hi),
        })
    return pd.DataFrame(ci_rows).sort_values("layer_idx").reset_index(
        drop=True)


# ============================================================
# CROSSOVER LAYER DETECTION
# ============================================================
def find_crossover_layer(df, model_name):
    """
    Crossover = first layer where entity_wins >= relation_wins
    after relation_wins was previously higher.
    Returns None if not detected.
    """
    sub = (
        df[df["model_name"] == model_name]
        .groupby("layer_idx")[["entity_wins","relation_wins"]]
        .mean()
        .reset_index()
        .sort_values("layer_idx")
    )
    prev_rel_higher = None
    for _, row in sub.iterrows():
        rel_higher = row["relation_wins"] > row["entity_wins"]
        if prev_rel_higher is not None \
                and prev_rel_higher and not rel_higher:
            return int(row["layer_idx"])
        prev_rel_higher = rel_higher
    return None


# ============================================================
# PLOTS
# ============================================================
def plot_model(df, ctrl_unrel_df, alt_donor_df,
               ci_df, model_name, crossover_layer):
    short  = model_name.split("/")[-1]
    sub    = df[df["model_name"] == model_name]
    agg    = (sub.groupby("layer_idx")[
        ["entity_wins","relation_wins","original_retained","mixed_or_other"]]
        .mean().reset_index().sort_values("layer_idx"))

    # ── Figure 1: entity vs relation with CI ─────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    if ci_df is not None and not ci_df.empty:
        ax.fill_between(ci_df["layer_idx"],
                        ci_df["rel_low"], ci_df["rel_high"],
                        alpha=0.15, color="tab:blue")
        ax.fill_between(ci_df["layer_idx"],
                        ci_df["ent_low"], ci_df["ent_high"],
                        alpha=0.15, color="tab:red")

    ax.plot(agg["layer_idx"], agg["relation_wins"],
            marker="o", color="tab:blue", linewidth=2,
            label="Relation wins")
    ax.plot(agg["layer_idx"], agg["entity_wins"],
            marker="o", color="tab:red", linewidth=2,
            label="Entity wins")

    # Controls
    if ctrl_unrel_df is not None and not ctrl_unrel_df.empty:
        cu = (ctrl_unrel_df[ctrl_unrel_df["model_name"] == model_name]
              .groupby("layer_idx")[["entity_wins","relation_wins"]]
              .mean().reset_index().sort_values("layer_idx"))
        if not cu.empty:
            ax.plot(cu["layer_idx"],
                    (cu["entity_wins"] + cu["relation_wins"]) / 2,
                    marker="^", linestyle="--", color="tab:gray",
                    alpha=0.6, label="Unrelated diagnostic")

    if crossover_layer is not None:
        ax.axvline(crossover_layer, linestyle=":",
                   color="black", linewidth=1.5,
                   label=f"Crossover L{crossover_layer}")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax.set_title(
        f"Both-Change: Entity vs Relation Dominance — {short}\n"
        f"95% CI shaded | crossover L{crossover_layer}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of cases")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = OUTPUT_DIR / f"{model_name.replace('/','_')}_entity_vs_relation.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[saved] {fname.name}")

    # ── Figure 2: stacked area ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    layers  = agg["layer_idx"].values
    ax.stackplot(
        layers,
        agg["original_retained"],
        agg["relation_wins"],
        agg["entity_wins"],
        agg["mixed_or_other"],
        labels=["Original", "Relation wins", "Entity wins", "Mixed"],
        colors=["tab:gray","tab:blue","tab:red","tab:orange"],
        alpha=0.75,
    )
    if crossover_layer is not None:
        ax.axvline(crossover_layer, linestyle=":",
                   color="black", linewidth=1.5,
                   label=f"Crossover L{crossover_layer}")
    ax.set_title(
        f"Both-Change: Outcome Breakdown — {short}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of cases")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fname = OUTPUT_DIR / f"{model_name.replace('/','_')}_stacked.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[saved] {fname.name}")

    # ── Figure 3: per family pair ─────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    palette = ["tab:blue","tab:orange","tab:green","tab:red",
               "tab:purple","tab:brown"]
    for color, (fp, fdf) in zip(
            palette, sub.groupby("family_pair")):
        fagg = (fdf.groupby("layer_idx")[
            ["entity_wins","relation_wins"]]
            .mean().reset_index().sort_values("layer_idx"))
        ax.plot(fagg["layer_idx"], fagg["entity_wins"],
                marker="o", color=color, linestyle="-",
                label=f"{fp} entity", alpha=0.8)
        ax.plot(fagg["layer_idx"], fagg["relation_wins"],
                marker="s", color=color, linestyle="--",
                alpha=0.8)
    if crossover_layer is not None:
        ax.axvline(crossover_layer, linestyle=":",
                   color="black", linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax.set_title(f"Both-Change by Family Pair — {short}\n"
                 f"(solid=entity, dashed=relation)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    fname = OUTPUT_DIR / f"{model_name.replace('/','_')}_by_family.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[saved] {fname.name}")


# ============================================================
# PRINT SUMMARIES
# ============================================================
def print_model_summary(df, model_name):
    sub = df[df["model_name"] == model_name]
    agg = (sub.groupby("layer_idx")[
        ["entity_wins","relation_wins","original_retained","mixed_or_other"]]
        .mean().reset_index().sort_values("layer_idx"))

    print(f"\n{'='*75}")
    print(f"BOTH-CHANGE RESULTS — {model_name.split('/')[-1]}")
    print(f"{'='*75}")
    print(f"  {'Layer':<8} {'entity_wins':>12} {'relation_wins':>14} "
          f"{'original':>10} {'mixed':>8}")
    print("  " + "─"*56)
    for _, row in agg.iterrows():
        dominant = ("ENTITY"   if row["entity_wins"]   > row["relation_wins"]
                    else "RELATION")
        print(f"  {int(row['layer_idx']):<8} "
              f"{row['entity_wins']:>12.3f} "
              f"{row['relation_wins']:>14.3f} "
              f"{row['original_retained']:>10.3f} "
              f"{row['mixed_or_other']:>8.3f}  "
              f"[{dominant}]")


def print_control_summary(df, ctrl_df, ctrl_name, model_name):
    if ctrl_df.empty: return
    real = (
        df[df["model_name"] == model_name]
        .groupby("layer_idx")[["entity_wins","relation_wins"]]
        .mean().reset_index()
    )
    ctrl = (
        ctrl_df[ctrl_df["model_name"] == model_name]
        .groupby("layer_idx")[["entity_wins","relation_wins"]]
        .mean().reset_index()
    )
    print(f"\n  [{ctrl_name}] max entity_wins: "
          f"real={real['entity_wins'].max():.3f}  "
          f"ctrl={ctrl['entity_wins'].max():.3f}")
    print(f"  [{ctrl_name}] max relation_wins: "
          f"real={real['relation_wins'].max():.3f}  "
          f"ctrl={ctrl['relation_wins'].max():.3f}")


# ============================================================
# MAIN LOOP
# ============================================================
all_main_rows        = []
all_unrel_ctrl_rows  = []
all_alt_ctrl_rows    = []   # alternate_donor (robustness, not negative ctrl)
all_noise_ctrl_rows  = []   # noise patch (true negative control)
all_self_ctrl_rows   = []   # self patch (sanity check)
onset_rows           = []

for model_name in MODELS:
    print(f"\n{'#'*80}\nMODEL: {model_name}\n{'#'*80}")
    runner = ModelRunner(model_name)

    # Cache
    cached_main  = cache_records(runner, BOTH_CHANGE_RECORDS,
                                  "Caching main")
    cached_unrel = cache_records(runner, UNRELATED_CTRL_RECORDS,
                                  "Caching unrelated ctrl")
    cached_alt   = cache_records(runner, ALTERNATE_DONOR_RECORDS,
                                  "Caching alternate-donor ctrl")

    # Run main + controls
    main_df  = run_experiment(runner, cached_main,
                               desc="Both-change main")
    unrel_df = run_experiment(runner, cached_unrel,
                               desc="Unrelated ctrl")
    alt_df   = run_experiment(runner, cached_alt,
                               desc="Alternate-donor ctrl")
    noise_df = run_noise_control(runner, cached_main,
                                  desc="Noise ctrl")
    self_df  = run_self_patch_control(runner, cached_main,
                                       desc="Self-patch ctrl")

    all_main_rows.append(main_df)
    all_unrel_ctrl_rows.append(unrel_df)
    all_alt_ctrl_rows.append(alt_df)
    all_noise_ctrl_rows.append(noise_df)
    all_self_ctrl_rows.append(self_df)

    # Per-model CSVs
    safe_name = model_name.replace("/", "_")
    main_df.to_csv(
        OUTPUT_DIR / f"{safe_name}_both_change_raw.csv", index=False)
    unrel_df.to_csv(
        OUTPUT_DIR / f"{safe_name}_unrelated_ctrl_raw.csv", index=False)
    alt_df.to_csv(
        OUTPUT_DIR / f"{safe_name}_alternate_donor_ctrl_raw.csv", index=False)
    noise_df.to_csv(
        OUTPUT_DIR / f"{safe_name}_noise_ctrl_raw.csv", index=False)
    self_df.to_csv(
        OUTPUT_DIR / f"{safe_name}_self_patch_ctrl_raw.csv", index=False)

    # Bootstrap CI
    ci_df = bootstrap_ci(main_df, model_name)
    ci_df.to_csv(
        OUTPUT_DIR / f"{safe_name}_ci.csv", index=False)

    # Crossover
    crossover = find_crossover_layer(main_df, model_name)
    print(f"\n  Crossover layer = {crossover}")

    onset_rows.append({
        "model_name":        model_name,
        "num_layers":        runner.num_layers,
        "crossover_layer":   crossover,
        "crossover_pct_depth": (
            round(crossover / runner.num_layers, 3)
            if crossover is not None else None),
    })

    # Print
    print_model_summary(main_df, model_name)
    print(f"\n  [OVERWRITE DIAGNOSTIC — unrelated donor]")
    print_control_summary(main_df, unrel_df, "unrelated_donor", model_name)
    print(f"\n  [NEGATIVE CONTROL — noise patch]")
    print_control_summary(main_df, noise_df, "noise_patch", model_name)
    print(f"\n  [SANITY CHECK — self patch "
          f"(original_retained should be ~100%)]")
    if not self_df.empty:
        sp = (self_df[self_df["model_name"] == model_name]
              .groupby("layer_idx")["original_retained"]
              .mean())
        print(f"    self_patch original_retained: "
              f"mean={sp.mean():.3f}  min={sp.min():.3f}")
    print(f"\n  [ROBUSTNESS — alternate donor (not a negative ctrl)]")
    print_control_summary(main_df, alt_df, "alternate_donor", model_name)

    # Plots
    plot_model(main_df, unrel_df, alt_df,
               ci_df, model_name, crossover)

    del runner.model, runner.tokenizer, runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# COMBINED RESULTS
# ============================================================
def safe_concat(dfs):
    valid = [d for d in dfs if not d.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()

all_main  = safe_concat(all_main_rows)
all_unrel = safe_concat(all_unrel_ctrl_rows)
all_alt   = safe_concat(all_alt_ctrl_rows)
all_noise = safe_concat(all_noise_ctrl_rows)
all_self  = safe_concat(all_self_ctrl_rows)

all_main.to_csv(OUTPUT_DIR / "both_change_all_models_raw.csv",         index=False)
all_unrel.to_csv(OUTPUT_DIR / "unrelated_ctrl_all_models_raw.csv",     index=False)
all_alt.to_csv(OUTPUT_DIR / "alternate_donor_all_models_raw.csv",      index=False)
all_noise.to_csv(OUTPUT_DIR / "noise_ctrl_all_models_raw.csv",         index=False)
all_self.to_csv(OUTPUT_DIR / "self_patch_ctrl_all_models_raw.csv",     index=False)

onset_df = pd.DataFrame(onset_rows)
onset_df.to_csv(OUTPUT_DIR / "both_change_crossover_table.csv", index=False)

summary = (
    all_main.groupby(["model_name","layer_idx"])[
        ["entity_wins","relation_wins","original_retained","mixed_or_other"]]
    .mean().reset_index()
)
summary.to_csv(OUTPUT_DIR / "both_change_summary.csv", index=False)

pair_summary = (
    all_main.groupby(["model_name","family_pair","layer_idx"])[
        ["entity_wins","relation_wins","original_retained","mixed_or_other"]]
    .mean().reset_index()
)
pair_summary.to_csv(OUTPUT_DIR / "both_change_by_pair_summary.csv", index=False)

# ── FINAL PRINT ──────────────────────────────────────────────
print("\n\n" + "="*80)
print("CROSSOVER TABLE — relation->entity dominance transition")
print("="*80)
print(onset_df.to_string(index=False))

print("\n" + "="*80)
print("PEAK entity_wins and relation_wins per model")
print("="*80)
for model_name in MODELS:
    sub = all_main[all_main["model_name"] == model_name]
    if sub.empty: continue
    agg = sub.groupby("layer_idx")[["entity_wins","relation_wins"]].mean()
    print(f"\n  {model_name.split('/')[-1]}")
    print(f"    peak entity_wins  : "
          f"{agg['entity_wins'].max():.3f} "
          f"at layer {agg['entity_wins'].idxmax()}")
    print(f"    peak relation_wins: "
          f"{agg['relation_wins'].max():.3f} "
          f"at layer {agg['relation_wins'].idxmax()}")

print("\n" + "="*80)
print("CONTROLS SUMMARY")
print("  unrelated_donor  = overwrite diagnostic / control for structured transfer")
print("  noise_patch      = true negative control (structured wins should be near zero)")
print("  self_patch       = sanity check (original_retained should be ~100%)")
print("  alternate_donor  = robustness check (NOT a negative control)")
print("="*80)
for model_name in MODELS:
    real  = all_main[all_main["model_name"] == model_name]
    if real.empty: continue
    re_ = real.groupby("layer_idx")[["entity_wins","relation_wins"]].mean()
    print(f"\n  {model_name.split('/')[-1]}")
    print(f"    real         entity_max={re_['entity_wins'].max():.3f}  "
          f"relation_max={re_['relation_wins'].max():.3f}")

    if not all_unrel.empty:
        uc = (all_unrel[all_unrel["model_name"] == model_name]
              .groupby("layer_idx")[["entity_wins","relation_wins"]].mean())
        if not uc.empty:
            print(f"    unrelated    entity_max={uc['entity_wins'].max():.3f}  "
                  f"relation_max={uc['relation_wins'].max():.3f}  "
                  f"[overwrite diagnostic — relation-wins should remain near zero]")

    if not all_noise.empty:
        nc = (all_noise[all_noise["model_name"] == model_name]
              .groupby("layer_idx")[["entity_wins","relation_wins"]].mean())
        if not nc.empty:
            print(f"    noise        entity_max={nc['entity_wins'].max():.3f}  "
                  f"relation_max={nc['relation_wins'].max():.3f}  "
                  f"[negative ctrl — should be ~0]")

    if not all_self.empty:
        sc = (all_self[all_self["model_name"] == model_name]
              .groupby("layer_idx")["original_retained"].mean())
        if not sc.empty:
            print(f"    self_patch   original_retained_mean={sc.mean():.3f}  "
                  f"[sanity — should be ~1.0]")

    if not all_alt.empty:
        ac = (all_alt[all_alt["model_name"] == model_name]
              .groupby("layer_idx")[["entity_wins","relation_wins"]].mean())
        if not ac.empty:
            print(f"    alt_donor    entity_max={ac['entity_wins'].max():.3f}  "
                  f"relation_max={ac['relation_wins'].max():.3f}  "
                  f"[robustness — crossover pattern should replicate]")

print(f"\n[done] All results saved to {OUTPUT_DIR}/")