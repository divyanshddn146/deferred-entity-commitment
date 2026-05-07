"""
run_exp1_relation_entity_transfer.py

Relation vs Entity Transfer with Control Diagnostics

This script runs final-token activation patching experiments for the
Deferred Entity Commitment project.

It compares:
1. Entity transfer: same relation, different argument.
2. Relation transfer: same argument, different relation.
3. Unrelated-random diagnostics.
4. Wrong-entity relation controls, separating relation-only transfer
   from donor-answer copying.

The script saves raw CSVs, summary CSVs, and diagnostic plots.

We use "entity" broadly to mean the prompt argument:
a country, verb, noun, adjective, or element depending on the task.

Onset thresholds are used only as descriptive layer markers.
Transfer curves are the primary evidence; onset summaries are secondary.
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

LOCAL_FILES_ONLY = False
USE_4BIT = False
USE_FP16_ON_CUDA = True
MAX_NEW_TOKENS = 12
LAYERS_TO_TEST_STEP = 2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "exp1_relation_entity_transfer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_SAMPLES = 300
BOOTSTRAP_ALPHA = 0.05

# Onset thresholds — used only as descriptive markers, not primary evidence.
# Transfer curves are reported directly in the paper.
REL_ONSET_THRESHOLD = 0.40
ENTITY_ONSET_THRESHOLD = 0.40

# ============================================================
# DATASETS
#
# We use "entity" broadly to mean the prompt argument:
# a country, verb, noun, adjective, or element depending on the task.
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

PLURAL_BANK = [
    ("cat",      "cats"),
    ("dog",      "dogs"),
    ("car",      "cars"),
    ("tree",     "trees"),
    ("river",    "rivers"),
    ("mountain", "mountains"),
    ("teacher",  "teachers"),
    ("table",    "tables"),
    ("chair",    "chairs"),
    ("window",   "windows"),
    ("apple",    "apples"),
    ("flower",   "flowers"),
    ("bird",     "birds"),
    ("cloud",    "clouds"),
    ("road",     "roads"),
    ("city",     "cities"),
    ("baby",     "babies"),
    ("story",    "stories"),
    ("bus",      "buses"),
    ("box",      "boxes"),
    ("match",    "matches"),
    ("dish",     "dishes"),
    ("watch",    "watches"),
    ("bottle",   "bottles"),
    ("camera",   "cameras"),
    ("pencil",   "pencils"),
    ("paper",    "papers"),
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

ELEMENT_BANK = [
    ("gold",      "au"),
    ("silver",    "ag"),
    ("iron",      "fe"),
    ("copper",    "cu"),
    ("calcium",   "ca"),
    ("carbon",    "c"),
    ("hydrogen",  "h"),
    ("potassium", "k"),
    ("chlorine",  "cl"),
    ("fluorine",  "f"),
    ("platinum",  "pt"),
    ("uranium",   "u"),
]

# ============================================================
# PROMPTS
# ============================================================
def make_capital_prompt(x):            return f"The capital of {x} is"
def make_language_prompt(x):           return f"The official language of {x} is"
def make_opposite_prompt(x):           return f"The opposite of {x} is"
def make_comparative_prompt(x):        return f"The comparative form of {x} is"
def make_symbol_prompt(x):             return f"The chemical symbol for {x} is"
def make_plural_prompt(x):             return f"The plural of {x} is"
def make_past_tense_prompt(x):         return f"The past tense of {x} is"
def make_present_participle_prompt(x): return f"The present participle of {x} is"

# ============================================================
# HELPERS
# ============================================================
def make_even(items):
    return items if len(items) % 2 == 0 else items[:-1]

def pair_up(items):
    items = make_even(items)
    return [(items[i], items[i + 1]) for i in range(0, len(items), 2)]

def normalize(x):
    return " ".join(str(x).lower().strip().replace("\n", " ").split())

def extract_continuation(prompt, full_text):
    return (full_text[len(prompt):].strip()
            if full_text.startswith(prompt) else full_text.strip())

def extract_first_phrase(prompt, full_text, max_words=3):
    cont = extract_continuation(prompt, full_text)
    piece = re.split(r"[,\.\n;:!?]", cont, maxsplit=1)[0].strip()
    return " ".join(piece.split()[:max_words]).lower().strip()

def extract_first_word(prompt, full_text):
    return extract_first_phrase(prompt, full_text, max_words=1)

def strip_prompt_from_ids(full_ids, prompt_ids):
    return full_ids[len(prompt_ids):]

def common_prefix_len(l1, l2):
    n = min(len(l1), len(l2))
    for i in range(n):
        if int(l1[i]) != int(l2[i]):
            return i
    return n

def classify_overlap(cp_a, cp_b, ov_a, ov_b):
    if cp_b > cp_a and ov_b > ov_a:
        return "DONOR-like"
    if cp_a > cp_b and ov_a > ov_b:
        return "ORIGINAL-like"
    return "MIXED"

def extract_slot(family, prompt, full_text):
    if family == "capital":
        return extract_first_phrase(prompt, full_text, max_words=2)
    return extract_first_word(prompt, full_text) or "<empty>"

def canonical_match(family, pred, expected):
    pred = normalize(pred)
    exp  = normalize(expected)
    return pred.split()[0] == exp.split()[0] if pred and exp else False

# ============================================================
# RECORD BUILDERS
# ============================================================
def build_full_donor_records():
    records = []

    for (a, a_cap, _), (b, b_cap, _) in pair_up(COUNTRY_BANK):
        records.append({
            "family": "capital", "record_type": "full_donor",
            "prompt_a": make_capital_prompt(a),
            "prompt_b": make_capital_prompt(b),
            "answer_a": a_cap, "answer_b": b_cap,
            "entity_a": a, "entity_b": b,
        })

    for (a, _, a_lang), (b, _, b_lang) in pair_up(COUNTRY_BANK):
        records.append({
            "family": "language", "record_type": "full_donor",
            "prompt_a": make_language_prompt(a),
            "prompt_b": make_language_prompt(b),
            "answer_a": a_lang, "answer_b": b_lang,
            "entity_a": a, "entity_b": b,
        })

    for (a_v, a_past, _), (b_v, b_past, _) in pair_up(TENSE_BANK):
        records.append({
            "family": "past_tense", "record_type": "full_donor",
            "prompt_a": make_past_tense_prompt(a_v),
            "prompt_b": make_past_tense_prompt(b_v),
            "answer_a": a_past, "answer_b": b_past,
            "entity_a": a_v, "entity_b": b_v,
        })

    for (a_v, _, a_part), (b_v, _, b_part) in pair_up(TENSE_BANK):
        records.append({
            "family": "present_participle", "record_type": "full_donor",
            "prompt_a": make_present_participle_prompt(a_v),
            "prompt_b": make_present_participle_prompt(b_v),
            "answer_a": a_part, "answer_b": b_part,
            "entity_a": a_v, "entity_b": b_v,
        })

    for (a_w, a_pl), (b_w, b_pl) in pair_up(PLURAL_BANK):
        records.append({
            "family": "plural", "record_type": "full_donor",
            "prompt_a": make_plural_prompt(a_w),
            "prompt_b": make_plural_prompt(b_w),
            "answer_a": a_pl, "answer_b": b_pl,
            "entity_a": a_w, "entity_b": b_w,
        })

    for (a_adj, a_opp, _), (b_adj, b_opp, _) in pair_up(ADJECTIVE_BANK):
        records.append({
            "family": "opposite", "record_type": "full_donor",
            "prompt_a": make_opposite_prompt(a_adj),
            "prompt_b": make_opposite_prompt(b_adj),
            "answer_a": a_opp, "answer_b": b_opp,
            "entity_a": a_adj, "entity_b": b_adj,
        })

    for (a_adj, _, a_comp), (b_adj, _, b_comp) in pair_up(ADJECTIVE_BANK):
        records.append({
            "family": "comparative", "record_type": "full_donor",
            "prompt_a": make_comparative_prompt(a_adj),
            "prompt_b": make_comparative_prompt(b_adj),
            "answer_a": a_comp, "answer_b": b_comp,
            "entity_a": a_adj, "entity_b": b_adj,
        })

    for (a_el, a_sym), (b_el, b_sym) in pair_up(ELEMENT_BANK):
        records.append({
            "family": "symbol", "record_type": "full_donor",
            "prompt_a": make_symbol_prompt(a_el),
            "prompt_b": make_symbol_prompt(b_el),
            "answer_a": a_sym, "answer_b": b_sym,
            "entity_a": a_el, "entity_b": b_el,
        })

    for i, r in enumerate(records, 1):
        r["record_id"] = i
    return records


def build_relation_donor_records():
    records = []

    for country, cap_ans, lang_ans in COUNTRY_BANK:
        records.append({
            "relation_pair_id": "cap_recv__lang_donor",
            "recipient_family": "capital", "donor_family": "language",
            "record_type": "relation_donor",
            "prompt_a": make_capital_prompt(country),
            "prompt_b": make_language_prompt(country),
            "answer_a": cap_ans, "answer_b": lang_ans,
            "entity_a": country, "entity_b": country,
        })

    for country, cap_ans, lang_ans in COUNTRY_BANK:
        records.append({
            "relation_pair_id": "lang_recv__cap_donor",
            "recipient_family": "language", "donor_family": "capital",
            "record_type": "relation_donor",
            "prompt_a": make_language_prompt(country),
            "prompt_b": make_capital_prompt(country),
            "answer_a": lang_ans, "answer_b": cap_ans,
            "entity_a": country, "entity_b": country,
        })

    for verb, past_ans, part_ans in TENSE_BANK:
        records.append({
            "relation_pair_id": "past_recv__part_donor",
            "recipient_family": "past_tense",
            "donor_family": "present_participle",
            "record_type": "relation_donor",
            "prompt_a": make_past_tense_prompt(verb),
            "prompt_b": make_present_participle_prompt(verb),
            "answer_a": past_ans, "answer_b": part_ans,
            "entity_a": verb, "entity_b": verb,
        })

    for verb, past_ans, part_ans in TENSE_BANK:
        records.append({
            "relation_pair_id": "part_recv__past_donor",
            "recipient_family": "present_participle",
            "donor_family": "past_tense",
            "record_type": "relation_donor",
            "prompt_a": make_present_participle_prompt(verb),
            "prompt_b": make_past_tense_prompt(verb),
            "answer_a": part_ans, "answer_b": past_ans,
            "entity_a": verb, "entity_b": verb,
        })

    for adj, opp_ans, comp_ans in ADJECTIVE_BANK:
        records.append({
            "relation_pair_id": "opp_recv__comp_donor",
            "recipient_family": "opposite", "donor_family": "comparative",
            "record_type": "relation_donor",
            "prompt_a": make_opposite_prompt(adj),
            "prompt_b": make_comparative_prompt(adj),
            "answer_a": opp_ans, "answer_b": comp_ans,
            "entity_a": adj, "entity_b": adj,
        })

    for adj, opp_ans, comp_ans in ADJECTIVE_BANK:
        records.append({
            "relation_pair_id": "comp_recv__opp_donor",
            "recipient_family": "comparative", "donor_family": "opposite",
            "record_type": "relation_donor",
            "prompt_a": make_comparative_prompt(adj),
            "prompt_b": make_opposite_prompt(adj),
            "answer_a": comp_ans, "answer_b": opp_ans,
            "entity_a": adj, "entity_b": adj,
        })

    for i, r in enumerate(records, 1):
        r["record_id"] = i
    return records


def build_unrelated_random_controls(full_records, relation_records):
    donor_pool = []
    for r in full_records:
        donor_pool.append({
            "family": r["family"],
            "prompt_b": r["prompt_b"],
            "answer_b": r["answer_b"],
            "entity_b": r["entity_b"],
        })
    for r in relation_records:
        donor_pool.append({
            "family": r["donor_family"],
            "prompt_b": r["prompt_b"],
            "answer_b": r["answer_b"],
            "entity_b": r["entity_b"],
        })

    rng = random.Random(SEED + 123)
    controls = []

    def sample_unrelated(recipient_family, entity_a):
        candidates = [
            d for d in donor_pool
            if d["family"] != recipient_family and d["entity_b"] != entity_a
        ]
        return rng.choice(candidates)

    for r in full_records:
        d = sample_unrelated(r["family"], r["entity_a"])
        controls.append({
            "control_type": "unrelated_random", "base_type": "full_donor",
            "family": r["family"],
            "prompt_a": r["prompt_a"], "answer_a": r["answer_a"],
            "entity_a": r["entity_a"],
            "prompt_b": d["prompt_b"], "answer_b": d["answer_b"],
            "entity_b": d["entity_b"], "donor_family": d["family"],
        })

    for r in relation_records:
        d = sample_unrelated(r["recipient_family"], r["entity_a"])
        controls.append({
            "control_type": "unrelated_random", "base_type": "relation_donor",
            "recipient_family": r["recipient_family"],
            "donor_family": d["family"],
            "prompt_a": r["prompt_a"], "answer_a": r["answer_a"],
            "entity_a": r["entity_a"],
            "prompt_b": d["prompt_b"], "answer_b": d["answer_b"],
            "entity_b": d["entity_b"],
        })

    for i, r in enumerate(controls, 1):
        r["record_id"] = i
    return controls


def build_relation_wrong_entity_controls():
    """
    Wrong-entity relation controls.

    Same donor family, different donor entity.
    Distinguishes two outcomes:
      relation_only_answer: donor relation applied to recipient entity
                            (genuine abstract relation transfer)
      answer_b:             donor answer for the wrong entity
                            (donor-answer copying, not relation transfer)
    This separation prevents donor-answer copying from inflating
    relation transfer estimates.
    """
    rng = random.Random(SEED + 456)
    controls = []

    country_cap_lookup  = [(c, cap)  for c, cap, _   in COUNTRY_BANK]
    country_lang_lookup = [(c, lang) for c, _,   lang in COUNTRY_BANK]
    verb_past_lookup    = [(v, past) for v, past, _   in TENSE_BANK]
    verb_part_lookup    = [(v, part) for v, _,   part in TENSE_BANK]
    adj_opp_lookup      = [(a, opp)  for a, opp, _    in ADJECTIVE_BANK]
    adj_comp_lookup     = [(a, comp) for a, _,   comp in ADJECTIVE_BANK]

    def pick_wrong(lookup, current_entity):
        candidates = [(e, ans) for e, ans in lookup if e != current_entity]
        return rng.choice(candidates)

    for country, cap_ans, lang_ans in COUNTRY_BANK:
        e_wrong, ans_wrong = pick_wrong(country_lang_lookup, country)
        controls.append({
            "control_type": "relation_wrong_entity",
            "recipient_family": "capital", "donor_family": "language",
            "prompt_a": make_capital_prompt(country), "answer_a": cap_ans,
            "entity_a": country,
            "prompt_b": make_language_prompt(e_wrong), "answer_b": ans_wrong,
            "entity_b": e_wrong, "relation_only_answer": lang_ans,
        })
        e_wrong, ans_wrong = pick_wrong(country_cap_lookup, country)
        controls.append({
            "control_type": "relation_wrong_entity",
            "recipient_family": "language", "donor_family": "capital",
            "prompt_a": make_language_prompt(country), "answer_a": lang_ans,
            "entity_a": country,
            "prompt_b": make_capital_prompt(e_wrong), "answer_b": ans_wrong,
            "entity_b": e_wrong, "relation_only_answer": cap_ans,
        })

    for verb, past_ans, part_ans in TENSE_BANK:
        e_wrong, ans_wrong = pick_wrong(verb_part_lookup, verb)
        controls.append({
            "control_type": "relation_wrong_entity",
            "recipient_family": "past_tense",
            "donor_family": "present_participle",
            "prompt_a": make_past_tense_prompt(verb), "answer_a": past_ans,
            "entity_a": verb,
            "prompt_b": make_present_participle_prompt(e_wrong),
            "answer_b": ans_wrong, "entity_b": e_wrong,
            "relation_only_answer": part_ans,
        })
        e_wrong, ans_wrong = pick_wrong(verb_past_lookup, verb)
        controls.append({
            "control_type": "relation_wrong_entity",
            "recipient_family": "present_participle",
            "donor_family": "past_tense",
            "prompt_a": make_present_participle_prompt(verb),
            "answer_a": part_ans, "entity_a": verb,
            "prompt_b": make_past_tense_prompt(e_wrong),
            "answer_b": ans_wrong, "entity_b": e_wrong,
            "relation_only_answer": past_ans,
        })

    for adj, opp_ans, comp_ans in ADJECTIVE_BANK:
        e_wrong, ans_wrong = pick_wrong(adj_comp_lookup, adj)
        controls.append({
            "control_type": "relation_wrong_entity",
            "recipient_family": "opposite", "donor_family": "comparative",
            "prompt_a": make_opposite_prompt(adj), "answer_a": opp_ans,
            "entity_a": adj,
            "prompt_b": make_comparative_prompt(e_wrong),
            "answer_b": ans_wrong, "entity_b": e_wrong,
            "relation_only_answer": comp_ans,
        })
        e_wrong, ans_wrong = pick_wrong(adj_opp_lookup, adj)
        controls.append({
            "control_type": "relation_wrong_entity",
            "recipient_family": "comparative", "donor_family": "opposite",
            "prompt_a": make_comparative_prompt(adj), "answer_a": comp_ans,
            "entity_a": adj,
            "prompt_b": make_opposite_prompt(e_wrong),
            "answer_b": ans_wrong, "entity_b": e_wrong,
            "relation_only_answer": opp_ans,
        })

    for i, r in enumerate(controls, 1):
        r["record_id"] = i
    return controls


# ============================================================
# BUILD RECORDS
# ============================================================
FULL_DONOR_RECORDS          = build_full_donor_records()
RELATION_DONOR_RECORDS      = build_relation_donor_records()
UNRELATED_RANDOM_CONTROLS   = build_unrelated_random_controls(
    FULL_DONOR_RECORDS, RELATION_DONOR_RECORDS)
RELATION_WRONG_ENTITY_CONTROLS = build_relation_wrong_entity_controls()

print(f"[records] full_donor                = {len(FULL_DONOR_RECORDS)}")
print(f"[records] relation_donor            = {len(RELATION_DONOR_RECORDS)}")
print(f"[records] unrelated_random_controls = {len(UNRELATED_RANDOM_CONTROLS)}")
print(f"[records] wrong_entity_controls     = {len(RELATION_WRONG_ENTITY_CONTROLS)}")

print("[relation pairs]")
for pid, cnt in Counter(
        r["relation_pair_id"] for r in RELATION_DONOR_RECORDS).items():
    print(f"  {pid}: {cnt}")

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
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
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
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.pad_token_id = \
                self.tokenizer.pad_token_id

        self.layers      = self._get_layers()
        self.num_layers  = len(self.layers)
        self.layers_to_test = list(
            range(0, self.num_layers, LAYERS_TO_TEST_STEP))
        if (self.num_layers - 1) not in self.layers_to_test:
            self.layers_to_test.append(self.num_layers - 1)

        print(f"[info] num_layers={self.num_layers}")
        print(f"[info] layers_to_test={self.layers_to_test}")

    def _get_layers(self):
        m = self.model
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "layers"):
            return m.layers
        raise ValueError(f"Unknown architecture: {self.model_name}")

    def tokenize(self, prompt):
        return {k: v.to(self.model.device)
                for k, v in self.tokenizer(
                    prompt, return_tensors="pt").items()}

    def get_forward(self, prompt):
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            out = self.model(**inputs, return_dict=True,
                             use_cache=False, output_hidden_states=True)
        return inputs, out

    def generate_text_and_ids(self, prompt):
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0], skip_special_tokens=True), out[0]

    def generate_with_patch(self, prompt, patch_vector, layer_idx, pos_idx):
        inputs = self.tokenize(prompt)
        done = {"v": False}

        def hook_fn(module, module_input, module_output):
            hidden = (module_output[0]
                      if isinstance(module_output, tuple)
                      else module_output)
            if not done["v"] and hidden.shape[1] > pos_idx:
                hidden = hidden.clone()
                hidden[:, pos_idx, :] = patch_vector.to(hidden.dtype)
                done["v"] = True
            return ((hidden,) + module_output[1:]
                    if isinstance(module_output, tuple) else hidden)

        handle = self.layers[layer_idx].register_forward_hook(hook_fn)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id)
        handle.remove()
        return (self.tokenizer.decode(out[0], skip_special_tokens=True),
                out[0], done["v"])


# ============================================================
# CACHING
# ============================================================
def build_cache(runner, records, desc):
    cached = []
    for rec in tqdm(records, desc=desc):
        inputs_a, out_a = runner.get_forward(rec["prompt_a"])
        inputs_b, out_b = runner.get_forward(rec["prompt_b"])

        pos_a = inputs_a["input_ids"].shape[1] - 1
        pos_b = inputs_b["input_ids"].shape[1] - 1

        normal_a_text, normal_a_ids = runner.generate_text_and_ids(rec["prompt_a"])
        normal_b_text, normal_b_ids = runner.generate_text_and_ids(rec["prompt_b"])

        prompt_a_ids = inputs_a["input_ids"][0].detach().clone()
        prompt_b_ids = inputs_b["input_ids"][0].detach().clone()

        x = dict(rec)
        x.update({
            "out_a": out_a, "out_b": out_b,
            "pos_a": pos_a, "pos_b": pos_b,
            "prompt_a_ids": prompt_a_ids, "prompt_b_ids": prompt_b_ids,
            "normal_a_text": normal_a_text, "normal_b_text": normal_b_text,
            "normal_a_ids": normal_a_ids, "normal_b_ids": normal_b_ids,
            "cont_a": strip_prompt_from_ids(normal_a_ids, prompt_a_ids),
            "cont_b": strip_prompt_from_ids(normal_b_ids, prompt_b_ids),
        })
        cached.append(x)
    return cached


# ============================================================
# PATCH EVALUATION
# ============================================================
def eval_patch(runner, rec, layer_idx):
    if layer_idx >= runner.num_layers - 1:
        return None

    donor_hidden = (rec["out_b"].hidden_states[layer_idx + 1]
                    [:, rec["pos_b"], :].clone())

    patched_text, patched_ids, _ = runner.generate_with_patch(
        rec["prompt_a"], donor_hidden, layer_idx, rec["pos_a"])

    cont_p = strip_prompt_from_ids(patched_ids, rec["prompt_a_ids"])
    cont_a = rec["cont_a"]
    cont_b = rec["cont_b"]

    cp_a = common_prefix_len(cont_p, cont_a)
    cp_b = common_prefix_len(cont_p, cont_b)
    n    = min(len(cont_p), len(cont_a), len(cont_b))
    ov_a = sum(int(cont_p[i]) == int(cont_a[i]) for i in range(n))
    ov_b = sum(int(cont_p[i]) == int(cont_b[i]) for i in range(n))

    return {
        "patched_text":  patched_text,
        "overlap_class": classify_overlap(cp_a, cp_b, ov_a, ov_b),
        "switch_strength": ov_b - ov_a,
    }


# ============================================================
# LABELS
# ============================================================
def label_entity_transfer(rec, patched_text, overlap_class):
    family = rec["family"]
    p_slot = extract_slot(family, rec["prompt_a"], patched_text)
    slot_ok = (not canonical_match(family, p_slot, rec["answer_a"])
               and canonical_match(family, p_slot, rec["answer_b"]))
    if slot_ok:
        return "entity_transfer"
    if overlap_class in {"DONOR-like", "MIXED"}:
        return "overwrite"
    return "failure"


def label_relation_transfer(rec, patched_text, overlap_class):
    rf     = rec["recipient_family"]
    p_slot = extract_slot(rf, rec["prompt_a"], patched_text)
    slot_ok = (not canonical_match(rf, p_slot, rec["answer_a"])
               and canonical_match(rf, p_slot, rec["answer_b"]))
    if slot_ok:
        return "relation_transfer"
    if overlap_class in {"DONOR-like", "MIXED"}:
        return "overwrite"
    return "failure"


def label_relation_wrong_entity_control(rec, patched_text, overlap_class):
    rf     = rec["recipient_family"]
    p_slot = extract_slot(rf, rec["prompt_a"], patched_text)
    if canonical_match(rf, p_slot, rec["relation_only_answer"]):
        return "relation_only_transfer"
    if canonical_match(rf, p_slot, rec["answer_b"]):
        return "donor_answer_copy"
    if overlap_class in {"DONOR-like", "MIXED"}:
        return "overwrite"
    return "failure"


# ============================================================
# PASSES
# ============================================================
def run_entity_transfer_pass(runner, cached_full):
    rows = []
    for rec_idx, rec in enumerate(
            tqdm(cached_full, desc="Entity transfer pass"), 1):
        for layer_idx in runner.layers_to_test:
            out = eval_patch(runner, rec, layer_idx)
            if out is None:
                continue
            label = label_entity_transfer(
                rec, out["patched_text"], out["overlap_class"])
            rows.append({
                "model_name": runner.model_name,
                "record_idx": rec_idx,
                "family": rec["family"],
                "layer_idx": layer_idx,
                "final_label": label,
                "entity_transfer": int(label == "entity_transfer"),
                "overwrite": int(label == "overwrite"),
                "failure": int(label == "failure"),
                "switch_strength": out["switch_strength"],
            })
    return pd.DataFrame(rows)


def run_relation_transfer_pass(runner, cached_rel):
    rows = []
    for rec_idx, rec in enumerate(
            tqdm(cached_rel, desc="Relation transfer pass"), 1):
        for layer_idx in runner.layers_to_test:
            out = eval_patch(runner, rec, layer_idx)
            if out is None:
                continue
            label = label_relation_transfer(
                rec, out["patched_text"], out["overlap_class"])
            rows.append({
                "model_name": runner.model_name,
                "record_idx": rec_idx,
                "relation_pair_id": rec.get("relation_pair_id", "unknown"),
                "recipient_family": rec["recipient_family"],
                "donor_family": rec["donor_family"],
                "layer_idx": layer_idx,
                "final_label": label,
                "relation_transfer": int(label == "relation_transfer"),
                "overwrite": int(label == "overwrite"),
                "failure": int(label == "failure"),
                "switch_strength": out["switch_strength"],
            })
    return pd.DataFrame(rows)


def run_relation_wrong_entity_control_pass(runner, cached_wrongent):
    rows = []
    for rec_idx, rec in enumerate(
            tqdm(cached_wrongent,
                 desc="Wrong-entity relation control pass"), 1):
        for layer_idx in runner.layers_to_test:
            out = eval_patch(runner, rec, layer_idx)
            if out is None:
                continue
            label = label_relation_wrong_entity_control(
                rec, out["patched_text"], out["overlap_class"])
            rows.append({
                "model_name": runner.model_name,
                "record_idx": rec_idx,
                "recipient_family": rec["recipient_family"],
                "donor_family": rec["donor_family"],
                "layer_idx": layer_idx,
                "final_label": label,
                "relation_only_transfer": int(label == "relation_only_transfer"),
                "donor_answer_copy": int(label == "donor_answer_copy"),
                "overwrite": int(label == "overwrite"),
                "failure": int(label == "failure"),
                "switch_strength": out["switch_strength"],
            })
    return pd.DataFrame(rows)


# ============================================================
# SUMMARIES
# ============================================================
def summarize_entity_transfer(df):
    return (df.groupby(
        ["model_name", "family", "layer_idx"], as_index=False)
        .agg(entity_score=("entity_transfer", "mean"),
             overwrite_pct=("overwrite", "mean"),
             failure_pct=("failure", "mean"),
             n=("record_idx", "count")))


def summarize_relation_transfer(df):
    return (df.groupby(
        ["model_name", "layer_idx"], as_index=False)
        .agg(relation_score=("relation_transfer", "mean"),
             overwrite_pct=("overwrite", "mean"),
             failure_pct=("failure", "mean"),
             n=("record_idx", "count")))


def summarize_relation_by_pair(df):
    return (df.groupby(
        ["model_name", "relation_pair_id", "layer_idx"], as_index=False)
        .agg(relation_transfer_pct=("relation_transfer", "mean"),
             overwrite_pct=("overwrite", "mean"),
             failure_pct=("failure", "mean"),
             n=("record_idx", "count")))


def summarize_control_entity(df):
    if df.empty:
        return pd.DataFrame()
    return (df.groupby(
        ["model_name", "layer_idx"], as_index=False)
        .agg(entity_transfer_pct=("entity_transfer", "mean"),
             overwrite_pct=("overwrite", "mean"),
             failure_pct=("failure", "mean"),
             n=("record_idx", "count")))


def summarize_control_relation(df):
    if df.empty:
        return pd.DataFrame()
    return (df.groupby(
        ["model_name", "layer_idx"], as_index=False)
        .agg(relation_transfer_pct=("relation_transfer", "mean"),
             overwrite_pct=("overwrite", "mean"),
             failure_pct=("failure", "mean"),
             n=("record_idx", "count")))


def summarize_wrong_entity_relation_control(df):
    if df.empty:
        return pd.DataFrame()
    return (df.groupby(
        ["model_name", "layer_idx"], as_index=False)
        .agg(relation_only_transfer_pct=("relation_only_transfer", "mean"),
             donor_answer_copy_pct=("donor_answer_copy", "mean"),
             overwrite_pct=("overwrite", "mean"),
             failure_pct=("failure", "mean"),
             n=("record_idx", "count")))


# ============================================================
# BOOTSTRAP + ONSETS
# ============================================================
def bootstrap_ci(entity_summary, rel_by_pair, model_name,
                 n_boot=BOOTSTRAP_SAMPLES):
    ent_m = entity_summary[
        entity_summary["model_name"] == model_name].copy()
    rel_m = rel_by_pair[
        rel_by_pair["model_name"] == model_name].copy()

    ent_families = sorted(ent_m["family"].unique())
    rel_pairs    = sorted(rel_m["relation_pair_id"].unique())
    layers       = sorted(set(ent_m["layer_idx"]) & set(rel_m["layer_idx"]))

    ent_lut = {(f, l): v for f, l, v in
               ent_m[["family", "layer_idx", "entity_score"]]
               .itertuples(index=False)}
    rel_lut = {(p, l): v for p, l, v in
               rel_m[["relation_pair_id", "layer_idx",
                       "relation_transfer_pct"]]
               .itertuples(index=False)}

    rng       = np.random.default_rng(SEED + 999)
    boot_rows = []
    for _ in range(n_boot):
        ent_sample = rng.choice(ent_families,
                                size=len(ent_families), replace=True)
        rel_sample = rng.choice(rel_pairs,
                                size=len(rel_pairs),    replace=True)
        for l in layers:
            av = [ent_lut[(f, l)] for f in ent_sample if (f, l) in ent_lut]
            rv = [rel_lut[(p, l)] for p in rel_sample if (p, l) in rel_lut]
            boot_rows.append({
                "layer_idx":      l,
                "entity_score":   float(np.mean(av)) if av else 0.0,
                "relation_score": float(np.mean(rv)) if rv else 0.0,
            })

    boot_df = pd.DataFrame(boot_rows)
    lo = 100 * (BOOTSTRAP_ALPHA / 2)
    hi = 100 * (1 - BOOTSTRAP_ALPHA / 2)
    ci_rows = []
    for l, sdf in boot_df.groupby("layer_idx"):
        ci_rows.append({
            "model_name": model_name, "layer_idx": l,
            "ent_low":  np.percentile(sdf["entity_score"],   lo),
            "ent_high": np.percentile(sdf["entity_score"],   hi),
            "rel_low":  np.percentile(sdf["relation_score"], lo),
            "rel_high": np.percentile(sdf["relation_score"], hi),
        })
    return (pd.DataFrame(ci_rows)
            .sort_values("layer_idx")
            .reset_index(drop=True))


def first_stable_onset(df, score_col, threshold, stable_steps=2):
    df   = df.sort_values("layer_idx").reset_index(drop=True)
    vals = df[score_col].tolist()
    lyrs = df["layer_idx"].tolist()
    for i in range(len(vals) - stable_steps + 1):
        if all(v >= threshold for v in vals[i:i + stable_steps]):
            return lyrs[i]
    for i, v in enumerate(vals):
        if v >= threshold:
            return lyrs[i]
    return None


# ============================================================
# PLOTS
# ============================================================
def plot_main_curves(entity_summary, rel_summary, ci_df,
                     model_name, rel_onset, entity_onset):
    ent_m = (entity_summary[entity_summary["model_name"] == model_name]
             .groupby("layer_idx", as_index=False)
             .agg(entity_score=("entity_score", "mean"))
             .sort_values("layer_idx"))
    rel_m = (rel_summary[rel_summary["model_name"] == model_name]
             .sort_values("layer_idx"))

    fig, ax = plt.subplots(figsize=(9, 5))
    if ci_df is not None and not ci_df.empty:
        ax.fill_between(ci_df["layer_idx"],
                        ci_df["rel_low"], ci_df["rel_high"],
                        alpha=0.15, color="tab:orange")
        ax.fill_between(ci_df["layer_idx"],
                        ci_df["ent_low"], ci_df["ent_high"],
                        alpha=0.15, color="tab:green")

    ax.plot(rel_m["layer_idx"], rel_m["relation_score"],
            marker="o", color="tab:orange", linewidth=2,
            label="Relation transfer")
    ax.plot(ent_m["layer_idx"], ent_m["entity_score"],
            marker="o", color="tab:green", linewidth=2,
            label="Entity transfer")

    if rel_onset is not None:
        ax.axvline(rel_onset, linestyle=":", linewidth=1.8,
                   color="tab:orange", label=f"Relation onset L{rel_onset}")
    if entity_onset is not None:
        ax.axvline(entity_onset, linestyle=":", linewidth=1.8,
                   color="tab:green", label=f"Entity onset L{entity_onset}")

    short = model_name.split("/")[-1]
    ax.set_title(f"Relation vs Entity Transfer — {short}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Transfer rate")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    fname = OUTPUT_DIR / f"{model_name.replace('/', '_')}_main_curves.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[saved] {fname.name}")


def plot_relation_by_pair(rel_by_pair_df, model_name):
    sub = rel_by_pair_df[rel_by_pair_df["model_name"] == model_name]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for pair_id, sdf in sub.groupby("relation_pair_id"):
        sdf = sdf.sort_values("layer_idx")
        ax.plot(sdf["layer_idx"], sdf["relation_transfer_pct"],
                marker="o", label=pair_id)
    short = model_name.split("/")[-1]
    ax.set_title(f"Relation Transfer by Pair — {short}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Relation transfer rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fname = OUTPUT_DIR / f"{model_name.replace('/', '_')}_relation_by_pair.png"
    plt.savefig(fname, dpi=220)
    plt.close()
    print(f"[saved] {fname.name}")


def plot_controls(entity_summary, rel_summary, ctrl_ent_summary,
                  ctrl_rel_summary, wrongent_summary, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ent_real = (entity_summary[entity_summary["model_name"] == model_name]
                .groupby("layer_idx", as_index=False)
                .agg(score=("entity_score", "mean"))
                .sort_values("layer_idx"))
    axes[0].plot(ent_real["layer_idx"], ent_real["score"],
                 marker="o", color="tab:green",
                 label="Entity transfer real")

    if not ctrl_ent_summary.empty:
        ctrl_ent = (ctrl_ent_summary[
            ctrl_ent_summary["model_name"] == model_name]
            .sort_values("layer_idx"))
        if not ctrl_ent.empty:
            axes[0].plot(ctrl_ent["layer_idx"],
                         ctrl_ent["entity_transfer_pct"],
                         marker="^", linestyle="--", color="tab:gray",
                         label="Unrelated entity diagnostic")

    axes[0].set_title("Entity Transfer: Real vs Diagnostic")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Transfer rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8)

    rel_real = (rel_summary[rel_summary["model_name"] == model_name]
                .sort_values("layer_idx"))
    axes[1].plot(rel_real["layer_idx"], rel_real["relation_score"],
                 marker="o", color="tab:orange",
                 label="Relation transfer real")

    if not ctrl_rel_summary.empty:
        ctrl_rel = (ctrl_rel_summary[
            ctrl_rel_summary["model_name"] == model_name]
            .sort_values("layer_idx"))
        if not ctrl_rel.empty:
            axes[1].plot(ctrl_rel["layer_idx"],
                         ctrl_rel["relation_transfer_pct"],
                         marker="^", linestyle="--", color="tab:gray",
                         label="Unrelated relation diagnostic")

    if not wrongent_summary.empty:
        we = (wrongent_summary[wrongent_summary["model_name"] == model_name]
              .sort_values("layer_idx"))
        if not we.empty:
            axes[1].plot(we["layer_idx"],
                         we["relation_only_transfer_pct"],
                         marker="x", linestyle="--", color="tab:red",
                         label="Wrong-entity: relation-only")
            axes[1].plot(we["layer_idx"],
                         we["donor_answer_copy_pct"],
                         marker="s", linestyle="--", color="tab:purple",
                         label="Wrong-entity: donor-copy")

    axes[1].set_title("Relation Transfer: Real vs Diagnostics")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Transfer rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=8)

    short = model_name.split("/")[-1]
    plt.suptitle(f"Control Diagnostics — {short}", fontsize=11)
    plt.tight_layout()
    fname = OUTPUT_DIR / f"{model_name.replace('/', '_')}_controls.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[saved] {fname.name}")


def plot_onset_comparison(onset_df):
    if onset_df.empty:
        return
    models = onset_df["model_name"].tolist()
    x      = list(range(len(models)))
    rel_on = onset_df["relation_onset"].tolist()
    ent_on = onset_df["entity_onset"].tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([xi - 0.2 for xi in x], rel_on, width=0.35,
           color="tab:orange", label="Relation onset")
    ax.bar([xi + 0.2 for xi in x], ent_on, width=0.35,
           color="tab:green",  label="Entity onset")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in models], fontsize=9)
    ax.set_ylabel("Layer")
    ax.set_title("Relation onset precedes entity onset across models")
    ax.legend()
    plt.tight_layout()
    fname = OUTPUT_DIR / "onset_comparison.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[saved] {fname.name}")


# ============================================================
# MAIN LOOP
# ============================================================
all_ent_rows       = []
all_rel_rows       = []
all_unrel_ent_rows = []
all_unrel_rel_rows = []
all_wrongent_rows  = []

for model_name in MODELS:
    print(f"\n{'=' * 80}\nMODEL: {model_name}\n{'=' * 80}")
    runner = ModelRunner(model_name)

    full_cached     = build_cache(runner, FULL_DONOR_RECORDS,
                                  "Caching full-donor")
    rel_cached      = build_cache(runner, RELATION_DONOR_RECORDS,
                                  "Caching relation-donor")
    unrelated_cached = build_cache(runner, UNRELATED_RANDOM_CONTROLS,
                                   "Caching unrelated controls")
    wrongent_cached  = build_cache(runner, RELATION_WRONG_ENTITY_CONTROLS,
                                   "Caching wrong-entity controls")

    ent_df     = run_entity_transfer_pass(runner, full_cached)
    rel_df     = run_relation_transfer_pass(runner, rel_cached)

    unrel_full = [r for r in unrelated_cached
                  if r.get("base_type") == "full_donor"]
    unrel_rel  = [r for r in unrelated_cached
                  if r.get("base_type") == "relation_donor"]

    unrel_ent_df  = (run_entity_transfer_pass(runner, unrel_full)
                     if unrel_full else pd.DataFrame())
    unrel_rel_df  = (run_relation_transfer_pass(runner, unrel_rel)
                     if unrel_rel else pd.DataFrame())
    wrongent_df   = (run_relation_wrong_entity_control_pass(
                         runner, wrongent_cached)
                     if wrongent_cached else pd.DataFrame())

    all_ent_rows.append(ent_df)
    all_rel_rows.append(rel_df)
    all_unrel_ent_rows.append(unrel_ent_df)
    all_unrel_rel_rows.append(unrel_rel_df)
    all_wrongent_rows.append(wrongent_df)

    del runner.model, runner.tokenizer, runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# AGGREGATE
# ============================================================
def safe_concat(dfs):
    valid = [d for d in dfs if not d.empty]
    return (pd.concat(valid, ignore_index=True)
            if valid else pd.DataFrame())

ent_all         = pd.concat(all_ent_rows, ignore_index=True)
rel_all         = pd.concat(all_rel_rows, ignore_index=True)
unrel_ent_all   = safe_concat(all_unrel_ent_rows)
unrel_rel_all   = safe_concat(all_unrel_rel_rows)
wrongent_rel_all = safe_concat(all_wrongent_rows)

entity_summary  = summarize_entity_transfer(ent_all)
rel_summary     = summarize_relation_transfer(rel_all)
rel_by_pair     = summarize_relation_by_pair(rel_all)
ctrl_ent_summary = summarize_control_entity(unrel_ent_all)
ctrl_rel_summary = summarize_control_relation(unrel_rel_all)
wrongent_summary = summarize_wrong_entity_relation_control(wrongent_rel_all)


# ============================================================
# SAVE CSVs
# ============================================================
for df, name in [
    (entity_summary,    "entity_transfer_summary"),
    (rel_summary,       "relation_transfer_summary"),
    (rel_by_pair,       "relation_transfer_by_pair"),
    (ent_all,           "entity_transfer_raw"),
    (rel_all,           "relation_transfer_raw"),
    (unrel_ent_all,     "unrelated_random_entity_raw"),
    (unrel_rel_all,     "unrelated_random_relation_raw"),
    (wrongent_rel_all,  "relation_wrong_entity_raw"),
    (ctrl_ent_summary,  "unrelated_random_entity_summary"),
    (ctrl_rel_summary,  "unrelated_random_relation_summary"),
    (wrongent_summary,  "relation_wrong_entity_summary"),
]:
    if not df.empty:
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        print(f"[saved] {name}.csv")


# ============================================================
# ONSETS + PLOTS
# ============================================================
onset_rows  = []
all_ci_rows = []

for model_name in MODELS:
    ent_m = (entity_summary[entity_summary["model_name"] == model_name]
             .groupby("layer_idx", as_index=False)
             .agg(entity_score=("entity_score", "mean"))
             .sort_values("layer_idx"))
    rel_m = (rel_summary[rel_summary["model_name"] == model_name]
             .sort_values("layer_idx"))

    if ent_m.empty or rel_m.empty:
        continue

    rel_onset    = first_stable_onset(
        rel_m, "relation_score", REL_ONSET_THRESHOLD)
    entity_onset = first_stable_onset(
        ent_m, "entity_score",   ENTITY_ONSET_THRESHOLD)

    gap      = (entity_onset - rel_onset
                if rel_onset is not None and entity_onset is not None
                else None)
    ordering = (rel_onset is not None and entity_onset is not None
                and rel_onset < entity_onset)

    onset_rows.append({
        "model_name":          model_name,
        "relation_onset":      rel_onset,
        "entity_onset":        entity_onset,
        "entity_minus_relation": gap,
        "ordering_holds":      int(ordering),
    })

    print(f"\n{model_name.split('/')[-1]}: "
          f"relation_onset={rel_onset} "
          f"entity_onset={entity_onset} "
          f"gap={gap} ordering={ordering}")

    ci_df = bootstrap_ci(entity_summary, rel_by_pair, model_name)
    all_ci_rows.append(ci_df)
    ci_df.to_csv(
        OUTPUT_DIR / f"{model_name.replace('/', '_')}_ci.csv",
        index=False)

    plot_main_curves(entity_summary, rel_summary, ci_df,
                     model_name, rel_onset, entity_onset)
    plot_relation_by_pair(rel_by_pair, model_name)
    plot_controls(entity_summary, rel_summary,
                  ctrl_ent_summary, ctrl_rel_summary,
                  wrongent_summary, model_name)

onset_df = pd.DataFrame(onset_rows)
onset_df.to_csv(OUTPUT_DIR / "onset_table.csv", index=False)
plot_onset_comparison(onset_df)

if all_ci_rows:
    pd.concat(all_ci_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_models_ci.csv", index=False)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================
print("\n\n" + "=" * 100)
print("ONSET TABLE — relation onset precedes entity onset")
print("=" * 100)
print(onset_df.to_string(index=False))

print("\n" + "=" * 100)
print("RELATION TRANSFER BY PAIR — peak per model")
print("=" * 100)
peak = rel_by_pair.loc[
    rel_by_pair.groupby(["model_name", "relation_pair_id"])
    ["relation_transfer_pct"].idxmax()
]
print(peak[["model_name", "relation_pair_id",
            "layer_idx", "relation_transfer_pct"]].to_string(index=False))

print("\n" + "=" * 100)
print("ENTITY TRANSFER — peak per model per family")
print("=" * 100)
ent_peak = entity_summary.loc[
    entity_summary.groupby(
        ["model_name", "family"])["entity_score"].idxmax()
]
print(ent_peak[["model_name", "family",
                "layer_idx", "entity_score"]].to_string(index=False))

print("\n" + "=" * 100)
print("CONTROL DIAGNOSTICS — separated labels")
print("Unrelated-random donors test structured-transfer artifacts; overwrite is reported separately.")
print("Wrong-entity donors separate relation-only transfer from donor-answer copying.")
print("=" * 100)

if not ctrl_ent_summary.empty:
    print("\n[unrelated random — entity]")
    pk = ctrl_ent_summary.loc[
        ctrl_ent_summary.groupby(
            "model_name")["entity_transfer_pct"].idxmax()
    ]
    print(pk[["model_name", "layer_idx",
              "entity_transfer_pct", "overwrite_pct",
              "failure_pct"]].to_string(index=False))

if not ctrl_rel_summary.empty:
    print("\n[unrelated random — relation]")
    pk = ctrl_rel_summary.loc[
        ctrl_rel_summary.groupby(
            "model_name")["relation_transfer_pct"].idxmax()
    ]
    print(pk[["model_name", "layer_idx",
              "relation_transfer_pct", "overwrite_pct",
              "failure_pct"]].to_string(index=False))

if not wrongent_summary.empty:
    print("\n[wrong-entity relation control]")
    pk = wrongent_summary.loc[
        wrongent_summary.groupby(
            "model_name")["relation_only_transfer_pct"].idxmax()
    ]
    print(pk[["model_name", "layer_idx",
              "relation_only_transfer_pct", "donor_answer_copy_pct",
              "overwrite_pct", "failure_pct"]].to_string(index=False))

print(f"\n[done] All results saved to {OUTPUT_DIR}/")