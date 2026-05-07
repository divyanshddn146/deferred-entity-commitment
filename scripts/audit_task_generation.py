"""
audit_task_generation.py

Optional task-generation audit for the Deferred Entity Commitment paper.

This script prints and saves greedy-generation checks for the programmatically
defined prompt banks. During the project, these checks were mainly inspected
manually through console output to confirm that model generations were sensible
before running patching experiments.

The CSV outputs are audit artifacts only. They were not used as the source of
the final prompt banks for the main patching experiments, which define their
task banks directly in code.

This script is not required for reproducing the main figures/tables.
"""


import gc
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
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
LOCAL_FILES_ONLY = False
MAX_NEW_TOKENS = 8

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results/task_generation_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Filtering thresholds
GOOD_TOP_RANK = 50
GOOD_MARGIN = 3.0
BORDERLINE_RANK = 500
BORDERLINE_MARGIN = 0.0


# ============================================================
# DATASETS
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
# FAMILY DATA
# ============================================================
FAMILY_DATA = {
    "capital": {
        "pair_name": "capital__language",
        "prompt_template": "The capital of {x} is",
        "items": [(country, cap) for country, cap, lang in COUNTRY_BANK],
    },
    "language": {
        "pair_name": "capital__language",
        "prompt_template": "The official language of {x} is",
        "items": [(country, lang) for country, cap, lang in COUNTRY_BANK],
    },
    "past_tense": {
        "pair_name": "past_tense__present_participle",
        "prompt_template": "The past tense of {x} is",
        "items": [(base, past) for base, past, part in TENSE_BANK],
    },
    "present_participle": {
        "pair_name": "past_tense__present_participle",
        "prompt_template": "The present participle of {x} is",
        "items": [(base, part) for base, past, part in TENSE_BANK],
    },
    "plural": {
        "pair_name": "plural",
        "prompt_template": "The plural of {x} is",
        "items": [(singular, plural) for singular, plural in PLURAL_BANK],
    },
    "opposite": {
        "pair_name": "opposite__comparative",
        "prompt_template": "The opposite of {x} is",
        "items": [(adj, opp) for adj, opp, comp in ADJECTIVE_BANK],
    },
    "comparative": {
        "pair_name": "opposite__comparative",
        "prompt_template": "The comparative form of {x} is",
        "items": [(adj, comp) for adj, opp, comp in ADJECTIVE_BANK],
    },
    "symbol": {
        "pair_name": "chemical_symbol",
        "prompt_template": "The chemical symbol for {x} is",
        "items": [(element, symbol) for element, symbol in ELEMENT_BANK],
    },
}


def make_prompt(family: str, x: str) -> str:
    return FAMILY_DATA[family]["prompt_template"].format(x=x)


# ============================================================
# TEXT CLEANING
# ============================================================
def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def normalize(x: str) -> str:
    x = x.lower().strip()
    x = x.strip('"').strip("'").strip("`")
    x = x.replace("\n", " ")
    x = strip_accents(x)
    return " ".join(x.split())


def extract_continuation(prompt: str, full_text: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text.strip()


def extract_first_phrase(prompt: str, full_text: str, max_words: int = 4) -> str:
    cont = extract_continuation(prompt, full_text)
    piece = re.split(r'[,.\n;:!?()"“”]', cont, maxsplit=1)[0].strip()
    return " ".join(piece.split()[:max_words]).lower().strip()


def clean_pred(pred: str) -> str:
    pred = normalize(pred)

    for bad in [" or ", " and ", " means ", " is "]:
        if bad in pred:
            pred = pred.split(bad)[0].strip()

    articles = {"a", "an", "the", "le", "la", "les", "un", "une", "el", "los", "las"}
    words = pred.split()
    if words and words[0] in articles:
        pred = " ".join(words[1:])

    return pred.strip()


def family_match(pred: str, gold: str) -> bool:
    """
    Normalized / first-word match (not strictly exact).
    Allows accent stripping and first-word prefix matching.
    Column is named normalized_answer_match to reflect this.
    """
    pred = clean_pred(pred)
    gold = normalize(gold)

    if not pred or not gold:
        return False

    if pred == gold:
        return True

    return pred.split()[0] == gold.split()[0]


# ============================================================
# MODEL WRAPPER
# ============================================================
class ModelRunner:
    def __init__(self, model_name: str):
        self.model_name = model_name

        print("\n" + "=" * 90)
        print(f"LOADING: {model_name}")
        print("=" * 90)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=LOCAL_FILES_ONLY,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        kwargs = {
            "trust_remote_code": True,
            "local_files_only": LOCAL_FILES_ONLY,
        }

        if USE_4BIT and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = (
                torch.float16 if torch.cuda.is_available() and USE_FP16_ON_CUDA
                else torch.float32
            )
            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def tokenize(self, prompt: str):
        toks = self.tokenizer(prompt, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in toks.items()}

    def generate(self, prompt: str) -> str:
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

    def gold_token_ids(self, gold_answer: str):
        # Try with leading space first, because causal LMs usually predict
        # " paris" not "paris"
        ids_space = self.tokenizer(
            " " + gold_answer, add_special_tokens=False)["input_ids"]
        ids_raw = self.tokenizer(
            gold_answer, add_special_tokens=False)["input_ids"]

        if ids_space:
            return ids_space, "leading_space"
        return ids_raw, "raw"

    def next_token_stats(self, prompt: str, gold_answer: str) -> Optional[dict]:
        gold_ids, gold_mode = self.gold_token_ids(gold_answer)
        if not gold_ids:
            return None

        first_gold_id = gold_ids[0]
        inputs = self.tokenize(prompt)

        with torch.no_grad():
            out = self.model(**inputs, return_dict=True, use_cache=False)

        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)

        topk = torch.topk(logits, k=10)
        top10_ids = topk.indices.tolist()
        top10_tokens = [
            self.tokenizer.decode([i]).replace("\n", "\\n") for i in top10_ids
        ]

        top1_id = int(torch.argmax(logits).item())
        top1_token = self.tokenizer.decode([top1_id]).strip()

        gold_logit = float(logits[first_gold_id].item())

        # first_token_mean_logit_margin (gold minus mean other)
        mean_other = float(
            ((logits.sum() - logits[first_gold_id]) / (logits.numel() - 1)).item()
        )
        first_token_mean_logit_margin = gold_logit - mean_other

        # first_token_top_competitor_margin (gold minus best non-gold)
        logits_copy = logits.clone()
        logits_copy[first_gold_id] = float("-inf")
        best_non_gold_logit = float(logits_copy.max().item())
        first_token_top_competitor_margin = gold_logit - best_non_gold_logit

        rank = int((logits > logits[first_gold_id]).sum().item() + 1)
        gold_prob = float(probs[first_gold_id].item())

        return {
            "gold_token_count": len(gold_ids),
            "gold_token_mode": gold_mode,
            "first_gold_token_id": first_gold_id,
            "first_gold_token_str": self.tokenizer.decode([first_gold_id]),
            "top1_token": top1_token,
            "top1_matches_first_gold": int(top1_id == first_gold_id),
            "first_token_mean_logit_margin": first_token_mean_logit_margin,
            "first_token_top_competitor_margin": first_token_top_competitor_margin,
            "first_token_rank": rank,
            "first_token_prob": gold_prob,
            "top10_tokens": " | ".join(top10_tokens),
        }


# ============================================================
# QUALITY LABELS
# ============================================================
def validation_label(row) -> str:
    if (
        row["normalized_answer_match"] == 1
        and row["first_token_rank"] <= GOOD_TOP_RANK
        and row["first_token_top_competitor_margin"] >= GOOD_MARGIN
    ):
        return "keep_strong"

    if (
        row["normalized_answer_match"] == 1
        and row["first_token_rank"] <= BORDERLINE_RANK
        and row["first_token_top_competitor_margin"] >= BORDERLINE_MARGIN
    ):
        return "keep_borderline"

    if row["normalized_answer_match"] == 1:
        return "generation_ok_but_logit_weak"

    return "drop_wrong_generation"


def family_quality(row) -> str:
    if row["keep_strong_rate"] >= 0.70 and row["normalized_answer_match_rate"] >= 0.85:
        return "good_to_use"
    if row["keep_any_rate"] >= 0.65 and row["normalized_answer_match_rate"] >= 0.75:
        return "maybe_use_after_manual_check"
    return "drop_or_rework"


# ============================================================
# MAIN
# ============================================================
all_rows = []

for model_name in MODELS:
    runner = ModelRunner(model_name)

    for family, cfg in FAMILY_DATA.items():
        print("\n" + "─" * 90)
        print(f"FAMILY: {family.upper()} | MODEL: {model_name}")
        print("─" * 90)

        for idx, (entity, gold) in enumerate(cfg["items"], start=1):
            prompt = make_prompt(family, entity)

            try:
                gen_text = runner.generate(prompt)
                pred_phrase = extract_first_phrase(prompt, gen_text, max_words=4)
                is_match = family_match(pred_phrase, gold)
                stats = runner.next_token_stats(prompt, gold) or {}
            except Exception as e:
                print(f"{idx:02d}. ERROR {entity}: {e}")
                all_rows.append({
                    "model_name": model_name,
                    "pair_name": cfg["pair_name"],
                    "family": family,
                    "item_index": idx,
                    "entity": entity,
                    "prompt": prompt,
                    "gold_answer": gold,
                    "error": str(e),
                })
                continue

            row = {
                "model_name": model_name,
                "pair_name": cfg["pair_name"],
                "family": family,
                "item_index": idx,
                "entity": entity,
                "prompt": prompt,
                "gold_answer": gold,
                "generated_text": gen_text,
                "pred_phrase": pred_phrase,
                "pred_clean": clean_pred(pred_phrase),
                "normalized_answer_match": int(is_match),
                **stats,
            }

            row["is_multi_token_gold"] = row.get("gold_token_count", 1) > 1

            row["validation_label"] = validation_label(row)

            mark = "✓" if is_match else "✗"
            print(
                f"{idx:02d}. {mark} {str(entity):<16} "
                f"gold={str(gold):<14} "
                f"pred={pred_phrase:<18} "
                f"rank={row.get('first_token_rank', -1):<6} "
                f"margin={row.get('first_token_mean_logit_margin', 0):>7.2f} "
                f"competitor_margin={row.get('first_token_top_competitor_margin', 0):>7.2f} "
                f"multi_tok={row['is_multi_token_gold']} "
                f"top1='{row.get('top1_token', '?')}' "
                f"label={row['validation_label']}"
            )

            all_rows.append(row)

    del runner.model, runner.tokenizer, runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# SAVE RESULTS
# ============================================================
results_df = pd.DataFrame(all_rows)

results_df.to_csv(OUTPUT_DIR / "logit_filter_all_items.csv", index=False)

valid_df = (
    results_df[results_df["error"].isna()]
    if "error" in results_df.columns
    else results_df
)

summary_df = (
    valid_df.groupby(["model_name", "pair_name", "family"], as_index=False)
    .agg(
        n=("entity", "count"),
        normalized_answer_match_rate=("normalized_answer_match", "mean"),
        mean_gold_token_count=("gold_token_count", "mean"),
        pct_single_token=("gold_token_count", lambda x: float((x == 1).mean())),
        pct_multi_token_gold=("is_multi_token_gold", "mean"),
        first_token_top1_rate=("top1_matches_first_gold", "mean"),
        mean_first_token_mean_logit_margin=("first_token_mean_logit_margin", "mean"),
        mean_first_token_top_competitor_margin=(
            "first_token_top_competitor_margin", "mean"),
        median_first_token_rank=("first_token_rank", "median"),
        keep_strong_rate=(
            "validation_label", lambda x: float((x == "keep_strong").mean())),
        keep_any_rate=(
            "validation_label",
            lambda x: float(
                x.isin(["keep_strong", "keep_borderline"]).mean())),
        drop_wrong_generation_rate=(
            "validation_label",
            lambda x: float((x == "drop_wrong_generation").mean())),
    )
)

summary_df["family_quality"] = summary_df.apply(family_quality, axis=1)
summary_df = (
    summary_df
    .sort_values(["model_name", "pair_name", "family"])
    .reset_index(drop=True)
)

summary_df.to_csv(OUTPUT_DIR / "logit_filter_family_summary.csv", index=False)

keep_df = valid_df[
    valid_df["validation_label"].isin(["keep_strong", "keep_borderline"])
].copy()
drop_df = valid_df[
    ~valid_df["validation_label"].isin(["keep_strong", "keep_borderline"])
].copy()

metadata = {
    "script": "01_logit_filter_and_task_validation.py",
    "models": MODELS,
    "thresholds": {
        "good_top_rank": GOOD_TOP_RANK,
        "good_top_competitor_margin": GOOD_MARGIN,
        "borderline_top_competitor_margin": BORDERLINE_MARGIN,
        "borderline_mean_logit_margin": BORDERLINE_MARGIN,
    },
    "max_new_tokens": MAX_NEW_TOKENS,
    "matching": (
        "normalized_answer_match: normalized / first-word match, "
        "not strictly exact. Allows accent stripping and first-word prefix matching."
    ),
    "margins": {
        "first_token_mean_logit_margin": (
            "gold logit minus mean of all other token logits"
        ),
        "first_token_top_competitor_margin": (
            "gold logit minus best non-gold token logit"
        ),
    },
    "is_multi_token_gold": (
        "True when gold answer tokenizes to more than one token "
        "(e.g. 'buenos aires', 'abu dhabi'). "
        "First-token logit stats are less reliable for these items."
    ),
    "note": (
    "Task banks are defined in code. CSVs are audit outputs only. "
    "During the project, greedy-generation behavior was primarily inspected "
    "manually from console output. These CSVs were not used as the source of "
    "the final task banks for patching experiments."
),
}

with open(OUTPUT_DIR / "validation_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\n\n" + "=" * 120)
print("FINAL FAMILY SUMMARY")
print("=" * 120)
print(summary_df.to_string(index=False))

print("\nSaved:")
print(f"  {OUTPUT_DIR / 'logit_filter_all_items.csv'}")
print(f"  {OUTPUT_DIR / 'logit_filter_family_summary.csv'}")
print(f"  {OUTPUT_DIR / 'validation_metadata.json'}")