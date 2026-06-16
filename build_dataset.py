"""
Unified EN→Tunisian (Derja) dataset builder.

Goals (see writing-plans/arabicnlp-dpo-dialect-fidelity.md):
  * Merge multiple sources (local CSV/JSONL + optional HF parallel dataset).
  * Aggressively clean: garbage, instruction-wrapper leakage ("ترجم ..."),
    length-ratio outliers, near-duplicates.
  * Track PROVENANCE per row (source, english origin synthetic/authentic,
    alignment score, length bucket) so the data section is honest.
  * Separate a SENTENCE-level track from a LEXICAL (dictionary) track, because
    BLEU on 1-word targets is meaningless and reviewers will check.
  * Build a held-out eval set drawn from the sentence track and FLAG it for
    human verification (never silently evaluate on synthetic English).
  * Optional, network-gated quality filters: semantic alignment (sentence
    embeddings) and dialect-fidelity (DID classifier). Core runs offline.

Usage:
  # Offline core on local data:
  python build_dataset.py --out data_v2

  # Add the HF parallel dataset (requires huggingface.co allowlisted):
  python build_dataset.py --hf_dataset hamzabouajila/<id> --out data_v2

  # Turn on heavy quality filters (need models / GPU / network):
  python build_dataset.py --out data_v2 --semantic_filter --dialect_filter
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent
SEED = 42

# Tashkeel (Arabic diacritics) range — stripped for the dedup/match key ONLY.
_TASHKEEL = re.compile(r"[ً-ٰٟ]")
_WS = re.compile(r"\s+")

# Instruction-wrapper leakage seen in the original data (both languages).
_INSTRUCTION_MARKERS = (
    "ترجم", "اكتب", "translat", "write in derja", "how do you say",
    "what is the word", "what does", "in derja", "can you give me", "write in",
)
_GARBAGE_PATTERNS = [
    re.compile(r"^[+\-=*/_.,;:\[\]()#@!?\"']+$"),
    re.compile(r"^\d+$"),
    re.compile(r"^[a-zA-Z]+$"),                      # latin-only on the TN side
    re.compile(r"ترجم\s+(لهنا|هذا|ل|هكا)"),          # the "translate here ..." leak
]


# --------------------------------------------------------------------------- #
# Normalization / quality predicates
# --------------------------------------------------------------------------- #
def norm_key(text: str) -> str:
    """Aggressive normalization used ONLY as a dedup/match key (not for output)."""
    text = str(text)
    text = _TASHKEEL.sub("", text)
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي")
    return _WS.sub(" ", text).strip().lower()


def is_garbage(text: str) -> bool:
    t = str(text).strip()
    if len(t) < 2:
        return True
    return any(p.match(t) for p in _GARBAGE_PATTERNS) or any(p.search(t) for p in _GARBAGE_PATTERNS[-1:])


def is_instruction(text: str) -> bool:
    t = str(text).lower().strip()
    return any(t.startswith(m) or m in t[:15] for m in _INSTRUCTION_MARKERS)


def length_ok(en: str, tn: str, lo=0.2, hi=5.0) -> bool:
    en_w, tn_w = len(str(en).split()), len(str(tn).split())
    if en_w == 0 or tn_w == 0:
        return False
    return lo <= tn_w / max(en_w, 1) <= hi


def length_bucket(en: str) -> str:
    w = len(str(en).split())
    return "lexical" if w <= 2 else "short" if w <= 4 else "medium" if w <= 8 else "long"


def is_sentence(en: str, tn: str) -> bool:
    """Sentence-grade = BOTH sides multi-word (Tunisian is compact, so TN>=2).

    This keeps BLEU/chrF meaningful and excludes dictionary glosses whose
    English definition is long but whose Tunisian target is a single word.
    """
    return len(str(en).split()) >= 4 and len(str(tn).split()) >= 2


def eval_reliability(r) -> int:
    """Higher = more trustworthy as a held-out reference (authentic + multi-word)."""
    score = 0
    if r.get("source") == "drejja_sentences":   # authentic parallel sentences
        score += 2
    if len(r["tn"].split()) >= 3:                # multi-token reference for BLEU
        score += 1
    if r.get("en_origin") == "authentic":        # non-synthetic English
        score += 1
    return score


# --------------------------------------------------------------------------- #
# Source loaders -> list of dicts {en, tn, source, en_origin}
# --------------------------------------------------------------------------- #
def load_csv(path: Path, source: str, en_origin: str):
    import csv
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # tolerate leading index column / arbitrary header order
        fields = {c.lower(): c for c in (reader.fieldnames or [])}
        en_c = fields.get("en") or fields.get("english") or fields.get("input")
        tn_c = fields.get("tn") or fields.get("tunisian") or fields.get("output")
        if not en_c or not tn_c:
            raise SystemExit(f"{path}: could not find en/tn columns in {reader.fieldnames}")
        for r in reader:
            rows.append({"en": (r.get(en_c) or "").strip(),
                         "tn": (r.get(tn_c) or "").strip(),
                         "source": source, "en_origin": en_origin})
    return rows


def load_jsonl(path: Path, source: str, en_origin: str, en_key="input", tn_key="output"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({"en": str(d.get(en_key, "")).strip(),
                         "tn": str(d.get(tn_key, "")).strip(),
                         "source": source, "en_origin": en_origin})
    return rows


def load_hf(dataset_id: str, en_key: str, tn_key: str, en_origin: str):
    """Load a parallel dataset from the HF hub. Requires network access."""
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split="train")
    rows = []
    for ex in ds:
        rows.append({"en": str(ex.get(en_key, "")).strip(),
                     "tn": str(ex.get(tn_key, "")).strip(),
                     "source": f"hf:{dataset_id}", "en_origin": en_origin})
    return rows


# --------------------------------------------------------------------------- #
# Optional heavy filters (network / GPU gated)
# --------------------------------------------------------------------------- #
def semantic_scores(rows, model_name="sentence-transformers/LaBSE", batch=256):
    """Cosine similarity between EN and TN embeddings; flags misaligned LLM pairs."""
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer(model_name)
    en = [r["en"] for r in rows]
    tn = [r["tn"] for r in rows]
    en_emb = model.encode(en, batch_size=batch, convert_to_tensor=True, show_progress_bar=True)
    tn_emb = model.encode(tn, batch_size=batch, convert_to_tensor=True, show_progress_bar=True)
    for i, r in enumerate(rows):
        r["align"] = float(util.cos_sim(en_emb[i], tn_emb[i]))
    return rows


def dialect_scores(rows, model_name="CAMeL-Lab/bert-base-arabic-camelbert-msa-did-madar-twitter5"):
    """Label each TN target by predicted Arabic variety; lets us drop MSA-leaning rows."""
    from transformers import pipeline
    clf = pipeline("text-classification", model=model_name, device=0)
    for r in rows:
        try:
            r["variety"] = clf(r["tn"][:512])[0]["label"]
        except Exception:
            r["variety"] = "UNK"
    return rows


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def clean(rows, args):
    stats = Counter()
    seen_pair, seen_en = set(), set()
    out = []
    for r in rows:
        en, tn = r["en"], r["tn"]
        stats["in"] += 1
        if not en or not tn:
            stats["drop_empty"] += 1; continue
        if is_garbage(en) or is_garbage(tn):
            stats["drop_garbage"] += 1; continue
        if is_instruction(en) or is_instruction(tn):
            stats["drop_instruction"] += 1; continue
        if not length_ok(en, tn):
            stats["drop_length"] += 1; continue
        ken, ktn = norm_key(en), norm_key(tn)
        if (ken, ktn) in seen_pair:
            stats["drop_dup_pair"] += 1; continue
        if ken in seen_en:                       # avoid one-EN -> many-TN noise
            stats["drop_dup_en"] += 1; continue
        seen_pair.add((ken, ktn)); seen_en.add(ken)
        r["len_bucket"] = length_bucket(en)
        r["track"] = "sentence" if is_sentence(en, tn) else "lexical"
        out.append(r)
        stats["kept"] += 1
    return out, stats


def split_and_save(rows, out_dir: Path, args):
    import random
    rng = random.Random(SEED)

    sent = [r for r in rows if r["track"] == "sentence"]
    lex = [r for r in rows if r["track"] == "lexical"]
    rng.shuffle(sent); rng.shuffle(lex)

    # Held-out eval is drawn from the MOST RELIABLE sentence pairs (authentic
    # source + multi-word target) so BLEU/chrF are meaningful; the rest train.
    sent.sort(key=eval_reliability, reverse=True)
    n_test = min(args.test_size, max(0, len(sent) // 5))
    n_val = min(args.val_size, max(0, (len(sent) - n_test) // 5))
    test = sent[:n_test]
    val = sent[n_test:n_test + n_val]
    train_sent = sent[n_test + n_val:]
    rng.shuffle(train_sent)

    for r in test:
        r["needs_human_verification"] = (r.get("en_origin") == "synthetic_en")

    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name, recs, minimal=True):
        p = out_dir / name
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                if minimal:
                    f.write(json.dumps({"input": r["en"], "output": r["tn"]}, ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(recs)}")

    print("\n[save] sentence track:")
    dump("sent_train.jsonl", train_sent)
    dump("sent_val.jsonl", val)
    dump("sent_test.jsonl", test)
    dump("sent_test.meta.jsonl", test, minimal=False)   # provenance + human-verify flag
    print("[save] lexical track:")
    dump("lex_all.jsonl", lex)
    dump("all.meta.jsonl", rows, minimal=False)          # full provenance dump
    return {"sent_train": len(train_sent), "sent_val": len(val),
            "sent_test": len(test), "lexical": len(lex)}


def report(rows):
    print("\n[composition]")
    by_track = Counter(r["track"] for r in rows)
    by_src = Counter(r["source"] for r in rows)
    by_bucket = Counter(r["len_bucket"] for r in rows)
    print("  by track :", dict(by_track))
    print("  by source:", dict(by_src))
    print("  by bucket:", dict(by_bucket))
    if rows and "align" in rows[0]:
        al = sorted(r["align"] for r in rows)
        print(f"  align    : min={al[0]:.2f} median={al[len(al)//2]:.2f} max={al[-1]:.2f}")
    if rows and "variety" in rows[0]:
        print("  variety  :", dict(Counter(r.get("variety") for r in rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_v2", help="output directory")
    ap.add_argument("--hf_dataset", help="HF parallel dataset id (needs network)")
    ap.add_argument("--hf_en_key", default="en")
    ap.add_argument("--hf_tn_key", default="tn")
    ap.add_argument("--hf_en_origin", default="synthetic_en",
                    help="mark HF English as synthetic (LLM TN->EN reversed) or authentic")
    ap.add_argument("--semantic_filter", action="store_true")
    ap.add_argument("--semantic_threshold", type=float, default=0.50)
    ap.add_argument("--dialect_filter", action="store_true",
                    help="drop TN targets the DID classifier does not call Tunisian")
    ap.add_argument("--test_size", type=int, default=400)
    ap.add_argument("--val_size", type=int, default=300)
    args = ap.parse_args()

    rows = []
    # Local sources (offline). en_origin unknown for these.
    drejja = DATA_DIR / "drejja_clean_sentences.jsonl"
    csv_raw = DATA_DIR / "tunsi_english_data.csv"
    if drejja.exists():
        rows += load_jsonl(drejja, "drejja_sentences", "unknown")
    if csv_raw.exists():
        rows += load_csv(csv_raw, "tunsi_csv", "unknown")
    if args.hf_dataset:
        print(f"[hf] loading {args.hf_dataset} ...")
        rows += load_hf(args.hf_dataset, args.hf_en_key, args.hf_tn_key, args.hf_en_origin)

    print(f"[load] {len(rows)} raw rows from {len(set(r['source'] for r in rows))} source(s)")

    rows, stats = clean(rows, args)
    print("[clean]", dict(stats))

    if args.semantic_filter:
        rows = semantic_scores(rows)
        before = len(rows)
        rows = [r for r in rows if r["align"] >= args.semantic_threshold]
        print(f"[semantic] kept {len(rows)}/{before} at align>={args.semantic_threshold}")
    if args.dialect_filter:
        rows = dialect_scores(rows)
        before = len(rows)
        rows = [r for r in rows if r.get("variety") in ("Tunisia", "TUN", "tn")]
        print(f"[dialect] kept {len(rows)}/{before} classified Tunisian")

    report(rows)
    counts = split_and_save(rows, DATA_DIR / args.out, args)
    print("\n[done]", counts)


if __name__ == "__main__":
    main()
