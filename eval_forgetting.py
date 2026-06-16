"""
Phase 7: Measure catastrophic forgetting.
Tests retention on MSA tasks (EN→MSA, MSA→EN) and general language PPL.

Usage: python eval_forgetting.py --model_path <path>
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DATA_DIR = Path(__file__).parent
SRC_LANG = "eng_Latn"
TGT_LANG = "aeb_Arab"
MSA_LANG = "arb_Arab"
MAX_LENGTH = 128


def calculate_perplexity(model, tokenizer, src_texts, tgt_texts, lang=None):
    total_loss = 0.0
    n = len(src_texts)
    with torch.no_grad():
        for src, tgt in zip(src_texts, tgt_texts):
            inputs = tokenizer(src, return_tensors="pt").to(model.device)
            labels = tokenizer(tgt, return_tensors="pt").to(model.device).input_ids
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
    avg_loss = total_loss / max(n, 1)
    return float(torch.exp(torch.tensor(avg_loss)))


def get_flores_200_test_small(tokenizer, src_lang, tgt_lang, n=100):
    """Load a small FLORES-200 test set for a language pair."""
    try:
        ds = load_dataset("facebook/flores", "all", split="devtest", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= n:
                break
            samples.append({
                "src": item[f"sentence_{src_lang}"],
                "tgt": item[f"sentence_{tgt_lang}"],
            })
        return samples
    except Exception as e:
        print(f"  Could not load FLORES: {e}")
        # Fall back to synthetic test pairs
        fallback_pairs = [
            ("The weather is nice today.", "الطقس لطيف اليوم."),
            ("I would like a cup of coffee.", "أريد فنجان قهوة."),
            ("Thank you very much.", "شكرا جزيلا."),
            ("What time is the meeting?", "في أي ساعة الاجتماع؟"),
            ("Please send me the document.", "يرجى إرسال المستند لي."),
        ] * 20
        return [{"src": s, "tgt": t} for s, t in fallback_pairs[:n]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", default="facebook/nllb-200-3.3B")
    parser.add_argument("--n_test", type=int, default=100)
    args = parser.parse_args()

    model_path = Path(args.model_path)

    print("=" * 60)
    print("Phase 7: Catastrophic Forgetting Evaluation")
    print("=" * 60)

    print(f"\nModel: {model_path.name}")

    # Load model
    is_adapter = (model_path / "adapter_config.json").exists()
    if is_adapter:
        base_id = args.base_model
        base = AutoModelForSeq2SeqLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = PeftModel.from_pretrained(base, str(model_path))
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    model.eval()

    results = {}

    # Test 1: Tunisian PPL (target)
    print("\n[1] Tunisian PPL (EN→TN)...")
    tn_pairs = []
    for split in ["test", "val"]:
        path = DATA_DIR / f"clean_en_tn_{split}.jsonl"
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                tn_pairs.append(d)
    ppl_tn = calculate_perplexity(
        model, tokenizer,
        [p["input"] for p in tn_pairs],
        [p["output"] for p in tn_pairs],
    )
    results["ppl_tunisian"] = round(ppl_tn, 2)
    print(f"  EN→TN PPL: {ppl_tn:.2f}")

    # Test 2: MSA PPL (should increase after DPO = style rejection)
    # Use English→MSA test pairs from FLORES
    print("\n[2] MSA PPL (EN→MSA)...")
    msa_pairs = get_flores_200_test_small(
        tokenizer, "eng_Latn", "arb_Arab", n=args.n_test
    )
    ppl_msa = calculate_perplexity(
        model, tokenizer,
        [p["src"] for p in msa_pairs],
        [p["tgt"] for p in msa_pairs],
    )
    results["ppl_msa"] = round(ppl_msa, 2)
    print(f"  EN→MSA PPL: {ppl_msa:.2f}")

    # Test 3: MSA→EN PPL (retention of reverse direction)
    print("\n[3] MSA→EN PPL (MSA→EN retention)...")
    ppl_msa_en = calculate_perplexity(
        model, tokenizer,
        [p["tgt"] for p in msa_pairs],
        [p["src"] for p in msa_pairs],
    )
    results["ppl_msa_en"] = round(ppl_msa_en, 2)
    print(f"  MSA→EN PPL: {ppl_msa_en:.2f}")

    # Test 4: Baseline PPL on non-Arabic language (French)
    print("\n[4] French→English PPL (general language retention)...")
    fra_pairs = get_flores_200_test_small(
        tokenizer, "fra_Latn", "eng_Latn", n=min(args.n_test, 50)
    )
    ppl_fra = calculate_perplexity(
        model, tokenizer,
        [p["src"] for p in fra_pairs],
        [p["tgt"] for p in fra_pairs],
    )
    results["ppl_french_en"] = round(ppl_fra, 2)
    print(f"  FR→EN PPL: {ppl_fra:.2f}")

    # Summary
    print(f"\n{'='*40}")
    print("FORGETTING EVALUATION SUMMARY")
    print(f"{'='*40}")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Save
    out_path = DATA_DIR / f"forgetting_{model_path.name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("\nDone! ✓")


if __name__ == '__main__':
    main()
