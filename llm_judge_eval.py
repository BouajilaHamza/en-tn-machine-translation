"""
Phase 9: LLM-as-judge evaluation for dialect translation quality.
Evaluates fluency, adequacy, and dialectal fidelity using an LLM judge.

Requires: OpenAI/Anthropic API key (or local model via vLLM).
Usage: python llm_judge_eval.py --model_path <path> [--api_key <key>]
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SEED = 42
random.seed(SEED)

SRC_LANG = "eng_Latn"
TGT_LANG = "aeb_Arab"
DATA_DIR = Path(__file__).parent
MAX_LENGTH = 128

EVAL_RUBRIC = """
Evaluate the translation on three criteria (1-5):
1. Fluency: Is the output natural and grammatical Arabic/Tunisian?
2. Adequacy: Does the output convey the meaning of the source?
3. Dialectal Fidelity: Does the output use Tunisian Derja (not MSA)?

Respond in JSON format:
{"fluency": X, "adequacy": X, "dialect_fidelity": X, "overall": X}
"""


def load_model(path: Path, base_model: str = "facebook/nllb-200-3.3B"):
    is_adapter = (path / "adapter_config.json").exists()
    if is_adapter:
        base = AutoModelForSeq2SeqLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = PeftModel.from_pretrained(base, str(path))
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(path), torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(path))

    model.eval()
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG
    return model, tokenizer


def generate_translations(model, tokenizer, texts, n_beams=4):
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_LENGTH,
            num_beams=n_beams,
            early_stopping=True,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def llm_judge(prompt: str, translation: str, api_key: str = None):
    """
    Query an LLM to judge the translation.
    Supports OpenAI/Anthropic API, or falls back to a simple heuristic.
    """
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EVAL_RUBRIC},
                    {"role": "user", "content": (
                        f"Source: {prompt}\n"
                        f"Translation: {translation}\n"
                        f"Please evaluate this translation."
                    )},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  LLM API error: {e}")
            return None
    else:
        # No API: return None (requires manual evaluation)
        return None


def heuristic_judge(prompt: str, translation: str, reference: str = None):
    """Simple heuristic scores when LLM is unavailable."""
    scores = {
        "fluency": 3.0,
        "adequacy": 3.0,
        "dialect_fidelity": 3.0,
    }

    if not translation or len(translation) < 3:
        return {"fluency": 1.0, "adequacy": 1.0,
                "dialect_fidelity": 1.0, "overall": 1.0,
                "note": "empty/too short"}

    # Dialect fidelity: check for Arabizi numerals (MSA marker)
    arabizi_numerals = sum(1 for c in translation if c in "23456789")
    if arabizi_numerals > 0:
        scores["dialect_fidelity"] -= 0.5

    # Check Arabic character density
    arabic_chars = sum(1 for c in translation if '\u0600' <= c <= '\u06FF')
    total_chars = len(translation.strip())
    if total_chars > 0:
        arabic_ratio = arabic_chars / total_chars
        if arabic_ratio < 0.3:
            scores["fluency"] -= 1.0

    scores["overall"] = sum(scores.values()) / 3
    return {**scores, "note": "heuristic"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = args.output or DATA_DIR / f"llm_eval_{model_path.name}.csv"

    print("=" * 60)
    print("Phase 9: LLM-as-Judge Evaluation")
    print("=" * 60)

    print(f"\nModel: {model_path.name}")
    model, tokenizer = load_model(model_path)

    # Load test pairs
    records = []
    for fname in ["clean_en_tn_test.jsonl", "clean_en_tn_val.jsonl"]:
        path = DATA_DIR / fname
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))

    random.shuffle(records)
    sample = records[:args.n_samples]
    print(f"  Evaluating {len(sample)} samples")

    # Generate translations
    srcs = [r["input"] for r in sample]
    refs = [r["output"] for r in sample]
    preds = generate_translations(model, tokenizer, srcs)

    # Judge each translation
    results = []
    for src, ref, pred in zip(srcs, refs, preds):
        judge_result = llm_judge(src, pred, args.api_key)
        if judge_result is None:
            judge_result = heuristic_judge(src, pred, ref)

        results.append({
            "source": src,
            "reference": ref,
            "prediction": pred,
            **judge_result,
        })

    # Summary
    df = pd.DataFrame(results)
    print(f"\nResults:")
    print(f"  Mean Fluency:          {df['fluency'].mean():.2f}")
    print(f"  Mean Adequacy:         {df['adequacy'].mean():.2f}")
    print(f"  Mean Dialect Fidelity: {df['dialect_fidelity'].mean():.2f}")
    print(f"  Mean Overall:          {df['overall'].mean():.2f}")

    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print("\nDone! ✓")


if __name__ == '__main__':
    main()
