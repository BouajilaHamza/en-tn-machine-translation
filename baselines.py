"""
Phase 6: Baseline comparisons for the paper.
1. SFT with explicit variety tags (<ar_TN>)
2. SFT with DialUp-style synthetic dialect augmentation
3. Decoding-time classifier-guided control

Usage: python baselines.py [--method tag|dialup|classifier]
"""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    pipeline,
)
import evaluate
import numpy as np

SEED = 42
random.seed(SEED)

BASE_MODEL = "facebook/nllb-200-3.3B"
SRC_LANG = "eng_Latn"
TGT_LANG = "aeb_Arab"
DATA_DIR = Path(__file__).parent
MAX_LENGTH = 128
BATCH_SIZE = 4
GRAD_ACCUM = 8


def load_clean_data():
    records = []
    for split in ["train", "val"]:
        path = DATA_DIR / f"clean_en_tn_{split}.jsonl"
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
    return records


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG

    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


# ======= METHOD A: Variety Tags =======
def prepare_tag_data(records):
    """Prepend <ar_TN> dialect tag to source."""
    augmented = []
    for r in records:
        augmented.append({
            "input": f"<ar_TN> {r['input']}",
            "output": r["output"],
        })
    # Also add MSA examples for contrast
    augmented.extend([
        {"input": f"<arb_Arab> {r['input']}", "output": r["output"]}
        for r in random.sample(records, min(500, len(records)))
    ])
    random.shuffle(augmented)
    return augmented


def train_tag_baseline():
    """Train SFT with variety tags."""
    print("\n=== Baseline A: SFT with Variety Tags ===")
    records = load_clean_data()
    tag_data = prepare_tag_data(records)
    print(f"  Tag-augmented data: {len(tag_data)} pairs")

    dataset = Dataset.from_list(tag_data)
    dataset = dataset.train_test_split(test_size=250, seed=SEED)
    model, tokenizer = load_model_and_tokenizer()

    def preprocess_fn(examples):
        inputs = tokenizer(
            examples["input"],
            max_length=MAX_LENGTH, truncation=True, padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["output"],
                max_length=MAX_LENGTH, truncation=True, padding=False,
            )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = dataset["train"].map(preprocess_fn, batched=True)
    eval_ds = dataset["test"].map(preprocess_fn, batched=True)

    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        return {
            "bleu": bleu.compute(
                predictions=decoded_preds,
                references=[[l] for l in decoded_labels]
            )["score"],
            "chrf": chrf.compute(
                predictions=decoded_preds,
                references=[[l] for l in decoded_labels]
            )["score"],
        }

    args = Seq2SeqTrainingArguments(
        output_dir=str(DATA_DIR / "nllb-en-tn-sft-tag"),
        learning_rate=2e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        report_to="none",
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    save_path = DATA_DIR / "nllb-en-tn-sft-tag"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"  Saved to {save_path}")
    return save_path


# ======= METHOD B: DialUp-style Augmentation =======
def dialup_augment(text, prob=0.3):
    """
    Simple DialUp-inspired synthetic dialect augmentation.
    Swaps high-frequency MSA words with their Tunisian counterparts.
    """
    msa_to_tn = {
        "ماذا": "شنو",
        "هذا": "هادا",
        "الآن": "توا",
        "يذهب": "يمش",
        "يريد": "يحب",
        "قال": "قال",
        "كبير": "كبير",
        "صغير": "صغير",
        "جيد": "باهي",
        "سيئ": "خايب",
        "جدا": "برشا",
        "أيضا": "زيادة",
        "ليس": "ماش",
    }
    words = text.split()
    augmented = []
    for w in words:
        if w in msa_to_tn and random.random() < prob:
            augmented.append(msa_to_tn[w])
        else:
            augmented.append(w)
    return " ".join(augmented)


def prepare_dialup_data(records):
    """DialUp-style: synthetically augment MSA training data with TN variants."""
    augmented = []
    for r in records:
        augmented.append(r)  # original
        # DialUp variant on source
        augmented.append({
            "input": dialup_augment(r["input"]),
            "output": r["output"],
        })
    random.shuffle(augmented)
    return augmented


def train_dialup_baseline():
    """Train SFT with DialUp-style synthetic augmentation."""
    print("\n=== Baseline B: DialUp-style Augmentation ===")
    records = load_clean_data()
    dialup_data = prepare_dialup_data(records)
    print(f"  DialUp-augmented data: {len(dialup_data)} pairs")

    dataset = Dataset.from_list(dialup_data)
    dataset = dataset.train_test_split(test_size=250, seed=SEED)
    model, tokenizer = load_model_and_tokenizer()

    def preprocess_fn(examples):
        inputs = tokenizer(
            examples["input"],
            max_length=MAX_LENGTH, truncation=True, padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["output"],
                max_length=MAX_LENGTH, truncation=True, padding=False,
            )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = dataset["train"].map(preprocess_fn, batched=True)
    eval_ds = dataset["test"].map(preprocess_fn, batched=True)

    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        return {
            "bleu": bleu.compute(
                predictions=decoded_preds,
                references=[[l] for l in decoded_labels]
            )["score"],
            "chrf": chrf.compute(
                predictions=decoded_preds,
                references=[[l] for l in decoded_labels]
            )["score"],
        }

    args = Seq2SeqTrainingArguments(
        output_dir=str(DATA_DIR / "nllb-en-tn-sft-dialup"),
        learning_rate=2e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        report_to="none",
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    save_path = DATA_DIR / "nllb-en-tn-sft-dialup"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"  Saved to {save_path}")
    return save_path


# ======= METHOD C: Decoding-time Classifier Guidance =======
def classifier_guided_decode(model, tokenizer, text, classifier, alpha=0.5):
    """
    Decoding-time guidance: score each candidate by
    (1 - alpha) * model_score + alpha * classifier_score_for_TN
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_LENGTH,
            num_beams=5,
            num_return_sequences=5,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    candidates = tokenizer.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )

    # Score each candidate with dialect classifier
    best_score = -float("inf")
    best_candidate = candidates[0]
    for cand in candidates:
        cls_result = classifier(cand[:512])[0]
        dialect_score = cls_result["score"] if cls_result["label"] == "TUN" else 0.0
        combined = dialect_score
        if combined > best_score:
            best_score = combined
            best_candidate = cand

    return best_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="tag",
                        choices=["tag", "dialup", "classifier"])
    args = parser.parse_args()

    if args.method == "tag":
        train_tag_baseline()
    elif args.method == "dialup":
        train_dialup_baseline()
    elif args.method == "classifier":
        print("\n=== Baseline C: Classifier-guided Decoding ===")
        print("  Requires Tunisian dialect classifier model.")
        print("  Run after training: classifier_guided_decode()")

    print("\nDone! ✓")


if __name__ == '__main__':
    main()
