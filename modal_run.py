"""
Modal experiment runner for EN→TN MT + DPO on T4 ($0.34/hr).
Each function runs in a separate container — data persisted via Modal Volume.
Usage:
    modal run modal_run.py               # Full pipeline
    modal run modal_run.py --mode eval   # Eval SFT + DPO
"""

import json
import os
import re
from pathlib import Path

import modal

app = modal.App("en-tn-dialect-dpo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "sentencepiece>=0.1.99",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "trl>=0.7.11",
        "bitsandbytes>=0.41.0",
        "evaluate>=0.4.0",
        "sacrebleu>=2.3.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "huggingface_hub>=0.20.0",
    )
)

vol = modal.Volume.from_name("en-tn-data", create_if_missing=True)
RESULTS_DIR = "/vol/en-tn"


def log(msg):
    print(f"[EN-TN] {msg}", flush=True)


def _prepare_data(csv_data: str):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import io

    df = pd.read_csv(io.StringIO(csv_data))
    df.columns = ['id', 'en', 'tn']
    df = df.dropna()
    df['en'] = df['en'].str.strip()
    df['tn'] = df['tn'].str.strip()

    def bad(t):
        if len(t) < 2:
            return True
        if re.match(r'^(ترجم|اكتب|translat|how do you say|what is|in derja|can you give)', t, re.I):
            return True
        return False

    df = df[~df['en'].apply(bad)]
    df = df[~df['tn'].apply(bad)]
    df = df.drop_duplicates(subset='en')
    df['ratio'] = df['tn'].str.split().str.len() / df['en'].str.split().str.len().clip(lower=1)
    df = df[df['ratio'].between(0.2, 5.0)]

    t, test = train_test_split(df, test_size=250, random_state=42)
    tr, val = train_test_split(t, test_size=250, random_state=42)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for n, s in [('train', tr), ('val', val), ('test', test)]:
        with open(f"{RESULTS_DIR}/clean_en_tn_{n}.jsonl", 'w') as f:
            for _, r in s.iterrows():
                f.write(json.dumps({"input": r['en'], "output": r['tn']}, ensure_ascii=False) + '\n')
    all_df = pd.concat([tr, val, test])
    with open(f"{RESULTS_DIR}/clean_en_tn_all.jsonl", 'w') as f:
        for _, r in all_df.iterrows():
            f.write(json.dumps({"input": r['en'], "output": r['tn']}, ensure_ascii=False) + '\n')

    log(f"Train: {len(tr)}, Val: {len(val)}, Test: {len(test)}")
    return {"train": len(tr), "val": len(val), "test": len(test)}


def _evaluate_model(model_path, pairs):
    import torch
    import evaluate as ev
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    log(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "aeb_Arab"

    sacrebleu = ev.load("sacrebleu")

    bs = 4
    preds = []
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i+bs]
        en = [p["input"] for p in batch]
        inputs = tokenizer(en, padding=True, truncation=True, max_length=128, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=2, early_stopping=True)
        preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # Sample prints
    log("  Samples:")
    for i in range(min(5, len(pairs))):
        log(f"    EN: {pairs[i]['input']}")
        log(f"    REF: {pairs[i]['output']}")
        log(f"    PRED: {preds[i]}")

    refs = [p["output"] for p in pairs]
    bleu = sacrebleu.compute(predictions=preds, references=[[r] for r in refs])["score"]

    losses = []
    for i in range(0, min(len(pairs), 200), bs):
        batch = pairs[i:i+bs]
        inp = tokenizer([p["input"] for p in batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(model.device)
        lbl = tokenizer([p["output"] for p in batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(model.device).input_ids
        with torch.no_grad():
            losses.append(model(**inp, labels=lbl).loss.item())
    ppl = float(torch.exp(torch.tensor(sum(losses) / len(losses))))

    # Free GPU before loading classifier
    del model, tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    log("  Loading dialect classifier...")
    classifier = pipeline("text-classification", model="CAMeL-Lab/bert-base-arabic-camelbert-msa-did-madar-twitter5", device=0)
    tun = sum(1 for p in preds if p and classifier(p[:512])[0]["label"] == "Tunisia")
    dialect_pct = round(tun / len(preds) * 100, 1) if preds else 0.0

    result = {"bleu": round(bleu, 2), "ppl": round(ppl, 2), "dialect_%": dialect_pct}
    log(f"  BLEU={bleu:.2f}  PPL={ppl:.2f}  Dialect={dialect_pct}%")
    return result


# ============================================================
# Functions
# ============================================================

@app.function(image=image, timeout=300, volumes={RESULTS_DIR: vol})
def prepare_data(csv_data: str):
    result = _prepare_data(csv_data)
    vol.commit()
    return result


@app.function(image=image, gpu="t4", timeout=3600 * 8, volumes={RESULTS_DIR: vol})
def train_sft(max_steps=600):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer,
        Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
    )
    import evaluate as ev
    import numpy as np

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log("Loading NLLB-200 3.3B + LoRA...")

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "aeb_Arab"

    def load_split(name):
        return Dataset.from_list([json.loads(l) for l in open(f"{RESULTS_DIR}/clean_en_tn_{name}.jsonl")])

    train_ds, val_ds = load_split("train"), load_split("val")

    def tok(ex):
        mi = tokenizer(ex["input"], max_length=128, truncation=True)
        mi["labels"] = tokenizer(ex["output"], max_length=128, truncation=True)["input_ids"]
        return mi

    train_ds = train_ds.map(tok, batched=True, remove_columns=["input", "output"])
    val_ds = val_ds.map(tok, batched=True, remove_columns=["input", "output"])

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-3.3B", dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
    ))
    model.print_trainable_parameters()

    bleu = ev.load("sacrebleu")

    def compute_metrics(p):
        preds, lbls = p
        lbls = np.where(lbls != -100, lbls, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        lbls = tokenizer.batch_decode(lbls, skip_special_tokens=True)
        return {"bleu": bleu.compute(predictions=[x.strip() for x in preds], references=[[x.strip()] for x in lbls])["score"]}

    args = Seq2SeqTrainingArguments(
        output_dir=f"{RESULTS_DIR}/nllb-en-tn-sft",
        learning_rate=2e-4,
        per_device_train_batch_size=4, per_device_eval_batch_size=4,
        gradient_accumulation_steps=8, max_steps=max_steps,
        eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=200,
        logging_steps=10, predict_with_generate=True, bf16=True,
        load_best_model_at_end=True, metric_for_best_model="bleu",
        report_to="none", seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        processing_class=tokenizer, compute_metrics=compute_metrics,
    )

    trainer.train()
    save = f"{RESULTS_DIR}/nllb-en-tn-sft"
    trainer.save_model(save)
    tokenizer.save_pretrained(save)
    vol.commit()

    final = compute_metrics(trainer.predict(val_ds))
    log(f"SFT done — BLEU: {round(final[0]['bleu'], 2)}")
    return {"bleu": round(final[0]["bleu"], 2)}


@app.function(image=image, gpu="t4", timeout=3600 * 4, volumes={RESULTS_DIR: vol})
def generate_dpo():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from tqdm import tqdm

    log("Generating MSA rejected translations...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    tokenizer.src_lang = "eng_Latn"

    model = AutoModelForSeq2SeqLM.from_pretrained(
        f"{RESULTS_DIR}/nllb-en-tn-sft",
        dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    records = list({json.loads(l)["input"]: json.loads(l) for l in open(f"{RESULTS_DIR}/clean_en_tn_all.jsonl")}.values())

    triplets = []
    bs = 8
    for i in tqdm(range(0, len(records), bs)):
        batch = records[i:i+bs]
        en = [r["input"] for r in batch]
        tn = [r["output"] for r in batch]

        inputs = tokenizer(en, padding=True, truncation=True, max_length=128, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("arb_Arab"),
                max_length=128, num_beams=4, early_stopping=True,
            )
        msa = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for e, t, m in zip(en, tn, msa):
            m, t = m.strip(), t.strip()
            if m and t and m != t:
                triplets.append({"prompt": e, "chosen": t, "rejected": m})

    import random
    random.shuffle(triplets)
    s = int(len(triplets) * 0.9)
    for n, d in [("train", triplets[:s]), ("test", triplets[s:])]:
        with open(f"{RESULTS_DIR}/dpo_en_tn_{n}.jsonl", "w") as f:
            for item in d:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    vol.commit()
    log(f"DPO triplets: {len(triplets)} ({s} train, {len(triplets)-s} test)")
    return {"total": len(triplets)}


@app.function(image=image, gpu="t4", timeout=3600 * 6, volumes={RESULTS_DIR: vol})
def train_dpo(beta=0.3, max_steps=400):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig

    log(f"DPO β={beta}...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "aeb_Arab"

    train_ds = Dataset.from_list([json.loads(l) for l in open(f"{RESULTS_DIR}/dpo_en_tn_train.jsonl")])
    test_ds = Dataset.from_list([json.loads(l) for l in open(f"{RESULTS_DIR}/dpo_en_tn_test.jsonl")])

    base = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-3.3B", dtype=torch.bfloat16, device_map="auto",
    )
    base.enable_input_require_grads()

    model = get_peft_model(base, LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
    ))

    sft_path = f"{RESULTS_DIR}/nllb-en-tn-sft"
    if os.path.isdir(sft_path):
        log("Loading SFT LoRA weights...")
        model = PeftModel.from_pretrained(base, sft_path)
        model.enable_input_require_grads()

    args = DPOConfig(
        output_dir=f"{RESULTS_DIR}/nllb-en-tn-dpo-beta{beta}",
        beta=beta, learning_rate=5e-6,
        per_device_train_batch_size=4, per_device_eval_batch_size=4,
        gradient_accumulation_steps=8, max_steps=max_steps,
        logging_steps=10, eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=200,
        bf16=True, load_best_model_at_end=True,
        report_to="none", seed=42,
    )

    trainer = DPOTrainer(
        model=model, ref_model=None, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    save = f"{RESULTS_DIR}/nllb-en-tn-dpo-beta{beta}"
    trainer.save_model(save)
    tokenizer.save_pretrained(save)
    vol.commit()

    log(f"DPO β={beta} done")
    return {"beta": beta}


@app.function(image=image, gpu="t4", timeout=3600 * 2, volumes={RESULTS_DIR: vol})
def evaluate(model_path: str = ""):
    import torch
    pairs = [json.loads(l) for l in open(f"{RESULTS_DIR}/clean_en_tn_test.jsonl")]
    return _evaluate_model(model_path if model_path else f"{RESULTS_DIR}/nllb-en-tn-sft", pairs)


# ============================================================
# Orchestrator
# ============================================================

@app.local_entrypoint()
def main(mode="full"):
    log("=" * 60)
    log(f"EN→TN Dialect DPO (mode={mode})")
    log("=" * 60)

    csv_path = Path(__file__).parent / "tunsi_english_data.csv"
    csv_data = csv_path.read_text()
    log(f"CSV: {len(csv_data)} bytes")

    if mode == "eval":
        sft = evaluate.remote(model_path=f"{RESULTS_DIR}/nllb-en-tn-sft")
        log(f"SFT: {sft}")
        dpo_path = f"{RESULTS_DIR}/nllb-en-tn-dpo-beta0.3"
        if os.path.isdir(dpo_path):
            dpo = evaluate.remote(model_path=dpo_path)
            log(f"DPO: {dpo}")
        return

    prepare_data.remote(csv_data)
    train_sft.remote(max_steps=600)
    generate_dpo.remote()
    train_dpo.remote(beta=0.3, max_steps=400)
    sft = evaluate.remote(model_path=f"{RESULTS_DIR}/nllb-en-tn-sft")
    dpo = evaluate.remote(model_path=f"{RESULTS_DIR}/nllb-en-tn-dpo-beta0.3")
    log(f"FINAL — SFT: {sft}  |  DPO: {dpo}")
