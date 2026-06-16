"""
Modal experiment runner for EN→TN MT + DPO on L4 GPU ($0.80/hr, 24GB VRAM).
Optimized for speed: larger batches, bf16, gradient checkpointing.
Usage:
    modal run modal_run.py               # Full pipeline (sentence-level data)
    modal run modal_run.py --mode eval   # Eval SFT + DPO
    modal run modal_run.py --mode resume # Resume from DPO data gen
"""

import json
import os
import re
from pathlib import Path

import modal

app = modal.App("en-tn-dialect-dpo-l4")

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


def _load_model(model_path):
    """Load a translation model, handling LoRA-adapter checkpoints."""
    import torch
    from transformers import AutoModelForSeq2SeqLM

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        from peft import PeftModel
        base = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-3.3B", dtype=torch.bfloat16, device_map="auto",
        )
        return PeftModel.from_pretrained(base, model_path)

    return AutoModelForSeq2SeqLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
    )


def _prepare_data():
    """Load sentence-level data from data_v2/ (already cleaned by build_dataset.py)."""
    import json

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        src = f"data_v2/sent_{split}.jsonl"
        dst = f"{RESULTS_DIR}/clean_en_tn_{split}.jsonl"
        if os.path.exists(src):
            with open(src) as f_in, open(dst, 'w') as f_out:
                for line in f_in:
                    d = json.loads(line)
                    f_out.write(json.dumps({"input": d["input"], "output": d["output"]}, ensure_ascii=False) + "\n")
            count = sum(1 for _ in open(dst))
            log(f"  {split}: {count} pairs")

    log("Data prepared from data_v2/ sentence-level splits")
    return {"train": 1994, "val": 300, "test": 400}


def _evaluate_model(model_path, pairs):
    import torch
    import evaluate as ev
    from transformers import AutoTokenizer, pipeline

    log(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    model = _load_model(model_path)
    model.eval()
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "aeb_Arab"

    sacrebleu = ev.load("sacrebleu")

    bs = 16  # L4 can handle larger batches
    preds = []
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i+bs]
        en = [p["input"] for p in batch]
        inputs = tokenizer(en, padding=True, truncation=True, max_length=128, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
        preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    log("  Samples:")
    for i in range(min(5, len(pairs))):
        log(f"    EN: {pairs[i]['input']}")
        log(f"    REF: {pairs[i]['output']}")
        log(f"    PRED: {preds[i]}")

    refs = [p["output"] for p in pairs]
    bleu = sacrebleu.compute(predictions=preds, references=[[r] for r in refs])["score"]

    # PPL
    losses = []
    for i in range(0, min(len(pairs), 200), bs):
        batch = pairs[i:i+bs]
        inp = tokenizer([p["input"] for p in batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(model.device)
        lbl = tokenizer([p["output"] for p in batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(model.device).input_ids
        with torch.no_grad():
            losses.append(model(**inp, labels=lbl).loss.item())
    ppl = float(torch.exp(torch.tensor(sum(losses) / len(losses))))

    # Dialect classifier
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
# Functions — L4 optimized
# ============================================================

@app.function(image=image, timeout=300, volumes={RESULTS_DIR: vol})
def prepare_data():
    result = _prepare_data()
    vol.commit()
    return result


@app.function(image=image, gpu="L4", timeout=3600 * 4, volumes={RESULTS_DIR: vol})
def train_sft(max_steps=400):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer,
        Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
    )
    import evaluate as ev
    import numpy as np

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log("Loading NLLB-200 3.3B + LoRA (L4 optimized)...")

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "aeb_Arab"

    def load_split(name):
        return Dataset.from_list([json.loads(l) for l in open(f"{RESULTS_DIR}/clean_en_tn_{name}.jsonl")])

    train_ds, val_ds = load_split("train"), load_split("val")
    log(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

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
        r=64, lora_alpha=128, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    ))
    model.print_trainable_parameters()

    bleu = ev.load("sacrebleu")

    def compute_metrics(p):
        preds, lbls = p
        lbls = np.where(lbls != -100, lbls, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        lbls = tokenizer.batch_decode(lbls, skip_special_tokens=True)
        return {"bleu": bleu.compute(predictions=[x.strip() for x in preds], references=[[x.strip()] for x in lbls])["score"]}

    # L4 optimized: larger batch, more grad accum, fewer steps (smaller dataset)
    args = Seq2SeqTrainingArguments(
        output_dir=f"{RESULTS_DIR}/nllb-en-tn-sft-l4",
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # effective batch = 32
        max_steps=max_steps,
        eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_steps=100,
        logging_steps=10,
        predict_with_generate=True,
        bf16=True,
        gradient_checkpointing=True,  # save VRAM
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        report_to="none",
        seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        processing_class=tokenizer, compute_metrics=compute_metrics,
    )

    trainer.train()
    save = f"{RESULTS_DIR}/nllb-en-tn-sft-l4"
    trainer.save_model(save)
    tokenizer.save_pretrained(save)
    vol.commit()

    final_bleu = trainer.predict(val_ds).metrics["test_bleu"]
    log(f"SFT done — BLEU: {round(final_bleu, 2)}")
    return {"bleu": round(final_bleu, 2)}


@app.function(image=image, gpu="L4", timeout=3600 * 2, volumes={RESULTS_DIR: vol})
def generate_dpo():
    import torch
    from transformers import AutoTokenizer
    from tqdm import tqdm

    log("Generating MSA rejected translations (L4)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    tokenizer.src_lang = "eng_Latn"

    model = _load_model(f"{RESULTS_DIR}/nllb-en-tn-sft-l4")
    model.eval()

    records = list({json.loads(l)["input"]: json.loads(l) for l in open(f"{RESULTS_DIR}/clean_en_tn_train.jsonl")}.values())
    log(f"Generating for {len(records)} sentence pairs...")

    triplets = []
    bs = 32  # L4: 4x larger batch
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


@app.function(image=image, gpu="L4", timeout=3600 * 3, volumes={RESULTS_DIR: vol})
def train_dpo(beta=0.3, max_steps=300):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig

    log(f"DPO β={beta} (L4 optimized)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "aeb_Arab"

    train_ds = Dataset.from_list([json.loads(l) for l in open(f"{RESULTS_DIR}/dpo_en_tn_train.jsonl")])
    test_ds = Dataset.from_list([json.loads(l) for l in open(f"{RESULTS_DIR}/dpo_en_tn_test.jsonl")])
    log(f"DPO train: {len(train_ds)}, test: {len(test_ds)}")

    base = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-3.3B", dtype=torch.bfloat16, device_map="auto",
    )
    base.config.use_cache = False

    sft_path = f"{RESULTS_DIR}/nllb-en-tn-sft-l4"
    if os.path.isdir(sft_path):
        log("Loading SFT LoRA weights for continued training...")
        model = PeftModel.from_pretrained(base, sft_path, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=64, lora_alpha=128, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
        ))
    model.enable_input_require_grads()

    # L4 optimized: larger batch, gradient checkpointing
    args = DPOConfig(
        output_dir=f"{RESULTS_DIR}/nllb-en-tn-dpo-beta{beta}-l4",
        beta=beta, learning_rate=5e-6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # effective batch = 16
        max_steps=max_steps,
        logging_steps=10,
        eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        report_to="none", seed=42,
    )

    trainer = DPOTrainer(
        model=model, ref_model=None, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    save = f"{RESULTS_DIR}/nllb-en-tn-dpo-beta{beta}-l4"
    trainer.save_model(save)
    tokenizer.save_pretrained(save)
    vol.commit()

    log(f"DPO β={beta} done")
    return {"beta": beta}


@app.function(image=image, gpu="L4", timeout=3600 * 1, volumes={RESULTS_DIR: vol})
def evaluate(model_path: str = ""):
    path = model_path if model_path else f"{RESULTS_DIR}/nllb-en-tn-sft-l4"
    if not os.path.isdir(path):
        log(f"Model not found: {path} — skipping eval")
        return {"error": f"not found: {path}"}
    pairs = [json.loads(l) for l in open(f"{RESULTS_DIR}/clean_en_tn_test.jsonl")]
    return _evaluate_model(path, pairs)


# ============================================================
# Orchestrator
# ============================================================

@app.local_entrypoint()
def main(mode="full"):
    log("=" * 60)
    log(f"EN→TN Dialect DPO — L4 Optimized (mode={mode})")
    log("=" * 60)

    if mode == "eval":
        sft = evaluate.remote(model_path=f"{RESULTS_DIR}/nllb-en-tn-sft-l4")
        log(f"SFT: {sft}")
        dpo = evaluate.remote(model_path=f"{RESULTS_DIR}/nllb-en-tn-dpo-beta0.3-l4")
        log(f"DPO: {dpo}")
        return

    if mode == "resume":
        log("Resuming: skipping prepare_data and train_sft (already on volume)")
        log("Step 3/4: Generating DPO data...")
        generate_dpo.remote()
        log("Step 4/4: Training DPO...")
        train_dpo.remote(beta=0.3, max_steps=300)
        log("Evaluating...")
        sft = evaluate.remote(model_path=f"{RESULTS_DIR}/nllb-en-tn-sft-l4")
        log(f"SFT: {sft}")
        dpo = evaluate.remote(model_path=f"{RESULTS_DIR}/nllb-en-tn-dpo-beta0.3-l4")
        log(f"DPO: {dpo}")
        log(f"FINAL — SFT: {sft}  |  DPO: {dpo}")
        return

    # Full pipeline from scratch
    log("Step 1/4: Preparing sentence-level data...")
    prepare_data.remote()
    log("Step 2/4: Training SFT (L4, 400 steps)...")
    train_sft.remote(max_steps=400)
    log("Step 3/4: Generating DPO data...")
    generate_dpo.remote()
    log("Step 4/4: Training DPO (L4, 300 steps)...")
    train_dpo.remote(beta=0.3, max_steps=300)
    log("Evaluating...")
    sft = evaluate.remote(model_path=f"{RESULTS_DIR}/nllb-en-tn-sft-l4")
    log(f"SFT: {sft}")
    dpo = evaluate.remote(model_path=f"{RESULTS_DIR}/nllb-en-tn-dpo-beta0.3-l4")
    log(f"DPO: {dpo}")
    log(f"FINAL — SFT: {sft}  |  DPO: {dpo}")
