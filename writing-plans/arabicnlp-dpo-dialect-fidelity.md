# Writing Plan — Preference Optimization for Dialect Fidelity (EN→Tunisian)

**Target venue:** ArabicNLP (SIGARAB), co-located with EMNLP/ACL.
**Status:** Resubmission after an ACL review. This plan re-frames the contribution,
grounds it in 2024–2026 literature, addresses every reviewer comment, and fixes the
data-validity problem uncovered during analysis.

---

## 0. TL;DR of the new strategy

1. **Kill the "first DPO for MT" claim** — it is false (DPO/CPO for MT is well established:
   ALMA-R/CPO, MBR-DPO, M²PO, CRPO). Re-position around a *narrow, true* contribution.
2. **New contribution:** an **annotation-free preference-optimization recipe for dialect
   *fidelity* control**, where the **rejected samples are the model's own MSA translations**
   (forced `arb_Arab` BOS) and the chosen samples are the Tunisian references. No human
   preference labels, no reward model.
3. **Lead with dialect fidelity, not BLEU.** Recent work (the "Accuracy Paradox") shows BLEU
   *rewards* MSA collapse and *penalizes* authentic dialect. This both motivates the paper and
   conveniently sidesteps the fact that our lexical data makes BLEU unreliable.
4. **Fix the data validity problem.** The cleaned MT splits are 53% single words (a lexicon,
   not MT). Build the sentence-level experiments on `drejja_clean_sentences.jsonl`
   (2,497 real EN→TN sentence pairs) and report data composition transparently.
5. **Target ArabicNLP**, not ACL main. It is the right home (dialectal scope, 41% accept rate,
   achievable bar) and the review we already got was about *rigor*, which is fixable.

---

## 1. Where to submit

**Primary: ArabicNLP (SIGARAB).** Scope explicitly includes dialectal Arabic; ArabicNLP 2025
(EMNLP 2025, Suzhou) accepted 39/95 (41%). EN→Tunisian dialect-fidelity work is squarely in
scope. Watch the **ArabicNLP 2026** CFP (likely co-located with EMNLP/ACL 2026; expect a
~Aug–Sep 2026 deadline — good timing from June 2026).

**Backups / alternates:**
- **WMT** research track or **LoResMT** / **VarDial** — dialect + low-resource MT fits.
- **EACL/ACL Findings** — only if eval is rigorous (fidelity classifier + human/LLM judge +
  COMET) and baselines are strong.

**Not recommended as first target:** ACL/EMNLP main. Single dialect + one method is thin;
would need multiple dialects, human eval, and a sharper novelty story.

---

## 2. Optimized framing & positioning (literature-grounded)

### 2.1 The one-sentence pitch
> Multilingual MT models collapse dialect requests toward MSA; we show that **preference
> optimization with self-generated MSA negatives** is a cheap, annotation-free way to steer
> English→Tunisian translation toward authentic dialect, and that this gain is only visible
> under **dialect-fidelity metrics**, not BLEU.

### 2.2 What is genuinely novel (defensible)
- **The negative-sampling trick:** use the *same* model's MSA output (forced `arb_Arab` BOS) as
  the `rejected` example and the dialect reference as `chosen`. This needs **no human preference
  annotation and no reward model** — distinct from CPO/ALMA-R (quality QE pairs) and
  MBR-DPO (MBR-selected pairs).
- **Objective = dialect *fidelity/register control*, not translation quality.** Prior PO-for-MT
  optimizes adequacy/quality; we optimize *which variety* the model speaks.
- **Under-studied direction & dialect:** EN→Tunisian (`aeb`). Most Tunisian MT work is the
  opposite direction (TN→MSA). Genuinely low-resource.

### 2.3 What to DROP or soften
- ❌ "First application of DPO to (dialectal) MT." → ✅ "We *adapt* preference optimization to the
  dialect-fidelity objective with self-supervised negatives."
- ❌ BLEU as the headline metric. → ✅ BLEU/chrF as *secondary*; fidelity + human/LLM judge primary.
- ❌ GRPO branding without rigor (the GRPO column in `final_sft_dpo_grpo.csv` is not yet credible).
  Either justify GRPO properly or cut it from the main story and keep DPO vs SFT vs baselines.

### 2.4 Positioning vs the closest related work (must cite & differentiate)
| Work | What it does | How we differ |
|---|---|---|
| **CPO / ALMA-R** (Xu et al., ICML 2024; 2401.08417) | PO for MT *quality* with QE/human pairs | We target *dialect fidelity*; negatives are self-generated MSA, no QE/reward model |
| **DPO-MBR for NMT** (2311.08380) | DPO pairs via MBR decoding | Different objective (quality vs variety); different, cheaper negatives |
| **M²PO / CRPO** (2510.13434; 2025.findings-acl.31) | Better reward signals/pairs for MT | Orthogonal; we use a single self-supervised contrast (dialect vs MSA) |
| **Context-Aware Dialectal Arabic MT** (2604.06456, 2026) | Controllable DA-MT via **rule-based data augmentation**; names the **"Accuracy Paradox"** | Concurrent; **PO instead of RBDA**. Use their Accuracy Paradox to justify our metrics. Cite as concurrent. |
| **DialUp** (Bafna et al., ACL 2025; 2501.16581) | Synthetic dialectal augmentation (M→D / D→M) | Baseline to compare against; we are augmentation-free |
| **Tradutor** (2502.14385) | Variety-specific MT (EU-Portuguese); **variety-classifier % as fidelity metric** | Precedent for our dialect-fidelity metric; we apply to Arabic + add PO |
| **AL-QASIDA** (2412.04193) | Systematic DA quality/accuracy analysis; variety ID | Methodological precedent for fidelity eval |

### 2.5 The "Accuracy Paradox" is our friend
Cite 2604.06456 and the dialect-metrics literature (2311.16865 — MT metrics on dialects w/o
standard orthography; CODET 2305.17267) to argue **BLEU/chrF are inadequate for unstandardized
dialects** and can *reward MSA collapse*. This (a) motivates a fidelity-first evaluation and
(b) pre-empts the reviewer who notices BLEU is low / unreliable on our lexical data.

---

## 3. The data problem and the fix (CRITICAL — do this first)

**Finding from analysis:**
- `clean_en_tn_{train,test}.jsonl`: target **median = 1 word**, **53–56% single-word**,
  94% ≤3 words. This is a **bilingual lexicon**, not sentence MT. BLEU here is near-meaningless.
- Raw `tunsi_english_data.csv` (13,037 rows): **66% single-word**, only **4% ≥4 words**.
- ✅ `drejja_clean_sentences.jsonl`: **2,497 genuine EN→TN sentence pairs** (EN mean 4.5 words).

**Status — `build_dataset.py` implemented (offline core run on local data):**
- Merges `drejja_clean_sentences.jsonl` + `tunsi_english_data.csv`; strips garbage/
  instruction leakage (`ترجم …`, 2,857 rows), dedups pairs + EN side (~3,000 rows),
  length-ratio filter. Output dir `data_v2/`, with per-row provenance in `*.meta.jsonl`.
- **Sentence track** (both sides multi-word): 2,694 → train 1,994 / val 300 / test 400.
- **Lexical track** (dictionary glosses): 5,797 (`lex_all.jsonl`).
- **Eval set is 100% authentic `drejja` sentences with ≥3-word targets** (reliable BLEU/chrF);
  synthetic-English rows are auto-flagged `needs_human_verification`.
- Optional, network-gated filters wired but not yet run: `--semantic_filter` (LaBSE alignment,
  drops bad LLM pairs) and `--dialect_filter` (CAMeL DID classifier, drops MSA-leaning targets).
- **Pending the HF parallel dataset** (`--hf_dataset <id>`, `en_origin=synthetic_en`) once
  `huggingface.co` is allowlisted — it will become the dominant sentence-level source.

**Plan:**
1. **Define two tasks/sets explicitly and report both transparently:**
   - **(A) Sentence-level EN→TN** — train/eval primarily on `drejja_clean_sentences.jsonl`
     (+ any ≥4-word pairs mined from the CSV: ~498). This is the honest "MT" track.
   - **(B) Lexical/term-level EN→TN** — the dictionary-style pairs, reported separately as
     *dialectal term translation* (still useful, but labeled correctly).
2. **Report data composition** in a table (length buckets, % single-word) — reviewers *will*
   check; owning it is far better than being caught.
3. **Hold out a clean sentence-level test set** for the headline results; bucket results by
   target length so single-word artifacts don't dominate.
4. If sentence data is too small for stable training, frame scope precisely ("word- and
   phrase-level dialect translation with a sentence-level evaluation subset") rather than
   overclaiming sentence MT.

---

## 4. Reviewer feedback → concrete actions (from `paper_revision_outline.md`)

| # | Reviewer ask | Action | Owner file |
|---|---|---|---|
| 1 | Use NLLB-200 **3.3B** not 600M | Done in `modal_run.py` / `train_sft.py` | training scripts |
| 2 | Eval = BLEU + chrF++ + **COMET** (not just PPL) | Add COMET to `evaluate_all.py`; keep as *secondary* metrics | `evaluate_all.py` |
| 3 | **Dialect fidelity** classifier (VID-style) | Use CAMeL-Lab DID classifier → % TUN vs MSA; report as **primary** (cf. Tradutor TDR) | `evaluate_all.py`, `modal_run.py` |
| 4 | Remove T5 paraphrase augmentation | Confirm removed; document | data section |
| 5 | Fix data section (sizes, script, MSA-rejected provenance) | Rewrite with §3 numbers; specify Arabic script, `arb_Arab` BOS negatives | paper |
| 6 | Measure **catastrophic forgetting** properly | `eval_forgetting.py`: EN→MSA PPL↑ (style rejection), MSA→EN & FR→EN PPL≈ (no forgetting) | `eval_forgetting.py` |
| 7 | **Baselines**: variety-tag, DialUp, classifier-guided, IPO, ORPO | Implement tag + DialUp + IPO/ORPO via TRL; classifier-guided optional | `baselines.py`, `train_dpo.py` |
| 8 | **Ablations**: β {0.1,0.3,0.5,0.7}; LoRA r {16,32,64}; DPO epochs | Sweep (already scaffolded in `train_dpo.py --sweep`) | `train_dpo.py` |
| 9 | **Stress tests**: beams {1,4,8}, T {0.7,0.9,1.0}, seeds {42,123,456}, hallucination rate | `stress_test.py` | `stress_test.py` |
| 10 | **Related Work** engagement | Write full §2 using bibliography in §7; engage (don't just cite) | paper |
| 11 | Operationalize Table 1 (replace vague labels) | Dialect-classifier acc, BLEU, PPL(TN)/PPL(MSA) ratio | paper |
| 12 | **Human / LLM-as-judge** eval (fluency, adequacy, dialect fidelity 1–5 + IAA) | `llm_judge_eval.py`; if possible, ≥1 native annotator on a 100-item subset | `llm_judge_eval.py` |
| 13 | Reproducibility appendix | Hyperparams, hardware (T4/L4), training time, seeds, licenses | appendix |

---

## 5. Experimental matrix (headline table)

Primary metric **bold** = dialect fidelity. Report on the **sentence-level** test set, bucketed by length.

| System | Dialect Acc (TUN%) ↑ | LLM-judge fidelity ↑ | chrF++ | BLEU | COMET | PPL(TN)↓ | PPL(MSA)↑ |
|---|---|---|---|---|---|---|---|
| NLLB-200 3.3B (zero-shot) | | | | | | | |
| SFT (LoRA) | | | | | | | |
| SFT + variety tag | | | | | | | |
| SFT + DialUp aug | | | | | | | |
| **DPO (ours, MSA-negatives)** | | | | | | | |
| IPO | | | | | | | |
| ORPO | | | | | | | |

Plus: β/rank/epoch ablations; stress-test (beam/temp/seed) hallucination rates; forgetting table.

**Expected story:** DPO yields the **largest dialect-fidelity gain** at a **small chrF/BLEU
cost**, beating tag/DialUp baselines on fidelity — i.e., preference optimization is competitive
with augmentation-heavy methods while being simpler and annotation-free.

---

## 6. Paper structure & section-by-section writing plan

- **Title (candidates):**
  - "Speak Tunisian, Not MSA: Preference Optimization for Dialect Fidelity in English→Tunisian MT"
  - "Self-Supervised Preference Optimization for Dialectal Machine Translation"
- **Abstract** — problem (MSA collapse) → method (self-negative DPO) → result (fidelity↑ at small
  BLEU cost) → metric insight (Accuracy Paradox).
- **§1 Introduction** — MSA-collapse problem; the metric paradox; contributions (3 bullets, no
  "first" overclaim).
- **§2 Related Work** — (a) PO for MT [CPO/ALMA-R, MBR-DPO, M²PO, CRPO]; (b) dialect MT &
  control [Context-Aware DA-MT/RBDA, DialUp, variety tags]; (c) dialect-fidelity evaluation
  [Tradutor, AL-QASIDA, CODET, no-standard-orthography metrics].
- **§3 Data** — provenance, cleaning, **composition table (§3)**, sentence vs lexical split,
  MSA-negative construction.
- **§4 Method** — SFT (LoRA on NLLB-3.3B); **self-supervised preference pairs**; DPO/IPO/ORPO
  objectives; why MSA-negatives.
- **§5 Evaluation setup** — metrics (fidelity primary; chrF/COMET secondary; PPL; LLM/human
  judge), why BLEU is secondary (cite paradox + no-orthography metrics).
- **§6 Results** — main table; ablations; stress tests; forgetting.
- **§7 Analysis** — qualitative examples (clean ones — current outputs leak "ترجم"; see §8.2),
  fidelity-vs-quality trade-off curve over β.
- **§8 Conclusion + Limitations** — single dialect, small sentence corpus, no native human eval
  (if so), lexical-data caveat.
- **Appendix** — reproducibility (§4 row 13).

---

## 7. Key bibliography (verified, 2023–2026)

**PO for MT:** Rafailov et al. 2023 DPO (arXiv:2305.18290); Xu et al. 2024 CPO/ALMA-R
(2401.08417, ICML); DPO-MBR for NMT (2311.08380); M²PO (2510.13434); CRPO (2025.findings-acl.31);
Backtranslation-augmented DPO for NMT (2604.25702).
**Dialect MT & control:** Context-Aware Dialectal Arabic MT / Accuracy Paradox / RBDA
(2604.06456, 2026 — concurrent); DialUp (2501.16581, ACL 2025); OSACT 2024 Task 2 DA→MSA
(2024.osact-1.12); Tunisian parallel resources (2020.wanlp-1.18).
**Dialect-fidelity evaluation:** Tradutor / variety classifier metric (2502.14385); AL-QASIDA
(2412.04193); CODET (2305.17267); MT metrics on dialects w/o standard orthography
(2311.16865, WMT 2023); Korean dialect steering / TDR (2511.06680).
**Backbone:** NLLB-200 (NLLB Team, 2022).

> Action: pull exact BibTeX from ACL Anthology / arXiv; verify the 2604.* IDs resolve (they are
> 2026 preprints surfaced in search) before citing as published.

---

## 8. Risks & immediate next steps

### 8.1 Risks
- **Sentence corpus is small (~2.5k).** Mitigate: combine with lexical data for training but
  evaluate on sentences; consider back-translation augmentation; be explicit about scope.
- **No native human eval.** Mitigate: LLM-as-judge with reported IAA; recruit ≥1 native speaker
  for a 100-item subset if at all possible (greatly strengthens acceptance odds).
- **Concurrent competitor (2604.06456).** Mitigate: cite as concurrent; differentiate (PO vs RBDA)
  and lean on the shared Accuracy-Paradox framing as validation, not threat.
- **GRPO results not credible yet.** Mitigate: drop from main claims or run properly.

### 8.2 Immediate next steps (in order)
1. **Re-clean data** into sentence-level (drejja + CSV ≥4w) vs lexical sets; produce the
   composition table. *(blocks everything)*
2. **Re-run SFT → MSA-negative generation → DPO** on the sentence set with the now-fixed
   `modal_run.py` (LoRA loading, metrics, DPO adapter bugs already fixed).
3. **Wire fidelity + COMET + LLM-judge** into `evaluate_all.py`; run the main table.
4. Run β/rank ablations + stress tests + forgetting.
5. Draft §2 Related Work and §3 Data first (least dependent on final numbers).
6. Investigate the **"ترجم" leakage** in sample outputs (instruction-word contamination) before
   producing qualitative examples — likely a decoding/forced-BOS or residual-instruction issue.

---

*Plan generated 2026-06-16. Literature grounded via web search (June 2026).*
