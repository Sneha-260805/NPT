# NPT — A Unified Multilingual **N**ormalisation‑**P**unctuation‑**T**ransliteration Model

> “One model, three pre‑processing super‑powers, 20 + Indian languages.”

---

## ✨ What is NPT?

NPT is a single **Gemma‑3 B** decoder‑only Transformer fine‑tuned to perform **three text‑clean‑up tasks** with a simple prompt switch:

| Task | Prompt keyword | Example (input ➜ output) |
|------|----------------|--------------------------|
| **Normalisation** | `Normalise:` | `pls send 5lakh by 5pm` ➜ `Please send ₹ 500 000 by 17:00.` |
| **Transliteration (to Latin)** | `Transliterate:` | `ध्यान` ➜ `dhyaan` |
| **Punctuation restoration** | `Punctuate:` | `lets eat grandma` ➜ `Let's eat, grandma.` |

It supports **12 scripts** (Devanagari, Tamil, Telugu, …) and **23 languages**, letting you clean noisy chat logs, OCR output, or ASR transcripts with one checkpoint.

---

## 🚀 Why another model?

* **One model ≘ one GPU** — no need to juggle task‑specific checkpoints.  
* **Shared learning** — transliteration helps punctuation, punctuation helps normalisation.  
* **Prompt‑conditioned** — adding a new task is as easy as defining a keyword and continuing training.

A detailed motivation, dataset description, and evaluation can be found in `Multilingual-Model-for-TEXT-PROCESSING.pdf`.

---

## 🗂️ Repository layout

```
NPT/
├─ scripts/              # train.py, inference.py, utils
├─ System_prompts/       # LLM prompt templates for data generation
├─ checkpoints/          # (ignored) fine-tuned weights land here
├─ Demo.ipynb            # Colab / Jupyter demo
└─ README.md             # you are here
```

---

## 📚 Dataset recipe (in brief)

1. **Public corpora** — Wiki Headlines, UD Treebanks, CFILT Indic-X.  
2. **Synthetic boost** — Gemini‑2.0‑Flash prompts in `System_prompts/normalization_synthetic_data.txt` generate *unnormalised ↔ normalised* pairs for low‑resource languages.  
3. **Merge utility** — (spec in `System_prompts/merge`) aligns parallel `.jsonl` trees into a single training file with `{task, unnormalised, normalised}` lines.

---

## 🛠️ Quick start

### 1. Install

```bash
git clone https://github.com/Sneha-260805/NPT
cd NPT
python -m venv .venv && source .venv/bin/activate        # optional
pip install -r requirements.txt                          # torch 2.2+, transformers 4.39+, datasets, wandb
```


### 3. Run inference

```bash
python scripts/inference.py   --checkpoint checkpoints/npt   --task normalization   --text "pls send 5lakh by 5pm"
```

Or open **`Demo (2).ipynb`** and try the three tasks interactively.

---

## 📊 Results (test‑set)

| Task | Metric | Score |
|------|--------|-------|
| Transliteration | Character Error Rate ↓ | **2.1 %** |
| Normalisation   | Slot‑level Exact Match ↑ | **93.4 %** |
| Punctuation     | F1 ↑ | **91.7 %** |

*(Macro‑averaged over 23 languages; full table in the PDF.)*



