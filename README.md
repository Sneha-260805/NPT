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

It supports **12 scripts** (Devanagari, Tamil, Telugu, …) for normalization, **23 languages** for punctuation restoration and **22 languages** for transliteration, letting you clean noisy chat logs, OCR output, or ASR transcripts with one checkpoint.

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

1. 1. **Public corpora**

   - **IndicCorp V2** — large monolingual dump covering 23 Indic languages.  
   - **BPCC (Bharat Parallel Corpus Collection)** — sentence-aligned parallel data for every scheduled Indian language.  
   - **IndicLLMSuite / IndicAlign** — high-quality transliteration triples used as gold supervision.  
   - **Mark-My-Words** — multilingual punctuation-restoration corpus.  
   - **Updesh-Beta** — multilingual question-answering & reasoning dataset used for robustness evaluation.

2. **Synthetic boost** — Gemini‑2.5‑Flash prompts in `System_prompts/normalization_synthetic_data.txt` generate *unnormalised ↔ normalised* pairs for low‑resource languages.  
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


### 2. **Run the interactive demo (recommended)**

Open **`Demo (2).ipynb`** locally (Jupyter Lab / VS Code) or in Google Colab:

1. **Load the checkpoint** – execute the first cell to mount (or download) `checkpoints/npt`.
2. **Pick a task keyword** – choose **`Normalise:`**, **`Transliterate:`**, or **`Punctuate:`** in the prompt cell.
3. **Run the examples** – watch the model clean up the sample sentences.
4. **Experiment** – paste your own text, adjust decoding parameters, and see live results.

> **Why start here?** The notebook shows all three tasks side-by-side, prints metrics, and lets you iterate without memorising CLI flags.






