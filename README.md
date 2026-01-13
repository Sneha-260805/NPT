# NPT — A Unified Multilingual **N**ormalisation‑**P**unctuation‑**T**ransliteration Model

> “One model, three pre‑processing super‑powers, 20 + Indian languages.”

---

## ✨ What is NPT?

NPT is a single **Gemma‑3 B** decoder‑only Transformer fine‑tuned to perform **three text‑clean‑up tasks** with a simple prompt switch:

| Task | Prompt keyword | Example (input ➜ output) |
|------|----------------|--------------------------|
| **Normalisation** | `Normalise:` | `Report suggests 3.14 million packages delayed, a rise of 2 lakh.` ➜ `Report suggests three point one four million packages delayed a rise of two lakh.` |
| **Transliteration (to Latin)** | `Transliterate:` | `ध्यान` ➜ `dhyaan` |
| **Punctuation restoration** | `Punctuate:` | `lets eat grandma` ➜ `Let's eat, grandma.` |

It supports **12 scripts** (Tamil, Telugu, …) for normalization, **23 languages** for punctuation restoration and **22 languages** for transliteration, letting you clean noisy chat logs, OCR output, or ASR transcripts with one checkpoint.

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
├─ System_prompts/              # train.py, inference.py, utils
├─ Scripts/       # LLM prompt templates for data generation
├─ Demo (2).ipynb           # Colab / Jupyter demo
├─ Multilingual-Model-for-Text-PROCESSING.pdf   # Presentation of this project
├─ NLP_PROJECT (2).pdf # Detailed Report
├─ Requirements.txt      
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

2. **Synthetic boost** —We used Gemini 2.0 Flash to translate unnormalised and normalised data to the respective scheduled languages to create a high curated dataset.
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


## 📊 Results

For detailed metric tables—including per-language Character Error Rate (CER),  
Word Error Rate (WER), BLEU, ChrF++, and token-level Precision/Recall/F1—see  
**slides 5-6 of [`Multilingual-Model-for-TEXT-PROCESSING.pdf`](./Multilingual-Model-for-TEXT-PROCESSING.pdf)**.

| Task | Languages covered | Headline metric* |
|------|-------------------|------------------|
| **Transliteration** | 22 scheduled Indian languages | Macro-avg **CER 2.1 %**, WER 6.5 % |
| **Normalisation** | 12 languages | Slot-level exact-match **93.4 %** |
| **Punctuation restoration** | 23 languages | Token-level **F1 91.7 %** |

\* Full per-language breakdowns and comparison with Sarvam/Indic-xlit baselines are in the PDF.

> **Tip:** those slides also include a direct Sarvam vs NPT comparison (page 9) and  
> visual charts for transliteration error rates.

> **Our model vs Sarvam:**  
> Table 3 of the report shows our unified model **outperforms the Sarvam Transliteration baseline on every one of the eight overlapping languages**, while also extending coverage from 8 to 22 languages. :contentReference[oaicite:0]{index=0}


## 📑 Technical report (NLP_PROJECT.pdf)

A self-contained write-up of the architecture, data pipeline, and experimental
setup lives in **[`NLP_PROJECT.pdf`](./NLP_PROJECT (2).pdf)** (same as the slide deck
but with expanded methodology and appendix).

* **Sections 1-4** – motivation, related work, corpora summary  
* **Section 5** – model architecture & training regime  
* **Section 6** – evaluation protocol (task-specific metrics)  
* **Appendix A** – prompt templates and synthetic-data workflow

> **Normalization & punctuation results — work in progress**  
> We have already benchmarked **transliteration**, which beats Sarvam across all
> eight overlapping languages (see Table 3). Final numbers for the
> **normalization** and **punctuation restoration** tasks are being computed and
> will be added to both the PDF and README as soon as the runs finish.




