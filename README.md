# UlsterGPT: A Tuition-Fee Assistant for Ulster University

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/beenishyousaf94-sketch/UlsterGPT/blob/main/UlsterGPT.ipynb)

UlsterGPT answers student questions about **Ulster University tuition fees and payment rules for 2026/27**: fees, deposits, instalment plans, payment methods and what happens if you don't pay. It combines a **QLoRA fine-tuned Mistral-7B** with **retrieval-augmented generation (RAG)**, so answers stay grounded in the university's published fee pages and Tuition Fees Payment Policy. Each answer cites its sources.

> ⚠️ This is a research and portfolio project, not official advice. Always confirm fees and deadlines with the Ulster University Fees Team.

---

## Highlights

- **Fine-tuned LLM:** Mistral-7B-Instruct-v0.2 trained with 4-bit QLoRA on 4,500 instruction examples.
- **Grounded answers:** BGE embeddings with a FAISS index retrieve the relevant policy text before generation.
- **Verified facts layer:** 50 hand-checked Q&As from the 2026/27 policy are added to the knowledge base, so official figures are always retrievable.
- **Evaluation:** 20 held-out, paraphrased questions scored for factual accuracy, ROUGE and BLEU, comparing the fine-tuned model with base Mistral using the same RAG.
- **Research pilot:** a controlled experiment on whether policy facts given to an LLM as **tables or as prose** change answer accuracy.
- **Demo:** an interactive Gradio chat interface.

---

## How it works

```
Student question
      │
      ▼
BGE-base embedding ──► FAISS search (top 15) ──► filter off-topic chunks ──► top 3 chunks
                                                                                   │
                                                                                   ▼
                                     Mistral-7B + QLoRA adapter  ◄── prompt: system + context + question
                                                                                   │
                                                                                   ▼
                                                                   Answer + cited sources
```

### 1. Data
| File | Contents |
|---|---|
| `ulster_fees_data.csv` | Text chunks scraped from Ulster University fee and payment pages (`url`, `page_title`, `chunk_text`) |
| `ulster_instructions.json` | 4,500 training examples: 4,000 generated from scraped text, plus 50 verified Q&As repeated ×10 |

The training examples cover instalment plans (including country-specific phrasings, because the same rule applies to every nationality), fee lookups, deposits and CAS, payment methods, and sanctions.

### 2. Fine-tuning (QLoRA)
| Setting | Value |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Quantisation | 4-bit NF4, double quantisation, bf16 compute |
| LoRA | r = 32, α = 16, dropout = 0.1, all attention and MLP projections |
| Training | 1 epoch, batch 4 × grad-accum 2, LR 2e-4 (cosine), max length 1024 |
| Optimiser | `paged_adamw_8bit` |
| Hardware | NVIDIA A100 80 GB (Google Colab) |

Only the LoRA adapter is saved.

### 3. Retrieval (RAG)
- Embedding model: `BAAI/bge-base-en-v1.5` (normalised, inner-product FAISS index).
- Knowledge base: the 50 verified policy facts plus all scraped chunks.
- Topic filtering: medicine-only rules (MBBS, clinical placement levy) are removed from the context unless the question is about medicine. This stops them leaking into general answers.

---

## Results

### Fine-tuned vs base model (20 held-out questions)
An answer counts as correct only if it contains all the required key facts (for example `£18,310`, or `01 December` and `01 March`).

| System | Factual accuracy | ROUGE-1 | ROUGE-L | BLEU |
|---|---|---|---|---|
| Base Mistral + RAG | 90% | 0.342 | 0.292 | 0.063 |
| **UlsterGPT (QLoRA + RAG)** | 90% | **0.380** | **0.330** | **0.099** |

Both systems reach the same **90% factual accuracy** (18/20). With good retrieval, the base model already finds the right facts. What fine-tuning adds is **closer alignment to reference answers**: ROUGE-1 +11%, ROUGE-L +13%, BLEU +57%. Answers are more concise and on-format. Retrieval drives correctness here; fine-tuning mostly shapes the style of the response.

### Pilot: does representation matter? (tables vs text)
The same verified 2026/27 facts were given to base Mistral-7B-Instruct (no fine-tuning, greedy decoding) **as prose** and **as Markdown tables**. It then answered 22 questions in three categories: lookup, conditional and arithmetic.

| Category | Text | Table |
|---|---|---|
| Lookup (8) | 100% | 100% |
| Conditional (8) | 100% | 100% |
| Arithmetic (6) | 66.7% | **83.3%** |
| **Overall** | 90.9% | **95.5%** |

Tables helped only on **multi-step arithmetic**, such as working out an instalment amount from a fee and a percentage. Paired comparison: one item was right only with the table and none was right only with the text (exact McNemar p = 1.0). This is a small pilot, so the difference is **not statistically significant**. It motivates a larger study, not a conclusion.

---

## Repository structure

```
UlsterGPT/
├── UlsterGPT.ipynb                    # full pipeline: data → fine-tune → RAG → eval → demo → pilot
├── requirements.txt
├── data/
│   ├── ulster_fees_data.csv
│   └── ulster_instructions.json
└── results/
    ├── evaluation_summary.csv
    ├── evaluation_results.csv
    ├── training_log.csv
    └── pilot_table_vs_text_results.csv
```

The trained LoRA adapter is hosted on Hugging Face: _[link to your adapter]_

---

## How to run

1. Open the notebook in Colab with the badge above. Choose **Runtime → Change runtime type → GPU** (A100 recommended; a T4 should work with smaller batch sizes).
2. Upload `data/ulster_fees_data.csv` to the Colab session.
3. Run the first cell to install packages, then **Runtime → Restart session**.
4. Run the remaining cells in order. The Gradio cell prints a public share link for the demo.

Running locally:
```bash
pip install -r requirements.txt
jupyter notebook UlsterGPT.ipynb
```
You need a CUDA GPU with at least 24 GB of memory for 4-bit training.

---

## Limitations

- Coverage is limited to fee information for **2026/27**, as published at the time of collection. Fees and policies change every year.
- The evaluation set is small (20 questions) and uses key-fact matching, which can miss correct paraphrases or accept partially wrong answers.
- Generated answers can still be long or include nearby but irrelevant rules from the retrieved context.
- Scraped content belongs to Ulster University and is used here for non-commercial research.

---

## Tech stack
Python · PyTorch · Hugging Face Transformers · PEFT · TRL · bitsandbytes · Sentence-Transformers · FAISS · Gradio · pandas · ROUGE / BLEU (NLTK)

## Author
**Beenish Yousaf**, MSc Computer Science and Technology with Business Development, Ulster University
[LinkedIn](https://www.linkedin.com/in/beenishyousaf)
