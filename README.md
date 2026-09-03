CNN/DailyMail Text Summarization

An end-to-end NLP project — from raw data to a fine-tuned, benchmarked, deployment-ready summarization model. Built with an emphasis on engineering rigor and documented decision-making, not just chasing a leaderboard number.

Live demo: Streamlit app →  ·  Model: BART on Hugging Face →  ·  Code: GitHub →

Task: summarize news articles. Models evaluated: TextRank (extractive baseline), fine-tuned BART-base, fine-tuned T5-small, and Pegasus (zero-shot specialist model). Deployed model: BART (fine-tuned) — see Deployment for why Pegasus is evaluated but not shipped in the live app.

Results at a Glance
Metric	TextRank	BART (final, deployed)	T5	Pegasus (zero-shot, eval only)
ROUGE-1	0.3008	0.3510	0.3254	0.3521
ROUGE-2	0.1221	0.1436	0.1244	0.1468
ROUGE-L	0.2556	0.2565	0.2307	0.2632
BERTScore F1	—	0.8612	—	0.8754

Fine-tuned BART beats the extractive baseline on every metric and is competitive with T5 (same scale, same fine-tuning budget). Pegasus — pretrained specifically for summarization and fine-tuned on the full dataset — leads across the board, for reasons explained in Model Selection Rationale, not ignored.

Deployment

Live app: a Streamlit interface where a pasted article is summarized by the fine-tuned BART model, with adjustable generation controls (max length, beam width) and two built-in example articles for quick testing.

Pegasus is evaluated throughout this project but is not part of the deployed app. google/pegasus-cnn_dailymail uses a SentencePiece-based tokenizer that fails to convert on Streamlit Community Cloud's current Python 3.14 / transformers environment (tiktoken-based fast-tokenizer conversion raises a ValueError specific to that platform — reproducible, not intermittent). Rather than pin an older transformers version across the whole app to work around one model's tokenizer and risk destabilizing BART inference in the process, I shipped BART-only to the live demo and kept the full BART/Pegasus/T5 comparison in the documented evaluation results above, computed separately on Kaggle/Colab where this conflict doesn't occur. This is a deliberate scope decision, not an oversight — see the engineering log below for the debugging path that led to it.

Stack: Streamlit (UI) · Hugging Face Hub (model hosting) · GitHub (CI: push → auto-redeploy)

Pipeline Architecture
Data → Preprocessing → Model Loading → Training → Evaluation → Inference → Deployment

Each stage is its own module (not a single notebook), so any part — data prep, a specific model, a metric — can be reused or swapped independently.

project/
├── configs/                     # YAML configs — one per experiment, fully reproducible
├── src/
│   ├── utils/                    # config loading, tokenization, dynamic-padding collator
│   ├── models/                   # BART & Pegasus loading (device-aware)
│   └── eval/                      # ROUGE and BERTScore scoring
├── train.py                      # config-driven training entry point
├── inference.py                   # production inference layer (BART)
├── app.py                         # Streamlit UI, deployed to Streamlit Community Cloud
└── README.md

No hyperparameter is hardcoded anywhere in the codebase — every experiment is fully reproducible from its config file alone.

Data & Baseline

Dataset: abisee/cnn_dailymail v3.0.0 — 287k train / 13k val / 11k test. Articles average 615 words; ~5% exceed BART's 1,024-token input limit and get truncated (documented limitation, not silently ignored). A 2,000-example quality check found 0 degenerate reference summaries — no cleaning step was needed, a conclusion based on evidence rather than assumption.

Baseline (TextRank, extractive, non-neural): built first, deliberately, as a control group — the question it answers is "does fine-tuning actually beat a free algorithm with zero training?" ROUGE-1: 0.30 / ROUGE-2: 0.12 / ROUGE-L: 0.26.

<details> <summary><strong>Full experiment log — 4 training runs (click to expand)</strong></summary>
Experiment	Train size	Epochs	LR	ROUGE-1	ROUGE-2	ROUGE-L
Baseline	5,000	3	5e-5	0.3421	0.1342	0.2415
LR variation	5,000	3	3e-5	0.3470	0.1392	0.2479
Full-scale	180,000	1	5e-5	0.3313	0.1305	0.2437
Best (final)	20,000	3	5e-5	0.3510	0.1436	0.2565

Key finding: the largest dataset (180k) underperformed a 9x-smaller dataset trained for 3x more epochs — validation loss on the 180k/1-epoch run hadn't converged. Epoch count mattered more than raw data volume, a genuine empirical finding, not an assumption carried in from the start.

</details>
Why I Stopped Optimizing ROUGE (Model Selection Rationale)

After four isolated experiments (one variable changed at a time), I deliberately stopped tuning further. The evidence:

Learning rate (5e-5 vs 3e-5): <1% difference — within normal noise.
Dataset size (5k vs 20k vs 180k): no monotonic relationship with performance; the best result came from a mid-sized dataset with more epochs, not the largest one.
BART vs T5, same scale: near-identical performance — consistent with both being general-purpose architectures with no summarization-specific pretraining advantage over each other.
The gap to Pegasus is consistent across every metric — that's not a hyperparameter problem. Pegasus was pretrained with a Gap Sentence Generation objective built specifically to simulate summarization, and fine-tuned on the full 287k dataset, not a subset.

Conclusion: further hyperparameter search on this architecture/scale had a low probability of closing that gap, at a real cost in limited GPU quota (Kaggle: 30 hrs/week) — time better spent finishing the inference pipeline and deployment. What would close the gap: starting from a summarization-pretrained checkpoint (e.g. distilbart-cnn) instead of general-purpose bart-base, training on the full dataset with a validated sufficient epoch count, or moving to bart-large. Each is a materially different investment than "try another learning rate" — and identifying which lever actually matters, rather than pulling the same one repeatedly, was the point of stopping here.

Known Limitations
Truncation: ~5% of articles exceed the 1,024-token limit; content past that point is never seen by the model.
Training scale: final model trained on 20k of 287k available examples, due to free-tier GPU quota constraints.
Base model size: bart-base (~140M params) chosen over bart-large (~400M) for memory reasons on free-tier T4 GPUs — caps achievable ROUGE relative to published bart-large-cnn results (~43 ROUGE-1).
No automated factual-consistency check — ROUGE/BERTScore measure overlap, not truth; only manual spot-checks were performed.
Domain-specific: trained only on CNN/DailyMail news text; untested on other domains.
Pegasus is not in the live app (see Deployment) due to a platform-specific tokenizer conversion failure — it remains fully evaluated in the offline comparison above.
A transformers-version bug (below) blocks automatic best-checkpoint selection; evaluation is done manually post-training instead.
Engineering Challenges (selected)
Seq2SeqTrainer.compute_metrics silently never invoked (transformers 5.x): confirmed via debug logging that the custom ROUGE function never fired despite correct config. Ruled out three hypotheses (tuple-wrapped predictions, multi-GPU gather failure, library version) with evidence before deciding not to keep debugging a framework internal — decoupled evaluation from the Trainer entirely and compute ROUGE manually post-training instead.
Pegasus tokenizer failure on Streamlit Community Cloud: AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail") raised a tiktoken-conversion ValueError specific to that platform's Python 3.14 / transformers combination (works correctly on Colab/Kaggle). Diagnosed through the dependency-installation logs rather than the app traceback, added the missing tiktoken and sentencepiece packages, then confirmed the remaining failure was a genuine slow→fast tokenizer conversion incompatibility, not a missing dependency. Scoped the fix to removing Pegasus from the deployed app rather than pinning framework versions app-wide, to avoid destabilizing the working BART path for the sake of a secondary model.
Session/checkpoint loss on Colab and Kaggle (ephemeral storage resets on disconnect) — solved by integrating Hugging Face Hub (push_to_hub=True) for durable checkpointing, and recovered one partially-lost run via resume_from_checkpoint.
Multi-GPU (Kaggle T4 x2) changes effective batch size and step counts vs. single-GPU runs — required recalculating training-time estimates from empirical throughput rather than reusing single-GPU numbers.
Dataset repo migration (cnn_dailymail → abisee/cnn_dailymail) mid-project — a reminder that hardcoded dataset paths are a real, recurring maintenance risk.

Requires a Hugging Face HF_TOKEN (write access) for Hub checkpointing during training, provided via secrets/env — never hardcoded.

Tech Stack

transformers (BART, T5, Pegasus) · PyTorch · datasets · rouge_score / bert-score · sumy (TextRank) · Hugging Face Hub · Streamlit · Colab & Kaggle (T4) · Git

Future Work
Fine-tune from a summarization-pretrained checkpoint (distilbart-cnn) instead of general-purpose bart-base.
Full-dataset training with a validated sufficient epoch count.
Resolve the Pegasus tokenizer conflict on Streamlit Cloud (likely via a pinned, isolated environment) and restore it to the live demo.
Systematic error categorization (hallucination / repetition / truncation-loss) on a larger sample.
Out-of-distribution robustness testing (short inputs, non-news text).