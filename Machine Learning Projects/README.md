# Machine Learning Mini-Project

# What Makes a Long-Context Monitor Fire Falsely?

**Authors:** Lucas Sebastian Hübner and Max Elliot Hübner

This repository contains the code and experiments for a small mechanistic-interpretability project studying how a linear activation monitor trained on short examples behaves when applied repeatedly inside longer contexts.

## Research question

We study a linear probe intended to detect:

> **“The user is trying to gain unauthorized access.”**

The probe is trained on residual-stream activations from short, single-segment examples. We then apply the same frozen probe and threshold to 33-segment contexts using max pooling,

\[
s(h)=w^\top h+b,
\qquad
s_{\mathrm{ctx}}=\max_k s(h_k).
\]

The benchmark contains:

- positive examples expressing unauthorized-access intent;
- benign **near misses**, such as quotations, classification tasks, policy discussion, and translation;
- unrelated neutral examples.

## Main findings

The short-context monitor performs well, but its false-positive rate increases strongly when many benign near-miss segments are placed in a long context.

In the strengthened Gemma 2 2B experiment:

- 504 distinct benign one-segment evaluation examples produced **0 false positives** in isolation;
- long contexts contain 33 segments;
- 1,020 long contexts are evaluated across near-miss counts
  \(m=0,2,\ldots,32\);
- the false-positive rate of near-miss contexts rises sharply as \(m\) increases, eventually approaching 1;
- holding the near-miss sentence fixed while increasing the amount of preceding neutral context produces a systematic positive shift in its probe score.

Thus ordinary resampling of the measured isolated score distribution does not explain the observed long-context failure in this benchmark. The monitor score itself is context dependent.

## SAE analysis

We additionally use a Gemma Scope sparse autoencoder to investigate whether the false positives can be attributed to a small number of interfering features.

The SAE has:

- transformer residual dimension: **2,304**;
- SAE dictionary size: **16,384**;
- overcompleteness factor: approximately **7.1**.

The SAE reconstructs false-positive activations about as well as other prediction classes, but we do **not** find evidence that a single SAE feature, or a small handful of SAE features, uniquely explains the false positives.

An \(m\)-matched comparison of false positives against true negatives instead suggests that the excess score is distributed across many SAE features. This disfavors a simple sparse-feature interference explanation, although more distributed mechanisms remain possible.

## Repository structure

```text
Mini-Project/
├── notebooks/
│   └── 00_main_research_workbook.ipynb
├── src/
│   └── long_context_monitor_lab/
├── tests/
├── data/
├── figures/
├── requirements.txt
└── README.md
