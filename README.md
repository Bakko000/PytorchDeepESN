# Deep Reservoir Computing with Linear Readout for Big Data

A PyTorch implementation of **Deep Echo State Networks (DeepESN)** and related reservoir computing models, designed for efficient sequence representation learning with a **fixed recurrent backbone** and a **linear ridge readout**.

This repository focuses on the following idea:

- use a **reservoir backbone** as a nonlinear temporal feature extractor;
- keep the recurrent layers **untrained**;
- train only a **linear readout**, making the approach particularly attractive for large-scale settings where full backpropagation through time may be expensive.

The core implementation is provided in [`deepesn.py`](./deepesn.py).

---

## Overview

This repository implements:

- **single-layer ESN**
- **deep ESN**
- **bidirectional DeepESN**
- **ridge-based linear readout**
- support for **sequence-to-sequence** and **sequence-to-one/sequence-to-vector** outputs
- support for (mean)pooled/last-state outputs
- support for **large datasets** through a readout fitting procedure based on sufficient statistics

---

## Repository Structure

```text
.
├── deepesn.py      # main DeepESN model
├── reservoir.py    # reservoir cell and bidirectional wrapper
├── readout.py      # linear readout + ridge fitting utilities
└── ...
