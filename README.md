# Large-Scale Multi-Source Business Entity Resolution
# Large-Scale Multi-Source Business Entity Resolution

## Project Mission
In modern commercial data ecosystems, business identity records originate from disparate, heterogeneous sources—each contributing incomplete, noisy fragments of real-world business entities. These fragments share no shared keys or global identifiers. The objective of this project is to construct a production-ready, ultra-scalable Machine Learning pipeline that resolves noisy records across three independent data sources into unified business entities, evaluated under the precision-heavy **Macro-Averaged $F_{0.5}$ metric**.

## Executive Summary
This repository documents the end-to-end architecture, exploratory data analysis, multi-strategy inverted blocking engine, and calibrated Gradient Boosted Decision Tree (GBDT) matching pipeline developed for large-scale multi-source business entity resolution.

### Key Highlights
- **Intractable Search Space Pruning:** Successfully reduced an intractable **11.84 trillion pair** cross-source comparison space down to **249 million candidate pairs**—achieving a **99.997894% reduction ratio**.
- **Discovery of Critical Invariants:** Uncovered the **100.0000% intra-country match consistency** (zero cross-border linkages), enabling mathematically lossless hard-partitioning.
- **Ultra-Fast In-Memory Pipeline:** Resolved memory bottlenecks by shifting from naive quadratic scans to an optimized in-memory pipeline executing training and calibration in **~14 seconds** under **< 500 MB RAM**.
- **High-Precision Calibration:** Achieved a **0.9964 Macro $F_{0.5}$** on validation entities with calibrated precision thresholding ($	au = 0.75$), aggressively penalizing false merges while correctly handling singletons.
