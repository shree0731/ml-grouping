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

## System Architecture Overview

```mermaid
flowchart TD
    subgraph Ingestion ["1. Data Ingestion & Partitioning"]
        S1["Source 1: Deduplicated Catalog<br>(2.2M Reference Records)"]
        S2["Source 2: Query Stream A<br>(5.0M Noisy Records)"]
        S3["Source 3: Query Stream B<br>(5.3M Noisy Records)"]
        CP["Country Hard-Partitioning<br>(100% Lossless Isolation)"]
        S1 --> CP
        S2 --> CP
        S3 --> CP
    end

    subgraph Preprocessing ["2. Unicode-Safe Normalization"]
        NORM["Multilingual Normalizer<br>• NFKC Decomposition<br>• Accent & Indic Matra Preservation<br>• Legal Suffix Stripping<br>• Web & Handle Cleansing<br>• Address Canonicalization"]
        CP --> NORM
    end

    subgraph Blocking ["3. Multi-Strategy Inverted Blocking"]
        B1["Exact Normalized Name"]
        B2["Suffix-Stripped Name"]
        B3["Compact Alphanumeric"]
        B4["Informative Tokens (freq ≤ 250)"]
        B5["Selective Prefix (5 chars, freq ≤ 150)"]
        NORM --> B1 & B2 & B3 & B4 & B5
        B1 & B2 & B3 & B4 & B5 --> DEDUP["Union & Deduplication<br>(99.9979% Reduction Ratio)"]
    end

    subgraph Matching ["4. Pairwise Feature Extraction & GBDT"]
        FE["15 Pairwise Features<br>• Name Token Jaccard & Containment<br>• Character 3-Gram Overlap<br>• Address Token Overlap<br>• House Number & PIN Match<br>• Interaction Signals"]
        XGB["XGBoost Classifier<br>(150 Trees, 8 Threads)"]
        DEDUP --> FE --> XGB
    end

    subgraph Decision ["5. Precision Calibration & Output"]
        THRESH["F0.5 Threshold Tuning<br>(Optimal τ = 0.75)"]
        OUT1["matching_results.tsv<br>(Final Entity Matches)"]
        OUT2["candidate_pairs.tsv<br>(Blocking Candidate Set)"]
        XGB --> THRESH --> OUT1 & OUT2
    end
```

## Problem Formulation

Let $\mathcal{S}_1 = \{e_1^{(1)}, e_2^{(1)}, \dots, e_{N_1}^{(1)}\}$ denote the deduplicated reference entity catalog from Source 1.
Let $\mathcal{S}_2 = \{e_1^{(2)}, \dots, e_{N_2}^{(2)}\}$ and $\mathcal{S}_3 = \{e_1^{(3)}, \dots, e_{N_3}^{(3)}\}$ denote noisy entity observation streams from Source 2 and Source 3 respectively.

Each record $e = (	ext{id}, 	ext{name}, 	ext{address}, 	ext{country})$ describes a business entity. There are no shared synthetic or natural primary keys across sources. The entity resolution objective is to learn a mapping:
$$\mathcal{M}: \mathcal{S}_1 ightarrow \mathcal{P}(\mathcal{S}_2 \cup \mathcal{S}_3)$$
that associates each reference entity $e_i^{(1)} \in \mathcal{S}_1$ with all matching records from $\mathcal{S}_2$ and $\mathcal{S}_3$ that denote the same physical real-world enterprise.

### Entity Resolution Topology

The data topology presents several specific structural properties:
1. **Deduplicated Reference Source:** Source 1 is internally deduplicated; no two rows in Source 1 represent the same physical business.
2. **One-to-Many Match Multiplicity:** A single Source 1 reference entity may match zero, one, or multiple records across Source 2 and Source 3 (e.g. branch representations, legal registration variations, alternate operating names).
3. **Presence of Singletons:** Many entities in Source 1 have zero matching records in Source 2 or Source 3. Correctly identifying singletons without generating false links is crucial under the competition scoring rules.
4. **Presence of Distractor Records:** Substantial portions of Source 2 and Source 3 denote businesses that do not exist in Source 1 (distractor noise).

## Evaluation Metric: Macro-Averaged $F_{0.5}$

Submissions are evaluated using the **Macro-Averaged $F_{0.5}$ Score**, defined across all $N_1$ entities in the evaluation set:

$$	ext{Macro } F_{0.5} = rac{1}{N_1} \sum_{i=1}^{N_1} F_{0.5}(e_i^{(1)})$$

where for each reference entity $e_i^{(1)}$:
$$F_{0.5}(e_i^{(1)}) = rac{1.25 	imes 	ext{Precision}(e_i^{(1)}) 	imes 	ext{Recall}(e_i^{(1)})}{0.25 	imes 	ext{Precision}(e_i^{(1)}) + 	ext{Recall}(e_i^{(1)})}$$

with:
$$	ext{Precision}(e_i^{(1)}) = rac{|\mathcal{T}_i \cap \mathcal{P}_i|}{|\mathcal{P}_i|}, \quad 	ext{Recall}(e_i^{(1)}) = rac{|\mathcal{T}_i \cap \mathcal{P}_i|}{|\mathcal{T}_i|}$$

where $\mathcal{T}_i$ is the ground-truth set of matching IDs, and $\mathcal{P}_i$ is the predicted set of matching IDs.

### Mathematical Implications of $F_{0.5}$ Weighting

1. **Precision Weighed 2× Over Recall:**
   The parameter $eta = 0.5$ places double the weight on precision relative to recall. In business entity resolution, false merges (joining two distinct companies together) pollute business registries far more severely than missing a link (false negative).
2. **Singleton Scoring Mechanics:**
   - If an entity is a true singleton ($\mathcal{T}_i = \emptyset$):
     - Predicting $\mathcal{P}_i = \emptyset$ yields a perfect score: $F_{0.5}(e_i^{(1)}) = 1.0$.
     - Predicting even a single false match ($\mathcal{P}_i 
e \emptyset$) drops the score immediately to: $F_{0.5}(e_i^{(1)}) = 0.0$.
   - Consequently, precision calibration and aggressive thresholding directly protect the substantial singleton score contribution.

## Exploratory Data Analysis & Scale Analysis

To analyze the 26.4 million tabular records without memory exhaustion, we leveraged DuckDB as an in-process vectorized SQL OLAP engine. This enabled rapid, zero-copy querying directly over tab-separated raw storage with multi-threaded streaming across 12 CPU cores.

### Global File Inventory & Scale

| Split | File Name | Role | Raw Size | Total Rows | Unique IDs | Duplicate IDs |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Train** | `train_source1.tsv` | Deduplicated Reference | 200.3 MB | 2,206,821 | 2,206,821 | 0 |
| **Train** | `train_source2.tsv` | Query Stream A | 466.6 MB | 5,034,616 | 5,034,616 | 0 |
| **Train** | `train_source3.tsv` | Query Stream B | 480.4 MB | 5,285,603 | 5,285,603 | 0 |
| **Train** | `train_ground_truth.tsv` | Match Labels | 121.1 MB | 2,206,821 | 2,206,821 | 0 |
| **Test** | `test_source1.tsv` | Test Reference | 166.9 MB | 1,732,544 | 1,732,544 | 0 |
| **Test** | `test_source2.tsv` | Test Query Stream A | 485.9 MB | 4,887,273 | 4,887,273 | 0 |
| **Test** | `test_source3.tsv` | Test Query Stream B | 482.6 MB | 5,082,316 | 5,082,316 | 0 |
| **Total** | **All 7 TSV Tables** | **Complete Dataset** | **~2.40 GB** | **26,436,000** | **26,436,000** | **0** |

### Data Integrity Guarantees

- **Primary Key Uniqueness:** Every source file exhibits 0 duplicate entity IDs. Entity IDs are strictly prefixed:
  - Source 1: `S1-`
  - Source 2: `S2-`
  - Source 3: `S3-`
- **Split Isolation:** Strict zero-overlap verified between train and test entity IDs:
  $$\mathcal{S}_1^{	ext{train}} \cap \mathcal{S}_1^{	ext{test}} = \emptyset, \quad \mathcal{S}_2^{	ext{train}} \cap \mathcal{S}_2^{	ext{test}} = \emptyset, \quad \mathcal{S}_3^{	ext{train}} \cap \mathcal{S}_3^{	ext{test}} = \emptyset$$

### Missingness Profile

- `entity_id`: 0 missing values across all 26.4M records (100% complete).
- `business_name`: 0 missing values across all 26.4M records (100% complete).
- `country`: 0 missing values across all 26.4M records (100% complete).
- `business_address`:
  - Source 1: 0% missing in both train and test.
  - Source 2: 3.36% missing in train (168,967 rows), 2.65% in test (129,408 rows).
  - Source 3: 3.33% missing in train (175,916 rows), 2.68% in test (136,098 rows).

*Implication:* Matching models cannot rely solely on address features; models must maintain robust matching capabilities when address tokens are completely absent.

### Geographic Distributions

The training data spans two major regional markets, while the test set introduces an open-set domain expansion:

| Split | Country | Source 1 Count (%) | Source 2 Count (%) | Source 3 Count (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Train** | United States (`US`) | 1,323,633 (59.98%) | 3,016,817 (59.92%) | 3,170,056 (59.98%) |
| **Train** | India (`India`) | 883,188 (40.02%) | 2,017,799 (40.08%) | 2,115,547 (40.02%) |
| **Test** | India (`India`) | 809,986 (46.75%) | 2,312,565 (47.32%) | 2,405,000 (47.32%) |
| **Test** | United States (`US`) | 663,106 (38.27%) | 1,871,330 (38.29%) | 1,945,701 (38.28%) |
| **Test** | France (`France`) | 259,452 (14.98%) | 703,378 (14.39%) | 731,615 (14.40%) |

### The 100% Intra-Country Invariant

During our Phase 1 exploratory data analysis, we verified the country linkage consistency across all **7,638,365 true matched pairs** in `train_ground_truth.tsv`:

- **Same-country matches:** $7,638,365$ ($100.0000\%$)
- **Cross-country matches:** $0$ ($0.0000\%$)

$$orall (e^{(1)}, e^{(	ext{cand})}) \in \mathcal{T}, \quad 	ext{country}(e^{(1)}) \equiv 	ext{country}(e^{(	ext{cand})})$$

> **Architectural Invariant:** Hard-blocking strictly on `country == country` is **100% lossless** with zero recall sacrifice, while reducing cross-country Cartesian pairs by over **50%** before any string operations.

### Handling Geographic Domain Shift (`France`)

The emergence of `France` in the test set (comprising ~15% of records) represents an open-set domain expansion. Our pipeline avoids overfitting to US/India-specific rules by:
1. Treating `country` as a dynamic string key without hardcoded conditional branches.
2. Handling French legal suffixes (`SARL`, `SAS`, `SASU`, `EURL`, `& Fils`).
3. Preserving Latin accented characters (`é`, `è`, `ê`, `à`, `ç`, `ô`, `ù`) via Unicode NFKC normalization.
4. Expanding European address tokens (`Boulevard`, `Rue`, `Avenue`, `bis`).

### Text Statistics Profile

| Field | Source Table | Min Chars | Max Chars | Mean Chars | Median Chars | Mean Words |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `business_name` | `train_source1` | 3 | 105 | 24.03 | 24.0 | 3.55 |
| `business_name` | `train_source2` | 2 | 104 | 25.10 | 25.0 | 3.61 |
| `business_name` | `train_source3` | 2 | 123 | 25.20 | 25.0 | 3.63 |
| `business_address` | `train_source1` | 11 | 256 | 52.07 | 41.0 | 8.03 |
| `business_address` | `train_source2` | 8 | 249 | 47.83 | 37.0 | 7.56 |
| `business_address` | `train_source3` | 2 | 240 | 48.32 | 42.0 | 7.42 |
| `business_name` | `test_source1` | 3 | 92 | 23.84 | 24.0 | 3.52 |
| `business_address` | `test_source1` | 11 | 268 | 57.21 | 50.0 | 8.59 |

### Ground Truth Match Multiplicity

Analyzing the distribution of matches per Source 1 reference entity in `train_ground_truth.tsv`:

| Matches per S1 Entity | Count of S1 Entities | Percentage | Cumulative % | Category |
| :---: | :---: | :---: | :---: | :--- |
| **0** | **123,247** | **5.5848%** | **5.58%** | **Singletons (No matches)** |
| 1 | 119,157 | 5.3995% | 10.98% | Single Match |
| 2 | 375,212 | 17.0024% | 27.99% | Moderate Linkage |
| **3** | **530,841** | **24.0546%** | **52.04%** | **Peak Frequency Mode** |
| 4 | 484,115 | 21.9372% | 73.98% | High Linkage |
| 5 | 321,957 | 14.5892% | 88.57% | High Linkage |
| 6 | 164,868 | 7.4708% | 96.04% | High Linkage |
| 7 to 11 | 87,424 | 3.9615% | 100.00% | Tail Multiplicity |

- **Mean Matches per S1:** $3.4613$
- **Median Matches per S1:** $3.0$
- **Maximum Matches observed:** $11$

### The 1-to-1 Reverse Exclusivity Guarantee

A crucial graph topology discovery emerged from checking link multiplicity in reverse:

$$	ext{Total Links in Ground Truth} = 7,638,365$$
$$	ext{Unique Matched Candidate IDs} = 7,638,365$$
$$	ext{Collisions / Multi-Parent Entities} = 0$$

> **Topology Invariant:** Every Source 2 and Source 3 entity in ground truth belongs to **at most one** Source 1 reference entity. There are zero multi-parent conflicts. This establishes that cross-source matching operates as an exclusive partition of query records to reference entities.

### Regional Singleton Invariance

The proportion of singletons (entities with 0 matches across S2 and S3) proved remarkably consistent across geographies:
- **United States:** $73,896$ singletons out of $1,323,633$ S1 entities (**$5.583\%$**)
- **India:** $49,351$ singletons out of $883,188$ S1 entities (**$5.588\%$**)

This structural invariant allows calibrated validation and baseline priors for test-set singleton estimation.

## Noise Taxonomy in Real-World Business Data

Systematic error auditing of the training data revealed five major noise categories that defeat naive exact-matching pipelines:

### 1. Cross-Script Phonological Transliteration
In the Indian partition, Source 1 business names are strictly rendered in the Latin alphabet, while substantial portions of Source 2 and Source 3 are phonetically transliterated into native Indic scripts:
- **English Reference:** `Red Ventures Private Limited`
- **Devanagari Observation:** `रेड वेंचर्स प्राइवेट लिमिटेड`
- **English Reference:** `Raj Investments LLP`
- **Tamil Observation:** `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`
- **English Reference:** `Big Real International Limited`
- **Telugu Observation:** `బిగ్ రియల్ ఇంటర్నేషనల్ లిమిటెడ్`

*Impact:* Character-level Jaccard or Levenshtein distance on raw text produces 0.0 similarity. However, their numeric street addresses and PIN codes remain tightly aligned.

### 2. OCR and Leet-Speak Typographical Corruptions
Digit-for-letter and visually ambiguous substitutions frequently corrupt initial characters:
- `Bhubaneswar Cement Limited` $\leftrightarrow$ `8hubaneswar Cement` (`B` $ightarrow$ `8`)
- `Olanelle Pinnacle` $\leftrightarrow$ `0LANELLE PINNACLE` (`O` $ightarrow$ `0`)
- `Ironclad LLC` $\leftrightarrow$ `lRONCLAD LLC` (Capital `I` $ightarrow$ lowercase `l`)
- `Medina and Thompson LLC` $\leftrightarrow$ `Medina and Tmthonm LLC` (Internal letter scrambling)

### 3. Address Component Re-ordering & Artifact Injection
Addresses frequently permute street, city, and state elements, while injecting synthetic placeholder strings:
- **Reference:** `630 45th Terrace, Kansas City, MO`
- **Observation:** `45ND TERRACE, null, KANSAS CITY, MO`
- **Reference:** `Af-684, Nandgram Near Mother India Public School, Ghaziabad, UP`
- **Observation:** `AF-0684, Uttar Pradesh, GHAZIABAD, 9487203`

*Impact:* Rigid sequence matching fails, but token-set Jaccard and numeric extraction preserve the match signal.
