# Drug Review Condition Classifier

## Overview
> This project uses NLP to classify patient drug reviews by the **condition** the review is about (30 conditions at baseline, 28 after two merges), based on the review text. Built and evaluated with a clinical lens to flag misclassifications that could be misleading or harmful, not just statistically wrong. Results are framed as patient-reported outcomes, not treatment recommendations.

**Status:** cleaning and preprocessing are complete; the text-only baseline and the class-weighting comparison have been run on a held-out validation set. The final model is fixed (text + drug name, logistic regression, 28 classes, validation macro F1 0.854). The final test has been run once on the cleaned Kaggle test file (macro F1 0.855, see *Test file cleaning and final test*); the remaining work is the write-up.

## Motivation
This project is modeled on a real application of NLP in healthcare: **pharmacovigilance**, 
the ongoing monitoring of drug safety after a drug reaches the market. Clinical trials 
are relatively small and short, so rare side effects, unexpected condition associations, 
and long-term outcomes often only surface once a drug is in wide use — and patients are 
frequently the first to notice and describe these effects, in their own words, before 
they show up in formal reporting systems. Regulators and pharmaceutical companies 
increasingly mine patient-written text (drug reviews, forums, social media) at scale to 
catch these signals faster than manual reporting alone would allow.

This project uses patient-written drug reviews as a stand-in for that kind of real-world 
text data, applying NLP to extract signal — here, the condition a patient is writing 
about — directly from what patients write.

- **What problem does this solve?** Demonstrates how patient-reported text can be 
  mined at scale to surface condition-level patterns that would be impractical to 
  review manually, one patient at a time.
- **The clinical background angle:** most data scientists evaluate a text classifier 
  purely on standard metrics (accuracy, F1). This project goes further with a 
  **clinical error analysis** — reviewing specific misclassifications to judge whether 
  they're clinically harmless (e.g., confusing two labels for the same underlying 
  condition) or clinically concerning (e.g., confusing conditions that would lead to 
  meaningfully different care) — a distinction a non-clinical reviewer of model errors 
  would likely miss.

## Dataset
- **Source:** UCI Drug Review Dataset (Drugs.com), accessed via the **Kaggle mirror** (UCI's own download endpoint returned a 403/dataset-not-found during retrieval, so the identical dataset was pulled from Kaggle instead)
- **Kaggle link:** https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018
- **Original UCI link (for citation):** https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com
- **Format:** `.csv` (comma-separated) — note this differs from UCI's native `.tsv` (tab-separated) files; the same data, just re-packaged
- **Size:** 161,297 rows in the training file (`drugsComTrain_raw.csv`), 53,766 rows in the test file (`drugsComTest_raw.csv`) — ~215,000 rows combined, matching the original dataset description
- **Features:** drug name, condition, review text, rating (1–10), date, useful count
- **Target variable:** `condition`, predicted from the review text (decision and evidence below). Sentiment/rating was the alternative and scored lower in the early comparison.
- **Citation:** Kallumadi, S. & Gräßer, F. (2018). Drug Reviews (Drugs.com) [Dataset]. UCI Machine Learning Repository.
- **License note:** Research use only — not for commercial use, and the raw data should not be redistributed. This applies regardless of whether it's sourced from UCI or the Kaggle mirror, since it's the same underlying dataset. Data is excluded from this repo via `.gitignore`; see Setup below for download instructions.
- **Data quality (verified via `.isna().sum()`):** 899 missing values in `condition` (train set). A full-row duplicate check found none, but a later check on review text found many near-duplicates (see findings below). The data is not perfectly clean, so cleaning is based on genuine data issues rather than synthetic ones.

## Exploratory Findings (before cleaning)

### EDA Findings
- **Rating is strongly bimodal/U-shaped:** most reviews cluster at the extremes (10 and 1), with far fewer in the middle. This means a 3-class sentiment bucketing (positive/neutral/negative) has a small, underrepresented "neutral" class.
- **Condition has a severe long tail:** 916 unique conditions total, but **Birth Control** alone accounts for ~38,000 reviews — several times more than the next few conditions combined (Depression, Pain, Anxiety, Acne, etc.). The long tail of rare conditions had to be filtered down to the top N classes, since a model has no realistic chance of learning from a handful of examples.

### Data Quality Finding: Condition Field Text Corruption

During EDA, systematic text corruption was discovered in the `condition` column, 
affecting approximately 2,450 rows (~1.5% of the train set). This was not part of 
the originally reported 899 missing values — these rows have a *present but corrupted* 
condition value.

**Discovery process:**
1. Initial condition frequency counts (`value_counts()`) surfaced conditions with 
   implausible names (e.g., `ibromyalgia`, `Bipolar Disorde`)
2. A regex check for condition names starting with a lowercase letter found ~35 
   additional malformed entries, mostly tied to conditions paired with combination-drug 
   descriptions (e.g., `ge (amlodipine / valsartan)`)
3. A systematic prefix-matching check (comparing all 884 unique conditions against 
   each other) was used to find additional corruption where names were truncated from 
   the *end* rather than the start (e.g., `Panic Disorde` → `Panic Disorder`). This 
   required manual review, since many prefix matches are legitimate distinct condition 
   pairs rather than corruption (e.g., `Diarrhea` vs. `Diarrhea, Acute`)

**Two truncation patterns identified:**
- Minor truncation (1-3 characters missing from start or end) — recoverable by 
  pattern/manual inspection
- Severe truncation (only 2-4 characters remaining, e.g., `Pe`, `Gas`, `mis`) — not 
  recoverable from text alone, since the fragment is ambiguous across multiple real 
  conditions

**Repair strategy identified:** the `drugName` column is not affected by this 
corruption. Cross-referencing corrupted rows against clean rows sharing the same drug 
name allows a lookup-based repair — but reliability varies by drug:
- High confidence: drugs with one dominant/exclusive condition in clean rows 
  (e.g., Budesonide/formoterol → Asthma, Maintenance)
- Low confidence / unsafe: multi-purpose drugs with several legitimate conditions 
  (e.g., Propranolol, Diclofenac) — auto-repair here would risk introducing incorrect 
  labels

**Status:** repaired in the cleaning phase (see *Data Cleaning* below). Limitation: rare end cut-offs were only checked among the top ~150 conditions, so corrupted names outside the final top 30 were not repaired.

### Data Quality Finding: Missing Condition Values Cluster by Drug and Rating

The 899 rows with a fully missing (`NaN`) `condition` value are not randomly 
distributed — they cluster in two connected ways.

**By drug:** the top drugs by missing-condition count are overwhelmingly hormonal 
contraceptives (e.g., Ethinyl estradiol / norethindrone: 100, Ethinyl estradiol / 
norgestimate: 84, Ethinyl estradiol / levonorgestrel: 79, Levonorgestrel: 26, 
Medroxyprogesterone: 24) — a concentration well beyond what Birth Control's overall 
share of the dataset alone would predict.

**By rating:** missing-condition rows skew toward the high end of the rating scale — 
rating 10 accounts for 244 of the 899 missing rows (the single largest bucket, more 
than double rating 1's count of 115), rather than an even spread across 1–10.

**Working hypothesis:** birth control reviewers who are satisfied (high rating) may 
be less likely to explicitly state their condition, since it's often self-evident 
from the drug itself — this could be a survey/form-design artifact at the data source 
(Drugs.com) rather than a random data entry gap.

**Implication for cleaning:** dropping or imputing these 899 rows without accounting 
for this pattern would disproportionately remove birth-control, high-rating reviews, 
and could skew the dataset's condition and rating balance.

**Status:** the 899 rows were set aside, not imputed, and saved in `data/missing_condition_rows.csv`. The skew is a known limitation of the final class set.

### Data Quality Finding: Near-Duplicate Reviews from Brand/Generic Drug Name Splits

A check for exact-duplicate review text (beyond the earlier full-row duplicate check, 
which found none) surfaced a real pattern: a number of long, detailed reviews appear 
more than once in the dataset, each time attached to a different `drugName`. In every 
case examined, the repeat count matches the number of distinct drug names involved 
(e.g., a review appearing twice is split across exactly 2 drug names) — consistent 
with the same underlying drug being listed under both its brand name and generic name 
(e.g., Retin-A / Tretinoin), rather than unrelated drugs coincidentally sharing text.

**Why the earlier duplicate check missed this:** `uniqueID` and `drugName` differ 
across these rows even though the review, condition, rating, and date are identical — 
so a standard full-row duplicate check does not flag them.

**Implication:** this inflates how often a given review's language/rating is 
represented in the dataset for the affected conditions. Worth deduplicating (e.g., by 
review text + condition + date) before modeling, so the same patient experience isn't 
counted multiple times.

**Status:** deduplicated in the cleaning phase (47,842 rows removed, plus 4 more after the "Pe" repair); the removed rows are saved in `data/duplicate_rows_removed.csv`.

### Decision: Handling Birth Control's Class Dominance

Birth Control accounts for ~28,788 reviews in the train set — several times larger 
than the next candidate condition (Depression, ~9,069). Including it in the top-15 
candidate condition set raises the imbalance ratio (largest class ÷ smallest class) 
to 13.7:1, versus 4.3:1 without it — a ratio generally considered likely to hurt a 
classifier's ability to learn minority classes.

**Decision: Birth Control is retained in the candidate set (not excluded or 
downsampled).** Excluding or downsampling it would reduce real-world relevance — 
Birth Control is a genuinely dominant condition in real drug-review data, not just an 
artifact of this dataset, and a pharmacovigilance-style tool that ignores it would be 
less representative of real patient-reported data.

**Mitigation approach instead of removing data:**
- **Class weighting** at the modeling stage (e.g., `class_weight='balanced'`), so the 
  model is penalized more for missing minority-class conditions, without discarding 
  any Birth Control reviews
- **Macro F1** (not accuracy) as the primary evaluation metric going forward, since 
  macro F1 treats all classes equally regardless of size and won't let strong 
  Birth Control performance mask weak performance on smaller conditions
- The Step 2 baseline-classifier viability check was run *with* class weighting 
  enabled, so the signal reflects the intended modeling approach rather than a naive 
  unweighted baseline

**Status after cleaning:** Birth Control is still the largest class (18,112 of 74,549 rows, 24.3%). Class weighting was tested and not adopted; downsampling Birth Control was not run.

### Finding: Condition Vocabulary Overlap (Step 2 — Condition Viability Test)

Word/n-gram analysis (unigrams + bigrams, English stopwords removed) was run across 
the top-15 candidate conditions to assess whether their review text is distinct enough 
to be learnable. Two preprocessing artifacts were identified and corrected for this 
check (not yet applied to raw data): `&#039;` HTML entities (decoded via 
`html.unescape`) and residual "ve"/apostrophe-split tokens from contractions (noted 
as a cleaning-phase tokenizer fix, not corrected here — negligible signal impact).

**Review length by condition:** means ranged narrowly (~66–105 words), suggesting 
length alone is a weak standalone signal, though mental-health conditions trended 
toward longer, more narrative reviews.

**Vocabulary overlap (Jaccard similarity, top-50 words per condition pair):**

| Pair | Jaccard | Decision |
|---|---|---|
| Weight Loss vs. Obesity | 0.69 | **Merged** into single class "Weight Management" |
| Depression vs. Anxiety | 0.69 | Kept separate |
| Depression vs. Bipolar Disorde | 0.69 | Kept separate |
| Anxiety vs. Bipolar Disorde | 0.64 | Kept separate |
| Birth Control vs. Abnormal Uterine Bleeding | 0.59 | Kept separate (watch-pair) |
| Birth Control vs. Emergency Contraception | 0.32 | Kept separate |
| Emergency Contraception vs. Abnormal Uterine Bleeding | 0.32 | Kept separate |

*(Note: "Disorde" is the truncated spelling present in the raw data at the time of this analysis; it was repaired in cleaning.)*

**Reasoning — why similar overlap scores led to different decisions:** vocabulary 
similarity alone does not determine whether classes should be merged. Weight Loss and 
Obesity were merged because they represent the same underlying patient experience 
under two different labels (a stated goal vs. a diagnosis), not genuinely distinct 
conditions. Depression, Anxiety, and Bipolar Disorder were kept separate despite equal 
or higher overlap scores, because they are clinically distinct diagnoses where 
conflating them has real consequences (e.g., treating undiagnosed Bipolar Disorder as 
straightforward Depression can trigger manic episodes). This distinction — vocabulary 
similarity vs. clinical distinctness — is treated as a deliberate output of the 
clinical-lens approach this project is built around, not an inconsistency.

**Planned use in later Clinical Error Analysis:** the mental-health cluster and the 
Birth Control/Abnormal Uterine Bleeding pair are logged as expected, watched 
confusion pairs — model errors here will be evaluated for clinical severity rather 
than treated as uniformly "wrong." (Results so far are in *Clinical Error Analysis* below.)

**Negation check (rating 1 vs. rating 10, strict extremes):** average negation-word 
count per review (not, no, didn't, doesn't, don't, never, wasn't, isn't, won't, 
couldn't, wouldn't) is nearly identical between rating 1 (3.66) and rating 10 (3.51). 
This confirms the sentiment vocabulary distinction found above is driven by genuinely 
distinct word choice, not by negation of otherwise-shared vocabulary (e.g., reviews 
aren't mainly saying "didn't work" — they use their own distinct negative words like 
"worst," "horrible"). This is a favorable result for a simple bag-of-words modeling 
approach, since the signal doesn't depend on the model understanding negation.

### Decision: Target Variable — Condition Prediction (Step 5 Go/No-Go)

A baseline classifier (TF-IDF features, unigrams + bigrams, Logistic Regression with 
`class_weight='balanced'`, 80/20 stratified train/test split) was run for both 
candidate targets to make a direct, evidence-based comparison.

| Target | Macro F1 |
|---|---|
| Condition (14 merged classes) | **0.829** |
| Sentiment (low/mid/high rating buckets) | 0.626 |

**Condition prediction selected as the primary target variable.** Reasoning:
- The gap (0.829 vs. 0.626) is decisive, not marginal
- Condition's weakest class (Abnormal Uterine Bleeding, F1 0.51) still substantially 
  outperforms sentiment's weakest class (mid-rating bucket, F1 0.26)
- Sentiment's low score is driven almost entirely by the small, inherently ambiguous 
  mid-rating bucket (1,648 test examples vs. 10,830 for high) — reviews in the 5-6 
  rating range often lack the strong, distinctive language that clearly positive 
  or negative reviews carry
- The merged "Weight Management" class scored particularly well (F1 0.93), validating 
  the earlier Weight Loss/Obesity merge decision
- Results for the mental-health cluster (Depression 0.78, Anxiety 0.76, Bipolar 
  Disorder 0.73) and the Birth Control/Abnormal Uterine Bleeding overlap matched 
  predictions from the earlier Jaccard vocabulary analysis, validating that approach

**Considered and rejected: dropping the mid-rating bucket to make sentiment a binary 
(low/high) problem.** This would likely have improved sentiment's macro F1, but was 
rejected because: (1) it discards the ability to detect ambivalent/mixed patient 
experiences, which are clinically meaningful; (2) it artificially simplifies the 
comparison rather than addressing why the 3-class version underperforms; (3) it 
reduces real-world applicability for the pharmacovigilance framing this project is 
built around, where detecting early/moderate dissatisfaction matters. A binary 
sentiment result may still be included later as a clearly-labeled secondary/supporting 
result, not as a substitute for this comparison.

**Status:** target variable decision finalized (`condition`).

**Comparability caveat:** the 0.829 figure came from an early viability check (14 merged classes, bigrams, a different split). The final pipeline uses 30 classes (28 after merges) with different cleaning and a stricter leak-free setup, so its macro F1 is **not directly comparable** with 0.829.

## Data Cleaning (`Cleaning.ipynb`)

161,297 raw rows reduced to **74,549 rows across 30 classes**. Every removed row is accounted for in `data/cleaning_row_log.csv` (the removed groups add back to 161,297).

| Step | Rows |
|---|---|
| Raw training rows | 161,297 |
| Missing condition, set aside | 899 |
| `</span>` junk in condition, set aside | 900 |
| Leftover cut-off drug fragments, set aside | 270 |
| Duplicates on review_clean + condition + date, removed | 47,842 |
| 4 further duplicates after the "Pe" repair, removed | 4 |
| Long texts under 2+ conditions, set aside | 55 |
| Outside the top 30 conditions, set aside | 36,155 |
| Reviews under 5 words, set aside | 623 |
| **Final** | **74,549** |

**Repairs:** truncated condition names (all "...Disorde" endings, 6,674 rows; start cut-offs via drug lookup, e.g. `ibromyalgia` → Fibromyalgia; rare end cut-offs such as Overactive Bladde, Breast Cance, Prostate Cance, Vertig, Stomach Ulce; `Pe` → Performance Anxiety).

**`review_clean`:** `html.unescape`, whitespace collapsed to single spaces, stripped. Use `review_clean`, not `review`. The original index is kept in the column `Unnamed: 0` to trace rows.

**Decisions:**
- **Class set:** top 30 conditions after dedup. Counts fall off gradually with no cliff; the smallest class has 706 rows.
- **Weight Loss + Obesity** merged into "Weight Management" (no review text appears under both labels).
- **Short reviews:** reviews of 1–4 words set aside (cutoff chosen from the word-count distribution); they are kept for a separate test.
- **Set-aside tables** (real drug names and ratings kept for the drug-ranking extension) are saved in `data/`: `missing_condition_rows.csv`, `junk_condition_rows.csv`, `leftover_condition_rows.csv`, `duplicate_rows_removed.csv` (47,846), `multi_condition_rows.csv`, `out_of_class_set_rows.csv`, `short_review_rows.csv`.

### Class counts after cleaning (30 classes)
Birth Control 18,112 · Depression 6,224 · Weight Management 5,066 · Pain 4,356 · Anxiety 4,269 · Acne 3,943 · Bipolar Disorder 2,855 · Insomnia 2,797 · ADHD 2,163 · Vaginal Yeast Infection 2,010 · High Blood Pressure 1,817 · Diabetes, Type 2 1,652 · Emergency Contraception 1,576 · Abnormal Uterine Bleeding 1,336 · Fibromyalgia 1,246 · Anxiety and Stress 1,225 · Smoking Cessation 1,217 · Migraine 1,176 · Bowel Preparation 1,169 · Constipation 1,114 · Migraine Prevention 1,103 · Major Depressive Disorder 1,061 · Chronic Pain 1,030 · Panic Disorder 1,001 · Urinary Tract Infection 944 · Muscle Spasm 894 · Generalized Anxiety Disorder 887 · Osteoarthritis 817 · Bacterial Infection 783 · Erectile Dysfunction 706. (Saved in `data/final_class_counts.csv`.)

## Approach

### 1. Preprocessing (`Preprocessing and Modelling.ipynb`)
- **Split:** 80/20 stratified on condition, `random_state=42` (chosen once, before seeing any results, and not changed). The 20% is the **validation** set (`val_df`), used for every modeling decision. Split on the whole dataframe so drug name and the original index stay attached to each row.
- **Result:** 59,639 train / 14,910 validation. Class shares match across the two parts (largest difference ~6e-05).
- **Normalisation:** lowercase only, done by the vectorizer. No stemming, no punctuation stripping.
- **Stop words:** none removed (negations such as *not*, *don't* stay in the text).
- **Leakage checks:**
  - 3 validation reviews (0.02%) also appeared in train with the same label; they were removed from validation, leaving 14,907 rows.
  - The condition name appears literally in many reviews, so part of the task is keyword spotting: share of reviews naming the condition ranged from 12% (Erectile Dysfunction) to 80% (Migraine) for the five classes checked.
  - Reviews arrive wrapped in quotation marks; the tokenizer ignores them. No class-specific boilerplate in the most frequent first and last words (top 20 only).
- **Merges decided on validation** (see Clinical Error Analysis): Major Depressive Disorder → Depression and Chronic Pain → Pain, stored in a new column `condition_merged` (the original `condition` is kept). This **changes two earlier locked decisions** (Chronic Pain / Pain was locked as separate).
- **After the merge:** 5 exact duplicates (text + merged label) were removed from train, keeping the first copy. Train is 59,634 rows, validation 14,907, 28 classes.

### 2. Feature Engineering
- TF-IDF, unigrams only, default vocabulary (34,186 tokens, including some junk tokens such as long digit strings), **fitted on train only**; validation is transformed with the train-fitted vectorizer.
- **Drug name:** one-hot encoded (`OneHotEncoder(handle_unknown='ignore')`), fitted on train only: 1,160 drugs in train; 53 validation rows (0.36%) have a drug unseen in train. Combined with the text matrix by `scipy.sparse.hstack`, unscaled.
- Not yet tried: `min_df=2`, bigrams.

### 3. Modeling
- **Baseline:** TF-IDF + Logistic Regression (`max_iter=1000`, no class weighting).
- **Class weighting:** `class_weight='balanced'` tested (macro F1 0.727 vs. 0.737 unweighted); not adopted.
- **Drug name:** drug-only and text + drug models, same plain logistic regression (see *Drug name vs. text* below).
- **Final model (chosen on validation):** text + drug name, logistic regression, 28 classes, macro F1 0.854. Birth Control was not downsampled.
- **Final test:** run once on the cleaned Kaggle test file: macro F1 0.855 (see *Test file cleaning and final test*).

### 4. Evaluation
- Primary metric: **macro F1** on validation, since it treats all classes equally. Per-class precision/recall/F1 and a row-normalised confusion matrix are used for error analysis.
- Why accuracy alone isn't sufficient: Birth Control alone is ~24% of rows.
- The Kaggle test file is **not touched** until the final model is chosen (see below).

## Results

All numbers are on the validation set (seed 42).

| Model | Classes | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 30 | 0.696 |
| Same, after Depression and Pain merges | 28 | 0.737 |
| Same + `class_weight='balanced'` | 28 | 0.727 |
| Drug-name-only | 28 | 0.741 |
| Text + drug name | 28 | **0.854** |
| Text + drug name, **final test** (10,236 reviews) | 28 | **0.855** |

**Do not read 0.696 → 0.737 as a model improvement.** Two of the weakest classes were merged away, so part of the rise is bookkeeping. Report both numbers side by side.

**Class weighting did not help macro F1** (0.737 → 0.727, one split and one seed, so the 0.010 gap is not conclusive either way). The per-class comparison is in *Class weighting vs. the plain baseline* below. The plain text-only model (0.737) stays the reference. The drug-only and text + drug rows use the drug name, so they are not pure text results; in real use the drug is usually known. On the 30 original labels, text-only scored 0.697 and drug-only 0.703 (see *30-class pair check* below).

## Clinical Error Analysis

Based on the 30-class baseline, per-class F1 on validation. The text separates classes with their own vocabulary and struggles where labels overlap.

- **Strongest:** Smoking Cessation (0.96), Vaginal Yeast Infection (0.95), Emergency Contraception (0.94), Birth Control (0.93).
- **Weakest:** Major Depressive Disorder (0.12), Generalized Anxiety Disorder (0.14), Chronic Pain (0.27), Anxiety and Stress (0.33), Panic Disorder (0.39), Abnormal Uterine Bleeding (0.43), Muscle Spasm (0.59).

**Where the weak classes' reviews go** (share of each true class, validation):

| True class | Where its reviews went |
|---|---|
| Major Depressive Disorder | 70% → Depression, 7% correct |
| Chronic Pain | 69% → Pain, 18% correct |
| Anxiety and Stress | 43% → Depression, 23% correct, 21% → Anxiety |
| Generalized Anxiety Disorder | 43% → Anxiety, 34% → Depression, 8% correct |
| Panic Disorder | 45% → Anxiety, 32% correct, 16% → Depression |
| Abnormal Uterine Bleeding | 68% → Birth Control, 30% correct |
| Muscle Spasm | 45% correct, 32% → Pain |
| Anxiety | 67% correct, 17% → Depression |
| Depression | 76% correct, 7% → Anxiety |

**Patterns and clinical reading:**
- **Same-condition label pairs (Depression / Major Depressive Disorder, Pain / Chronic Pain):** the model sends most of the smaller class to the larger one. These errors are clinically harmless, because the pair describes largely the same patient experience, so the pairs were merged.
- **Anxiety family (Anxiety, Anxiety and Stress, Generalized Anxiety Disorder, Panic Disorder):** reviews spread across Anxiety and Depression. Merging Anxiety and Stress into Anxiety would not fix this, so all four were **kept separate**. After the drug-name comparison they remain the weakest classes (F1 0.38 to 0.54 with text + drug), and the decision to keep them separate is final; they are listed as a limitation. The confusion between Depression and the anxiety classes is the more clinically relevant one, since the diagnoses are distinct.
- **Abnormal Uterine Bleeding vs. Birth Control:** confirms the watch-pair predicted by the Jaccard analysis. Likely a real overlap, since hormonal contraceptives are used for bleeding disorders and reviews may mainly describe the pill. Not merged (different conditions).
- **Short generic reviews** (e.g. "This medicine saved my life.") carry no condition words, so a text-only model has little to go on. The 623 short reviews that were set aside confirm this (see *Short reviews* below).
- **Why the drug alone does not solve the anxiety family:** the same drug is filed under several anxiety labels (and under Depression), so the drug cannot say which one a reviewer chose. A spot check of one drug (escitalopram) on today's Drugs.com page fits this: it lists Anxiety, Depression, Generalized Anxiety Disorder and Major Depressive Disorder as separate entries, with no Panic Disorder or Anxiety and Stress entry. Whether this comes from the site's condition list or from prescribing practice cannot be settled from this data. On the 30 original labels the drug-only model does not separate the two merged pairs either (see *30-class pair check*).

### Anxiety family: where text + drug sends the reviews (28 classes, validation)

| Class | Correct | To the other 3 anxiety classes | To Depression |
|---|---|---|---|
| Anxiety | 0.78 | 0.04 | 0.13 |
| Anxiety and Stress | 0.49 | 0.09 | 0.39 |
| Generalized Anxiety Disorder | 0.27 | 0.41 | 0.24 |
| Panic Disorder | 0.39 | 0.46 | 0.14 |

- **Where the reviews go (top predictions):** Generalized Anxiety Disorder → Anxiety 0.34 (above its own 0.27), Depression 0.24; Panic Disorder → Anxiety 0.38 (its own is 0.39), Depression 0.14; Anxiety and Stress → Depression 0.39 (its own is 0.49), Anxiety 0.05.
- **Options considered:** keep all four; merge Generalized Anxiety Disorder into Anxiety; merge both Generalized Anxiety Disorder and Panic Disorder into Anxiety.
- **Decision: keep all four separate.** The pattern is weaker than for the two merged pairs (Generalized Anxiety Disorder sends 34% to Anxiety, versus 70% and 69% for Major Depressive Disorder and Chronic Pain). Generalized Anxiety Disorder still sends 24% to Depression, which a merge would not fix. Panic Disorder is a clinically distinct diagnosis. Merging would also raise macro F1 by bookkeeping (removing a weak class), not by a better model.
- **Consequence:** these four classes keep pulling macro F1 down. This reflects overlapping labels in patient language, not only a weak model.
- Anxiety itself rarely leaks to its siblings (4%), so the larger class is not harmed by the smaller ones.
- Anxiety and Stress read like Depression reviews to the model. This is a clinically relevant confusion (distinct diagnoses), but the reason is not tested here.
- **Generalized Anxiety Disorder → Depression (24%):** about 40 of the 43 reviews are about eight antidepressants (e.g. venlafaxine, escitalopram, sertraline) that are labelled Depression far more often than Generalized Anxiety Disorder in train. With little text to go on, the model follows the drug's most common label. The review text was not read, so this is an explanation, not proof. Brand and generic names (Effexor XR / venlafaxine, Lexapro / escitalopram) are separate categories for the encoder, which splits the drug signal.

### Class weighting vs. the plain baseline (28 classes, validation)

`class_weight='balanced'` makes the model more willing to guess small classes. Of the 28 classes, 11 gained F1 and 17 lost it (macro F1 0.737 → 0.727).

| Class | F1 plain → balanced | What moved |
|---|---|---|
| Generalized Anxiety Disorder | 0.12 → 0.29 | recall 0.07 → 0.42, precision 0.48 → 0.22 |
| Anxiety and Stress | 0.33 → 0.40 | recall 0.22 → 0.51, precision 0.64 → 0.32 |
| Muscle Spasm | 0.57 → 0.61 | recall 0.43 → 0.69, precision 0.85 → 0.55 |
| Erectile Dysfunction | 0.88 → 0.80 | precision 0.97 → 0.70 |
| Bacterial Infection | 0.77 → 0.69 | precision 0.81 → 0.59 |
| Urinary Tract Infection | 0.80 → 0.74 | precision 0.87 → 0.68 |
| Anxiety | 0.61 → 0.56 | recall 0.66 → 0.48 |
| Birth Control | 0.93 → 0.87 | recall 0.97 → 0.80 |
| Depression | 0.69 → 0.65 | recall 0.80 → 0.56 |

- **Pattern:** small classes that overlap with a larger sibling gain recall and lose precision; the larger siblings (Anxiety, Depression, Birth Control) lose recall.
- **Abnormal Uterine Bleeding:** recall roughly doubled (0.31 → 0.69) but precision fell (0.74 → 0.30), so F1 barely moved (-0.019). Bleeding reviews are pulled out of Birth Control at the cost of many false alarms.
- **Generalized Anxiety Disorder** is still weak after weighting (F1 0.29, precision 0.22).
- **Depression** recall fell (0.80 → 0.56). Where those reviews went has not been checked.
- Small classes have 141–245 validation rows, so small F1 shifts may be noise. The large moves are more reliable.
- **Reading:** the changes mostly shuffle reviews between overlapping labels, which fits the idea that the text alone cannot separate these labels. This motivates the drug-name comparison.

### Drug name vs. text (28 classes, validation)

| Model | Macro F1 |
|---|---|
| Text-only | 0.737 |
| Drug-only | 0.741 |
| Text + drug | 0.854 |

- Text + drug has a higher F1 than both single models in all 28 classes. Five classes differ by less than 0.01 against drug-only (Vaginal Yeast Infection, Bowel Preparation, Constipation, ADHD, Erectile Dysfunction); there the drug alone is already near the ceiling (F1 0.95 to 0.99), so the text adds little and the differences are too small to claim on a single split.
- **Drug-only is strong where drugs are distinctive** (F1: Muscle Spasm 0.90, Migraine 0.93, Pain 0.90) **and collapses where drugs overlap:** Abnormal Uterine Bleeding recall 0.02 (97% predicted as Birth Control), Generalized Anxiety Disorder and Panic Disorder F1 0.00.
- Text and drug are complementary. Unlike class weighting, the combined model did not trade precision for recall: the larger classes also improved (Depression 0.69 → 0.80, Anxiety 0.61 → 0.75, Birth Control 0.93 → 0.96).
- Drug columns did not take over the text: Abnormal Uterine Bleeding recall (0.36) is above both single models, so no scaling experiment was needed.

| Class (text-only → text + drug) | Recall | Precision | F1 |
|---|---|---|---|
| Generalized Anxiety Disorder | 0.07 → 0.27 | 0.48 → 0.65 | 0.12 → 0.38 |
| Panic Disorder | 0.32 → 0.39 | 0.52 → 0.63 | 0.39 → 0.48 |
| Anxiety and Stress | 0.22 → 0.49 | 0.64 → 0.60 | 0.33 → 0.54 |
| Abnormal Uterine Bleeding | 0.31 → 0.36 | 0.74 → 0.74 | 0.44 → 0.49 |
| Muscle Spasm | 0.43 → 0.94 | 0.85 → 0.93 | 0.57 → 0.94 |

The four weakest classes after adding the drug are still Generalized Anxiety Disorder (0.38), Panic Disorder (0.48), Abnormal Uterine Bleeding (0.49) and Anxiety and Stress (0.54). Many classes are near-perfect with the drug (e.g. Erectile Dysfunction), because a drug identifies them; this reflects drug identity, not language understanding.

### 30-class pair check (original labels, validation)

Text-only and drug-only models refitted on the original 30 labels (before the two merges): macro F1 0.697 (text-only) and 0.703 (drug-only).

| Class | Validation rows | Recall text → drug | F1 text → drug |
|---|---|---|---|
| Depression | 1,245 | 0.756 → 0.697 | 0.628 → 0.620 |
| Major Depressive Disorder | 212 | 0.066 → 0.094 | 0.118 → 0.165 |
| Pain | 871 | 0.837 → 0.871 | 0.709 → 0.805 |
| Chronic Pain | 206 | 0.175 → 0.209 | 0.270 → 0.320 |

- Neither the text nor the drug separates Major Depressive Disorder from Depression (recall under 0.10 in both) or Chronic Pain from Pain (under 0.21 in both). This supports the two merges, which were decided on the text evidence.
- The drug helps Pain overall (F1 0.71 → 0.81) but does not rescue Chronic Pain, so the drug alone does not determine the label for these pairs either.
- Limitation: a text + drug model was not fitted on the 30 labels, so it is not known whether the combination separates these pairs.

### Where the text + drug model sends the anxiety family (28 classes, validation)

Share of each class's validation reviews (rows sum to about 1; the remainder goes to other classes):

| True class | Correct | To the other three anxiety classes | To Depression |
|---|---|---|---|
| Anxiety | 0.78 | 0.04 | 0.13 |
| Anxiety and Stress | 0.49 | 0.09 | 0.39 |
| Generalized Anxiety Disorder | 0.27 | 0.41 | 0.24 |
| Panic Disorder | 0.39 | 0.46 | 0.14 |

- **Generalized Anxiety Disorder and Panic Disorder:** most misses stay inside the anxiety family (about 56% and 75% of their misses). These are errors between related labels.
- **Anxiety and Stress:** most misses go to Depression (about 76% of its misses), so merging inside the family would not address its main confusion.
- **Decision (made before looking, on validation):** a mixed picture means the four classes stay separate. Merging would only remove weak classes from the average and would need a rebuild and refit. The confusion with Depression is the more clinically relevant error, since the diagnoses differ, and the model's errors inside the family are arguably less consequential.

### Short reviews (623 reviews of 1 to 4 words, report only)

These reviews were set aside in cleaning. They were not used for training, tuning or choosing the model: the three finished models were run on them with transform only (no refitting). Labels use the same two merges, and all 623 labels fall within the 28 training classes. 7 rows (1.1%) have a drug unseen in train.

| Model | Accuracy |
|---|---|
| Text-only | 0.281 |
| Drug-only | 0.785 |
| Text + drug | 0.809 |
| Reference: always predict the largest class (Pain, 129 of 623) | 0.207 |

- **The class mix differs from the main data:** Birth Control is only 21 of 623 reviews (3.4%, versus 24% in train), and Pain, Depression and Anxiety make up 44%. Short reviews come from different conditions than long ones.
- **The text carries almost nothing here:** text-only is only a little above the majority guess. The drug carries nearly all the signal. Adding the text to the drug gains 0.024 (about 15 reviews), too small to read much into on 623 reviews.
- Many classes have fewer than 10 reviews, so per-class scores are not reported.
- This result is reported, not used to change the model; text + drug remains the best model.

### Test file cleaning and final test

**Cleaning.** The Kaggle test file (`drugsComTest_raw.csv`, 53,766 rows) was cleaned by replaying the training recipe in the same order, using the same fixed dictionaries and rules. The 30 classes were taken from train, and nothing was learned from the test file. The two merges were then applied, as for train.

| Step | Rows removed |
|---|---|
| Missing condition | 295 |
| `</span>` junk | 271 |
| Leftover cut-off values | 88 |
| Duplicates (review text + condition + date) | 5,335 |
| New duplicates after the "Pe" repair | 1 |
| Multi-condition long texts | 13 |
| Outside the 30 classes | 15,156 |
| Reviews under 5 words | 270 |
| Duplicates after the merges | 1 |
| Text also in train | 17,728 |
| Text also in validation | 4,372 |
| **Final test set** | **10,236 rows, 28 classes** |

- **Twins across train and test.** All 17,728 test rows removed for matching a train review were the same review under a different drug name, with the same condition and date (100% on each of the three checks). About 55% of the test rows that survived cleaning were copies of training reviews, because the Kaggle split was made by row and brand/generic twin pairs landed on both sides. Scoring a model on the raw test file would overstate its performance.
- **Validation twins.** A further 4,372 rows matching validation reviews were removed, because validation was used to choose the model. The final test set contains no review text seen in train or validation.
- **Class sizes.** The smallest test classes are Erectile Dysfunction (71 rows), Osteoarthritis (100), Bowel Preparation (107) and Abnormal Uterine Bleeding (112), so their per-class scores are noisy.

**Final test result.** The frozen pipeline (text + drug name, logistic regression) was run once, with no refitting:

| | Macro F1 |
|---|---|
| Validation | 0.854 |
| **Final test** | **0.855** |

- The result matches validation, so there is no sign that the model choices were overfitted to the validation set. A difference of about 0.01 either way would also have been normal on a test set this size.
- **The same four classes are weakest** on the test set as on validation: Generalized Anxiety Disorder 0.449, Panic Disorder 0.455, Anxiety and Stress 0.526 and Abnormal Uterine Bleeding 0.555. The overlapping-label finding is therefore not a validation-set accident.
- **The strongest classes** are the ones with their own drugs or vocabulary: Vaginal Yeast Infection 0.998, Emergency Contraception 0.989, Erectile Dysfunction 0.986 and Weight Management 0.986.

| Class | Validation F1 | Test F1 |
|---|---|---|
| Generalized Anxiety Disorder | 0.378 | 0.449 |
| Panic Disorder | 0.483 | 0.455 |
| Anxiety and Stress | 0.543 | 0.526 |
| Abnormal Uterine Bleeding | 0.486 | 0.555 |
| Anxiety | 0.745 | 0.754 |
| Depression | 0.801 | 0.742 |

Depression is the largest drop among the big classes (recall 0.850 → 0.804, precision 0.758 → 0.689); the reason was not investigated. Only the final model was run on the test set, so the text-only (0.737) and drug-only (0.741) comparisons rest on validation alone.

## Limitations
- **Overlapping labels limit macro F1.** The anxiety family, Abnormal Uterine Bleeding vs. Birth Control, and the merged pairs share vocabulary and often share drugs. Part of the label seems to be a choice among the conditions a drug is listed for, which text alone cannot recover.
- **The drug-name models are not pure NLP results.** In real use the drug is usually known, but the write-up reports the text-only and drug-only models separately.
- **Brand and generic drug names are separate categories** for the encoder, which splits the drug signal.
- **Short reviews (1 to 4 words):** text alone is nearly useless (accuracy 0.28 vs. 0.81 with the drug); the model relies on the drug.
- **Rare cut-off condition names** outside the top 30 were not repaired in cleaning. On the test file, none of the rows dropped as "outside the 30 classes" looked like a shortened class name.
- **One split and one test run.** The model choices were made on a single validation split, and the comparisons between models rest on validation only.
- **Small test classes** (71 to 112 rows) have noisy per-class scores.
- **How Drugs.com assigns conditions was not verified**, and the dataset dates from 2018, so the site's current condition lists may differ.

## Demo / App
A Streamlit app (a review goes in, a predicted condition with a confidence score comes out) is planned but not built yet, so there is no demo link.

## Project Structure
```
drug-review-condition-classifier/
├── data/                                           # git-ignored: raw data, cleaned files, checkpoints
├── EDA.ipynb                                       # exploratory analysis of the raw training data
├── Cleaning.ipynb                                  # cleaning the training data
├── Preprocessing and Modelling.ipynb               # split, features, models, validation results
├── Test file cleaning and model testing.ipynb      # test file cleaning and the final test
├── drug_review_target_distribution.png             # figure from the EDA
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup
```bash
git clone https://github.com/praiseemma/drug-review-condition-classifier.git
cd drug-review-condition-classifier
conda create -n drug-review python=3.10
conda activate drug-review
pip install -r requirements.txt
```

**Dataset download (not included in repo):**
1. Visit https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018 (free Kaggle account required)
2. Click **Download**, extract the zip
3. Place `drugsComTrain_raw.csv` and `drugsComTest_raw.csv` in `data/` (this folder is git-ignored)

**Run the notebooks in this order:**
1. `EDA.ipynb` explores the raw training data and, at the end, saves it as `data/Drug Train Dataframe.csv`.
2. `Cleaning.ipynb` reads `data/Drug Train Dataframe.csv`, then writes `data/cleaned_train.csv` and the set-aside tables.
3. `Preprocessing and Modelling.ipynb` reads `data/cleaned_train.csv` and saves its results to `data/checkpoints/`.
4. `Test file cleaning and model testing.ipynb` reads `data/drugsComTest_raw.csv` and the saved checkpoints, and runs the final test once.

## Planned Extension: Drug Ranking by Patient-Reported Outcomes

Beyond the core classification task (condition prediction from review text), a 
second, applied layer is planned: **ranking which drugs have the best 
patient-reported outcomes for a given condition**, built on top of the trained 
NLP model rather than replacing it.

**Why this is a layer on top of the classifier, not a substitute for it:** ranking 
drugs by raw average rating alone (`groupby(drug, condition)['rating'].mean()`) would 
not require any NLP and would sideline the review text entirely — a weaker use of 
this dataset given the project's focus. Instead, the ranking will draw on richer 
signals extracted from review text by the trained model (e.g., mentions of efficacy, 
side effects, onset time), producing a more nuanced ranking than the star rating 
alone can offer. The set-aside tables keep real drug names and ratings for this purpose.

**Two known challenges to address if this extension is built:**
- **Small-sample bias:** a drug with only a handful of reviews, all highly rated, 
  would naively outrank a drug with thousands of reviews at a slightly lower average. 
  A weighted/Bayesian average (accounting for review volume, not just the mean) will 
  be needed instead of a simple mean.
- **No ground truth to validate against:** unlike the classifier (which can be 
  scored with accuracy/F1 on held-out data), a ranking has no obvious "correct 
  answer." Validation will likely rely on sanity checks — e.g., correlation with 
  `usefulCount`, stability across different data splits — rather than a single 
  accuracy metric.

**Framing note:** results should be described as *patient-reported outcomes* (i.e., 
what reviewers say worked for them), not treatment recommendations or clinical 
efficacy — this is self-reported survey data, not a clinical trial, and the writeup 
should be careful not to imply medical advice.

## Future Work
- [ ] Complete the model comparisons listed under *Pending decisions and next steps*
- [ ] Try transformer embeddings (e.g. DistilBERT) beyond TF-IDF
- [ ] Expand clinical error analysis with more examples (read actual misclassified reviews, not just rates)
- [ ] Streamlit app (review in, predicted condition out)
- [ ] Drug ranking by patient-reported outcomes

## Author
**Praise**, a third-year medical student. This project combines clinical judgement with NLP: data cleaning, leak-free modelling and a clinical error analysis.

[LinkedIn](https://www.linkedin.com/in/praise-emma-2a5b3b265/) · [GitHub](https://github.com/praiseemma) · emmaiwuozopraise@gmail.com
