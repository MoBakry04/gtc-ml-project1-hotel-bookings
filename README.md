# Hotel Booking Cancellation — Preprocessing Pipeline

**Objective:** Build a robust data preprocessing pipeline for a hotel booking cancellation prediction model. The business problem: last-minute booking cancellations reduce revenue — this pipeline prepares raw bookings data so a downstream model can reliably learn cancellation patterns.

---

## Table of contents

- [Overview](#overview)
- [What this repo contains](#what-this-repo-contains)
- [Quickstart (Colab / Local)](#quickstart-colab--local)
- [Dependencies](#dependencies)
- [What the preprocessing does (step-by-step)](#what-the-preprocessing-does-step-by-step)
- [Outputs](#outputs)
---

## Overview

This project contains a preprocessing pipeline that cleans, imputes, engineers features, encodes categorical variables, handles outliers, scales numeric features, and produces a reproducible train/test split for a hotel bookings dataset. The goal is **data quality** — not model training — so the downstream model (classification for `is_canceled`) will have good inputs.

---

## What this repo contains

- `notebook` / `script` (the provided code): loads `hotel_bookings.csv`, inspects missing values and duplicates, imputes missing values, does feature engineering, encodes categories, clips outliers, drops a few columns, and creates an 80/20 train/test split.

---

## Quickstart (Colab / Local)

### Colab

1. Upload `hotel_bookings.csv` to your Colab session or mount Google Drive.
2. Install dependencies (if missing):

```bash
!pip install pandas numpy scikit-learn matplotlib seaborn missingno
```

3. Run the notebook cells. A cleaned CSV `updated_dataset.csv` will be produced.

### Local

1. Create a virtual environment and activate it:

```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Place `hotel_bookings.csv` in the repo root and run the preprocessing script or notebook.

---

## Dependencies

A minimal `requirements.txt` example:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
missingno
```

Pin versions if you need exact reproducibility.

---

## What the preprocessing does (step-by-step)

1. **Load data** from `hotel_bookings.csv`.
2. **Initial EDA**: `df.describe()`, `df.info()`, missing-value counts, simple plots.
3. **Impute missing values** for selected columns (`children`, `country`, `agent`, `company`).
4. **Drop duplicates** and reset index.
5. **Feature engineering**:
   - `total_guests = adults + children + babies`
   - `total_nights = stays_in_weekend_nights + stays_in_week_nights`
   - `is_family (df['children'] + df['babies']) > 0`
6. **Outlier detection & handling** for `adr` and `lead_time` (IQR & percentile clipping attempted).
7. **Type casting** for object/string and numeric columns.
8. **Categorical encoding**: one-hot encode many categorical fields (e.g. `meal`, `market_segment`, `reserved_room_type`, etc.)
9. **Low-frequency consolidation** for `country` (countries with low counts are grouped into `Other`), and `country_freq` is computed.
10. **Drop leakage-prone / unwanted cols** (`country`, `reservation_status`, `reservation_status_date`), save `updated_dataset.csv`.
11. **Train/test split** with `train_test_split(..., test_size=0.2, random_state=42)`.

---

## Outputs

- `updated_dataset.csv` — cleaned dataset (as produced by the original notebook).
