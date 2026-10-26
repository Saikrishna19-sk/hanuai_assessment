# HanuAi Machine Learning Assessment

## Project Overview

This project contains two data analysis tasks:

1. **Customer Review Analysis** — analyzes review text and assigns sentiment labels.
2. **Maintenance Data Analysis** — explores maintenance records, summarizes numeric fields, categorizes common issues, and analyzes text descriptions.

## Project Structure

```text
hanuai_assessment/
├── data/
│   ├── input/
│   │   ├── reviews.csv
│   │   └── maintenance_data.csv
│   └── processed/
├── outputs/
│   ├── charts/
│   └── reports/
├── src/
│   ├── task1_reviews.py
│   ├── task2_eda.py
│   ├── generate_summary.py
│   ├── validate_data.py
│   └── text_mining.py
└── README.md
```

## Requirements

Install the required packages:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn vaderSentiment openpyxl
```

## Input Data

- `reviews.csv` should include review text and rating columns.
- `maintenance_data.csv` should include asset, description, downtime hours, and maintenance cost.

Use authorized datasets for the final assessment. Clearly identify synthetic data if it is used for testing.

## How to Run

Run these commands from the project root directory:

```bash
python src/validate_data.py
python src/task1_reviews.py
python src/task2_eda.py
python src/text_mining.py
python src/generate_summary.py
```

## Outputs

Generated outputs may include:

- Sentiment-labeled customer reviews
- Maintenance analysis CSV files
- Numeric summary and missing-value reports
- Keyword analysis CSV
- Charts and stakeholder reports
- Final assessment summary

## Data Integrity

Check that all output files are generated from the intended input data. Do not treat sample or synthetic data as actual customer feedback or real maintenance records.
