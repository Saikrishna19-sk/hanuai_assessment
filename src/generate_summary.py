
from pathlib import Path
import pandas as pd
from datetime import datetime

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

TASK1_FILE = PROCESSED_DIR / "task1_reviews_with_sentiment.csv"
TASK2_FILE = PROCESSED_DIR / "task2_maintenance_analysis.csv"
NUMERIC_FILE = PROCESSED_DIR / "task2_numeric_summary.csv"
MISSING_FILE = PROCESSED_DIR / "task2_missing_values.csv"

OUTPUT_FILE = REPORTS_DIR / "assessment_summary.txt"


def read_csv_safely(file_path):
    """Read a CSV safely; return None if missing, empty, or invalid."""
    if not file_path.exists():
        return None

    if file_path.stat().st_size == 0:
        return None

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        if df.empty or len(df.columns) == 0:
            return None
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError,
            UnicodeDecodeError, OSError) as exc:
        print(f"Could not read {file_path.name}: {exc}")
        return None


def find_column(df, possible_names):
    """Find a column using case-insensitive names."""
    if df is None:
        return None

    lookup = {str(col).strip().lower(): col for col in df.columns}

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def add_task1_summary(lines):
    lines.append("=" * 60)
    lines.append("TASK 1: PRODUCT REVIEW SENTIMENT ANALYSIS")
    lines.append("=" * 60)

    df = read_csv_safely(TASK1_FILE)

    if df is None:
        lines.append(
            "Task 1 results are unavailable because the output CSV "
            "is missing, empty, or invalid."
        )
        lines.append(
            "Check data/input/reviews.csv and run "
            "python src/task1_reviews.py"
        )
        lines.append("")
        return

    lines.append(f"Total reviews analysed: {len(df)}")

    sentiment_col = find_column(
        df, ["sentiment", "predicted_sentiment", "sentiment_label"]
    )

    if sentiment_col:
        counts = df[sentiment_col].fillna("Unknown").astype(str).value_counts()

        lines.append("\nSentiment distribution:")
        for sentiment, count in counts.items():
            percentage = count / len(df) * 100
            lines.append(
                f"  {sentiment}: {count} reviews ({percentage:.1f}%)"
            )

        positive_count = sum(
            count for sentiment, count in counts.items()
            if "positive" in sentiment.lower()
        )
        negative_count = sum(
            count for sentiment, count in counts.items()
            if "negative" in sentiment.lower()
        )
        neutral_count = sum(
            count for sentiment, count in counts.items()
            if "neutral" in sentiment.lower()
        )

        lines.append("\nKey findings:")
        lines.append(
            f"- Positive reviews: {positive_count} "
            f"({positive_count / len(df) * 100:.1f}%)"
        )
        lines.append(
            f"- Negative reviews: {negative_count} "
            f"({negative_count / len(df) * 100:.1f}%)"
        )
        lines.append(
            f"- Neutral reviews: {neutral_count} "
            f"({neutral_count / len(df) * 100:.1f}%)"
        )
    else:
        lines.append(
            "\nSentiment column was not found. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )

    rating_col = find_column(df, ["rating", "score", "stars"])

    if rating_col:
        ratings = pd.to_numeric(df[rating_col], errors="coerce").dropna()
        if not ratings.empty:
            lines.append(f"\nAverage rating: {ratings.mean():.2f}")

    lines.append(
        "\nRecommendation: Review recurring negative feedback "
        "to identify product-quality and customer-experience improvements."
    )
    lines.append("")


def add_task2_summary(lines):
    lines.append("=" * 60)
    lines.append("TASK 2: MAINTENANCE DATA ANALYSIS")
    lines.append("=" * 60)

    df = read_csv_safely(TASK2_FILE)

    if df is None:
        lines.append(
            "Task 2 results are unavailable because the output CSV "
            "is missing, empty, or invalid."
        )
        lines.append(
            "Add the maintenance dataset to data/input/ and run "
            "python src/task2_eda.py"
        )
        lines.append("")
    else:
        lines.append(f"Total maintenance records analysed: {len(df)}")
        lines.append(f"Number of columns: {len(df.columns)}")

        lines.append("\nColumns in the dataset:")
        for column in df.columns:
            lines.append(f"- {column}")

        issue_col = find_column(
            df, ["issue_category", "issue_type", "category", "issue"]
        )

        if issue_col:
            counts = (
                df[issue_col]
                .fillna("Unspecified")
                .astype(str)
                .value_counts()
            )
            lines.append("\nMost common issue categories:")
            for category, count in counts.head(10).items():
                lines.append(f"- {category}: {count}")

        lines.append("")

    numeric_df = read_csv_safely(NUMERIC_FILE)
    if numeric_df is not None:
        lines.append("Numeric summary:")
        lines.append(numeric_df.head(20).to_string(index=False))
        lines.append("")

    missing_df = read_csv_safely(MISSING_FILE)
    if missing_df is not None:
        lines.append("Missing-value summary:")
        lines.append(missing_df.head(20).to_string(index=False))
        lines.append("")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "HANUAI ML ASSESSMENT - COMBINED SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Project directory: {BASE_DIR}",
        "",
    ]

    add_task1_summary(lines)
    add_task2_summary(lines)

    lines.extend([
        "=" * 60,
        "OVERALL NOTES",
        "=" * 60,
        "- Verify that input data is genuine and authorised for use.",
        "- Treat sentiment labels and automated issue categories as "
        "estimates that may require manual validation.",
        "- Do not report findings from missing data as actual results.",
        "",
        "End of report.",
    ])

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\nSummary report created successfully:")
    print(OUTPUT_FILE)

    print("\nInput/output status:")
    for path in [TASK1_FILE, TASK2_FILE, NUMERIC_FILE, MISSING_FILE]:
        status = "available" if path.exists() and path.stat().st_size > 0 \
            else "missing or empty"
        print(f"- {path.name}: {status}")


if __name__ == "__main__":
    main()
