
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "input" / "reviews.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHART_DIR = BASE_DIR / "outputs" / "charts"
REPORT_DIR = BASE_DIR / "outputs" / "reports"


def main():
    for folder in [PROCESSED_DIR, CHART_DIR, REPORT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists() or INPUT_FILE.stat().st_size == 0:
        print(f"ERROR: Reviews dataset not found or empty: {INPUT_FILE}")
        print("Place your authorised review dataset at data/input/reviews.csv")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as exc:
        print(f"Could not read reviews CSV: {exc}")
        return

    if df.empty:
        print("ERROR: reviews.csv contains no review rows.")
        return

    # Automatically identify a likely review-text column.
    candidates = [
        "review", "reviews", "review_text", "text",
        "content", "comment", "comments"
    ]
    column_lookup = {str(c).strip().lower(): c for c in df.columns}
    review_col = next(
        (column_lookup[name] for name in candidates if name in column_lookup),
        None
    )

    if review_col is None:
        print("ERROR: Could not identify the review text column.")
        print("Available columns:", list(df.columns))
        print("Rename your review text column to 'review_text' and retry.")
        return

    df = df.dropna(subset=[review_col]).copy()
    df[review_col] = df[review_col].astype(str).str.strip()
    df = df[df[review_col] != ""]

    if df.empty:
        print("ERROR: No non-empty reviews found.")
        return

    analyzer = SentimentIntensityAnalyzer()

    def analyse_sentiment(text):
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            label = "Positive"
        elif score <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return pd.Series([score, label])

    df[["sentiment_score", "sentiment"]] = df[review_col].apply(
        analyse_sentiment
    )

    output_csv = PROCESSED_DIR / "task1_reviews_with_sentiment.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    counts = df["sentiment"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"], fill_value=0
    )

    ax = counts.plot(kind="bar", rot=0)
    ax.set_title("Product Review Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "sentiment_distribution.png", dpi=150)
    plt.close()

    report_file = REPORT_DIR / "task1_stakeholder_report.txt"
    total = len(df)

    with report_file.open("w", encoding="utf-8") as report:
        report.write("TASK 1: PRODUCT REVIEW SENTIMENT ANALYSIS\n")
        report.write("=" * 45 + "\n")
        report.write(f"Reviews analysed: {total}\n\n")

        for sentiment, count in counts.items():
            report.write(
                f"{sentiment}: {count} ({count / total * 100:.1f}%)\n"
            )

        report.write(
            "\nMethod: VADER lexicon-based sentiment analysis.\n"
        )
        report.write(
            "Note: Automated sentiment labels are estimates and may "
            "require manual review.\n"
        )

    print(f"Task 1 complete. Analysed {total} reviews.")
    print(f"CSV: {output_csv}")
    print(f"Chart: {CHART_DIR / 'sentiment_distribution.png'}")
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()
