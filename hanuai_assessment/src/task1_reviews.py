
"""
HANUAI ASSESSMENT - TASK 1
Review sentiment analysis.

Input:
    data/input/reviews.csv

Important:
    Use an authorized review export or dataset.
    Automated scraping of Best Buy Canada requires permission.
    This script does not bypass CAPTCHA or access restrictions.
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "input" / "reviews.csv"
OUTPUT = ROOT / "data" / "processed"
CHARTS = ROOT / "outputs" / "charts"
REPORTS = ROOT / "outputs" / "reports"

for folder in [OUTPUT, CHARTS, REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)


def find_column(df, candidates):
    """Find a column despite common naming variations."""
    normalised = {
        re.sub(r"[^a-z0-9]", "", str(c).lower()): c
        for c in df.columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalised:
            return normalised[key]
    return None


def run():
    """Load reviews, calculate sentiment, and export results."""
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT}\n"
            "Add an authorized review dataset to data/input/reviews.csv."
        )

    df = pd.read_csv(INPUT, encoding="utf-8-sig")

    text_col = find_column(
        df, ["review_text", "review", "text", "comment", "content", "body"]
    )
    title_col = find_column(df, ["title", "review_title", "headline"])
    date_col = find_column(df, ["date", "review_date", "created_at"])
    rating_col = find_column(df, ["rating", "stars", "score"])
    name_col = find_column(df, ["reviewer_name", "reviewer", "author"])
    source_col = find_column(df, ["source", "website", "platform"])
    id_col = find_column(df, ["id", "review_id", "primary_key"])

    if text_col is None:
        raise ValueError(
            "Review text column not found. Use a column such as review_text."
        )

    def text(value):
        return "" if pd.isna(value) else re.sub(
            r"\s+", " ", str(value)
        ).strip()

    result = pd.DataFrame()
    result["ID"] = (
        df[id_col].astype(str)
        if id_col
        else [f"REVIEW_{i:06d}" for i in range(1, len(df) + 1)]
    )
    result["Title"] = df[title_col].map(text) if title_col else ""
    result["Review Text"] = df[text_col].map(text)

    result["Date"] = (
        pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
        if date_col else pd.NA
    )
    result["Rating"] = (
        pd.to_numeric(df[rating_col], errors="coerce")
        if rating_col else float("nan")
    )
    result["Rating"] = result["Rating"].where(
        result["Rating"].between(1, 5)
    )
    result["Source"] = (
        df[source_col].map(text)
        if source_col else "Not provided in input"
    )
    result["Reviewer Name"] = df[name_col].map(text) if name_col else ""

    # VADER provides a transparent baseline for English-language sentiment.
    analyzer = SentimentIntensityAnalyzer()

    def classify(review):
        if not review:
            return pd.Series([float("nan"), "Unknown"])

        score = analyzer.polarity_scores(review)["compound"]
        if score >= 0.05:
            label = "Positive"
        elif score <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return pd.Series([score, label])

    result[["Sentiment Score", "Sentiment"]] = (
        result["Review Text"].apply(classify)
    )

    # Keyword themes help stakeholders interpret review sentiment.
    theme_rules = {
        "Quality / durability": r"broken|defect|durab|crack|failed",
        "Performance": r"performance|slow|fast|not working|works well",
        "Ease of use": r"easy to use|difficult|setup|install|assemble",
        "Price / value": r"price|expensive|overpriced|worth it|value",
        "Delivery / packaging": r"delivery|shipping|late|packaging",
        "Customer service": r"support|refund|return|warranty",
        "Battery / power": r"battery|charging|charger|power issue",
        "Design / appearance": r"design|colour|color|appearance|compact",
    }

    def themes(review):
        found = [
            name for name, pattern in theme_rules.items()
            if re.search(pattern, review, flags=re.IGNORECASE)
        ]
        return "; ".join(found) if found else "Other / unclassified"

    result["Review Themes"] = result["Review Text"].map(themes)

    result.to_csv(
        OUTPUT / "task1_reviews_with_sentiment.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save a sentiment chart.
    counts = result["Sentiment"].value_counts()
    counts.plot(kind="bar", title="Customer Review Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Review count")
    plt.tight_layout()
    plt.savefig(CHARTS / "sentiment_distribution.png", dpi=150)
    plt.close()

    # Save a concise report with counts calculated from the actual data.
    theme_counts = (
        result["Review Themes"].str.split("; ").explode().value_counts()
    )
    with open(REPORTS / "task1_stakeholder_report.txt", "w",
              encoding="utf-8") as file:
        file.write("HANUAI - TASK 1 REVIEW ANALYSIS\n\n")
        file.write(f"Input review rows: {len(result)}\n\n")
        file.write("Sentiment distribution:\n")
        file.write(counts.to_string())
        file.write("\n\nTheme distribution:\n")
        file.write(theme_counts.to_string())
        file.write(
            "\n\nCaveat: VADER sentiment and keyword themes are estimates. "
            "Validate against manually labelled reviews before making "
            "business decisions.\n"
        )

    print("Task 1 completed successfully.")
    print(f"Reviews processed: {len(result)}")
    print(f"CSV: {OUTPUT / 'task1_reviews_with_sentiment.csv'}")
    print(f"Chart: {CHARTS / 'sentiment_distribution.png'}")
    print(f"Report: {REPORTS / 'task1_stakeholder_report.txt'}")


if __name__ == "__main__":
    run()
