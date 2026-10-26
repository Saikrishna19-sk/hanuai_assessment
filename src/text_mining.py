
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "input" / "maintenance_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "maintenance_keywords.csv"


def main():
    if not INPUT_FILE.exists() or INPUT_FILE.stat().st_size == 0:
        print("Maintenance input CSV missing or empty:", INPUT_FILE)
        return

    df = pd.read_csv(INPUT_FILE)

    if "description" not in df.columns:
        print("ERROR: CSV must contain a 'description' column.")
        return

    descriptions = df["description"].fillna("").astype(str)

    vectorizer = CountVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        max_features=30,
    )

    try:
        matrix = vectorizer.fit_transform(descriptions)
    except ValueError:
        print("No usable text found in descriptions.")
        return

    totals = matrix.sum(axis=0).A1
    keywords = vectorizer.get_feature_names_out()

    result = pd.DataFrame({
        "keyword": keywords,
        "frequency": totals,
    }).sort_values("frequency", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("Keyword analysis completed.")
    print("Output:", OUTPUT_FILE)
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
