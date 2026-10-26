
"""
HANUAI ASSESSMENT - TASK 2
Exploratory Data Analysis and maintenance text mining.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"
OUTPUT_DIR = ROOT / "data" / "processed"
CHART_DIR = ROOT / "outputs" / "charts"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [OUTPUT_DIR, CHART_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def read_input():
    """Load the first supported dataset in data/input."""
    files = sorted(
        list(INPUT_DIR.glob("*.csv"))
        + list(INPUT_DIR.glob("*.xlsx"))
        + list(INPUT_DIR.glob("*.xls"))
    )
    # Ignore the Task 1 review dataset.
    files = [f for f in files if f.name.lower() != "reviews.csv"]

    if not files:
        raise FileNotFoundError(
            "Place your maintenance dataset (CSV or Excel) in data/input."
        )

    path = files[0]
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        df = pd.read_excel(path)

    df.columns = [
        re.sub(r"\s+", "_", str(c).strip().lower())
        for c in df.columns
    ]
    print(f"Loaded {path.name}: {df.shape}")
    return df


def identify_text_columns(df):
    """Identify likely free-text fields from their length and variability."""
    columns = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        values = df[col].dropna().astype(str)
        if not values.empty:
            avg_length = values.str.len().mean()
            unique_ratio = values.nunique() / len(values)
            if avg_length >= 20 and unique_ratio > 0.05:
                columns.append(col)
    return columns


def extract_tags(value):
    """Extract indicative maintenance tags using explainable rules."""
    text = "" if pd.isna(value) else str(value).lower()

    rules = {
        "component_failure": [
            r"\bbroken\b", r"\bfailure\b", r"\bfailed\b",
            r"\bcrack(?:ed|ing)?\b", r"\bfracture\b",
            r"\bworn out\b", r"\bdamaged\b", r"\bmalfunction"
        ],
        "electrical_issue": [
            r"\belectrical\b", r"\bshort circuit\b", r"\bwiring\b",
            r"\bfuse\b", r"\bvoltage\b", r"\bbattery\b",
            r"\bpower outage\b"
        ],
        "mechanical_issue": [
            r"\bvibration\b", r"\bnoise\b", r"\bbearing\b",
            r"\bmisalignment\b", r"\bleak(?:age|ing)?\b",
            r"\boverheat(?:ing)?\b", r"\bjammed\b", r"\bgear\b"
        ],
        "road_surface_issue": [
            r"\bpothole", r"\basphalt\b", r"\bpavement\b",
            r"\brutting\b", r"\broad surface\b", r"\bsurface crack"
        ],
        "water_drainage_issue": [
            r"\bflood", r"\bdrain", r"\bwaterlogging\b",
            r"\bstanding water\b", r"\bblocked culvert\b"
        ],
        "corrosion_issue": [
            r"\bcorrosion\b", r"\brust(?:ed|ing)?\b",
            r"\boxidation\b"
        ],
        "safety_issue": [
            r"\bsafety\b", r"\bhazard\b", r"\bunsafe\b",
            r"\baccident\b", r"\binjury\b"
        ],
        "maintenance_delay": [
            r"\bdelay(?:ed)?\b", r"\boverdue\b",
            r"\bbacklog\b", r"\bnot repaired\b"
        ],
    }

    matches = [
        tag
        for tag, patterns in rules.items()
        if any(re.search(p, text) for p in patterns)
    ]
    return "; ".join(matches) if matches else "unclassified"


def run():
    df = read_input()

    # Profile missing values, data types and duplicate records.
    profile = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [
            round(float(df[c].isna().mean() * 100), 2)
            for c in df.columns
        ],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    profile.to_csv(OUTPUT_DIR / "task2_eda_summary.csv", index=False)

    # Preserve the raw input; create a separate cleaned dataset.
    cleaned = df.drop_duplicates().copy()
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = cleaned[col].map(
            lambda v: re.sub(r"\s+", " ", v).strip()
            if isinstance(v, str) else v
        )
    cleaned.to_csv(
        OUTPUT_DIR / "task2_cleaned_data.csv",
        index=False, encoding="utf-8-sig"
    )

    numeric = cleaned.select_dtypes(include=np.number)
    if not numeric.empty:
        numeric.describe().T.to_csv(
            OUTPUT_DIR / "task2_numeric_statistics.csv"
        )
        corr = numeric.corr()
        corr.to_csv(OUTPUT_DIR / "task2_numeric_correlations.csv")

        if len(numeric.columns) >= 2:
            plt.figure(figsize=(10, 7))
            plt.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar(label="Correlation")
            plt.xticks(range(len(corr.columns)), corr.columns,
                       rotation=90)
            plt.yticks(range(len(corr.index)), corr.index)
            plt.title("Numeric Feature Correlations")
            plt.tight_layout()
            plt.savefig(CHART_DIR / "correlation_heatmap.png", dpi=150)
            plt.close()

    text_columns = identify_text_columns(cleaned)
    pd.DataFrame({"identified_text_column": text_columns}).to_csv(
        OUTPUT_DIR / "task2_text_columns.csv", index=False
    )

    if not text_columns:
        raise ValueError(
            "No likely free-text columns detected. "
            "Inspect task2_eda_summary.csv and identify the correct "
            "description/incident column."
        )

    # Combine free-text columns so every row gets one consolidated
    # description for transparent tag extraction.
    cleaned["combined_text"] = cleaned[text_columns].fillna("").astype(str).agg(
        " ".join, axis=1
    )
    cleaned["issue_tags"] = cleaned["combined_text"].map(extract_tags)

    # Keyword tags are initial hypotheses, not confirmed root causes.
    cleaned["primary_issue_category"] = cleaned["issue_tags"].map(
        lambda s: s.split("; ")[0]
    )

    cleaned.to_csv(
        OUTPUT_DIR / "task2_text_mining_results.csv",
        index=False, encoding="utf-8-sig"
    )

    # Summarise tag frequencies.
    tag_counts = (
        cleaned["issue_tags"].str.split("; ").explode()
        .value_counts().rename_axis("tag").reset_index(name="count")
    )
    tag_counts.to_csv(
        OUTPUT_DIR / "task2_issue_frequency.csv", index=False
    )

    plt.figure(figsize=(10, 6))
    top = tag_counts.head(15).sort_values("count")
    plt.barh(top["tag"], top["count"])
    plt.title("Most Frequent Maintenance Issue Tags")
    plt.xlabel("Number of records")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "issue_frequency.png", dpi=150)
    plt.close()

    # TF-IDF + KMeans groups similar descriptions.
    # Very small datasets may not support meaningful clustering.
    docs = cleaned["combined_text"].fillna("").astype(str)
    usable = docs.str.strip().ne("")

    if usable.sum() >= 3 and docs[usable].nunique() >= 3:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=3000,
            ngram_range=(1, 2),
            min_df=1
        )
        matrix = vectorizer.fit_transform(docs[usable])

        if matrix.shape[1] >= 2:
            k = min(5, int(usable.sum()) - 1)
            if k >= 2:
                model = KMeans(
                    n_clusters=k, random_state=42, n_init=10
                )
                labels = model.fit_predict(matrix)
                cleaned["cluster"] = pd.NA
                cleaned.loc[usable, "cluster"] = labels

                terms = vectorizer.get_feature_names_out()
                with open(
                    REPORT_DIR / "task2_cluster_keywords.txt",
                    "w", encoding="utf-8"
                ) as f:
                    for cluster_id, centre in enumerate(model.cluster_centers_):
                        top_ids = centre.argsort()[::-1][:10]
                        keywords = [terms[i] for i in top_ids]
                        f.write(
                            f"Cluster {cluster_id}: "
                            + ", ".join(keywords) + "\n"
                        )

                cleaned.to_csv(
                    OUTPUT_DIR / "task2_cluster_results.csv",
                    index=False, encoding="utf-8-sig"
                )
    else:
        print("Not enough distinct non-empty texts for clustering.")

    # Build a report using observed dataset statistics only.
    with open(
        REPORT_DIR / "task2_stakeholder_report.txt",
        "w", encoding="utf-8"
    ) as f:
        f.write("HANUAI - TASK 2 EDA AND TEXT MINING\n\n")
        f.write(f"Original rows: {len(df)}\n")
        f.write(f"Rows after exact duplicate removal: {len(cleaned)}\n")
        f.write(f"Exact duplicate rows removed: {len(df) - len(cleaned)}\n\n")
        f.write("Column quality profile:\n")
        f.write(profile.to_string(index=False))
        f.write("\n\nIdentified text columns:\n")
        f.write(", ".join(text_columns))
        f.write("\n\nIssue tag counts:\n")
        f.write(tag_counts.to_string(index=False))
        f.write(
            "\n\nRecommendations:\n"
            "- Validate extracted tags with maintenance specialists.\n"
            "- Prioritise recurring high-severity issues, not frequency alone.\n"
            "- Compare issue rates over time after accounting for exposure.\n"
            "- Investigate possible root causes using work orders, asset age, "
            "environment, component identifiers and repair history.\n"
            "- Treat unsupervised clusters as exploratory groups, not "
            "confirmed root causes.\n"
        )

    print("Task 2 completed.")
    print(f"Detected text columns: {text_columns}")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
