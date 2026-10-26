from pathlib import Path
import re
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "input"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHART_DIR = BASE_DIR / "outputs" / "charts"
REPORT_DIR = BASE_DIR / "outputs" / "reports"


def classify_issue(text):
    text = str(text).lower()

    rules = {
        "Electrical": [
            "electrical", "wiring", "power", "voltage", "circuit"
        ],
        "Mechanical": [
            "motor", "bearing", "vibration", "gear", "overheating"
        ],
        "Hydraulic": [
            "hydraulic", "pressure", "pump", "leak", "fluid"
        ],
        "Preventive Maintenance": [
            "inspection", "lubrication", "preventive", "routine"
        ],
        "Software/Control": [
            "software", "sensor", "controller", "alarm"
        ],
    }

    for category, keywords in rules.items():
        if any(word in text for word in keywords):
            return category

    return "Other/Review Required"


def load_dataset():
    # Ignore the product review dataset.
    candidates = [
        p for p in INPUT_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in [".csv", ".xlsx", ".xls"]
        and p.name.lower() not in ["reviews.csv"]
        and p.stat().st_size > 0
    ]

    if not candidates:
        raise FileNotFoundError(
            "No non-empty maintenance dataset found in data/input. "
            "Create data/input/maintenance_data.csv first."
        )

    # Prefer the explicitly named CSV, otherwise use the first non-empty file.
    candidates.sort(
        key=lambda p: (
            p.name.lower() != "maintenance_data.csv",
            p.name.lower() != "maintenance_data.xlsx",
            p.name.lower(),
        )
    )

    file_path = candidates[0]

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    if df.empty:
        raise ValueError(f"{file_path.name} contains no data rows.")

    return file_path, df


def main():
    for folder in [PROCESSED_DIR, CHART_DIR, REPORT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    try:
        input_file, df = load_dataset()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return

    print(f"Loaded input file: {input_file.name}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    # Basic data cleaning
    df.columns = [str(col).strip() for col in df.columns]
    df = df.drop_duplicates().copy()

    # EDA outputs
    numeric_df = df.select_dtypes(include="number")
    numeric_summary = numeric_df.describe().transpose()
    numeric_summary.to_csv(
        PROCESSED_DIR / "task2_numeric_summary.csv",
        encoding="utf-8-sig"
    )

    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": [int(df[col].isna().sum()) for col in df.columns],
    })
    missing_df["missing_percent"] = (
        missing_df["missing_count"] / len(df) * 100
    )
    missing_df.to_csv(
        PROCESSED_DIR / "task2_missing_values.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Missing-value chart
    plt.figure()
    df.isna().sum().plot(kind="bar")
    plt.title("Missing Values by Column")
    plt.xlabel("Column")
    plt.ylabel("Missing values")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "task2_missing_values.png", dpi=150)
    plt.close()

    # Find a suitable text column
    text_candidates = [
        "description", "issue_description", "problem",
        "issue", "maintenance_notes", "notes", "text"
    ]
    lookup = {col.lower(): col for col in df.columns}
    text_col = next(
        (lookup[name] for name in text_candidates if name in lookup),
        None
    )

    if text_col:
        df[text_col] = df[text_col].fillna("").astype(str)
        df["issue_category"] = df[text_col].apply(classify_issue)

        # Text clustering requires multiple distinct non-empty descriptions.
        nonempty = df[text_col].str.strip().ne("")
        text_series = df.loc[nonempty, text_col]

        if len(text_series) >= 2 and text_series.nunique() >= 2:
            try:
                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    max_features=1000
                )
                matrix = vectorizer.fit_transform(text_series)

                if matrix.shape[1] > 0:
                    n_clusters = min(3, len(text_series))
                    model = KMeans(
                        n_clusters=n_clusters,
                        random_state=42,
                        n_init=10
                    )
                    labels = model.fit_predict(matrix)
                    df["text_cluster"] = pd.NA
                    df.loc[text_series.index, "text_cluster"] = labels
            except ValueError as exc:
                print(f"Clustering skipped: {exc}")

        category_counts = df["issue_category"].value_counts()

        plt.figure()
        category_counts.plot(kind="bar")
        plt.title("Maintenance Issue Categories")
        plt.xlabel("Category")
        plt.ylabel("Record count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "task2_issue_categories.png", dpi=150)
        plt.close()

    # Save processed records
    output_csv = PROCESSED_DIR / "task2_maintenance_analysis.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Create report
    report_lines = [
        "TASK 2: MAINTENANCE DATA ANALYSIS",
        "=" * 45,
        f"Source file: {input_file.name}",
        f"Records analysed: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Duplicate rows after cleaning: {int(df.duplicated().sum())}",
        "",
        "Column names:",
        *[f"- {col}" for col in df.columns],
        "",
        "Missing values:",
        missing_df.to_string(index=False),
        "",
    ]

    if "issue_category" in df.columns:
        report_lines.extend([
            "Issue category counts:",
            df["issue_category"].value_counts().to_string(),
            "",
        ])

    report_lines.extend([
        "Limitations:",
        "- Categories are assigned using simple keyword rules.",
        "- Clusters are exploratory groups, not confirmed root causes.",
        "- Review results with a maintenance expert before making decisions.",
    ])

    report_file = REPORT_DIR / "task2_eda_report.txt"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nTASK 2 SUCCESS")
    print(f"Processed records: {output_csv}")
    print(f"Numeric summary: {PROCESSED_DIR / 'task2_numeric_summary.csv'}")
    print(f"Missing values: {PROCESSED_DIR / 'task2_missing_values.csv'}")
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()
