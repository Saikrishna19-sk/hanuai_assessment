
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    "data/input/reviews.csv",
    "data/input/maintenance_data.csv",
    "data/processed/task1_reviews_with_sentiment.csv",
    "data/processed/task2_maintenance_analysis.csv",
    "data/processed/task2_numeric_summary.csv",
    "data/processed/task2_missing_values.csv",
    "outputs/charts/sentiment_distribution.png",
    "outputs/reports/task1_stakeholder_report.txt",
    "outputs/reports/task2_stakeholder_report.txt",
    "outputs/reports/assessment_summary.txt",
]


def main():
    print("HANUAI PROJECT - FINAL FILE CHECK\n")
    missing = []
    empty = []

    for relative_path in REQUIRED_FILES:
        path = BASE_DIR / relative_path

        if not path.exists():
            missing.append(relative_path)
            print("[MISSING]", relative_path)
        elif path.stat().st_size == 0:
            empty.append(relative_path)
            print("[EMPTY]  ", relative_path)
        else:
            print("[OK]     ", relative_path)

    print("\nVerification summary")
    print("Missing files:", len(missing))
    print("Empty files:", len(empty))

    if missing or empty:
        print("STATUS: Some files need attention.")
    else:
        print("STATUS: Required files exist and are non-empty.")
        print("Note: File existence does not prove analysis correctness.")


if __name__ == "__main__":
    main()
