
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "input"

REQUIRED_COLUMNS = {
    "reviews.csv": ["review_text", "rating"],
    "maintenance_data.csv": [
        "asset",
        "description",
        "downtime_hours",
        "maintenance_cost",
    ],
}


def validate_csv(file_path, required_columns):
    print(f"\nChecking: {file_path.name}")

    if not file_path.exists():
        print("ERROR: File not found.")
        return False

    try:
        if file_path.stat().st_size == 0:
            print("ERROR: File is empty.")
            return False

        df = pd.read_csv(file_path)

        if df.empty:
            print("ERROR: No data rows found.")
            return False

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:
            print("ERROR: Missing columns:", missing_columns)
            print("Available columns:", df.columns.tolist())
            return False

        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())
        print("Missing values:\n", df.isnull().sum())
        print("PASS: Dataset structure is valid.")
        return True

    except Exception as exc:
        print("ERROR:", exc)
        return False


def main():
    all_valid = True

    for filename, columns in REQUIRED_COLUMNS.items():
        valid = validate_csv(INPUT_DIR / filename, columns)
        all_valid = all_valid and valid

    print("\nOverall result:", "PASS" if all_valid else "CHECK ERRORS")


if __name__ == "__main__":
    main()
