"""Preprocess HMDA loan data: create binary target, drop 'other' outcomes,
and exclude rows missing sensitive demographic attributes."""

import pandas as pd

DATA_PATH = r"c:\Users\db234\OneDrive\Documents\Vector\Data\bmo_nationwide.csv"
OUTPUT_PATH = r"c:\Users\db234\OneDrive\Documents\Vector\Data\bmo_nationwide_preprocessed.csv"

SENSITIVE_COLS = ["derived_ethnicity", "derived_race", "derived_sex"]


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    initial_rows = len(df)

    # Create 0/1 target: 1 = loan originated (action_taken == 1),
    # 0 = application denied (action_taken == 3), 2 = any other value
    df["target"] = 2
    df.loc[df["action_taken"] == 1, "target"] = 1
    df.loc[df["action_taken"] == 3, "target"] = 0

    # Remove rows labeled 2 (any other action taken)
    df = df[df["target"] != 2]

    # Exclude rows missing sensitive attributes
    for col in SENSITIVE_COLS:
        df = df[df[col].notna()]
        df = df[df[col].astype(str).str.strip() != ""]
        df = df[~df[col].astype(str).str.upper().isin(["NA", "NAN", "NULL"])]

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Initial rows:        {initial_rows}")
    print(f"Preprocessed rows:   {len(df)}")
    print(f"Rows removed:        {initial_rows - len(df)}")
    print(f"Target distribution:\n{df['target'].value_counts().sort_index()}")
    print(f"\nRow total of new dataset: {len(df)}")


if __name__ == "__main__":
    main()
