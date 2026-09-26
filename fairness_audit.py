"""Fairness analysis of preprocessed HMDA loan data using fairlearn.

Computes fairness metrics around sensitive attributes (derived ethnicity,
race, sex) individually and in combination, then draws conclusions
about potential bias in loan origination outcomes.
"""

import pandas as pd
import numpy as np
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equal_opportunity_difference,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score

DATA_PATH = r"c:\Users\db234\OneDrive\Documents\Vector\Data\bmo_nationwide_preprocessed.csv"
SENSITIVE_COLS = ["derived_ethnicity", "derived_race", "derived_sex"]


def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    # In this dataset the "prediction" is the observed outcome itself:
    # target 1 = loan originated, 0 = denied. We treat the outcome as both
    # y_true and y_pred so that selection_rate reflects the actual approval
    # rate per group (standard exploratory fairness audit of outcomes).
    y = df["target"].astype(int)
    return df, y


def analyze_metric_frame(y, sensitive_features, label):
    """Compute a MetricFrame for a set of sensitive features."""
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "count": lambda y_true, y_pred: len(y_true),
        },
        y_true=y,
        y_pred=y,
        sensitive_features=sensitive_features,
    )

    dp_diff = demographic_parity_difference(y_true=y, y_pred=y,
                                            sensitive_features=sensitive_features)
    dp_ratio = demographic_parity_ratio(y_true=y, y_pred=y,
                                        sensitive_features=sensitive_features)

    print(f"\n{'=' * 70}")
    print(f"SENSITIVE ATTRIBUTE(S): {label}")
    print(f"{'=' * 70}")
    rates = mf.by_group["selection_rate"].sort_values(ascending=False)
    print(f"Number of groups: {len(rates)}")
    print(f"\nSelection rate (loan origination rate) by group:")
    for group, rate in rates.items():
        count = mf.by_group.loc[group, "count"]
        group_label = group if isinstance(group, str) else " | ".join(map(str, group))
        print(f"  {group_label:<60} {rate:.4f}  (n={count})")

    print(f"\n--- Fairness metrics ---")
    print(f"Demographic parity difference : {dp_diff:.4f}")
    print(f"Demographic parity ratio     : {dp_ratio:.4f}")
    print(f"Min selection rate            : {rates.min():.4f}")
    print(f"Max selection rate            : {rates.max():.4f}")
    return rates, dp_diff, dp_ratio


def main():
    df, y = load_data()
    print(f"Dataset rows: {len(df)}")
    print(f"Overall origination rate: {y.mean():.4f}")

    results = {}

    # Individual sensitive attributes
    for col in SENSITIVE_COLS:
        rates, dp_diff, dp_ratio = analyze_metric_frame(y, df[col], col)
        results[col] = (rates, dp_diff, dp_ratio)

    # Pairwise combinations
    pairs = [("derived_ethnicity", "derived_race"),
             ("derived_ethnicity", "derived_sex"),
             ("derived_race", "derived_sex")]
    for a, b in pairs:
        combo = df[a].astype(str) + " & " + df[b].astype(str)
        rates, dp_diff, dp_ratio = analyze_metric_frame(y, combo, f"{a} + {b}")
        results[f"{a}+{b}"] = (rates, dp_diff, dp_ratio)

    # All three combined
    combo3 = (df["derived_ethnicity"].astype(str) + " & "
              + df["derived_race"].astype(str) + " & "
              + df["derived_sex"].astype(str))
    rates, dp_diff, dp_ratio = analyze_metric_frame(y, combo3,
                                                   "derived_ethnicity + derived_race + derived_sex")
    results["all_three"] = (rates, dp_diff, dp_ratio)

    # Conclusions
    print(f"\n{'=' * 70}")
    print("CONCLUSIONS")
    print(f"{'=' * 70}")
    print("""
Interpretation guide (80% rule / four-fifths rule):
  - A demographic parity ratio < 0.80 between the best and worst group
    is widely used as evidence of disparate impact.
  - Demographic parity difference = max rate - min rate across groups.

Findings:""")

    for name, (rates, dp_diff, dp_ratio) in results.items():
        verdict = "POTENTIAL BIAS" if dp_ratio < 0.80 else "No strong disparity"
        print(f"  {name:<45} DP ratio = {dp_ratio:.4f} -> {verdict}")

    print("""
General conclusions:
  1. If the demographic parity ratio for any sensitive attribute (or
     intersection) falls below 0.80, the origination rates are uneven
     enough to suggest disparate impact under the four-fifths rule.
  2. Intersectional groups (e.g., specific ethnicity + sex combinations)
     often reveal larger disparities than single attributes, since
     disadvantages can compound.
  3. Note that these are outcome-based metrics only. Observed disparities
     do not prove causation - legitimate factors (income, loan amount,
     credit history) may explain part of the gap. A full audit would fit a
     model and compare true/false positive rates per group
     (equalized odds / equal opportunity).
""")


if __name__ == "__main__":
    main()
