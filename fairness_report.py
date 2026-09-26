"""Generate an HTML fairness audit report from the preprocessed HMDA data.

Computes fairlearn fairness metrics per sensitive attribute (and
combinations), writes per-metric conclusions, and saves a styled HTML report.
"""

import numpy as np
import pandas as pd
from scipy import stats
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
)

DATA_PATH = r"c:\Users\db234\OneDrive\Documents\Vector\Data\bmo_nationwide_preprocessed.csv"
REPORT_PATH = r"c:\Users\db234\OneDrive\Documents\Vector\fairness_report.html"
SENSITIVE_COLS = ["derived_ethnicity", "derived_race", "derived_sex"]


def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for a proportion.

    Unlike the naive (normal) interval, the Wilson interval behaves well
    for small or skewed samples, so it is well suited for comparing
    selection rates of groups with unequal sizes.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def chi_square_test(y, sensitive):
    """Chi-square test of independence between group and outcome.

    Tests the null hypothesis that origination rate is independent of group
    membership. Robust to sample size in the sense that it accounts for
    expected cell counts; groups with < 5 records are pooled into an
    'Other (small)' bucket to keep the test valid.
    """
    counts = sensitive.value_counts()
    small_groups = counts[counts < 5].index
    s = sensitive.copy()
    if len(small_groups) > 0:
        s[s.isin(small_groups)] = "Other (small n)"
    table = pd.crosstab(s, y)
    if table.shape[0] < 2:
        return None, None
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return chi2, p


def compute_metrics(y, sensitive):
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate,
                 "count": lambda y_true, y_pred: len(y_true)},
        y_true=y, y_pred=y, sensitive_features=sensitive,
    )
    rates = mf.by_group["selection_rate"].sort_values(ascending=False)
    counts = mf.by_group["count"]
    dp_diff = demographic_parity_difference(y_true=y, y_pred=y, sensitive_features=sensitive)
    dp_ratio = demographic_parity_ratio(y_true=y, y_pred=y, sensitive_features=sensitive)

    # Wilson 95% CI per group
    cis = {}
    for group in rates.index:
        n = int(counts.loc[group])
        k = int(round(rates.loc[group] * n))
        cis[group] = wilson_ci(k, n)

    chi2, p_value = chi_square_test(y, sensitive)
    return rates, counts, dp_diff, dp_ratio, cis, chi2, p_value


def conclusion_for(dp_ratio, dp_diff, rates, counts, cis, p_value):
    """Generate a textual conclusion for one attribute/combination."""
    best_group = rates.index[0]
    worst_group = rates.index[-1]
    best_label = best_group if isinstance(best_group, str) else " & ".join(map(str, best_group))
    worst_label = worst_group if isinstance(worst_group, str) else " & ".join(map(str, worst_group))

    # Statistical significance: do the CIs of best and worst groups overlap?
    best_ci = cis[best_group]
    worst_ci = cis[worst_group]
    significant = best_ci[0] > worst_ci[1]  # lower bound of best > upper bound of worst

    if dp_ratio >= 0.95:
        severity = "No significant disparity"
        css = "ok"
        detail = (f"Origination rates are nearly uniform across groups "
                  f"(ratio {dp_ratio:.3f}). No evidence of disparate impact.")
    elif dp_ratio >= 0.80:
        severity = "Moderate disparity"
        css = "warn"
        detail = (f"The ratio ({dp_ratio:.3f}) is within the four-fifths rule "
                  f"threshold (>= 0.80), but a gap of {dp_diff:.1%} exists between "
                  f"the highest and lowest groups. Monitor closely.")
    else:
        severity = "POTENTIAL BIAS - fails four-fifths rule"
        css = "bad"
        detail = (f"The ratio ({dp_ratio:.3f}) falls below the 0.80 four-fifths "
                  f"threshold, indicating disparate impact. The highest-rate group "
                  f"({best_label}, {rates.iloc[0]:.1%}) is approved at "
                  f"{dp_ratio:.1%} of the rate of the lowest-rate group "
                  f"({worst_label}, {rates.iloc[-1]:.1%}).")

    # Statistical significance assessment
    if significant and p_value is not None and p_value < 0.05:
        detail += (f" <strong>Statistically significant:</strong> the 95% Wilson "
                   f"confidence intervals of the highest and lowest groups do not "
                   f"overlap, and the chi-square test of independence rejects the "
                   f"null hypothesis of equal rates (p = {p_value:.2e}). The gap is "
                   f"unlikely to be a sampling artifact.")
    elif p_value is not None and p_value < 0.05:
        detail += (f" The chi-square test rejects independence "
                   f"(p = {p_value:.2e}), though the extreme groups' confidence "
                   f"intervals overlap - the disparity is significant overall but "
                   f"may be driven by mid-sized groups rather than the extremes.")
    else:
        detail += (" <strong>Not statistically significant:</strong> the gap may be "
                   "attributable to sampling variability given the group sizes, so "
                   "treat the four-fifths verdict with caution.")

    # Small-group caveat
    min_n = counts.min()
    if min_n < 50:
        detail += (" <strong>Caveat:</strong> at least one group has fewer than 50 "
                   "records, so its rate may be statistically unreliable.")

    return severity, css, detail, best_label, worst_label


def group_rows_html(rates, counts, cis):
    rows = []
    for group, rate in rates.items():
        label = group if isinstance(group, str) else " & ".join(map(str, group))
        n = int(counts.loc[group])
        lo, hi = cis[group]
        bar = int(rate * 100)
        color = "#2a9d8f" if rate >= rates.max() * 0.8 else "#e76f51"
        rows.append(f"""
        <tr>
          <td class="group">{label}</td>
          <td>{n:,}</td>
          <td>{rate:.4f}</td>
          <td>[{lo:.3f}, {hi:.3f}]</td>
          <td class="bar-cell"><div class="bar" style="width:{bar}%; background:{color}"></div></td>
        </tr>""")
    return "\n".join(rows)


METRIC_EXPLANATIONS = {
    "selection_rate": """<strong>Selection Rate.</strong> The proportion of records in a group
    that received the positive outcome (loan originated). This is the raw group-level
    approval rate and the basis for all demographic parity metrics.""",
    "dp": """<strong>Demographic Parity Difference &amp; Ratio.</strong> The difference is
    the gap between the highest and lowest group selection rates; the ratio is the lowest
    rate divided by the highest. The <em>four-fifths (80%) rule</em> treats a ratio below
    0.80 as evidence of disparate impact. <em>Limitation:</em> these are purely descriptive
    thresholds and are sensitive to sample size - small groups can produce misleading
    ratios purely by chance.""",
    "wilson": """<strong>Wilson 95% Confidence Intervals.</strong> For each group we compute
    a Wilson score interval around its selection rate. Unlike the naive normal interval,
    the Wilson interval remains accurate for small or skewed samples. If the intervals of
    two groups do not overlap, their rates differ with at least 95% confidence. This makes
    disparity claims robust to sample size: a large gap between huge groups is significant,
    while the same gap between tiny groups may not be.""",
    "chi2": """<strong>Chi-Square Test of Independence.</strong> Tests the null hypothesis
    that group membership and loan outcome are statistically independent (i.e., all groups
    share the same underlying approval rate). A p-value below 0.05 means the observed
    differences in rates across groups are unlikely to have arisen by chance alone.
    Unlike the four-fifths rule, this test weighs evidence strength by sample size rather
    than flagging any raw gap. Groups with fewer than 5 records are pooled to keep the
    test valid.""",
}


def section_html(title, rates, counts, dp_diff, dp_ratio, cis, chi2, p_value):
    severity, css, detail, best_label, worst_label = conclusion_for(
        dp_ratio, dp_diff, rates, counts, cis, p_value)
    p_text = f"{p_value:.2e}" if p_value is not None else "n/a"
    chi2_text = f"{chi2:.1f}" if chi2 is not None else "n/a"
    return f"""
    <section class="metric-section">
      <h2>{title}</h2>
      <div class="explanation">
        <h3>How these metrics work</h3>
        <p>{METRIC_EXPLANATIONS['selection_rate']}</p>
        <p>{METRIC_EXPLANATIONS['dp']}</p>
        <p>{METRIC_EXPLANATIONS['wilson']}</p>
        <p>{METRIC_EXPLANATIONS['chi2']}</p>
      </div>
      <div class="metric-cards">
        <div class="card"><span class="card-label">DP Difference</span><span class="card-value">{dp_diff:.4f}</span></div>
        <div class="card"><span class="card-label">DP Ratio</span><span class="card-value">{dp_ratio:.4f}</span></div>
        <div class="card"><span class="card-label">Chi-Square</span><span class="card-value">{chi2_text}</span></div>
        <div class="card"><span class="card-label">p-value</span><span class="card-value">{p_text}</span></div>
        <div class="card"><span class="card-label">Groups</span><span class="card-value">{len(rates)}</span></div>
        <div class="card {css}-card"><span class="card-label">Verdict</span><span class="card-value">{severity}</span></div>
      </div>
      <div class="conclusion {css}">
        <strong>Conclusion:</strong> {detail}
      </div>
      <table>
        <thead><tr><th>Group</th><th>Count</th><th>Selection Rate</th><th>95% CI (Wilson)</th><th style="width:30%">Rate</th></tr></thead>
        <tbody>{group_rows_html(rates, counts, cis)}</tbody>
      </table>
    </section>"""


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    y = df["target"].astype(int)

    # --- Handling 'Joint' in derived_sex ---
    # In HMDA, derived_sex == 'Joint' means the application was filed jointly
    # (e.g., two applicants of different sexes, such as a married couple).
    # It is NOT a missing or unknown value - it is a legitimate category
    # describing a mixed-sex joint application. We therefore keep it as its
    # own group rather than dropping it, and note in the report that its
    # outcome reflects a two-applicant household, which is not directly
    # comparable to single-applicant sex groups.
    joint_note = """
    <div class="explanation">
      <h3>Note on the 'Joint' value in derived_sex</h3>
      <p>In HMDA data, <strong>Joint</strong> in <code>derived_sex</code> indicates a
      <strong>joint application</strong> - typically two applicants (e.g., a married
      couple) reported together, where the sexes of the primary and secondary
      applicants differ or are combined. It is not a missing or unknown value.
      This report keeps <em>Joint</em> as its own group because it is a legitimate,
      distinct category, but its origination rate reflects two-applicant households
      (which often have higher combined income and different approval dynamics),
      so it is not directly comparable to single-applicant sex groups. If a strict
      single-applicant comparison is desired, filter out <code>Joint</code> records
      before comparing Male vs. Female rates.</p>
    </div>"""

    analyses = []
    for col in SENSITIVE_COLS:
        analyses.append((col, df[col]))
    pairs = [("derived_ethnicity", "derived_race"),
             ("derived_ethnicity", "derived_sex"),
             ("derived_race", "derived_sex")]
    for a, b in pairs:
        analyses.append((f"{a} + {b}", df[a].astype(str) + " & " + df[b].astype(str)))
    analyses.append(("derived_ethnicity + derived_race + derived_sex",
                     df["derived_ethnicity"].astype(str) + " & "
                     + df["derived_race"].astype(str) + " & "
                     + df["derived_sex"].astype(str)))

    sections = []
    summary_rows = []
    for title, sensitive in analyses:
        rates, counts, dp_diff, dp_ratio, cis, chi2, p_value = compute_metrics(y, sensitive)
        sections.append(section_html(title, rates, counts, dp_diff, dp_ratio, cis, chi2, p_value))
        severity, css, _, *_ = conclusion_for(dp_ratio, dp_diff, rates, counts, cis, p_value)
        summary_rows.append(
            f"<tr><td>{title}</td><td>{dp_diff:.4f}</td><td>{dp_ratio:.4f}</td>"
            f"<td class='{css}-text'>{severity}</td></tr>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fairness Audit Report - HMDA Loan Data</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f4f6f8; color: #22303c; }}
  header {{ background: #1d3557; color: #fff; padding: 30px 40px; }}
  header h1 {{ margin: 0 0 6px 0; }}
  header p {{ margin: 0; opacity: 0.85; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 30px 20px; }}
  .metric-section {{ background: #fff; border-radius: 10px; padding: 25px; margin-bottom: 25px;
                     box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  h2 {{ margin-top: 0; color: #1d3557; border-bottom: 2px solid #e0e4e8; padding-bottom: 8px; }}
  .metric-cards {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 15px; }}
  .card {{ background: #f0f4f8; border-radius: 8px; padding: 12px 18px; min-width: 130px; }}
  .card-label {{ display: block; font-size: 0.75em; text-transform: uppercase; color: #6b7a8c; }}
  .card-value {{ display: block; font-size: 1.15em; font-weight: 600; }}
  .ok-card {{ background: #e6f4ea; }} .warn-card {{ background: #fff4e0; }} .bad-card {{ background: #fdecea; }}
  .conclusion {{ border-radius: 8px; padding: 14px 18px; margin-bottom: 18px; line-height: 1.5; }}
  .ok {{ background: #e6f4ea; border-left: 5px solid #2a9d8f; }}
  .warn {{ background: #fff8e6; border-left: 5px solid #e9c46a; }}
  .bad {{ background: #fdecea; border-left: 5px solid #e76f51; }}
  .ok-text {{ color: #2a9d8f; font-weight: 600; }}
  .warn-text {{ color: #d4a017; font-weight: 600; }}
  .bad-text {{ color: #e76f51; font-weight: 600; }}
  .explanation {{ background: #f0f4f8; border-radius: 8px; padding: 15px 20px; margin-bottom: 18px; }}
  .explanation h3 {{ margin: 0 0 8px 0; color: #1d3557; font-size: 1.05em; }}
  .explanation p {{ margin: 6px 0; line-height: 1.55; font-size: 0.92em; }}
  code {{ background: #e2e8ee; padding: 1px 5px; border-radius: 4px; font-size: 0.9em; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92em; }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #e8ecf0; }}
  th {{ background: #f0f4f8; }}
  td.group {{ font-weight: 500; }}
  .bar-cell {{ padding: 4px 10px; }}
  .bar {{ height: 14px; border-radius: 4px; min-width: 2px; }}
  footer {{ text-align: center; color: #8a97a5; font-size: 0.85em; padding: 20px; }}
</style>
</head>
<body>
<header>
  <h1>Fairness Audit Report</h1>
  <p>HMDA Loan Data (BMO Nationwide) &middot; Generated with fairlearn &middot; {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
</header>
<main>
  <section class="metric-section">
    <h2>Executive Summary</h2>
    <p>Dataset rows: <strong>{len(df):,}</strong> &middot; Overall origination rate: <strong>{y.mean():.4f}</strong></p>
    <p>This report evaluates fairness of loan origination outcomes across sensitive
    demographic attributes (derived ethnicity, race, and sex), individually and in
    intersectional combinations, using fairlearn's demographic parity metrics
    supplemented with <strong>statistical significance tests</strong> (Wilson confidence
    intervals and chi-square tests) that are robust to unequal group sample sizes.
    The <strong>four-fifths (80%) rule</strong> is applied as a descriptive threshold:
    a selection-rate ratio below 0.80 between the highest and lowest groups is treated
    as evidence of potential disparate impact, but each verdict is cross-checked against
    statistical significance before a bias conclusion is drawn.</p>
    {joint_note}
    <table>
      <thead><tr><th>Attribute(s)</th><th>DP Difference</th><th>DP Ratio</th><th>Verdict</th></tr></thead>
      <tbody>{''.join(summary_rows)}</tbody>
    </table>
  </section>
  {''.join(sections)}
  <section class="metric-section">
    <h2>Overall Conclusions</h2>
    <ul>
      <li><strong>Four-fifths rule (descriptive):</strong> attributes or combinations whose
      demographic parity ratio falls below <strong>0.80</strong> are flagged as showing
      potential disparate impact in origination rates.</li>
      <li><strong>Statistical significance (sample-size robust):</strong> each verdict is
      cross-checked with Wilson 95% confidence intervals and a chi-square test of
      independence. A disparity is only treated as strong evidence of bias when the
      extreme groups' confidence intervals do not overlap and the chi-square test
      rejects independence (p &lt; 0.05). This prevents small groups from producing
      false alarms (or hiding real gaps) purely due to sample size.</li>
      <li><strong>'Joint' sex category:</strong> records with <code>derived_sex = Joint</code>
      represent joint applications (typically two applicants of different sexes).
      They are kept as a distinct group but are not directly comparable to
      single-applicant sex groups; filter them out for a strict Male vs. Female
      comparison.</li>
      <li><strong>Intersectional groups</strong> (e.g., ethnicity + sex) often show
      larger disparities than single attributes, because disadvantages can compound
      across multiple protected characteristics.</li>
      <li>These are <strong>outcome-based metrics only</strong>. Observed disparities do
      not prove discriminatory intent or causation; legitimate underwriting factors
      (income, loan amount, credit history) may explain part of the gap. A complete
      audit would fit a predictive model and compare per-group error rates
      (equalized odds / equal opportunity).</li>
      <li>Groups with very small sample sizes should be interpreted cautiously;
      the Wilson intervals in each section make this uncertainty explicit.</li>
    </ul>
  </section>
</main>
<footer>Generated automatically by fairness_report.py using fairlearn.</footer>
</body>
</html>"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
