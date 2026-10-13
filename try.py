from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.company_stock_filter import is_company_stock_candidate
except ModuleNotFoundError:  # direct execution: python scripts/build_balanced_events.py
    from company_stock_filter import is_company_stock_candidate


GENERIC_MANAGER_NAMES = {
    "", "nan", "none", "unknown", "unknown manager", "team managed",
    "team manager", "management team", "multiple managers", "not disclosed",
    "n/a", "na",
}


IDENTITY_REQUIRED_FIELDS = [
    "event_id", "crsp_portno", "month_end", "report_date_original",
    "event_month", "analysis_entity", "analysis_entity_level", "family_name",
    "allocation_source", "allocation_quality",
]

ALLOCATION_REQUIRED_FIELDS = ["stock_weight", "bond_weight", "cash_weight"]

CHANGE_REQUIRED_FIELDS = [
    "delta_stock_pp", "delta_bond_pp", "delta_cash_pp",
    "allocation_change_l1_pp", "allocation_turnover_pp",
    "gap_months_from_prev_report",
]



def step1_input_paths(data_root: Path) -> list[Path]:
    """Return exactly the source files consumed by the Step 1 builder."""
    candidates = [
        data_root / "crsp" / "fund_level" / "balanced_before2010.csv",
        data_root / "crsp" / "fund_level" / "balanced_after2010.csv",
        data_root / "crsp" / "holdings_raw" / "stock berfore 2010_new___.csv",
        data_root / "crsp" / "holdings_raw" / "stock between 2010_2014_new___.csv",
        data_root / "crsp" / "holdings_raw" / "stock between 2015_2019_new___.csv",
        data_root / "crsp" / "holdings_raw" / "stock between 2020_2026_new___.csv",
        data_root / "market" / "treasury_10y_1997_2025.csv",
        data_root / "market" / "F-F_Research_Data_Factors.csv",
        data_root / "market" / "F-F_Research_Data_Factors(1).csv",
        data_root / "market" / "yahoo_equity_adj_close_monthly.csv",
        data_root / "part5_non_individual_holdings" / "part5_excluded_two_group_enriched.csv",
        data_root
        / "part5_non_individual_holdings"
        / "part5_excluded_individual_stock_like_removed_audit.csv",
    ]
    return [p for p in candidates if p.exists()]


def file_hash_manifest(paths: list[Path], base: Path) -> list[dict]:
    """Create deterministic SHA-256 provenance records without mutating inputs."""
    rows: list[dict] = []
    for path in sorted(paths, key=lambda p: p.as_posix().lower()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        stat = path.stat()
        rows.append({
            "filename": path.relative_to(base).as_posix(),
            "sha256": digest.hexdigest(),
            "size_bytes": int(stat.st_size),
            "modified_time": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        })
    return rows


@dataclass(frozen=True)
class BuildConfig:
    research_change_threshold_pp: float = 0.50
    change_epsilon_pp: float = 1e-6
    holdings_proxy_min_coverage: float = 0.70
    holdings_proxy_max_coverage: float = 1.20
    treasury_primary_modified_duration: float = 8.50
    treasury_duration_sensitivity_low: float = 7.50
    treasury_duration_sensitivity_high: float = 9.50
    max_change_gap_months: int = 6
    no_action_warn_coverage: float = 0.85
    reported_bond_discrepancy_warn_pp: float = 5.0
    cash_return_fallback: float = 0.0


def month_end(s: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(s, errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )


def normalize_name(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.strip()
    )


def is_specific_manager_value(v: object) -> bool:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return str(v).strip().lower() not in GENERIC_MANAGER_NAMES


def compound_forward_by_calendar(
    group: pd.DataFrame,
    value_col: str,
    out_col: str,
    months: int = 6,
) -> pd.DataFrame:
    """Compound t+1...t+months only when every calendar month is available."""
    g = group.sort_values("month_end").copy()
    if g.empty:
        g[out_col] = np.nan
        return g

    calendar = pd.date_range(
        g["month_end"].min(),
        g["month_end"].max(),
        freq="ME",
    )
    s = (
        g.set_index("month_end")[value_col]
        .reindex(calendar)
        .pipe(pd.to_numeric, errors="coerce")
    )
    vals = s.to_numpy(dtype=float)

    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        window = vals[i + 1 : i + 1 + months]
        if len(window) == months and np.isfinite(window).all():
            out[i] = np.prod(1.0 + window) - 1.0

    m = pd.DataFrame({"month_end": calendar, out_col: out})
    return g.merge(m, on="month_end", how="left")


def _weighted_group_value(
    df: pd.DataFrame,
    keys: list[str],
    value_col: str,
    weight_col: str,
    out_name: str,
) -> pd.DataFrame:
    v = pd.to_numeric(df[value_col], errors="coerce")
    w = pd.to_numeric(df[weight_col], errors="coerce").clip(lower=0)
    valid = v.notna() & w.notna() & w.gt(0)

    num = (
        (v.where(valid) * w.where(valid))
        .groupby([df[k] for k in keys])
        .sum(min_count=1)
    )
    den = (
        w.where(valid)
        .groupby([df[k] for k in keys])
        .sum(min_count=1)
    )
    out = (num / den.replace(0, np.nan)).rename(out_name).reset_index()

    # Deterministic equal-weight fallback when positive weights are unavailable.
    mean = (
        v.groupby([df[k] for k in keys])
        .mean()
        .rename(out_name + "_mean")
        .reset_index()
    )
    out = out.merge(mean, on=keys, how="outer")
    out[out_name] = out[out_name].fillna(out[out_name + "_mean"])
    return out[keys + [out_name]]


def _expense_weight_coverage(
    df: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """
    Vectorized fraction of return-eligible share-class weight for which
    expense ratio is observed. If no positive TNA weight is available
    for a portfolio-month, fall back to an equal-count coverage ratio.
    """
    mret_ok = pd.to_numeric(df["mret"], errors="coerce").notna()
    exp_ok = pd.to_numeric(df["exp_ratio"], errors="coerce").notna()
    w = pd.to_numeric(df["w_tna"], errors="coerce").clip(lower=0)

    groupers = [df[k] for k in keys]

    eligible_w = w.where(mret_ok & w.gt(0))
    den = eligible_w.groupby(groupers).sum(min_count=1)
    num = (
        eligible_w.where(exp_ok, 0.0)
        .groupby(groupers)
        .sum(min_count=1)
    )
    weighted = num / den.replace(0, np.nan)

    count_den = (
        mret_ok.astype(float)
        .groupby(groupers)
        .sum(min_count=1)
    )
    count_num = (
        (mret_ok & exp_ok).astype(float)
        .groupby(groupers)
        .sum(min_count=1)
    )
    equal_count = count_num / count_den.replace(0, np.nan)

    coverage = weighted.fillna(equal_count)
    return coverage.rename(
        "expense_ratio_weight_coverage"
    ).reset_index()


def load_fund_level(data_root: Path) -> pd.DataFrame:
    folder = data_root / "crsp" / "fund_level"
    usecols = [
        "crsp_fundno", "crsp_portno", "fund_name", "ticker",
        "mgmt_name", "mgr_name", "adv_name", "caldt",
        "mret", "mtna", "exp_ratio", "age_tmp", "turn_ratio",
    ]

    frames = [
        pd.read_csv(
            folder / "balanced_before2010.csv",
            usecols=usecols,
            low_memory=False,
        ),
        pd.read_csv(
            folder / "balanced_after2010.csv",
            usecols=usecols,
            low_memory=False,
        ),
    ]
    df = pd.concat(frames, ignore_index=True)

    df["caldt_original"] = pd.to_datetime(df["caldt"], errors="coerce")
    df["month_end"] = month_end(df["caldt"])
    df = df.dropna(
        subset=["crsp_portno", "crsp_fundno", "month_end"]
    ).copy()

    df["crsp_portno"] = (
        pd.to_numeric(df["crsp_portno"], errors="coerce").astype("Int64")
    )
    df["crsp_fundno"] = (
        pd.to_numeric(df["crsp_fundno"], errors="coerce").astype("Int64")
    )

    # One share-class observation per calendar month: retain the latest source row.
    df = (
        df.sort_values(
            ["crsp_fundno", "month_end", "caldt_original"]
        )
        .drop_duplicates(["crsp_fundno", "month_end"], keep="last")
        .copy()
    )

    # Lagged TNA only when the prior share-class observation is exactly one month earlier.
    prev_me = df.groupby("crsp_fundno")["month_end"].shift(1)
    lag_tna = pd.to_numeric(
        df.groupby("crsp_fundno")["mtna"].shift(1),
        errors="coerce",
    )
    gap = (
        df["month_end"].dt.year * 12
        + df["month_end"].dt.month
        - (prev_me.dt.year * 12 + prev_me.dt.month)
    )
    curr_tna = pd.to_numeric(df["mtna"], errors="coerce")
    df["w_tna"] = np.where(
        gap.eq(1) & lag_tna.gt(0),
        lag_tna,
        curr_tna,
    )
    df["w_tna"] = (
        pd.to_numeric(df["w_tna"], errors="coerce")
        .clip(lower=0)
    )

    df["mret"] = pd.to_numeric(df["mret"], errors="coerce")
    df["exp_ratio"] = pd.to_numeric(df["exp_ratio"], errors="coerce")

    # Gross-return approximation remains explicit:
    # unknown annual expense ratio contributes 0 to the add-back,
    # and coverage is separately disclosed.
    df["gross_return_approx_shareclass"] = (
        df["mret"] + df["exp_ratio"].fillna(0.0) / 12.0
    )

    keys = ["crsp_portno", "month_end"]

    base = df.groupby(keys, as_index=False).agg(
        share_class_count=("crsp_fundno", "nunique"),
        portfolio_tna=(
            "mtna",
            lambda x: (
                pd.to_numeric(x, errors="coerce")
                .clip(lower=0)
                .sum(min_count=1)
            ),
        ),
    )

    # Core weighted series.
    for value_col, out_col in [
        ("mret", "fund_net_return"),
        ("gross_return_approx_shareclass", "fund_gross_return_approx"),
        ("age_tmp", "fund_age_years"),
        ("turn_ratio", "turnover_ratio"),
        ("exp_ratio", "expense_ratio_known_annual"),
    ]:
        base = base.merge(
            _weighted_group_value(
                df, keys, value_col, "w_tna", out_col
            ),
            on=keys,
            how="left",
        )

    base = base.merge(
        _expense_weight_coverage(df, keys),
        on=keys,
        how="left",
    )
    base["expense_ratio_complete"] = (
        pd.to_numeric(
            base["expense_ratio_weight_coverage"], errors="coerce"
        ).ge(1.0 - 1e-12)
    )
    base["expense_ratio_missing_reason"] = np.select(
        [
            base["expense_ratio_complete"],
            base["expense_ratio_weight_coverage"].isna(),
            base["expense_ratio_weight_coverage"].lt(1.0 - 1e-12),
        ],
        [
            "complete",
            "no_return_eligible_expense_coverage",
            "partial_share_class_expense_coverage",
        ],
        default="unresolved_expense_coverage",
    )

    # This field exactly matches the expense add-back that is actually embedded
    # in fund_gross_return_approx, avoiding the old multi-share-class mismatch.
    base["expense_ratio_annual"] = 12.0 * (
        base["fund_gross_return_approx"]
        - base["fund_net_return"]
    )

    # Flow diagnostic: remove the mechanical effect of return from TNA change.
    base = base.sort_values(["crsp_portno", "month_end"]).copy()
    base["prev_portfolio_tna"] = (
        base.groupby("crsp_portno")["portfolio_tna"].shift(1)
    )
    prev_me = base.groupby("crsp_portno")["month_end"].shift(1)
    base["fund_month_gap"] = (
        base["month_end"].dt.year * 12
        + base["month_end"].dt.month
        - (prev_me.dt.year * 12 + prev_me.dt.month)
    )

    flow_ok = (
        base["fund_month_gap"].eq(1)
        & base["prev_portfolio_tna"].gt(0)
        & base["portfolio_tna"].notna()
        & base["fund_net_return"].notna()
    )
    base["estimated_net_flow"] = np.nan
    base.loc[flow_ok, "estimated_net_flow"] = (
        base.loc[flow_ok, "portfolio_tna"]
        - base.loc[flow_ok, "prev_portfolio_tna"]
        * (1.0 + base.loc[flow_ok, "fund_net_return"])
    )
    base["estimated_flow_ratio"] = (
        base["estimated_net_flow"]
        / base["prev_portfolio_tna"].replace(0, np.nan)
    )
    base["flow_available"] = flow_ok
    base["flow_missing_reason"] = np.select(
        [
            flow_ok,
            base["fund_month_gap"].isna(),
            base["fund_month_gap"].ne(1),
            base["prev_portfolio_tna"].isna(),
            base["prev_portfolio_tna"].le(0),
            base["portfolio_tna"].isna(),
            base["fund_net_return"].isna(),
        ],
        [
            "available",
            "no_previous_calendar_month",
            "nonconsecutive_fund_month",
            "missing_previous_tna",
            "nonpositive_previous_tna",
            "missing_current_tna",
            "missing_fund_net_return",
        ],
        default="unresolved_flow_input",
    )

    # Pick metadata deterministically: prefer a specific manager,
    # then the latest source row within the month.
    df["manager_specific"] = (
        df["mgr_name"].map(is_specific_manager_value).astype(int)
    )
    meta = (
        df.sort_values(
            keys + ["manager_specific", "caldt_original"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(keys)
        [
            keys
            + [
                "fund_name", "ticker", "mgr_name",
                "mgmt_name", "adv_name",
            ]
        ]
        .rename(
            columns={
                "ticker": "fund_ticker",
                "mgr_name": "manager_raw",
                "mgmt_name": "family_name",
                "adv_name": "advisor_name",
            }
        )
    )
    out = base.merge(meta, on=keys, how="left")

    manager_specific = out["manager_raw"].map(
        is_specific_manager_value
    )
    advisor_ok = (
        out["advisor_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("nan", "")
        .ne("")
    )
    family_ok = (
        out["family_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("nan", "")
        .ne("")
    )

    out["analysis_entity"] = np.select(
        [
            manager_specific,
            ~manager_specific & advisor_ok,
            ~manager_specific & ~advisor_ok & family_ok,
        ],
        [
            out["manager_raw"].fillna(""),
            "TEAM/FAMILY :: " + out["advisor_name"].fillna(""),
            "FAMILY :: " + out["family_name"].fillna(""),
        ],
        default="UNRESOLVED MANAGER/FAMILY",
    )
    out["analysis_entity_level"] = np.select(
        [
            manager_specific,
            ~manager_specific & advisor_ok,
            ~manager_specific & ~advisor_ok & family_ok,
        ],
        [
            "manager",
            "advisor_team_fallback",
            "family_fallback",
        ],
        default="unresolved",
    )

    # Forward 6M fund return for each portfolio.
    pieces = []
    for _, g in out.groupby("crsp_portno", sort=False):
        g = compound_forward_by_calendar(
            g,
            "fund_gross_return_approx",
            "future_6m_fund_gross_return",
            6,
        )
        g = compound_forward_by_calendar(
            g,
            "fund_net_return",
            "future_6m_fund_net_return",
            6,
        )
        pieces.append(g)

    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(keys)
    )


def _find_ff_factors_path(data_root: Path) -> Path:
    """Locate the Fama-French Research Data Factors CSV used for the cash RF sleeve."""
    candidates = [
        data_root / "market" / "F-F_Research_Data_Factors.csv",
        data_root / "market" / "F-F_Research_Data_Factors(1).csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing F-F_Research_Data_Factors.csv under data/market. "
        "Teacher revision requires the monthly RF column; zero-cash fallback is no longer allowed."
    )


def _load_ff_rf_cash_return(
    data_root: Path,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.Series, str, pd.DataFrame]:
    """
    Parse the MONTHLY section of Kenneth French's F-F Research Data Factors CSV.

    The source stores dates as YYYYMM and RF in PERCENT units (e.g. 0.22 means 0.22%).
    We convert the month key to the calendar month-end and RF to decimal monthly return.
    Annual rows (YYYY) and descriptive/header rows are ignored explicitly.
    """
    path = _find_ff_factors_path(data_root)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            key = str(row[0]).strip()
            if len(key) != 6 or not key.isdigit() or len(row) < 5:
                continue
            try:
                rf_pct = float(str(row[4]).strip())
            except ValueError:
                continue
            dt = pd.Period(key, freq="M").to_timestamp("M")
            rows.append({
                "source_month": key,
                "month_end": dt,
                "ff_rf_pct": rf_pct,
                "cash_proxy_return": rf_pct / 100.0,
            })
    if not rows:
        raise ValueError(f"No monthly YYYYMM RF rows found in {path}")

    parsed = (
        pd.DataFrame(rows)
        .sort_values("month_end")
        .drop_duplicates("month_end", keep="last")
    )
    series = parsed.set_index("month_end")["cash_proxy_return"].reindex(calendar)
    method = f"{path.name}::RF_monthly_percent_to_decimal_month_end"
    return series, method, parsed


def load_market(
    data_root: Path,
    cfg: BuildConfig,
) -> pd.DataFrame:
    tr = pd.read_csv(
        data_root / "market" / "treasury_10y_1997_2025.csv",
        usecols=["Date", "DGS10_Yield"],
    )

    tr["source_date_treasury"] = pd.to_datetime(
        tr["Date"], errors="coerce"
    )
    tr["month_end"] = month_end(tr["Date"])
    tr["treasury10y_yield_pct"] = pd.to_numeric(
        tr["DGS10_Yield"], errors="coerce"
    )
    tr = (
        tr.sort_values("source_date_treasury")
        .drop_duplicates("month_end", keep="last")
    )

    start = tr["month_end"].min()
    end = tr["month_end"].max()
    calendar = pd.date_range(start, end, freq="ME")

    m = pd.DataFrame({"month_end": calendar})
    m = m.merge(
        tr[
            [
                "month_end", "source_date_treasury",
                "treasury10y_yield_pct",
            ]
        ],
        on="month_end",
        how="left",
    )

    y = m["treasury10y_yield_pct"] / 100.0
    m["treasury10y_yield_change_bps"] = (
        m["treasury10y_yield_pct"].diff() * 100.0
    )
    prev_y = y.shift(1)
    dy = y.diff()

    durations = [
        cfg.treasury_duration_sensitivity_low,
        cfg.treasury_primary_modified_duration,
        cfg.treasury_duration_sensitivity_high,
    ]
    for d in durations:
        label = str(d).replace(".", "p")
        m[f"treasury10y_proxy_return_d{label}"] = (
            prev_y / 12.0 - d * dy
        )

    primary = (
        "treasury10y_proxy_return_d"
        + str(cfg.treasury_primary_modified_duration).replace(".", "p")
    )
    m["treasury10y_proxy_return"] = m[primary]

    cash_series, cash_method, ff_rf_table = _load_ff_rf_cash_return(
        data_root,
        calendar,
    )
    m["cash_proxy_return"] = cash_series.to_numpy()
    m["cash_return_method"] = cash_method
    m["cash_return_is_fallback"] = False
    m["source_date_rf"] = m["month_end"].where(m["cash_proxy_return"].notna())

    # Exact 6M forward returns for the Bond and Cash legs, reusing the
    # synthetic Treasury proxy and F-F RF monthly series -- needed by the
    # No-Action passive-drift model's Bond/Cash sleeves.
    m = compound_forward_by_calendar(
        m,
        "treasury10y_proxy_return",
        "future_6m_treasury10y_proxy_return",
        6,
    )
    m = compound_forward_by_calendar(
        m,
        "cash_proxy_return",
        "future_6m_cash_proxy_return",
        6,
    )
    return m.sort_values("month_end")


def build_classification_maps(
    data_root: Path,
) -> tuple[pd.DataFrame, set[str]]:
    folder = data_root / "part5_non_individual_holdings"
    epath = folder / "part5_excluded_two_group_enriched.csv"
    rpath = (
        folder
        / "part5_excluded_individual_stock_like_removed_audit.csv"
    )

    if not epath.exists():
        return pd.DataFrame(), set()

    e = pd.read_csv(
        epath,
        usecols=[
            "holding_security_name",
            "teacher_category",
            "teacher_subcategory",
            "classification_confidence",
        ],
        low_memory=False,
    )
    e["name_key"] = normalize_name(e["holding_security_name"])
    e = (
        e[
            e["classification_confidence"]
            .astype(str)
            .str.lower()
            .eq("high")
        ]
        .drop_duplicates("name_key")
    )

    removed: set[str] = set()
    if rpath.exists():
        r = pd.read_csv(
            rpath,
            usecols=["holding_security_name"],
            low_memory=False,
        )
        removed = set(
            normalize_name(r["holding_security_name"]).tolist()
        )

    return (
        e[
            [
                "name_key",
                "teacher_category",
                "teacher_subcategory",
            ]
        ],
        removed,
    )


def load_holdings_events(
    data_root: Path,
    cfg: BuildConfig,
) -> pd.DataFrame:
    folder = data_root / "crsp" / "holdings_raw"
    cols = [
        "crsp_portno", "fund_ticker", "fund_name",
        "fund_percent_common_stock", "fund_percent_bond",
        "fund_percent_cash", "report_dt", "holding_percent_tna",
        "holding_security_name", "holding_ticker", "holding_permno",
    ]
    paths = [
        "stock berfore 2010_new___.csv",
        "stock between 2010_2014_new___.csv",
        "stock between 2015_2019_new___.csv",
        "stock between 2020_2026_new___.csv",
    ]

    h = pd.concat(
        [
            pd.read_csv(folder / name, usecols=cols, low_memory=False)
            for name in paths
        ],
        ignore_index=True,
    )
    h["report_date_original"] = pd.to_datetime(h["report_dt"], errors="coerce")
    h["month_end"] = month_end(h["report_dt"])
    h = h.dropna(subset=["crsp_portno", "report_date_original", "month_end"]).copy()
    h["crsp_portno"] = pd.to_numeric(h["crsp_portno"], errors="coerce").astype("Int64")

    # One observed report per portfolio-calendar-month: keep the last actual report date.
    latest = h.groupby(["crsp_portno", "month_end"])["report_date_original"].transform("max")
    h = h[h["report_date_original"].eq(latest)].copy()

    maps, removed = build_classification_maps(data_root)
    h["name_key"] = normalize_name(h["holding_security_name"])
    if not maps.empty:
        h = h.merge(maps, on="name_key", how="left")
    else:
        h["teacher_category"] = np.nan
        h["teacher_subcategory"] = np.nan

    h["removed_false_equity"] = h["name_key"].isin(removed)
    h["bucket"] = "unknown"
    h.loc[h["holding_permno"].notna() & ~h["removed_false_equity"], "bucket"] = "equity"
    h.loc[h["teacher_category"].eq("Equity Fund / Stock-fund-like"), "bucket"] = "equity"

    cash_like = (
        h["teacher_subcategory"]
        .fillna("")
        .str.contains(r"cash|money market|liquidity|deposit", case=False, regex=True)
    )
    bond_side = h["teacher_category"].eq("Bond / Credit / Money-related")
    h.loc[bond_side & cash_like, "bucket"] = "cash"
    h.loc[bond_side & ~cash_like, "bucket"] = "bond"
    h["holding_weight"] = pd.to_numeric(h["holding_percent_tna"], errors="coerce") / 100.0

    keys = ["crsp_portno", "month_end", "report_date_original"]
    summary = h.groupby(keys, as_index=False).agg(
        fund_ticker_holdings=("fund_ticker", "first"),
        fund_name_holdings=("fund_name", "first"),
        reported_stock_pct_raw=("fund_percent_common_stock", "first"),
        reported_bond_pct_raw=("fund_percent_bond", "first"),
        reported_cash_pct_raw=("fund_percent_cash", "first"),
        holding_record_count=("holding_security_name", "size"),
    )

    pivot = h.pivot_table(
        index=keys,
        columns="bucket",
        values="holding_weight",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for c in ["equity", "bond", "cash", "unknown"]:
        if c not in pivot:
            pivot[c] = 0.0
    pivot = pivot.rename(columns={
        "equity": "holdings_equity_proxy",
        "bond": "holdings_bond_proxy",
        "cash": "holdings_cash_proxy",
        "unknown": "holdings_unknown_proxy",
    })
    out = summary.merge(pivot, on=keys, how="left")

    for raw, dest in [
        ("reported_stock_pct_raw", "stock_reported"),
        ("reported_bond_pct_raw", "bond_reported"),
        ("reported_cash_pct_raw", "cash_reported"),
    ]:
        out[dest] = pd.to_numeric(out[raw], errors="coerce") / 100.0

    # IMPORTANT RESEARCH RULE (teacher-requested):
    # For reported allocation summaries, do NOT trust a tiny/non-missing reported bond
    # merely because it exists. Whenever reported stock and reported cash are available,
    # construct the research bond sleeve as the residual 1 - stock - cash.
    # The original reported bond is NEVER destroyed; it is retained as an audit field.
    out["bond_residual_from_stock_cash"] = (
        1.0 - out["stock_reported"] - out["cash_reported"]
    )
    out["reported_bond_discrepancy_pp"] = (
        out["bond_reported"] - out["bond_residual_from_stock_cash"]
    ) * 100.0
    out["reported_bond_discrepancy_flag"] = (
        out["bond_reported"].notna()
        & out["bond_residual_from_stock_cash"].notna()
        & out["reported_bond_discrepancy_pp"].abs().gt(
            cfg.reported_bond_discrepancy_warn_pp
        )
    )

    reported_primary = out[["stock_reported", "bond_residual_from_stock_cash", "cash_reported"]]
    reported_complete = reported_primary.notna().all(axis=1)
    reported_primary_sane = (
        reported_complete
        & reported_primary.ge(0.0).all(axis=1)
        & reported_primary.le(1.0).all(axis=1)
    )
    out["reported_sum_raw"] = out[["stock_reported", "bond_reported", "cash_reported"]].sum(
        axis=1, min_count=3
    )
    out["reported_sum_research"] = reported_primary.sum(axis=1, min_count=3)

    out["reported_quality"] = "missing"
    out.loc[reported_complete & ~reported_primary_sane, "reported_quality"] = "review"
    out.loc[reported_primary_sane, "reported_quality"] = "high"
    out.loc[
        reported_primary_sane & out["reported_bond_discrepancy_flag"],
        "reported_quality",
    ] = "residual_review"

    # Holdings proxy remains classification-based. We do NOT force it to 100%, because
    # unknown holdings must not be silently reclassified as bonds/cash.
    out["proxy_known"] = out[
        ["holdings_equity_proxy", "holdings_bond_proxy", "holdings_cash_proxy"]
    ].sum(axis=1)
    proxy_sane = (
        out["proxy_known"].between(
            cfg.holdings_proxy_min_coverage,
            cfg.holdings_proxy_max_coverage,
        )
        & out[["holdings_equity_proxy", "holdings_bond_proxy", "holdings_cash_proxy"]]
        .ge(-0.10).all(axis=1)
        & out[["holdings_equity_proxy", "holdings_bond_proxy", "holdings_cash_proxy"]]
        .le(1.20).all(axis=1)
    )

    use_report = out["reported_quality"].isin(["high", "residual_review"])
    use_proxy = ~use_report & proxy_sane

    # Research rule: S+B+C must sum to exactly 1 within the research universe.
    # The raw classified proxy shares (holdings_equity_proxy etc.) only sum to
    # proxy_known (allowed to be in [0.70, 1.20] to tolerate unknown/unclassified
    # holdings), so the proxy path must be normalized by proxy_known before it is
    # used as stock_weight/bond_weight/cash_weight. allocation_known_coverage
    # (set below) separately preserves proxy_known so data-quality is never lost,
    # it is just no longer conflated with the allocation weights themselves.
    out["stock_weight"] = np.where(
        use_report,
        out["stock_reported"],
        np.where(use_proxy, out["holdings_equity_proxy"] / out["proxy_known"], np.nan),
    )
    out["bond_weight"] = np.where(
        use_report,
        out["bond_residual_from_stock_cash"],
        np.where(use_proxy, out["holdings_bond_proxy"] / out["proxy_known"], np.nan),
    )
    out["cash_weight"] = np.where(
        use_report,
        out["cash_reported"],
        np.where(use_proxy, out["holdings_cash_proxy"] / out["proxy_known"], np.nan),
    )
    out["allocation_source"] = np.select(
        [use_report, use_proxy],
        ["reported_stock_cash_residual_bond", "holdings_proxy_high_confidence"],
        default="unavailable",
    )
    out["allocation_quality"] = np.select(
        [use_report, use_proxy],
        [out["reported_quality"], "proxy"],
        default="missing",
    )
    out["allocation_known_coverage"] = np.where(
        use_report,
        1.0,
        np.where(use_proxy, out["proxy_known"], np.nan),
    )
    total = out[["stock_weight", "bond_weight", "cash_weight"]].sum(axis=1, min_count=3)
    out["other_unclassified_weight"] = np.where(use_report, 0.0, 1.0 - total)

    return out.sort_values(["crsp_portno", "month_end"])


def add_changes(
    events: pd.DataFrame,
    cfg: BuildConfig,
) -> pd.DataFrame:
    """Teacher research rule: Observed Delta compares the current allocation to
    the previous OBSERVED ALLOCATION EVENT, i.e. the nearest earlier report in the
    same portfolio that itself had a usable Stock/Bond/Cash allocation. A raw
    intervening report with unavailable allocation is skipped when looking
    backward -- it is not treated as breaking comparability by itself, as long as
    the resulting gap to that earlier usable report is within
    cfg.max_change_gap_months. (Previously this used the immediate prior raw
    report row via a plain groupby shift, which silently dropped comparability
    for any event whose immediately preceding report lacked allocation, even when
    an earlier usable report existed within the allowed gap.)"""
    e = events.sort_values(
        ["crsp_portno", "month_end"]
    ).reset_index(drop=True).copy()

    allocation_ok = e[["stock_weight", "bond_weight", "cash_weight"]].notna().all(axis=1)
    e["_unusable_allocation_report"] = (~allocation_ok).astype(int)
    e["_cum_unusable_upto_here"] = (
        e.groupby("crsp_portno")["_unusable_allocation_report"].cumsum()
    )

    lookup_cols = [
        "stock_weight", "bond_weight", "cash_weight",
        "other_unclassified_weight", "allocation_known_coverage",
        "allocation_source", "allocation_quality",
        "_cum_unusable_upto_here",
    ]
    if "portfolio_tna" in e.columns:
        lookup_cols.append("portfolio_tna")

    valid = e.loc[allocation_ok, ["crsp_portno", "month_end"] + lookup_cols].copy()
    rename_map = {c: f"prev_{c}" for c in lookup_cols}
    rename_map["month_end"] = "prev_observed_allocation_month"
    rename_map["_cum_unusable_upto_here"] = "_cum_unusable_at_prev_valid"
    rename_map["portfolio_tna"] = "prev_event_portfolio_tna"
    valid = valid.rename(columns=rename_map)
    valid = valid.sort_values("prev_observed_allocation_month")

    # merge_asof(direction="backward") finds, for each row, the nearest earlier
    # valid-allocation report in the same portfolio -- this is the vectorized
    # equivalent of "search backward past unusable reports for the last valid one".
    e = pd.merge_asof(
        e.sort_values("month_end"),
        valid,
        left_on="month_end",
        right_on="prev_observed_allocation_month",
        by="crsp_portno",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values(["crsp_portno", "month_end"]).reset_index(drop=True)

    e["prev_month_end"] = e["prev_observed_allocation_month"]
    e["gap_months_from_prev_report"] = (
        e["month_end"].dt.year * 12
        + e["month_end"].dt.month
        - (
            e["prev_observed_allocation_month"].dt.year * 12
            + e["prev_observed_allocation_month"].dt.month
        )
    )
    # Explicit teacher-requested alias for the same value.
    e["observed_change_gap_months"] = e["gap_months_from_prev_report"]
    # How many intervening raw reports (in the same portfolio, strictly between
    # the previous valid allocation event and the current one) were skipped
    # because they had unusable allocation.
    e["skipped_unusable_reports_count"] = (
        e["_cum_unusable_upto_here"] - e["_unusable_allocation_report"]
        - e["_cum_unusable_at_prev_valid"]
    )
    e.loc[e["prev_observed_allocation_month"].isna(), "skipped_unusable_reports_count"] = np.nan
    e = e.drop(columns=["_cum_unusable_upto_here", "_unusable_allocation_report", "_cum_unusable_at_prev_valid"])

    for c, short in [
        ("stock_weight", "stock"),
        ("bond_weight", "bond"),
        ("cash_weight", "cash"),
        ("other_unclassified_weight", "other_unclassified"),
    ]:
        e[f"delta_{short}_pp"] = (
            e[c] - e[f"prev_{c}"]
        ) * 100.0

    dcols = [
        "delta_stock_pp",
        "delta_bond_pp",
        "delta_cash_pp",
    ]

    all_components = e[dcols].notna().all(axis=1)
    e["allocation_change_l1_pp"] = np.where(
        all_components,
        e[dcols].abs().sum(axis=1),
        np.nan,
    )
    e["allocation_turnover_pp"] = (
        0.5 * e["allocation_change_l1_pp"]
    )

    max_abs = e[dcols].abs().max(axis=1, skipna=False)
    e["has_allocation_change"] = (
        all_components & max_abs.gt(cfg.change_epsilon_pp)
    )
    e["is_research_change"] = (
        all_components
        & max_abs.ge(cfg.research_change_threshold_pp)
    )

    e["comparable_change"] = (
        e[dcols].notna().all(axis=1)
        & e["gap_months_from_prev_report"].between(
            1, cfg.max_change_gap_months
        )
    )

    ds = e["delta_stock_pp"]
    e["allocation_direction"] = np.select(
        [
            ds >= 1,
            ds <= -1,
            ds.abs() < 1,
        ],
        [
            "risk_on_more_equity",
            "defensive_less_equity",
            "roughly_stable_equity",
        ],
        default="mixed_or_missing",
    )

    return e


def build_ticker_monthly_return_panel(price_cache: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker monthly return panel, reindexed onto each ticker's own full
    monthly calendar (one row per (yahoo_ticker, month_end) with that month's
    simple return). Feeds both the forward-looking 6M compounded return table
    (t -> t+6, the No-Action outcome legs) and the backward passive
    buy-and-hold drift walk (t- -> t, used to passively re-establish the
    previous portfolio at t per the 2026-09 formula correction)."""
    parts = []
    for ticker, g in price_cache.groupby("yahoo_ticker", sort=False):
        gg = g[["month_end", "adj_close"]].drop_duplicates("month_end", keep="last").sort_values("month_end")
        if gg.empty:
            continue
        calendar = pd.date_range(gg["month_end"].min(), gg["month_end"].max(), freq="ME")
        gg = gg.set_index("month_end").reindex(calendar)
        gg.index.name = "month_end"
        gg["monthly_return"] = gg["adj_close"].pct_change(fill_method=None)
        gg = gg.reset_index()
        gg["yahoo_ticker"] = ticker
        parts.append(gg[["yahoo_ticker", "month_end", "monthly_return"]])
    if not parts:
        return pd.DataFrame(columns=["yahoo_ticker", "month_end", "monthly_return"])
    return pd.concat(parts, ignore_index=True)


def walk_passive_drift(
    initial_weights: dict,
    monthly_return_lookup: dict,
    months: "pd.DatetimeIndex",
) -> tuple:
    """Recursively carry `initial_weights` (observed at t-, keyed by component
    id -- an individual stock ticker, or "B"/"C" for the bond/cash sleeves)
    forward through each calendar month in `months` (t-+1 .. t) under a pure
    buy-and-hold assumption (no rebalancing, no manager trades):

        R_BH_m  = sum_i w_i,m-1 * r_i,m
        w_i,m   = w_i,m-1 * (1 + r_i,m) / (1 + R_BH_m)

    `monthly_return_lookup[component_id]` must be a pandas Series indexed by
    month_end. Returns (drifted_weights, True) on success, or (None, False)
    if any component's return is missing in any intervening month -- this is
    a strict complete-case rule (matching the rest of this pipeline): a
    missing drift-period return is never replaced by zero, it invalidates the
    whole passive-drift reconstruction for that event.
    """
    weights = dict(initial_weights)
    if len(months) == 0:
        return weights, True
    for m in months:
        rets = {}
        for cid in weights:
            series = monthly_return_lookup.get(cid)
            r = series.get(m) if series is not None else None
            if r is None or pd.isna(r):
                return None, False
            rets[cid] = float(r)
        r_bh = sum(weights[cid] * rets[cid] for cid in weights)
        denom = 1.0 + r_bh
        if not np.isfinite(denom) or denom == 0:
            return None, False
        weights = {cid: weights[cid] * (1.0 + rets[cid]) / denom for cid in weights}
    return weights, True


def _compound_series_between(
    series: pd.Series,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
) -> float:
    """Compound calendar-month returns over (start_month, end_month]."""
    if pd.isna(start_month) or pd.isna(end_month) or end_month <= start_month:
        return np.nan
    months = pd.date_range(
        start_month + pd.offsets.MonthEnd(1),
        end_month,
        freq="ME",
    )
    vals = pd.to_numeric(series.reindex(months), errors="coerce")
    if len(vals) != len(months) or vals.isna().any():
        return np.nan
    return float(np.prod(1.0 + vals.to_numpy(dtype=float)) - 1.0)


def add_flow_interval_diagnostics(
    events: pd.DataFrame,
    fund: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add event-interval and forward-six-month flow diagnostics.

    Flow is never silently subtracted from manager skill. We estimate total external
    flow from TNA and returns, then report it separately. The cumulative formulation
    also verifies that every calendar month in the requested interval is present.
    """
    e = events.copy()
    f = fund[["crsp_portno", "month_end", "estimated_net_flow"]].copy()
    f["crsp_portno"] = pd.to_numeric(f["crsp_portno"], errors="coerce").astype("Int64")
    f["month_end"] = pd.to_datetime(f["month_end"], errors="coerce")
    f = f.dropna(subset=["crsp_portno", "month_end"]).sort_values(["crsp_portno", "month_end"])
    # A zero is used only as a guarded cumulative-sum increment. The validity
    # counter below prevents any interval containing a missing flow from being
    # emitted as an observed zero-flow interval.
    f["flow_increment_for_guarded_cumsum"] = f["estimated_net_flow"].where(
        f["estimated_net_flow"].notna(), 0.0
    )
    f["flow_cum"] = f["flow_increment_for_guarded_cumsum"].groupby(
        f["crsp_portno"]
    ).cumsum()
    f["flow_valid_cum"] = f["estimated_net_flow"].notna().astype(int).groupby(f["crsp_portno"]).cumsum()

    flow_cum = f.set_index(["crsp_portno", "month_end"])["flow_cum"]
    valid_cum = f.set_index(["crsp_portno", "month_end"])["flow_valid_cum"]

    ports = pd.to_numeric(e["crsp_portno"], errors="coerce").astype("Int64")
    cur = pd.to_datetime(e["month_end"], errors="coerce")
    prev = pd.to_datetime(e["prev_month_end"], errors="coerce")
    future6 = cur + pd.offsets.MonthEnd(6)

    def lookup(series: pd.Series, months: pd.Series) -> pd.Series:
        mi = pd.MultiIndex.from_arrays([ports, months], names=["crsp_portno", "month_end"])
        return pd.Series(series.reindex(mi).to_numpy(), index=e.index, dtype=float)

    cur_cum = lookup(flow_cum, cur)
    prev_cum = lookup(flow_cum, prev)
    cur_valid = lookup(valid_cum, cur)
    prev_valid = lookup(valid_cum, prev)
    future_cum = lookup(flow_cum, future6)
    future_valid = lookup(valid_cum, future6)

    gap = pd.to_numeric(e["gap_months_from_prev_report"], errors="coerce")
    interval_ok = gap.gt(0) & (cur_valid - prev_valid).eq(gap)
    interval_flow = (cur_cum - prev_cum).where(interval_ok)

    future_ok = (future_valid - cur_valid).eq(6)
    future_flow = (future_cum - cur_cum).where(future_ok)

    prev_event_tna = pd.to_numeric(e.get("prev_event_portfolio_tna"), errors="coerce")
    cur_tna = pd.to_numeric(e.get("portfolio_tna"), errors="coerce")

    e["interval_estimated_net_flow"] = interval_flow
    e["interval_estimated_flow_ratio"] = interval_flow / prev_event_tna.where(prev_event_tna.gt(0))
    e["future_6m_estimated_net_flow"] = future_flow
    e["future_6m_estimated_flow_ratio"] = future_flow / cur_tna.where(cur_tna.gt(0))
    e["interval_flow_available"] = (
        interval_ok & prev_event_tna.gt(0) & interval_flow.notna()
    )
    e["future_6m_flow_available"] = (
        future_ok & cur_tna.gt(0) & future_flow.notna()
    )
    e["interval_flow_missing_reason"] = np.select(
        [
            e["interval_flow_available"],
            prev.isna(),
            ~interval_ok,
            prev_event_tna.isna(),
            prev_event_tna.le(0),
        ],
        [
            "available",
            "no_previous_report_event",
            "incomplete_interval_flow_calendar",
            "missing_previous_event_tna",
            "nonpositive_previous_event_tna",
        ],
        default="unresolved_interval_flow_input",
    )
    e["future_6m_flow_missing_reason"] = np.select(
        [
            e["future_6m_flow_available"],
            ~future_ok,
            cur_tna.isna(),
            cur_tna.le(0),
        ],
        [
            "available",
            "insufficient_or_incomplete_future_flow_calendar",
            "missing_current_event_tna",
            "nonpositive_current_event_tna",
        ],
        default="unresolved_future_flow_input",
    )
    return e


def add_exact_event_future_fund_returns(
    events: pd.DataFrame,
    fund: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach t+1...t+6 returns to every holdings event.

    An event can lack a fund-level row in month t while all six future monthly
    returns exist. Future-window availability therefore must not depend on an
    inner/current-month fund record.
    """
    e = events.copy()
    ports = pd.to_numeric(e["crsp_portno"], errors="coerce").astype("Int64")
    for value_col, output_col in [
        ("fund_gross_return_approx", "future_6m_fund_gross_return"),
        ("fund_net_return", "future_6m_fund_net_return"),
    ]:
        values = (
            fund.dropna(subset=["crsp_portno", "month_end"])
            .set_index(["crsp_portno", "month_end"])[value_col]
        )
        wealth = pd.Series(1.0, index=e.index, dtype=float)
        valid = pd.Series(True, index=e.index, dtype=bool)
        for k in range(1, 7):
            target = e["month_end"] + pd.offsets.MonthEnd(k)
            key = pd.MultiIndex.from_arrays(
                [ports, target], names=["crsp_portno", "month_end"]
            )
            monthly = pd.Series(
                values.reindex(key).to_numpy(), index=e.index, dtype=float
            )
            valid &= monthly.notna()
            wealth = (wealth * (1.0 + monthly)).where(valid)
        e[output_col] = wealth - 1.0
    return e


def load_stock_selection_holdings(
    data_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build CURRENT-event holdings for Case B / No-Action.

    w_j,t = MV_j,t / (sum equity MV + sum bond MV + sum cash MV).
    `holding_market_val` is already shares * price. If an event lacks usable
    market values, normalized `holding_percent_tna` is an explicit fallback.

    2026-08-30: the holdings schema gained `holding_nbr_shares` (share count).
    It is carried through to the per-stock detail output as an audit/reference
    field only -- it is NOT used to compute or cross-check `cs_item_weight`,
    because the raw files still have no per-share price series, so
    shares * price cannot be independently reconstructed and verified against
    `holding_market_val`. Bond/cash holdings can also populate a "units" style
    quantity in the same column, so treating a positive share count as an
    equity signal on its own would be an unjustified classification rule.
    """
    folder = data_root / "crsp" / "holdings_raw"
    cols = [
        "crsp_portno", "report_dt", "holding_market_val", "holding_percent_tna",
        "holding_security_name", "holding_ticker", "holding_permno", "holding_nbr_shares",
    ]
    paths = [
        "stock berfore 2010_new___.csv",
        "stock between 2010_2014_new___.csv",
        "stock between 2015_2019_new___.csv",
        "stock between 2020_2026_new___.csv",
    ]
    h = pd.concat(
        [pd.read_csv(folder / name, usecols=cols, low_memory=False) for name in paths],
        ignore_index=True,
    )
    h["report_date_original"] = pd.to_datetime(h["report_dt"], errors="coerce")
    h["month_end"] = month_end(h["report_dt"])
    h = h.dropna(subset=["crsp_portno", "report_date_original", "month_end"]).copy()
    h["crsp_portno"] = pd.to_numeric(h["crsp_portno"], errors="coerce").astype("Int64")
    latest = h.groupby(["crsp_portno", "month_end"])["report_date_original"].transform("max")
    h = h[h["report_date_original"].eq(latest)].copy()

    maps, removed = build_classification_maps(data_root)
    h["name_key"] = normalize_name(h["holding_security_name"])
    if not maps.empty:
        h = h.merge(maps, on="name_key", how="left")
    else:
        h["teacher_category"] = np.nan
        h["teacher_subcategory"] = np.nan

    h["removed_false_equity"] = h["name_key"].isin(removed)
    h["bucket"] = "unknown"
    h.loc[h["holding_permno"].notna() & ~h["removed_false_equity"], "bucket"] = "equity"
    h.loc[h["teacher_category"].eq("Equity Fund / Stock-fund-like"), "bucket"] = "equity"
    cash_like = (
        h["teacher_subcategory"].fillna("")
        .str.contains(r"cash|money market|liquidity|deposit", case=False, regex=True)
    )
    bond_side = h["teacher_category"].eq("Bond / Credit / Money-related")
    h.loc[bond_side & cash_like, "bucket"] = "cash"
    h.loc[bond_side & ~cash_like, "bucket"] = "bond"

    upper_name = h["holding_security_name"].fillna("").astype(str).str.upper()
    cash_fallback = upper_name.str.contains(
        r"CASH|MONEY MARKET|DEPOSIT|REPURCHASE|COMMERCIAL PAPER|CERTIFICATE OF DEPOSIT|EURODOLLAR",
        regex=True,
    )
    bond_fallback = upper_name.str.contains(
        r"TREASURY|\bBOND\b|\bNOTE\b|DEBENTURE|MUNICIPAL|MORTGAGE|ASSET BACKED|\bMBS\b|\bCMBS\b|FNMA|FHLMC|GNMA|HIGH YIELD|FLOATING RATE",
        regex=True,
    )
    h.loc[h["bucket"].eq("unknown") & cash_fallback, "bucket"] = "cash"
    h.loc[h["bucket"].eq("unknown") & ~cash_fallback & bond_fallback, "bucket"] = "bond"

    h["holding_market_val"] = pd.to_numeric(h["holding_market_val"], errors="coerce")
    h["holding_percent_tna"] = pd.to_numeric(h["holding_percent_tna"], errors="coerce")
    h["holding_nbr_shares"] = pd.to_numeric(h["holding_nbr_shares"], errors="coerce")
    h["holding_ticker"] = h["holding_ticker"].fillna("").astype(str).str.strip().str.upper()
    h["yahoo_ticker"] = h["holding_ticker"].str.replace(".", "-", regex=False)
    # 2026-08-30 data-quality fix: a handful of historical holdings tickers have
    # since been reassigned by exchanges to an unrelated, later company, so a
    # naive current-day Yahoo lookup under the OLD ticker string silently
    # returns the WRONG company's prices (or nothing at all) instead of the
    # historically-held one. Verified from this project's own data before
    # aliasing -- do not add an entry without the same standard of evidence:
    # the security name transitions cleanly across the corporate action inside
    # holdings_raw itself, AND the successor ticker's cached price series
    # starts exactly at the transition with no gap. Confirmed so far:
    #   FB -> META: `holding_security_name` for ticker FB reads "FACEBOOK INC"
    #   through 2022 and "META PLATFORMS INC" from 2022 onward (the June 2022
    #   Facebook -> Meta Platforms rename); the "FB" ticker was later reassigned
    #   to an unrelated small-cap starting mid-2025, while "META" is already
    #   cached with a continuous series from 2012-05 (Facebook's IPO month)
    #   through the present.
    #   HWP -> HPQ: `holding_security_name` is uniformly "HEWLETT PACKARD CO"
    #   (the NYSE ticker changed from HWP to HPQ in 2002 after the Compaq
    #   merger); "HPQ" is cached continuously from 2002-08 with prices
    #   consistent with HP's known split-adjusted history (~$3-5 in 2002-03,
    #   ~$12-15 in 2010).
    #   MHP -> SPGI: `holding_security_name` is uniformly "MCGRAW HILL COS
    #   INC" -- the same legal entity/CIK renamed McGraw Hill Financial (2013)
    #   then S&P Global (2016), keeping continuous listing throughout; "SPGI"
    #   is cached continuously from 2002-08 with prices consistent with
    #   McGraw-Hill's known history (~$18-20 in 2002-03, ~$20-26 in 2010).
    # Only add an entry here after confirming BOTH of these from data already
    # in hand -- do not add a ticker rename from memory/general knowledge
    # alone. Several superficially similar candidates were checked and
    # rejected: EMC, CAM, VAL, LMNX, APC, WB were acquired and delisted (no
    # legitimate successor ticker); ABX (Barrick Gold, ABX->GOLD in 2019) and
    # DELL (went private 2013, re-IPO'd 2018) have gaps that predate their
    # successor ticker's own cached history, so remapping would not help;
    # UN/UL (Unilever) were two separately-priced dual-listed entities before
    # their 2020 unification, so aliasing would misprice pre-2020 events.
    YAHOO_TICKER_RENAME_ALIASES = {"FB": "META", "HWP": "HPQ", "MHP": "SPGI"}
    h["yahoo_ticker"] = h["yahoo_ticker"].replace(YAHOO_TICKER_RENAME_ALIASES)
    keys = ["crsp_portno", "month_end", "report_date_original"]

    # Total raw market value is retained to quantify how much of the holdings
    # inventory is classifiable as known S/B/C before normalization.
    all_mv = (
        h.assign(_mv=h["holding_market_val"].clip(lower=0))
        .groupby(keys, as_index=False)["_mv"].sum(min_count=1)
        .rename(columns={"_mv": "cs_total_holding_market_value"})
    )

    known = h[h["bucket"].isin(["equity", "bond", "cash"])].copy()
    if known.empty:
        return pd.DataFrame(), pd.DataFrame()
    grp = known.groupby(keys, sort=False)
    known["_known_mv_total"] = grp["holding_market_val"].transform(lambda x: x.clip(lower=0).sum(min_count=1))
    known["_known_pct_total"] = grp["holding_percent_tna"].transform(lambda x: x.clip(lower=0).sum(min_count=1))
    known["_mv_all_present"] = grp["holding_market_val"].transform(lambda x: x.notna().all())
    known["_use_mv"] = known["_mv_all_present"] & known["_known_mv_total"].gt(0)
    known["cs_item_weight"] = np.where(
        known["_use_mv"],
        known["holding_market_val"].clip(lower=0) / known["_known_mv_total"],
        known["holding_percent_tna"].clip(lower=0) / known["_known_pct_total"].replace(0, np.nan),
    )
    known["weight_source"] = np.where(
        known["_use_mv"], "holding_market_val_direct", "holding_percent_tna_normalized_fallback"
    )

    weight_pivot = (
        known.pivot_table(index=keys, columns="bucket", values="cs_item_weight", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    for col in ["equity", "bond", "cash"]:
        if col not in weight_pivot.columns:
            weight_pivot[col] = 0.0
    weight_pivot = weight_pivot.rename(columns={
        "equity": "cs_stock_weight", "bond": "cs_bond_weight", "cash": "cs_cash_weight"
    })
    weight_pivot["cs_sbc_weight_sum"] = weight_pivot[["cs_stock_weight", "cs_bond_weight", "cs_cash_weight"]].sum(axis=1)
    # A valid teacher-formula event must have a usable denominator.  If both
    # market-value and holding-percent denominators are unusable, pivot_table
    # would otherwise materialize three zeros.  Zero/zero is not a 0%
    # allocation; keep the event explicitly missing instead.
    valid_weight_sum = weight_pivot["cs_sbc_weight_sum"].sub(1.0).abs().le(1e-9)
    weight_pivot.loc[
        ~valid_weight_sum,
        ["cs_stock_weight", "cs_bond_weight", "cs_cash_weight", "cs_sbc_weight_sum"],
    ] = np.nan

    stats = grp.agg(
        cs_known_market_value=("holding_market_val", lambda x: x.clip(lower=0).sum(min_count=1)),
        cs_weight_source=("weight_source", "first"),
    ).reset_index()
    event_summary = weight_pivot.merge(stats, on=keys, how="left").merge(all_mv, on=keys, how="left")
    event_summary["cs_known_coverage"] = (
        event_summary["cs_known_market_value"]
        / event_summary["cs_total_holding_market_value"].replace(0, np.nan)
    )

    company_mask = pd.Series(
        [
            is_company_stock_candidate(name, ticker)
            for name, ticker in zip(known["holding_security_name"], known["holding_ticker"])
        ],
        index=known.index,
        dtype=bool,
    )
    company = known[
        known["bucket"].eq("equity")
        & known["holding_permno"].notna()
        & known["yahoo_ticker"].ne("")
        & ~known["removed_false_equity"]
        & company_mask
    ].copy()
    if company.empty:
        return pd.DataFrame(), event_summary
    detail = (
        company.groupby(keys + ["yahoo_ticker"], as_index=False)
        .agg(
            cs_stock_weight=("cs_item_weight", "sum"),
            holding_market_val=("holding_market_val", "sum"),
            holding_nbr_shares=("holding_nbr_shares", "sum"),
            holding_security_name=("holding_security_name", "first"),
            holding_ticker=("holding_ticker", "first"),
            holding_permno=("holding_permno", "first"),
            weight_source=("weight_source", "first"),
        )
    )
    return detail, event_summary


def load_yahoo_equity_adj_close_cache(data_root: Path) -> pd.DataFrame:
    """Read the reproducible Yahoo adjusted-close cache produced by prepare_yahoo_equity_prices.py."""
    path = data_root / "market" / "yahoo_equity_adj_close_monthly.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run prepare_yahoo_equity_prices.py --data-root <data> first. "
            "The builder does not silently query the network so the paper build remains reproducible."
        )
    px = pd.read_csv(path, low_memory=False)
    required = {"yahoo_ticker", "month_end", "adj_close"}
    missing = required - set(px.columns)
    if missing:
        raise ValueError(f"Yahoo monthly price cache missing columns: {sorted(missing)}")
    px["yahoo_ticker"] = px["yahoo_ticker"].fillna("").astype(str).str.upper().str.strip()
    px["month_end"] = month_end(px["month_end"])
    px["adj_close"] = pd.to_numeric(px["adj_close"], errors="coerce")
    return px.dropna(subset=["yahoo_ticker", "month_end", "adj_close"]).sort_values(["yahoo_ticker", "month_end"])


def build_no_action_model(
    events: pd.DataFrame,
    market: pd.DataFrame,
    stock_detail: pd.DataFrame,
    price_cache: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    公式修正.pdf's single No-Action model. The previous (t-) Stock/Bond/Cash
    allocation -- Stock broken into its previously-identified individual
    stocks -- is passively buy-and-hold drifted forward month-by-month from
    t- to t (S3.1: no rebalancing, no manager trades), and the DRIFTED
    weights at t become the No-Action starting point, evaluated t -> t+6 by
    re-pricing the same stocks plus the Bond/Cash proxies. Also computes the
    stock-level Action features: observed CURRENT holdings vs. the drifted
    No-Action holdings, both at t (S4).
    """
    e = events.copy()
    keys = ["crsp_portno", "month_end", "report_date_original"]
    if stock_detail.empty:
        nan_cols = [
            "cs_company_stock_weight_sum",
            "drift_stock_weight", "drift_bond_weight", "drift_cash_weight",
            "future_6m_no_action_stock_contribution",
            "future_6m_no_action_stock_terminal_contribution",
            "future_6m_no_action_bond_contribution", "future_6m_no_action_cash_contribution",
            "future_6m_no_action_return", "future_6m_no_action_terminal_wealth",
            "future_6m_excess_vs_no_action", "action_magnitude_stock",
            "delta_stock_action_pp", "delta_bond_action_pp", "delta_cash_action_pp",
            "normalized_actual_start_value", "normalized_actual_end_value_6m",
            "normalized_actual_wealth_m0", "normalized_actual_wealth_m6",
        ]
        for c in nan_cols:
            e[c] = np.nan
        e["drift_complete"] = False
        e["drift_stock_return_complete"] = False
        e["no_action_complete"] = False
        e["no_action_definition"] = ""
        e["stock_action_features_definition"] = ""
        empty_action_detail = pd.DataFrame(columns=[
            "crsp_portno", "month_end", "report_date_original", "yahoo_ticker",
            "q_action", "q_no_action", "w_action", "w_no_action",
            "delta_q_action", "delta_w_action", "action_type",
        ])
        return e, stock_detail, empty_action_detail

    d = stock_detail.copy()
    current_sum = (
        d.groupby(keys, sort=False)["cs_stock_weight"].sum(min_count=1)
        .rename("cs_company_stock_weight_sum").reset_index()
    )
    e = e.merge(current_sum, on=keys, how="left")

    monthly_panel = build_ticker_monthly_return_panel(price_cache)
    px_parts = []
    for ticker, gg in monthly_panel.groupby("yahoo_ticker", sort=False):
        gg = compound_forward_by_calendar(
            gg[["month_end", "monthly_return"]], "monthly_return", "stock_6m_return", 6
        )
        gg["yahoo_ticker"] = ticker
        px_parts.append(gg[["yahoo_ticker", "month_end", "stock_6m_return"]])
    forward = pd.concat(px_parts, ignore_index=True) if px_parts else pd.DataFrame(
        columns=["yahoo_ticker", "month_end", "stock_6m_return"]
    )

    # Previous event's individual stock holdings, to be re-priced over the
    # CURRENT event's t -> t+6 window once passively drifted forward to t
    # (the `forward` table has 6M-forward returns for every (ticker,
    # month_end) the price cache covers, regardless of whether that ticker
    # was actually held at that particular month_end).
    prev_detail = d[["crsp_portno", "month_end", "yahoo_ticker", "cs_stock_weight"]].rename(columns={
        "month_end": "prev_observed_allocation_month",
        "cs_stock_weight": "prev_event_ticker_weight",
    })
    # `keys` uniquely identifies a current event here -- `event_id` is not
    # assigned until later in build_all.
    na_detail = e[keys + ["prev_observed_allocation_month"]].merge(
        prev_detail, on=["crsp_portno", "prev_observed_allocation_month"], how="inner"
    )

    # --- Passive buy-and-hold drift, t- -> t (公式修正.pdf S3.1) ---
    # The previous portfolio (Stock broken into its previously-identified
    # individual stocks, plus Bond/Cash) is carried forward from t- to t
    # under a pure buy-and-hold assumption (price-induced weight drift only,
    # no rebalancing, no manager trades). The drifted weights at t are the
    # No-Action starting point for the t -> t+6 evaluation.
    market_bond_m = market.set_index("month_end")["treasury10y_proxy_return"]
    market_cash_m = market.set_index("month_end")["cash_proxy_return"]
    ticker_monthly_lookup = (
        monthly_panel.set_index(["yahoo_ticker", "month_end"])["monthly_return"].sort_index()
        if not monthly_panel.empty else pd.Series(dtype=float)
    )
    _ticker_series_cache: dict = {}

    def _ticker_series(name):
        if name not in _ticker_series_cache:
            try:
                s = ticker_monthly_lookup.loc[name]
                _ticker_series_cache[name] = s if isinstance(s, pd.Series) else pd.Series(dtype=float)
            except KeyError:
                _ticker_series_cache[name] = pd.Series(dtype=float)
        return _ticker_series_cache[name]

    na_detail_by_key = (
        {k: g for k, g in na_detail.groupby(keys, sort=False)} if not na_detail.empty else {}
    )
    drift_cols = [
        "crsp_portno", "month_end", "report_date_original",
        "drift_stock_weight", "drift_bond_weight", "drift_cash_weight",
        "drift_complete",
    ]
    _prev_cols = e[keys + [
        "prev_observed_allocation_month", "prev_stock_weight", "prev_bond_weight",
        "prev_cash_weight", "comparable_change",
    ]]
    drift_rows = []
    drift_stock_weight_detail = {}
    for row in _prev_cols.itertuples(index=False, name=None):
        crsp_portno, month_end, report_date_original, prev_month, w_s, w_b, w_c, comparable = row
        key = (crsp_portno, month_end, report_date_original)
        result = dict.fromkeys(drift_cols[3:], np.nan)
        result["drift_complete"] = False
        result.update({
            "crsp_portno": crsp_portno, "month_end": month_end,
            "report_date_original": report_date_original,
        })
        valid_start = (
            bool(comparable) and pd.notna(prev_month)
            and pd.notna(w_s) and pd.notna(w_b) and pd.notna(w_c)
        )
        if not valid_start:
            drift_rows.append(result)
            continue
        months = pd.date_range(prev_month + pd.offsets.MonthEnd(1), month_end, freq="ME")

        # Per-ticker + bond + cash drift, using the previous individual-stock
        # basket re-scaled to whole-portfolio weight via prev_stock_weight.
        g = na_detail_by_key.get(key)
        if g is not None and not g.empty:
            ticker_group_sum = g["prev_event_ticker_weight"].sum()
            if ticker_group_sum and np.isfinite(ticker_group_sum) and ticker_group_sum > 0:
                # NOTE: bond/cash use non-ticker-like sentinel keys, NOT bare
                # "B"/"C" -- several real stock tickers are single letters that
                # collide with those (e.g. Citigroup's ticker IS "C"), which
                # would otherwise silently overwrite that stock's weight with
                # the cash weight and break the sum-to-one invariant.
                _BOND_KEY, _CASH_KEY = "\x00__BOND__\x00", "\x00__CASH__\x00"
                init_weights = {_BOND_KEY: w_b, _CASH_KEY: w_c}
                lookup = {_BOND_KEY: market_bond_m, _CASH_KEY: market_cash_m}
                tickers = []
                for _, r in g.iterrows():
                    tkr = r["yahoo_ticker"]
                    w_j = w_s * (r["prev_event_ticker_weight"] / ticker_group_sum)
                    init_weights[tkr] = w_j
                    lookup[tkr] = _ticker_series(tkr)
                    tickers.append(tkr)
                drifted, ok = walk_passive_drift(init_weights, lookup, months)
                if ok:
                    stock_w = {t: drifted[t] for t in tickers}
                    result["drift_stock_weight"] = sum(stock_w.values())
                    result["drift_bond_weight"] = drifted[_BOND_KEY]
                    result["drift_cash_weight"] = drifted[_CASH_KEY]
                    result["drift_complete"] = True
                    drift_stock_weight_detail[key] = stock_w
        drift_rows.append(result)

    drift_df = pd.DataFrame(drift_rows, columns=drift_cols) if drift_rows else pd.DataFrame(columns=drift_cols)
    e = e.merge(drift_df, on=keys, how="left")
    e["drift_complete"] = e["drift_complete"].fillna(False).astype(bool)

    # Evaluate the drifted weights t -> t+6, per-ticker + bond + cash, using
    # the same `forward` (stock_6m_return) table built above.
    drift_stock_rows = []
    for key, stock_w in drift_stock_weight_detail.items():
        crsp_portno, month_end, report_date_original = key
        for tkr, w_j in stock_w.items():
            drift_stock_rows.append({
                "crsp_portno": crsp_portno, "month_end": month_end,
                "report_date_original": report_date_original,
                "yahoo_ticker": tkr, "w_j_drift": w_j,
            })
    drift_detail = pd.DataFrame(
        drift_stock_rows,
        columns=["crsp_portno", "month_end", "report_date_original", "yahoo_ticker", "w_j_drift"],
    )
    if not drift_detail.empty:
        drift_detail = drift_detail.merge(forward, on=["yahoo_ticker", "month_end"], how="left")
        drift_detail["return_available"] = drift_detail["stock_6m_return"].notna()
        drift_detail["contribution"] = drift_detail["w_j_drift"] * drift_detail["stock_6m_return"]
        drift_detail["terminal_contribution"] = (
            drift_detail["w_j_drift"] * (1.0 + drift_detail["stock_6m_return"])
        )
        drift_summary_rows = []
        for key, g in drift_detail.groupby(keys, sort=False):
            positive = pd.to_numeric(g["w_j_drift"], errors="coerce").fillna(0).gt(0)
            complete = bool((~positive | g["return_available"]).all() and positive.any())
            drift_summary_rows.append({
                "crsp_portno": key[0], "month_end": key[1], "report_date_original": key[2],
                "drift_stock_return_complete": complete,
                "future_6m_no_action_stock_contribution": float(g["contribution"].sum()) if complete else np.nan,
                "future_6m_no_action_stock_terminal_contribution": float(g["terminal_contribution"].sum()) if complete else np.nan,
            })
        drift_summary = pd.DataFrame(drift_summary_rows)
    else:
        drift_summary = pd.DataFrame(columns=keys + [
            "drift_stock_return_complete",
            "future_6m_no_action_stock_contribution",
            "future_6m_no_action_stock_terminal_contribution",
        ])
    e = e.merge(drift_summary, on=keys, how="left")
    e["drift_stock_return_complete"] = e["drift_stock_return_complete"].fillna(False).astype(bool)
    e["future_6m_no_action_bond_contribution"] = e["drift_bond_weight"] * e["future_6m_treasury10y_proxy_return"]
    e["future_6m_no_action_cash_contribution"] = e["drift_cash_weight"] * e["future_6m_cash_proxy_return"]
    _na_components = e[[
        "future_6m_no_action_stock_contribution",
        "future_6m_no_action_bond_contribution",
        "future_6m_no_action_cash_contribution",
    ]]
    e["no_action_complete"] = (
        e["drift_complete"] & e["drift_stock_return_complete"] & _na_components.notna().all(axis=1)
    )
    e["future_6m_no_action_return"] = np.where(
        e["no_action_complete"], _na_components.sum(axis=1), np.nan,
    )
    e["future_6m_no_action_terminal_wealth"] = np.where(
        e["no_action_complete"],
        e["future_6m_no_action_stock_terminal_contribution"]
        + e["drift_bond_weight"] * (1.0 + e["future_6m_treasury10y_proxy_return"])
        + e["drift_cash_weight"] * (1.0 + e["future_6m_cash_proxy_return"]),
        np.nan,
    )
    e["future_6m_excess_vs_no_action"] = (
        e["future_6m_fund_gross_return"] - e["future_6m_no_action_return"]
    )
    e["no_action_definition"] = (
        "No-Action counterfactual (公式修正.pdf): at t-, initialize buy-and-hold "
        "weights for every previously identified individual stock plus Bond/Cash; "
        "passively drift them month-by-month from t- to t using each stock's own "
        "monthly return (Bond=synthetic Treasury duration/carry proxy, Cash=F-F RF, "
        "no rebalancing, no manager trades); the drifted weights at t are the "
        "No-Action starting point, evaluated t->t+6 by re-pricing the SAME stocks. "
        "Excess = Actual future 6M fund gross return - No-Action 6M return. Excess "
        "is not alpha and not a measure of manager skill (公式修正.pdf S9): it is a "
        "single decomposition of the realized outcome, not a risk-adjusted estimate. "
        "Missing drift-period or forward-period returns are never zero-filled."
    )

    # --- Stock-level Action features: Actual holdings at t vs. passively ---
    # --- drifted No-Action holdings at t (公式修正.pdf S4). Additive; does ---
    # --- not affect any return/Excess computed above. Union universe      ---
    # --- U_t = H_t (actual current holdings) u H_t- (previous holdings,   ---
    # --- passively drifted forward to t).                                 ---
    d_by_key = {k: g for k, g in d.groupby(keys, sort=False)}
    stock_weight_by_key = e.set_index(keys)["stock_weight"]
    cs_sum_by_key = e.set_index(keys)["cs_company_stock_weight_sum"]
    action_rows = []
    for key, drift_w in drift_stock_weight_detail.items():
        w_s_current = stock_weight_by_key.get(key)
        cs_sum_current = cs_sum_by_key.get(key)
        w_s_drift = sum(drift_w.values())
        if w_s_drift <= 0:
            continue
        # Actual (current) within-sleeve weights: q_Action_j,t = MV_j,t / sum_l MV_l,t
        # over the CURRENT identified company-stock holdings only.
        q_action = {}
        w_action = {}
        g_cur = d_by_key.get(key)
        if g_cur is not None and pd.notna(cs_sum_current) and cs_sum_current > 0 and pd.notna(w_s_current):
            for _, r in g_cur.iterrows():
                if pd.to_numeric(r["cs_stock_weight"], errors="coerce") and r["cs_stock_weight"] > 0:
                    q_action[r["yahoo_ticker"]] = r["cs_stock_weight"] / cs_sum_current
                    w_action[r["yahoo_ticker"]] = w_s_current * (r["cs_stock_weight"] / cs_sum_current)
        q_na = {tkr: w / w_s_drift for tkr, w in drift_w.items()}
        w_na = dict(drift_w)
        union = set(q_action) | set(q_na)
        for tkr in union:
            qa, qn = q_action.get(tkr, 0.0), q_na.get(tkr, 0.0)
            wa, wn = w_action.get(tkr, 0.0), w_na.get(tkr, 0.0)
            if qa > 0 and qn == 0:
                action_type = "observed_entry"
            elif qa == 0 and qn > 0:
                action_type = "observed_drop"
            elif qa > 0 and qn > 0 and qa > qn:
                action_type = "observed_weight_increase"
            elif qa > 0 and qn > 0 and qa < qn:
                action_type = "observed_weight_decrease"
            else:
                action_type = "unchanged"
            action_rows.append({
                "crsp_portno": key[0], "month_end": key[1], "report_date_original": key[2],
                "yahoo_ticker": tkr,
                "q_action": qa, "q_no_action": qn, "w_action": wa, "w_no_action": wn,
                "delta_q_action": qa - qn, "delta_w_action": wa - wn,
                "action_type": action_type,
            })
    stock_action_detail = pd.DataFrame(action_rows, columns=[
        "crsp_portno", "month_end", "report_date_original", "yahoo_ticker",
        "q_action", "q_no_action", "w_action", "w_no_action",
        "delta_q_action", "delta_w_action", "action_type",
    ])
    if not stock_action_detail.empty:
        action_magnitude = (
            stock_action_detail.assign(abs_dq=lambda x: x["delta_q_action"].abs())
            .groupby(keys)["abs_dq"].sum().mul(0.5).rename("action_magnitude_stock").reset_index()
        )
    else:
        action_magnitude = pd.DataFrame(columns=keys + ["action_magnitude_stock"])
    e = e.merge(action_magnitude, on=keys, how="left")
    e["delta_stock_action_pp"] = np.where(
        e["drift_complete"], (e["stock_weight"] - e["drift_stock_weight"]) * 100.0, np.nan,
    )
    e["delta_bond_action_pp"] = np.where(
        e["drift_complete"], (e["bond_weight"] - e["drift_bond_weight"]) * 100.0, np.nan,
    )
    e["delta_cash_action_pp"] = np.where(
        e["drift_complete"], (e["cash_weight"] - e["drift_cash_weight"]) * 100.0, np.nan,
    )
    e["stock_action_features_definition"] = (
        "Stock-level Action features (公式修正.pdf S4): compares the observed CURRENT "
        "holdings report H_t against the passively drifted No-Action holdings at t "
        "(previous holdings H_t- carried forward under buy-and-hold from t- to t, no "
        "rebalancing). q_action/q_no_action are within-stock-sleeve weights; "
        "w_action/w_no_action are whole-portfolio weights; delta_* = Action - No-Action. "
        "action_type is one of observed_entry/observed_drop/observed_weight_increase/"
        "observed_weight_decrease/unchanged -- described as an OBSERVED change in the "
        "disclosed (usually Top-10) holdings set, not an exact reconstruction of a buy "
        "or sell transaction. action_magnitude_stock = 0.5 * sum|delta_q_action| over "
        "the union of current and drifted-previous holdings. delta_{stock,bond,cash}_"
        "action_pp are the asset-class-level Actual-minus-drifted-No-Action gaps, in "
        "percentage points; computed only when the passive drift is complete."
    )

    e["normalized_actual_start_value"] = np.where(e["future_6m_fund_gross_return"].notna(), 1.0, np.nan)
    e["normalized_actual_end_value_6m"] = 1.0 + e["future_6m_fund_gross_return"]
    # Endpoint-only aliases used by the UI. Intermediate months are intentionally not exposed.
    e["normalized_actual_wealth_m0"] = e["normalized_actual_start_value"]
    e["normalized_actual_wealth_m6"] = e["normalized_actual_end_value_6m"]
    return e, d, stock_action_detail


def strict_no_action_required_fields() -> list[str]:
    """公式修正.pdf's No-Action model: Actual versus the previous individual-
    stock/Bond/Cash mix after being passively buy-and-hold drifted t- -> t."""
    return (
        IDENTITY_REQUIRED_FIELDS
        + ALLOCATION_REQUIRED_FIELDS
        + CHANGE_REQUIRED_FIELDS
        + [
            "drift_stock_weight", "drift_bond_weight", "drift_cash_weight",
            "drift_complete", "drift_stock_return_complete",
            "future_6m_fund_gross_return",
            "future_6m_no_action_stock_contribution", "future_6m_no_action_bond_contribution",
            "future_6m_no_action_cash_contribution", "future_6m_no_action_return",
            "future_6m_no_action_terminal_wealth", "future_6m_excess_vs_no_action",
        ]
    )


def add_strict_eligibility(
    events: pd.DataFrame,
    market: pd.DataFrame,
    fund: pd.DataFrame,
) -> pd.DataFrame:
    """Create the complete-case No-Action sample (公式修正.pdf's single model)."""
    e = events.copy()

    e["identity_complete"] = e[IDENTITY_REQUIRED_FIELDS].notna().all(axis=1)
    e["allocation_complete"] = e[ALLOCATION_REQUIRED_FIELDS].notna().all(axis=1)
    e["change_complete"] = e[CHANGE_REQUIRED_FIELDS].notna().all(axis=1)
    e["future6m_fund_complete"] = e["future_6m_fund_gross_return"].notna()
    e["flow_complete"] = (
        e["flow_available"].fillna(False).astype(bool)
        & e["interval_flow_available"].fillna(False).astype(bool)
        & e["future_6m_flow_available"].fillna(False).astype(bool)
    )

    comparable = e["comparable_change"].fillna(False).astype(bool)

    # The No-Action counterfactual needs the Bond/Cash proxy 6M forward legs
    # (the Stock leg's own completeness is tracked by drift_stock_return_complete).
    no_action_market_complete = e[[
        "future_6m_treasury10y_proxy_return", "future_6m_cash_proxy_return"
    ]].notna().all(axis=1)
    e["no_action_market_complete"] = no_action_market_complete

    eligible_no_action = (
        e["identity_complete"] & e["allocation_complete"] & e["change_complete"] & comparable
        & e["future6m_fund_complete"] & no_action_market_complete
        & e["no_action_complete"].fillna(False).astype(bool)
        & e["future_6m_no_action_return"].notna()
        & e["future_6m_no_action_terminal_wealth"].notna()
        & e["future_6m_excess_vs_no_action"].notna()
    )
    e["eligible_no_action"] = eligible_no_action

    reasons = pd.Series("", index=e.index, dtype="object")

    def add_reason(mask: pd.Series, label: str) -> None:
        active = mask.fillna(False).astype(bool)
        prior = reasons.loc[active]
        reasons.loc[active] = np.where(prior.eq(""), label, prior + ";" + label)

    target6 = e["month_end"] + pd.offsets.MonthEnd(6)
    fund_max = e["crsp_portno"].map(fund.groupby("crsp_portno")["month_end"].max())
    market_max = pd.to_datetime(market["month_end"], errors="coerce").max()
    tail = target6.gt(fund_max) | target6.gt(market_max)

    add_reason(~e["identity_complete"], "identity_incomplete")
    add_reason(~e["allocation_complete"], "current_allocation_unavailable")
    add_reason(~e["change_complete"] | ~comparable, "previous_valid_allocation_event_unavailable")
    add_reason(~e["future6m_fund_complete"] & tail, "actual_6m_insufficient_future_calendar")
    add_reason(~e["future6m_fund_complete"] & ~tail, "actual_6m_fund_return_incomplete")
    no_action_future_incomplete = ~eligible_no_action & no_action_market_complete & comparable & e["change_complete"]
    add_reason(no_action_future_incomplete & tail, "no_action_insufficient_future_calendar")
    add_reason(
        no_action_future_incomplete & ~tail & ~e["drift_complete"].fillna(False).astype(bool),
        "no_action_drift_price_or_weight_incomplete",
    )
    add_reason(
        no_action_future_incomplete
        & ~tail
        & e["drift_complete"].fillna(False).astype(bool)
        & ~e["drift_stock_return_complete"].fillna(False).astype(bool),
        "no_action_stock_6m_forward_price_incomplete",
    )
    add_reason(
        e["change_complete"] & comparable & e["future_6m_treasury10y_proxy_return"].isna(),
        "bond_6m_path_incomplete",
    )
    add_reason(
        e["change_complete"] & comparable & e["future_6m_cash_proxy_return"].isna(),
        "cash_6m_rf_path_incomplete",
    )
    e["missing_reason"] = reasons.where(reasons.ne(""), "eligible")
    e["excluded_from_no_action"] = ~eligible_no_action
    e["exclusion_reason_no_action"] = np.where(eligible_no_action, "eligible", e["missing_reason"])
    e["excluded_from_strict_sample"] = ~eligible_no_action
    e["exclusion_reason"] = e["exclusion_reason_no_action"]
    return e


def _json_ready_subset(
    df: pd.DataFrame,
    cols: list[str],
) -> list[dict]:
    x = df[[c for c in cols if c in df.columns]].copy()

    for c in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = x[c].dt.strftime("%Y-%m-%d")

    x = x.replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return json.loads(
        x.to_json(
            orient="records",
            date_format="iso",
        )
    )


def write_baseline_jsons(
    events: pd.DataFrame,
    derived: Path,
) -> None:
    """Project the persisted master into the endpoint-only No-Action JSON (公式修正.pdf's single model)."""
    common = [
        "event_id", "analysis_entity", "analysis_entity_level", "family_name", "crsp_portno",
        "report_date_original", "month_end", "baseline_anchor_month", "past_6m_context_start",
        "stock_weight", "bond_weight", "cash_weight", "allocation_source", "allocation_quality",
        "allocation_known_coverage", "other_unclassified_weight", "stock_reported", "bond_reported",
        "cash_reported", "holdings_equity_proxy", "holdings_bond_proxy", "holdings_cash_proxy",
        "holdings_unknown_proxy", "delta_stock_pp", "delta_bond_pp", "delta_cash_pp",
        "allocation_change_l1_pp", "allocation_turnover_pp", "gap_months_from_prev_report",
        "eligible_no_action", "missing_reason", "excluded_from_strict_sample", "exclusion_reason",
        "interval_estimated_flow_ratio", "future_6m_estimated_flow_ratio",
        "flow_available", "flow_missing_reason", "interval_flow_available",
        "interval_flow_missing_reason", "future_6m_flow_available", "future_6m_flow_missing_reason",
        "expense_ratio_weight_coverage", "expense_ratio_complete", "expense_ratio_missing_reason",
        "cash_return_method", "cash_return_is_fallback", "reported_bond_pct_raw",
        "bond_residual_from_stock_cash", "reported_bond_discrepancy_pp",
        "reported_bond_discrepancy_flag",
    ]

    no_action_cols = common + [
        "no_action_definition", "stock_action_features_definition",
        "prev_stock_weight", "prev_bond_weight", "prev_cash_weight",
        "drift_stock_weight", "drift_bond_weight", "drift_cash_weight",
        "drift_complete", "drift_stock_return_complete",
        "future_6m_fund_gross_return",
        "future_6m_no_action_stock_contribution", "future_6m_no_action_bond_contribution",
        "future_6m_no_action_cash_contribution", "future_6m_no_action_return",
        "future_6m_no_action_terminal_wealth", "future_6m_excess_vs_no_action",
        "action_magnitude_stock", "delta_stock_action_pp", "delta_bond_action_pp",
        "delta_cash_action_pp",
        "no_action_market_complete", "no_action_complete",
        "exclusion_reason_no_action",
    ]
    (derived / "baseline_no_action_events.json").write_text(
        json.dumps(_json_ready_subset(events, no_action_cols), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_all(
    data_root: Path,
    cfg: BuildConfig | None = None,
) -> dict:
    cfg = cfg or BuildConfig()
    data_root = Path(data_root)
    derived = data_root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    input_paths = step1_input_paths(data_root)
    input_manifest_before = file_hash_manifest(input_paths, data_root)

    fund = load_fund_level(data_root)
    market = load_market(data_root, cfg)
    holdings = load_holdings_events(data_root, cfg)
    cs_stock_detail, _cs_event_weights = load_stock_selection_holdings(data_root)
    price_cache = load_yahoo_equity_adj_close_cache(data_root)

    e = holdings.merge(
        fund,
        on=["crsp_portno", "month_end"],
        how="left",
    )
    e = add_exact_event_future_fund_returns(e, fund)
    # Holdings reports without a matching fund-level observation retain missing
    # flow values, but their availability/provenance fields are always explicit.
    e["flow_available"] = e["flow_available"].where(
        e["flow_available"].notna(), False
    ).astype(bool)
    e["flow_missing_reason"] = e["flow_missing_reason"].where(
        e["flow_missing_reason"].notna(),
        "no_matching_portfolio_month_fund_record",
    )
    e["expense_ratio_complete"] = e["expense_ratio_complete"].where(
        e["expense_ratio_complete"].notna(), False
    ).astype(bool)
    e["expense_ratio_missing_reason"] = e["expense_ratio_missing_reason"].where(
        e["expense_ratio_missing_reason"].notna(),
        "no_matching_portfolio_month_fund_record",
    )
    e = e.merge(
        market,
        on="month_end",
        how="left",
    )
    e = add_changes(e, cfg)

    e = add_flow_interval_diagnostics(e, fund)
    e, cs_detail_result, stock_action_detail = build_no_action_model(
        e, market, cs_stock_detail, price_cache
    )

    e["event_id"] = (
        "BAL_"
        + e["crsp_portno"].astype(str)
        + "_"
        + e["month_end"].dt.strftime("%Y%m")
    )
    e["event_month"] = (
        e["month_end"].dt.strftime("%Y-%m")
    )
    e["baseline_anchor_month"] = e["event_month"]
    e["past_6m_context_start"] = e["month_end"] - pd.offsets.MonthEnd(6)

    e["analysis_entity"] = e["analysis_entity"].fillna(
        "UNRESOLVED MANAGER/FAMILY"
    )
    e["analysis_entity_level"] = (
        e["analysis_entity_level"].fillna("unresolved")
    )
    e["family_name"] = e["family_name"].fillna(
        "Unknown Family"
    )
    if not cs_detail_result.empty:
        cs_detail_result = cs_detail_result.merge(
            e[["event_id", "crsp_portno", "month_end", "report_date_original", "analysis_entity", "family_name"]],
            on=["crsp_portno", "month_end", "report_date_original"],
            how="left",
        )
    if not stock_action_detail.empty:
        stock_action_detail = stock_action_detail.merge(
            e[["event_id", "crsp_portno", "month_end", "report_date_original"]],
            on=["crsp_portno", "month_end", "report_date_original"],
            how="left",
        )
    e = add_strict_eligibility(e, market, fund)

    master_path = derived / "balanced_allocation_events.csv"
    e.to_csv(master_path, index=False)
    fund.to_csv(
        derived / "balanced_portfolio_month_returns.csv",
        index=False,
    )
    market.to_csv(
        derived / "market_monthly.csv",
        index=False,
    )

    cs_detail_result.to_csv(derived / "no_action_stock_detail.csv", index=False)
    stock_action_detail.to_csv(derived / "no_action_stock_action_detail.csv", index=False)

    # The baseline JSON files are projections of the persisted master CSV only.
    # Re-reading the master prevents an in-memory parallel calculation/schema.
    master_for_json = pd.read_csv(master_path, low_memory=False)
    write_baseline_jsons(master_for_json, derived)

    cash_method = (
        market["cash_return_method"].dropna().iloc[0]
        if market["cash_return_method"].notna().any()
        else "unknown"
    )

    audit = {
        "build_config": asdict(cfg),
        "method_warning": (
            "The synthetic Bond proxy is a duration/carry proxy derived from H.15 "
            "DGS10 yield, not an official bond total-return series."
        ),
        "no_action_definition": (
            "公式修正.pdf's single No-Action model: at t-, initialize buy-and-hold weights for "
            "every previously identified individual stock plus Bond/Cash; passively drift them "
            "month-by-month from t- to t (no rebalancing, no manager trades); the drifted weights "
            "at t are the No-Action starting point, evaluated t->t+6 by re-pricing the same stocks "
            "plus the Bond/Cash proxies. Excess = Actual - No-Action; not alpha, not skill."
        ),
        "allocation_change_definition": (
            "Observed Delta: current ACTUAL allocation minus previous ACTUAL observed allocation "
            "(used for eligibility gating and the stock-level Action features' 'Actual' side)."
        ),
        "endpoint_display_rule": (
            "Selected-event charts expose only t-, t, and t+6 endpoints; intermediate monthly points are not shown."
        ),
        "reported_allocation_rule": (
            "For reported summaries, research bond weight is always 1 - reported stock - reported cash "
            "when stock and cash are available. The original reported bond is preserved only for audit "
            "and discrepancy flags; it is not trusted merely because it is non-missing."
        ),
        "cash_return_method": cash_method,
        "event_rows": int(len(e)),
        "unique_event_ids": int(e["event_id"].nunique()),
        "event_date_min": str(e["month_end"].min().date()),
        "event_date_max": str(e["month_end"].max().date()),
        "allocation_available": int(e["stock_weight"].notna().sum()),
        "allocation_quality_counts": {
            str(k): int(v)
            for k, v in e[
                "allocation_quality"
            ].value_counts(dropna=False).items()
        },
        "allocation_source_counts": {
            str(k): int(v)
            for k, v in e[
                "allocation_source"
            ].value_counts(dropna=False).items()
        },
        "reported_bond_discrepancy_flag_count": int(
            e["reported_bond_discrepancy_flag"].fillna(False).astype(bool).sum()
        ),
        "change_events": int(
            e["has_allocation_change"].fillna(False).sum()
        ),
        "research_change_events": int(
            e["is_research_change"].fillna(False).sum()
        ),
        "eligible_future_6m_events": int(
            e["future6m_fund_complete"].fillna(False).sum()
        ),
        "eligible_no_action_events": int(
            e["eligible_no_action"].fillna(False).sum()
        ),
        "specific_manager_events": int(
            e["analysis_entity_level"].eq("manager").sum()
        ),
        "fallback_manager_events": int(
            e["analysis_entity_level"].ne("manager").sum()
        ),
        "fund_return_date_min": str(
            fund["month_end"].min().date()
        ),
        "fund_return_date_max": str(
            fund["month_end"].max().date()
        ),
        "market_date_min": str(
            market["month_end"].min().date()
        ),
        "market_date_max": str(
            market["month_end"].max().date()
        ),
    }

    (
        derived / "balanced_allocation_events_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        derived / "STEP1_DATA_DICTIONARY.md"
    ).write_text(
        """# Step 1 Data Dictionary — No-Action Passive-Drift Model (公式修正.pdf) 2026-09

- One Event = one actually observed holdings report for a portfolio in a calendar month. Missing report months are not fabricated.
- Observed Delta uses REAL observed data only: `delta_*_pp = current_weight - previous_observed_weight`. This is the "Actual" side of the stock-level Action features, not the No-Action counterfactual itself.
- `report_date_original` is preserved; `month_end` is the analysis key.
- Reported-summary rule: research `bond_weight = 1 - stock_reported - cash_reported`; raw reported bond remains audit-only.
- Cash return uses the monthly `RF` column of `F-F_Research_Data_Factors.csv`; YYYYMM is mapped to true month-end and percent units are divided by 100. The Bond leg is a synthetic duration/carry proxy derived from H.15 DGS10 yield.
- No-Action model (公式修正.pdf S3.1): at t-, initialize buy-and-hold weights for every previously identified individual stock plus Bond/Cash; passively drift them month-by-month from t- to t using each component's own monthly return (no rebalancing, no manager trades). The drifted weights at t (`drift_stock_weight`/`drift_bond_weight`/`drift_cash_weight`) are the No-Action starting point, evaluated t->t+6 by re-pricing the same stocks plus the Bond/Cash proxies: `future_6m_no_action_return`.
- `future_6m_excess_vs_no_action` = Actual future 6M fund gross return - `future_6m_no_action_return`. Per 公式修正.pdf S9, this is a decomposition, not alpha and not a measure of manager skill.
- Missing drift-period or forward-period returns are never zero-filled; incomplete Events remain ineligible (`eligible_no_action`).
- Stock-level Action features (公式修正.pdf S4): `no_action_stock_action_detail.csv` compares observed CURRENT holdings at t (`q_action`/`w_action`) against the passively drifted No-Action holdings at t (`q_no_action`/`w_no_action`), classifying each stock as `observed_entry`/`observed_drop`/`observed_weight_increase`/`observed_weight_decrease`/`unchanged`. `action_magnitude_stock = 0.5 * sum|delta_q_action|` over the union of current and drifted-previous holdings.
- `no_action_stock_detail.csv` stores each stock's CURRENT-event weight (used as the "Actual" side of the Action features).
""",
        encoding="utf-8",
    )

    input_manifest_after = file_hash_manifest(input_paths, data_root)
    inputs_unchanged = input_manifest_before == input_manifest_after
    input_hash_payload = {
        "algorithm": "SHA-256",
        "scope": "files consumed by scripts/build_balanced_events.py",
        "unchanged_during_build": inputs_unchanged,
        "files": input_manifest_after,
    }
    (derived / "STEP1_INPUT_HASHES.json").write_text(
        json.dumps(input_hash_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not inputs_unchanged:
        raise RuntimeError(
            "A Step 1 input changed while the build was running; outputs are not reproducible."
        )

    builder_output_names = [
        "balanced_allocation_events.csv",
        "balanced_portfolio_month_returns.csv",
        "market_monthly.csv",
        "baseline_no_action_events.json",
        "no_action_stock_detail.csv",
        "no_action_stock_action_detail.csv",
        "balanced_allocation_events_audit.json",
        "STEP1_DATA_DICTIONARY.md",
        "STEP1_INPUT_HASHES.json",
    ]
    output_payload = {
        "algorithm": "SHA-256",
        "scope": "Step 1 builder outputs (hash manifest excludes itself)",
        "files": file_hash_manifest(
            [derived / name for name in builder_output_names], derived
        ),
    }
    (derived / "STEP1_OUTPUT_HASHES.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    print(
        json.dumps(
            build_all(Path(args.data_root)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
