"""Data access and filtering helpers for QKash.

The app uses the World Bank RPW-style CSV in ``data/``.  Filtering is kept
separate from scoring so the UI can enforce the required flow:

    load dataset -> filter corridor/options -> score -> Pareto prune -> optimize
"""

from __future__ import annotations

# Imports.
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


# Variable descriptions.
# PROJECT_ROOT is the repository root; it is used to find the default CSV.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# PROCESSED_DATA_PATH is preferred when a separate processed folder is present.
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "remittance_east_africa_clean.csv"

# ROOT_DATA_PATH is the audited RPW extract currently present in this project.
ROOT_DATA_PATH = PROJECT_ROOT / "data" / "remittance_east_africa_clean.csv"

# Also accept the legacy/export RPW file `RPW_dataset.csv` if present and prefer
# it over the default root CSV when the processed export is absent. This lets the
# app use `data/RPW_dataset.csv` as requested by the user.
RPW_LEGACY_PATH = PROJECT_ROOT / "data" / "RPW_dataset.csv"
# DEFAULT_DATA_PATH resolves to the available audited RPW extract, preferring
# processed exports, then the legacy `RPW_dataset.csv`, then the root CSV.
if PROCESSED_DATA_PATH.exists():
    DEFAULT_DATA_PATH = PROCESSED_DATA_PATH
elif RPW_LEGACY_PATH.exists():
    DEFAULT_DATA_PATH = RPW_LEGACY_PATH
else:
    DEFAULT_DATA_PATH = ROOT_DATA_PATH

# TEXT_COLUMNS are normalized to clean strings during load.
TEXT_COLUMNS = (
    "source_name",
    "destination_name",
    "firm",
    "firm_type",
    "access point",
    "speed actual",
    "transparent",
    "receiving network coverage",
    "pickup method",
    "corridor",
    "payment instrument",
    "period",
)

# REQUIRED_NUMERIC_COLUMNS are needed to compute transaction fee and FX losses.
REQUIRED_NUMERIC_COLUMNS = (
    "cc1 lcu amount",
    "cc1 denomination amount",
    "cc1 lcu fee",
    "cc1 fx margin",
    "cc1 total cost %",
    "cc2 lcu amount",
    "cc2 denomination amount",
    "cc2 lcu fee",
    "cc2 fx margin",
    "cc2 total cost %",
)

# AMOUNT_TIERS map the app's amount input to the available RPW fee columns.
AMOUNT_TIERS = ("cc1", "cc2")

# SERVICE_KEY_COLUMNS define one provider-and-transfer-method candidate.
SERVICE_KEY_COLUMNS = ("firm", "payment instrument", "pickup method")

# COLUMN_ALIASES allow common underscore/case variants to map into the CSV names.
COLUMN_ALIASES = {
    "access_point": "access point",
    "speed_actual": "speed actual",
    "receiving_network_coverage": "receiving network coverage",
    "pickup_method": "pickup method",
    "payment_instrument": "payment instrument",
    "pick-up method": "pickup method",
    "pick_up_method": "pickup method",
}

# SOURCE_COLUMN_FLAGS record whether optional modeling columns came from the CSV or were created
# as empty compatibility columns by the loader.
SOURCE_COLUMN_FLAGS = {
    "transparent": "_source_has_transparent",
    "receiving network coverage": "_source_has_receiving_network_coverage",
    "access point": "_source_has_access_point",
}


# DATE_FORMATS are the source date spellings the loader accepts, in priority order.
DATE_FORMATS = ("%d/%b/%Y", "%Y-%m-%d")


@dataclass(frozen=True)
class FilterSpec:
    """UI-selected filters applied before Pareto pruning."""

    source_name: str | None = None
    destination_name: str | None = None
    corridor: str | None = None
    firm_type: str | None = None
    pickup_method: str | None = None
    payment_instrument: str | None = None
    access_point: str | None = None
    period_range: tuple[str, str] | None = None


@dataclass(frozen=True)
class AmountMatch:
    """Mapping between a user amount and the closest RPW amount tier."""

    requested_amount: float
    amount_tier: str
    matched_denomination: float
    distance: float


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and lightly normalize the CSV without changing its meaning."""

    csv_path = Path(path) if path else DEFAULT_DATA_PATH
    df = pd.read_csv(csv_path)
    df = ensure_supported_columns(df)

    df["date_parsed"] = parse_dates(df.get("date", pd.Series([], dtype=object)))
    df["period_order"] = df["period"].map(period_order)
    return df


def parse_dates(values: object) -> pd.Series:
    """Parse source dates, accepting every spelling in ``DATE_FORMATS``.

    The RPW extracts do not agree on a date format: earlier periods use
    ``24/Jan/2011`` while the audited export and the 2025 periods use ISO
    ``2011-01-24``.  Each format is tried in turn against the values still
    unparsed, so a column mixing both resolves completely instead of silently
    becoming ``NaT``.
    """

    series = pd.Series(values, dtype=object)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for date_format in DATE_FORMATS:
        remaining = parsed.isna()
        if not remaining.any():
            break
        parsed.loc[remaining] = pd.to_datetime(
            series.loc[remaining],
            format=date_format,
            errors="coerce",
        )
    return parsed


def period_order(label: str | float | int | None) -> int:
    """Convert labels like ``2025_3Q`` into sortable integers."""

    text = "" if label is None else str(label)
    match = re.match(r"^(\d{4})_(\d)Q$", text.strip())
    if not match:
        return -1
    year, quarter = match.groups()
    return int(year) * 10 + int(quarter)


def sorted_periods(df: pd.DataFrame) -> list[str]:
    """Return available period labels in chronological order."""

    periods = [value for value in df["period"].dropna().unique() if str(value).strip()]
    return sorted(periods, key=period_order)


def unique_values(df: pd.DataFrame, column: str) -> list[str]:
    """Return sorted non-empty display values for a filter widget."""

    df = ensure_supported_columns(df, require_numeric=False)
    if column not in df.columns:
        return []

    values: set[str] = set()
    for raw_value in df[column].dropna().astype(str):
        for value in raw_value.split(","):
            cleaned = value.strip()
            if cleaned:
                values.add(cleaned)
    return sorted(values, key=str.lower)


def filter_dataset(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    """Apply all user filters before any scoring or Pareto pruning."""

    filtered = ensure_supported_columns(df, require_numeric=False)

    filtered = _filter_exact(filtered, "source_name", spec.source_name)
    filtered = _filter_exact(filtered, "destination_name", spec.destination_name)
    filtered = _filter_exact(filtered, "corridor", spec.corridor)
    filtered = _filter_exact(filtered, "firm_type", spec.firm_type)
    filtered = _filter_exact(filtered, "pickup method", spec.pickup_method)
    filtered = _filter_token(filtered, "payment instrument", spec.payment_instrument)
    filtered = _filter_token(filtered, "access point", spec.access_point)
    if spec.period_range:
        start, end = spec.period_range
        start_order = period_order(start)
        end_order = period_order(end)
        filtered = filtered[
            (filtered["period_order"] >= start_order) & (filtered["period_order"] <= end_order)
        ]

    return filtered.reset_index(drop=True)


def match_transfer_amount(df: pd.DataFrame, transfer_amount: float) -> AmountMatch:
    """Choose the RPW amount tier closest to the user-entered transfer amount."""

    amount = float(transfer_amount)
    if amount <= 0:
        raise ValueError("transfer_amount must be positive")

    normalized = ensure_supported_columns(df, require_numeric=False)
    tier_denominations: dict[str, float] = {}
    for tier in AMOUNT_TIERS:
        column = f"{tier} denomination amount"
        if column not in normalized.columns:
            continue
        values = pd.to_numeric(normalized[column], errors="coerce").dropna()
        if not values.empty:
            tier_denominations[tier] = float(values.median())

    if not tier_denominations:
        raise ValueError("Dataset does not include supported denomination amounts")

    amount_tier, denomination = min(
        tier_denominations.items(),
        key=lambda item: abs(item[1] - amount),
    )
    return AmountMatch(
        requested_amount=amount,
        amount_tier=amount_tier,
        matched_denomination=denomination,
        distance=abs(denomination - amount),
    )


def prepare_candidates(
    df: pd.DataFrame,
    amount_tier: str = "cc1",
    latest_per_firm: bool = False,
) -> pd.DataFrame:
    """Create optimization-ready numeric features for the selected amount tier."""

    prefix = amount_tier.lower()
    if prefix not in {"cc1", "cc2"}:
        raise ValueError("amount_tier must be either 'cc1' or 'cc2'")

    prepared = ensure_supported_columns(df)
    prepared["amount_tier"] = prefix
    prepared["amount_lcu"] = pd.to_numeric(prepared[f"{prefix} lcu amount"], errors="coerce")
    prepared["denomination_amount"] = pd.to_numeric(
        prepared[f"{prefix} denomination amount"], errors="coerce"
    )
    prepared["fee_lcu"] = pd.to_numeric(prepared[f"{prefix} lcu fee"], errors="coerce")
    prepared["fx_margin"] = pd.to_numeric(prepared[f"{prefix} fx margin"], errors="coerce")
    prepared["total_cost_pct"] = pd.to_numeric(
        prepared[f"{prefix} total cost %"],
        errors="coerce",
    )
    prepared["speed_score"] = prepared["speed actual"].map(speed_score).fillna(0.35)
    prepared["settlement_days"] = prepared["speed actual"].map(settlement_days).fillna(3.0)
    prepared["coverage_score"] = prepared["receiving network coverage"].map(coverage_score).fillna(
        0.35
    )
    prepared["transparent_score"] = np.where(
        prepared["transparent"].str.lower().eq("yes"),
        1.0,
        0.0,
    )

    prepared = prepared.dropna(
        subset=["firm", "amount_lcu", "fee_lcu", "fx_margin", "total_cost_pct"]
    )
    prepared = prepared[prepared["firm"].astype(str).str.len() > 0]

    if latest_per_firm and not prepared.empty:
        prepared = (
            prepared.sort_values(["firm", "date_parsed"], ascending=[True, False])
            .drop_duplicates("firm", keep="first")
            .reset_index(drop=True)
        )

    return prepared.reset_index(drop=True)


def latest_service_options(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest observation for each provider/payment/receiving method."""

    prepared = ensure_supported_columns(df, require_numeric=False)
    if prepared.empty:
        return prepared.copy()

    return (
        prepared.sort_values(
            ["firm", "payment instrument", "pickup method", "period_order", "date_parsed"],
            ascending=[True, True, True, False, False],
            kind="mergesort",
        )
        .drop_duplicates(list(SERVICE_KEY_COLUMNS), keep="first")
        .sort_values(["firm", "payment instrument", "pickup method"], kind="mergesort")
        .reset_index(drop=True)
    )


def ensure_supported_columns(
    df: pd.DataFrame,
    require_numeric: bool = True,
) -> pd.DataFrame:
    """Normalize expected column names and create optional text columns."""

    normalized = df.copy()
    normalized.columns = [_normalize_column_label(column) for column in normalized.columns]
    normalized = normalized.rename(
        columns={
            column: COLUMN_ALIASES.get(column, column)
            for column in normalized.columns
        }
    )
    source_columns = set(normalized.columns)
    source_presence = {
        column: _existing_source_presence(normalized, flag_column, column in source_columns)
        for column, flag_column in SOURCE_COLUMN_FLAGS.items()
    }

    for column in TEXT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()

    for column, flag_column in SOURCE_COLUMN_FLAGS.items():
        normalized[flag_column] = bool(source_presence[column])

    if require_numeric:
        missing = [column for column in REQUIRED_NUMERIC_COLUMNS if column not in normalized.columns]
        if missing:
            raise ValueError(f"Dataset missing required numeric columns: {', '.join(missing)}")

    if "date_parsed" not in normalized.columns:
        normalized["date_parsed"] = parse_dates(
            normalized.get("date", pd.Series([], dtype=object))
        )
    if "period_order" not in normalized.columns:
        normalized["period_order"] = normalized["period"].map(period_order)

    return normalized


def source_column_available(df: pd.DataFrame, column: str) -> bool:
    """Return whether an optional source column is present and has usable values."""

    normalized_column = COLUMN_ALIASES.get(
        _normalize_column_label(column),
        _normalize_column_label(column),
    )
    flag_column = SOURCE_COLUMN_FLAGS.get(normalized_column)
    if normalized_column not in df.columns:
        return False

    if flag_column and flag_column in df.columns:
        has_source = _truthy_series(df[flag_column]).any()
    else:
        has_source = True
    if not bool(has_source):
        return False

    values = df[normalized_column].fillna("").astype(str).str.strip()
    return bool(values.ne("").any())


def speed_score(label: str | None) -> float:
    """Map delivery speed text to a higher-is-better score."""

    value = (label or "").strip().lower()
    if "less than one hour" in value:
        return 1.0
    if "same day" in value:
        return 0.82
    if "next day" in value:
        return 0.62
    if "2 days" in value:
        return 0.42
    if "3-5 days" in value or "3 - 5 days" in value:
        return 0.18
    return 0.35


def settlement_days(label: str | None) -> float:
    """Map delivery speed text to a conservative upper-bound day count."""

    value = (label or "").strip().lower()
    if "less than one hour" in value:
        return 1.0 / 24.0
    if "same day" in value:
        return 1.0
    if "next day" in value:
        return 1.0
    if "2 days" in value:
        return 2.0
    if "3-5 days" in value or "3 - 5 days" in value:
        return 5.0
    return 3.0


def coverage_score(label: str | None) -> float:
    """Map receiving network coverage text to a higher-is-better score."""

    value = (label or "").strip().lower()
    if "nationwide" in value:
        return 1.0
    if "major cities" in value:
        return 0.65
    if "main city" in value:
        return 0.35
    return 0.35


def _matches_token(series: pd.Series, selected: str) -> pd.Series:
    """Match one selected value against comma-separated CSV fields."""

    wanted = selected.strip().lower()

    def matches(raw_value: str) -> bool:
        tokens = {token.strip().lower() for token in str(raw_value).split(",")}
        return wanted in tokens

    return series.fillna("").astype(str).map(matches)


def _filter_exact(df: pd.DataFrame, column: str, selected: str | None) -> pd.DataFrame:
    """Apply a single-value exact filter if a selection was made."""

    if not selected:
        return df
    if column not in df.columns:
        return df.iloc[0:0].copy()
    return df[df[column] == selected]


def _filter_token(df: pd.DataFrame, column: str, selected: str | None) -> pd.DataFrame:
    """Apply a single-value token filter if a selection was made."""

    if not selected:
        return df
    if column not in df.columns:
        return df.iloc[0:0].copy()
    return df[_matches_token(df[column], selected)]


def _normalize_column_label(column: object) -> str:
    """Normalize a raw CSV column label for matching."""

    return str(column).strip().lower()


def _existing_source_presence(
    frame: pd.DataFrame,
    flag_column: str,
    fallback: bool,
) -> bool:
    """Preserve loader provenance when an already-normalized frame is normalized again."""

    if flag_column not in frame.columns:
        return bool(fallback)
    return bool(_truthy_series(frame[flag_column]).any())


def _truthy_series(series: pd.Series) -> pd.Series:
    """Coerce boolean-like provenance flag values safely."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})
