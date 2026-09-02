"""Validated loading for the cleaned, deduplicated CSV dataset."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..config import CLEANED_DATASET_PATH


EXPECTED_COLUMNS = (
    "PERIOD",
    "SOURCE_CODE",
    "SOURCE_NAME",
    "DESTINATION_CODE",
    "DESTINATION_NAME",
    "FIRM",
    "FIRM_TYPE",
    "PAYMENT INSTRUMENT",
    "SPEED ACTUAL",
    "CC1 LCU AMOUNT",
    "CC1 DENOMINATION AMOUNT",
    "CC1 LCU CODE",
    "CC1 LCU FEE",
    "CC1 LCU FX RATE",
    "CC1 FX MARGIN",
    "CC1 TOTAL COST %",
    "CC2 LCU AMOUNT",
    "CC2 DENOMINATION AMOUNT",
    "CC2 LCU CODE",
    "CC2 LCU FEE",
    "CC2 LCU FX RATE",
    "CC2 FX MARGIN",
    "CC2 TOTAL COST %",
    "INTER LCU BANK FX",
    "TRANSPARENT",
    "NOTE1",
    "NOTE2",
    "PICK-UP METHOD",
    "DATE",
    "CORRIDOR",
)


class DatasetLoadError(RuntimeError):
    """Raised when the cleaned dataset is missing or has an unexpected schema."""


@lru_cache(maxsize=1)
def load_cleaned_dataset(path: Path = CLEANED_DATASET_PATH) -> pd.DataFrame:
    """Load the small cleaned CSV; the malformed raw workbook is never read here."""

    if not path.is_file():
        raise DatasetLoadError(f"Cleaned dataset not found: {path}")

    try:
        dataset = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise DatasetLoadError(f"Unable to read cleaned dataset: {path}") from exc

    actual_columns = tuple(dataset.columns)
    if actual_columns != EXPECTED_COLUMNS:
        raise DatasetLoadError(
            "Cleaned dataset schema mismatch: "
            f"expected {len(EXPECTED_COLUMNS)} ordered columns, "
            f"found {len(actual_columns)}."
        )

    return dataset
