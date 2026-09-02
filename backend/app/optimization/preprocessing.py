"""Deterministic summary calculations over cleaned remittance records."""

import re

import pandas as pd

from ..schemas import CountrySummary, DatasetSummaryResponse


PERIOD_PATTERN = re.compile(r"^(?P<year>\d{4})_(?P<quarter>[1-4])Q$")


def _period_sort_key(period: object) -> tuple[int, int]:
    match = PERIOD_PATTERN.fullmatch(str(period))
    if match is None:
        raise ValueError(f"Invalid reporting period: {period!r}")
    return int(match.group("year")), int(match.group("quarter"))


def _country_summaries(
    dataset: pd.DataFrame, code_column: str, name_column: str
) -> list[CountrySummary]:
    countries = (
        dataset[[code_column, name_column]]
        .drop_duplicates()
        .sort_values([name_column, code_column])
    )
    return [
        CountrySummary(code=str(code), name=str(name))
        for code, name in countries.itertuples(index=False, name=None)
    ]


def build_dataset_summary(dataset: pd.DataFrame) -> DatasetSummaryResponse:
    """Build the API summary without modifying the cached source dataframe."""

    if dataset.empty:
        raise ValueError("The cleaned dataset is empty.")

    latest_period = max(dataset["PERIOD"], key=_period_sort_key)
    kenya_to_tanzania = dataset[
        (dataset["SOURCE_CODE"] == "KEN")
        & (dataset["DESTINATION_CODE"] == "TZA")
    ]

    return DatasetSummaryResponse(
        record_count=int(len(dataset)),
        corridor_count=int(dataset["CORRIDOR"].nunique()),
        latest_reporting_period=str(latest_period),
        sending_countries=_country_summaries(
            dataset, "SOURCE_CODE", "SOURCE_NAME"
        ),
        receiving_countries=_country_summaries(
            dataset, "DESTINATION_CODE", "DESTINATION_NAME"
        ),
        kenya_to_tanzania_record_count=int(len(kenya_to_tanzania)),
    )
