"""Download, verify and split the Bank Marketing dataset."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import (
    CSV_PATH, DATA_URL, EXPECTED_ROWS, INNER_ZIP, LEAKED_COLUMNS,
    MACRO_COLUMNS, SPLIT, TARGET,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW = DATA_DIR / "raw"


def download() -> Path:
    """The UCI archive nests a zip inside a zip."""
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / "bank-additional-full.csv"
    if target.exists():
        return target

    print(f"downloading {DATA_URL}")
    archive = RAW / "bank+marketing.zip"
    urllib.request.urlretrieve(DATA_URL, archive)
    with zipfile.ZipFile(archive) as outer:
        inner_bytes = outer.read(INNER_ZIP)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        target.write_bytes(inner.read(CSV_PATH))
    return target


def load(drop_leakage: bool = True, drop_macro: bool = False) -> pd.DataFrame:
    """Load the campaign table.

    `drop_leakage` removes `duration`, which is only known after the call.
    `drop_macro` is off here so the leakage study can measure the macro
    columns' effect; the model itself never sees them, because they are absent
    from NUMERIC. See config.py for why.
    """
    frame = pd.read_csv(download(), sep=";")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS:,} rows, found {len(frame):,}")

    frame["converted"] = (frame[TARGET] == "yes").astype(int)
    frame = frame.drop(columns=[TARGET])
    if drop_leakage:
        frame = frame.drop(columns=list(LEAKED_COLUMNS))
    if drop_macro:
        frame = frame.drop(columns=list(MACRO_COLUMNS))
    return frame


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / validation / test, in file order.

    The file is ordered by campaign date, so slicing it in order is a time
    split. Shuffling would leak later macroeconomic conditions into training.
    """
    n = len(frame)
    test_start = int(n * (1 - SPLIT.test_fraction))
    validation_start = int(test_start * (1 - SPLIT.validation_fraction))
    return (
        frame.iloc[:validation_start].copy(),
        frame.iloc[validation_start:test_start].copy(),
        frame.iloc[test_start:].copy(),
    )
