"""Feature preparation.

Kept deliberately thin. The point of this project is a deployable scoring
service with honest evaluation, not feature engineering, and every transform
here has to be reproducible inside the API at request time.
"""

from __future__ import annotations

import pandas as pd

from .config import CATEGORICAL, NUMERIC

FEATURES = list(CATEGORICAL) + list(NUMERIC)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready features.

    Categoricals stay as pandas `category` dtype: LightGBM consumes them
    natively, which avoids one-hot encoding and, more importantly, avoids a
    fitted encoder that the API would have to keep in sync with the model.
    """
    prepared = frame[FEATURES].copy()
    for column in CATEGORICAL:
        prepared[column] = prepared[column].astype("category")
    return prepared


def align_categories(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Force a batch to carry the training category sets.

    Without this, a request whose `job` is "admin." would be encoded against a
    category index built from that one request rather than from training, and
    the prediction would be silently wrong instead of loudly broken.

    A value the training window never contained is mapped to NaN explicitly,
    and LightGBM treats it as a missing category. This is not hypothetical: the
    chronological split means `month` in {mar, apr, sep, dec} appears only after
    the training cutoff, so a perfectly valid March lead is a category the model
    has never seen. The schema still accepts it — March is a real month — and
    the model falls back to the rest of the lead's features rather than failing.
    """
    aligned = frame.copy()
    for column in CATEGORICAL:
        known = reference[column].cat.categories
        # Blank out unseen values first, so the cast never has to silently
        # discard them (pandas 4 raises on that).
        values = aligned[column].where(aligned[column].isin(known))
        aligned[column] = values.astype(pd.CategoricalDtype(categories=known))
    return aligned[reference.columns]
