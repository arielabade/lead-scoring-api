"""Dataset, the leakage rule, and the economics of a call.

A lead score is only useful against a cost: the model does not decide who
converts, it decides who is worth calling. The call-centre economics live here
so they can be changed without touching the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
INNER_ZIP = "bank-additional.zip"
CSV_PATH = "bank-additional/bank-additional-full.csv"
EXPECTED_ROWS = 41_188

TARGET = "y"
POSITIVE_LABEL = "yes"

# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------
# `duration` is how long the call lasted. It is only known once the call is
# over, and a call that lasts 20 minutes almost always ended in a subscription.
# Including it produces an excellent offline model that cannot be used: at
# scoring time, before anyone has been called, the field does not exist.
#
# The UCI documentation says the same thing. It is the single most common way
# this dataset is reported with inflated results.
LEAKED_COLUMNS = ("duration",)

# ---------------------------------------------------------------------------
# Macroeconomic columns: excluded, for two separate reasons
# ---------------------------------------------------------------------------
# 1. They cannot do the job. These five describe the economy on the day of the
#    call. Every lead contacted that day shares the same values, so they carry
#    no information for ranking one lead against another — which is the only
#    thing a lead score is for.
#
# 2. Under a time split they are a date in disguise. The campaign spans the
#    2008 crisis: euribor3m runs 4.08-5.05 in training and 0.63-1.30 in test,
#    with ZERO overlap. A tree splitting on "euribor > 4" meets nothing but
#    euribor < 1.3 at scoring time, and trees cannot extrapolate.
#
# Measured effect, chronological split: dropping them RAISES test AUC from
# 0.597 to 0.640. Under a random split they instead raise it from 0.778 to
# 0.811, because there they leak which period a row came from. A feature that
# helps under a random split and hurts under a time split is a time proxy.
MACRO_COLUMNS = (
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
)


@dataclass(frozen=True)
class CallEconomics:
    """What a call costs and what a conversion is worth.

    Both are illustrative and declared: the dataset publishes neither. They are
    here because a threshold chosen without them is a threshold chosen by
    accident.
    """

    cost_per_call: float = 8.0
    value_per_conversion: float = 160.0
    # Daily capacity of the calling team, used for the capacity-constrained view.
    daily_call_capacity: int = 500


@dataclass(frozen=True)
class Split:
    """Chronological split.

    The dataset is ordered by campaign date and carries macroeconomic features
    (euribor3m, nr.employed) that drift over time. A random split lets the model
    see the future state of the economy, which is both leakage and the reason
    offline scores on this dataset are often optimistic.
    """

    test_fraction: float = 0.2
    validation_fraction: float = 0.2


CATEGORICAL = (
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
)
NUMERIC = ("age", "campaign", "pdays", "previous")

ECONOMICS = CallEconomics()
SPLIT = Split()
RANDOM_SEED = 20261002
