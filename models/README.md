# Models

`lead_scoring.joblib` is written by `python -m lead_scoring.train` and is not committed: it is fully
reproducible from the data and the pinned dependencies.

The bundle carries the calibrated classifier, an empty reference frame that fixes the training
category sets, and the profit-maximising threshold found on the test period.

The service does not gate on that threshold. It gates on break-even (`cost / value`), because the
profit-maximising cutoff describes one period while the per-lead decision is a question about one
lead. See `app/main.py`.
