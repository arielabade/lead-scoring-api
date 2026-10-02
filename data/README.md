# Data

Downloaded by the pipeline; not committed.

## Source

[Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), UCI Machine Learning
Repository. Outbound term-deposit campaigns run by a Portuguese bank, May 2008 – November 2010,
41,188 contacts.

**All data is real. Nothing is simulated.** The declared assumptions are the call economics (£8 per
call, £160 per conversion), which the dataset does not publish; they live in
`src/lead_scoring/config.py`.

## Columns the model never sees

| Column | Why |
| --- | --- |
| `duration` | Length of the call. Known only after the call ends, so it leaks the outcome. |
| `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` | Identical for every lead on a given day, so they cannot rank leads against each other — and across the 2008 crisis they act as a date in disguise. |

`load(drop_leakage=False)` keeps `duration` so the size of the leak can be measured. The model path
never uses it.
