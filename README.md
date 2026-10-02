# Lead Scoring API

A deployable lead-scoring service, and an honest account of what the model is actually worth.

---

## 1. Business problem

A bank's outbound team has more leads than it can call. Each call costs money; each subscription is
worth more than the call. The team needs an order to work the list in, decided before anyone picks up
the phone.

The modelling question is narrower than it looks: not "who will convert" but "who should be called
first, using only what is known at dialling time".

---

## 2. Key results

**The number this dataset is usually reported with is wrong, twice over.**

| Setup | Test AUC |
| --- | --- |
| Random split, with `duration` and macro features | **0.954** |
| Random split, no `duration`, with macro | 0.811 |
| Random split, no `duration`, no macro | 0.778 |
| Chronological split, with `duration` and macro | 0.728 |
| Chronological split, no `duration`, with macro | 0.597 |
| **Chronological split, no `duration`, no macro — the deployable model** | **0.640** |

Two separate leaks sit between 0.954 and 0.640:

**`duration` is the length of the call.** It only exists once the call is over, and a twenty-minute
call almost always ended in a subscription. It is the single largest driver of inflated results on
this dataset, and it cannot be used by a model that scores leads *before* dialling.

**The macroeconomic columns are a date in disguise.** `euribor3m` runs 4.08–5.05 in the training
period and 0.63–1.30 in the test period, with **zero overlap** — the campaign spans the 2008 crisis.
A tree that learns "euribor above 4 means low conversion" meets nothing but euribor below 1.3 at
scoring time, and trees cannot extrapolate. Dropping them **raises** chronological AUC from 0.597 to
0.640 while **lowering** random-split AUC from 0.811 to 0.778. A feature that helps under a random
split and hurts under a time split is a time proxy, not a signal.

They also fail a simpler test: every lead called on the same day shares the same euribor. A feature
constant across the leads being ranked cannot rank them.

### What the deployable model is worth

| | AUC | PR-AUC | Brier |
| --- | --- | --- | --- |
| LightGBM, calibrated | 0.658 | 0.425 | **0.237** |
| LightGBM, raw | **0.661** | 0.438 | 0.280 |
| Logistic regression | 0.643 | **0.443** | 0.264 |
| Prior (base rate) | 0.500 | 0.308 | 0.281 |

Gradient boosting barely beats logistic regression on ranking. It earns its place on **calibration**:
Brier 0.237 against 0.264, which matters because the scores drive a dialling decision priced in money,
not just a sort order.

**The decision this supports.** With a call costing £8 and a subscription worth £160, break-even is a
**5% conversion probability**. The test period's base rate is 30.8%, so *every* lead clears
break-even — calling everyone is profitable, and the model does not decide who to skip.

What it decides is who goes first, which is the real constraint when capacity is fixed:

| Capacity | Conversions captured | Lift vs random order | Net value |
| --- | --- | --- | --- |
| Top 10% | 15.9% | **1.59x** | £58,056 |
| Top 20% | 31.3% | 1.57x | £114,184 |
| Top 30% | 46.1% | 1.54x | £167,752 |
| Top 50% | 66.5% | 1.33x | £237,128 |

A team with capacity for 30% of the list reaches 46% of the conversions. That is the business case,
and it is a long way from what a 0.954 AUC would promise.

---

## 3. Data

| | |
| --- | --- |
| Source | Bank Marketing — UCI Machine Learning Repository ([link](https://archive.ics.uci.edu/dataset/222/bank+marketing)) |
| Content | Portuguese bank, outbound term-deposit campaigns, May 2008 – Nov 2010 |
| Size | 41,188 contacts · 20 original features |
| Split | 26,360 train · 6,590 validation · 8,238 test, **in file order** |

**All data is real. Nothing is simulated.** The declared assumptions are the call economics — £8 per
call, £160 per conversion — which the dataset does not publish and which live in `config.py`.

> The brief for this project originally named Olist's marketing-funnel dataset. It is only available
> through Kaggle, which requires credentials, and has no public mirror. Bank Marketing is a real
> marketing funnel with a conversion outcome, and it clones and runs without a login.

### The shift that governs everything

Conversion runs **4.8% in training and 30.8% in test**. By decile of file order it climbs from 2.8%
to 45.9%. The campaign ran through the financial crisis, and term deposits became dramatically easier
to sell. Any single reported "base rate" for this dataset is an average over two different worlds.

Four months — March, April, September, December — appear **only** after the training cutoff. They are
valid business inputs the model has never seen; the service maps them to a missing category and falls
back on the lead's other features rather than failing. There is a test for it.

---

## 4. Approach

**Chronological split, not random.** The file is ordered by campaign date. A random split lets the
model see the future state of the economy, and is the reason offline scores on this dataset are
routinely optimistic.

**Isotonic calibration, fitted on validation.** Scores feed a value calculation, so they must behave
like probabilities. Calibrating on training data calibrates a model to its own overconfidence, so the
booster is frozen and the isotonic map is fitted on held-out data.

**A logistic baseline, kept in the report even though it nearly wins.** A model that cannot clearly
beat logistic regression should not be carrying an API, a container and a CI pipeline without saying
so. Here boosting justifies itself on calibration, not on ranking — and the README says that rather
than hiding the baseline.

**Break-even as the per-lead gate, not 0.5.** Call when `p × value ≥ cost`, i.e. `p ≥ cost / value`.
The 0.5 default is an artefact of balanced-class tutorials and has no business meaning.

**MLflow on SQLite.** The file store is in maintenance mode upstream; a local database is the
smallest backend that still supports the model registry.

---

## 5. Business metrics

```
break_even_probability = cost_per_call / value_per_conversion      = 8 / 160 = 0.05
expected_value(call)   = p * value_per_conversion - cost_per_call
net_value(top K)       = conversions_captured * value - K * cost
lift_vs_random_order   = share_of_conversions_captured / share_of_list_called
```

| Assumption | Value | Note |
| --- | --- | --- |
| Cost per call | £8 | Agent time, telephony, overhead. |
| Value per conversion | £160 | Contribution from a term deposit. |
| Daily capacity | 500 calls | Drives the capacity view. |

Change these in `config.py` and the threshold, the capacity table and the recommendation all move.

---

## 6. Limitations and next steps

- **AUC 0.640 is a weak model, and that is the finding.** Without `duration` and without the time
  proxies, the remaining demographic and campaign-history features carry limited signal. The honest
  framing is a 1.5x lift on call ordering, not a predictive breakthrough.
- **The test period is not the deployment period.** Training on 2008–2009 and scoring 2010 is exactly
  the problem a production model faces, and it is why the model needs retraining on a rolling window
  rather than a one-off fit. Early stopping at iteration 7 is itself a symptom: the model is kept
  deliberately shallow because deeper fits do not transfer across the shift.
- **Call economics are assumed.** At £20 per call, break-even rises to 12.5% and the model starts
  genuinely excluding leads rather than only ordering them.
- **No uplift modelling.** This predicts who converts, not who converts *because they were called*.
  Some high-scoring leads would have subscribed anyway, and calling them adds cost without adding
  revenue. Separating the two needs an experiment — see
  [ab-testing-toolkit](https://github.com/arielabade/ab-testing-toolkit).
- **Next step:** retrain on a rolling window with time-based validation, and add a monitoring job
  that alerts when the live score distribution drifts from the training distribution. On this
  dataset, that alarm would have fired loudly.

---

## 7. How to run

```bash
git clone https://github.com/arielabade/lead-scoring-api
cd lead-scoring-api

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m lead_scoring.train        # downloads data, trains, logs to MLflow, writes models/
pytest                              # 18 tests
uvicorn app.main:app --reload       # http://localhost:8000/docs
```

### Container

```bash
docker build -t lead-scoring-api .
docker run -p 8000:8000 lead-scoring-api
curl localhost:8000/health
```

### Scoring a lead

```bash
curl -X POST localhost:8000/score \
  -H 'content-type: application/json' \
  -d '{"age":30,"job":"student","marital":"single","education":"university.degree",
       "default":"no","housing":"no","loan":"no","contact":"cellular","month":"mar",
       "day_of_week":"thu","campaign":1,"pdays":3,"previous":2,"poutcome":"success"}'

# {"probability":0.236,"call":true,"priority":"high"}
```

`duration` is not in the schema. It cannot be supplied, because at scoring time it does not exist.

### Layout

```
src/lead_scoring/   config (assumptions + leakage rules), data, features, train, evaluate, schema
app/                FastAPI service
tests/              leakage rules, evaluation maths, API contract
.github/workflows/  train, test, build the image, boot it and score through it
```

CI trains the model, runs the suite, builds the image, starts the container and scores a lead through
it — so a green check means the thing actually serves.
