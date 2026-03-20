# BharatFed

BharatFed is a platform for turning Account Aggregator financial data into practical tools for:

- personal finance analytics,
- explainable cashflow forecasting, and
- P2P/micro-credit workflows.

The original repository contains legacy experiments for privacy-preserving ML, FIU integration, and loan processing. This codebase has now been extended with a lightweight, dependency-minimal backend service that makes the repo usable without requiring the old TensorFlow stack.

## What is now available

### Modernized backend service

The repository now includes a standard-library HTTP API that exposes:

- `/health` for service status,
- `/analytics/overview` for account summaries from the sample FIU payload,
- `/analytics/forecast?days=30` for rule-based recurring cashflow projections,
- `/profiles` and `/profiles/<user_id>` for borrower/lender profile access,
- `/loans`, `/loans/requests`, and `/portfolio/summary` for lending operations, and
- `POST /loans/requests` to register new loan requests safely in SQLite.

Run it locally with:

```bash
python app.py
```

Or customize paths/host/port:

```bash
python app.py --host 0.0.0.0 --port 8080 --sample-data data_response_bharatfed99@finvu_1yr.txt --database bharat_fed.db
```

## Repository structure

- `bharatfed/analytics.py` – FIU payload parsing, analytics summaries, and forecast generation.
- `bharatfed/lending.py` – SQLite-backed profile, loan request, and portfolio helpers.
- `bharatfed/service.py` – application service layer used by the API.
- `bharatfed/api.py` – dependency-light HTTP server.
- `app.py` – local server entrypoint.
- `tests/` – built-in `unittest` coverage for analytics, lending, and HTTP endpoints.

## Legacy ML scripts

The original ML and experimentation scripts are still present:

1. `train_tfe_model.py`
2. `train_dp_model.py`
3. `federated_model_gen.py`
4. `test_fiu_api.py`
5. `test_1yr_data_req.py`

These scripts rely on older dependencies listed in `requirements.txt`. The new backend layer is intentionally written using the Python standard library so it can be exercised immediately in constrained environments.

## Running tests

Use Python’s built-in unittest discovery:

```bash
python -m unittest discover -s tests -v
```

## Data sources

- Sample FIU responses are included in `data_response.txt` and `data_response_bharatfed99@finvu_1yr.txt`.
- Lending data is backed by the checked-in SQLite database `bharat_fed.db`.

## Modernization Task Plan

A detailed execution task for upgrading BharatFed for rural credit accessibility is available in `TASK_BHARATFED_V2_RURAL_CREDIT.md`.
