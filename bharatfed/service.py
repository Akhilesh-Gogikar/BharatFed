"""High-level BharatFed service orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .analytics import extract_transactions, forecast_cashflow, load_fiu_payload, summarize_transactions
from .lending import LendingRepository


class BharatFedService:
    """Single facade used by the CLI/tests/API."""

    def __init__(self, *, sample_data_path: str, database_path: str) -> None:
        self.sample_data_path = sample_data_path
        self.database_path = database_path
        self.lending = LendingRepository(database_path)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "sample_data_path": str(Path(self.sample_data_path)),
            "database_path": str(Path(self.database_path)),
        }

    def analytics_overview(self) -> dict[str, Any]:
        payload = load_fiu_payload(self.sample_data_path)
        transactions = extract_transactions(payload)
        summary = summarize_transactions(transactions)
        summary["data_source"] = Path(self.sample_data_path).name
        return summary

    def analytics_forecast(self, days: int = 30) -> dict[str, Any]:
        payload = load_fiu_payload(self.sample_data_path)
        transactions = extract_transactions(payload)
        forecast = forecast_cashflow(transactions, days=days)
        forecast["data_source"] = Path(self.sample_data_path).name
        return forecast

    def list_profiles(self) -> list[dict[str, Any]]:
        return self.lending.list_profiles()

    def get_profile(self, user_id: str) -> dict[str, Any] | None:
        return self.lending.get_profile(user_id)

    def list_loan_requests(self, status: str | None = None) -> list[dict[str, Any]]:
        return self.lending.list_loan_requests(status=status)

    def list_loans(self) -> list[dict[str, Any]]:
        return self.lending.list_loans()

    def portfolio_summary(self) -> dict[str, Any]:
        return self.lending.portfolio_summary()

    def create_loan_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        required_fields = {
            "borrower_id",
            "lender_id",
            "loan_amount",
            "interest_rate",
            "loan_period_months",
            "text",
        }
        missing = sorted(required_fields - payload.keys())
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        return self.lending.create_loan_request(
            borrower_id=str(payload["borrower_id"]),
            lender_id=str(payload["lender_id"]),
            loan_amount=float(payload["loan_amount"]),
            interest_rate=float(payload["interest_rate"]),
            loan_period_months=int(payload["loan_period_months"]),
            text=str(payload["text"]),
        )
