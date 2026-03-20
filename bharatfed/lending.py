"""SQLite-backed lending workflows for BharatFed."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import math
import secrets
import sqlite3
from typing import Any


class LendingRepository:
    """Small repository abstraction over the legacy SQLite database."""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def list_profiles(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT userid, first_name, last_name, Pan, phone, email, last_balance,
                       last_txn_amt, credit_score, bill_income_ratio, profile_text,
                       frugality_factor, fin_goals
                FROM profiles
                ORDER BY userid
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_profile(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT userid, first_name, last_name, Pan, phone, email, last_balance,
                       last_txn_amt, credit_score, bill_income_ratio, profile_text,
                       frugality_factor, fin_goals
                FROM profiles
                WHERE userid = ?
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_loan_requests(self, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM loan_reqs"
        parameters: tuple[Any, ...] = ()
        if status:
            query += " WHERE req_status = ?"
            parameters = (status,)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def list_loans(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM loans ORDER BY last_updated DESC").fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def calculate_monthly_emi(principal: float, annual_interest_rate: float, months: int) -> float:
        """Calculate EMI using the standard amortization formula."""

        if principal <= 0:
            raise ValueError("principal must be positive")
        if months <= 0:
            raise ValueError("months must be positive")
        monthly_rate = annual_interest_rate / 12 / 100
        if monthly_rate == 0:
            return round(principal / months, 2)
        emi = principal * monthly_rate * math.pow(1 + monthly_rate, months)
        emi /= math.pow(1 + monthly_rate, months) - 1
        return round(emi, 2)

    def create_loan_request(
        self,
        *,
        borrower_id: str,
        lender_id: str,
        loan_amount: float,
        interest_rate: float,
        loan_period_months: int,
        text: str,
    ) -> dict[str, Any]:
        borrower = self.get_profile(borrower_id)
        lender = self.get_profile(lender_id)
        if not borrower:
            raise ValueError(f"unknown borrower_id: {borrower_id}")
        if not lender:
            raise ValueError(f"unknown lender_id: {lender_id}")
        if borrower_id == lender_id:
            raise ValueError("borrower and lender must be different users")
        if loan_amount <= 0:
            raise ValueError("loan_amount must be positive")
        if interest_rate < 0:
            raise ValueError("interest_rate cannot be negative")
        if loan_period_months <= 0:
            raise ValueError("loan_period_months must be positive")

        loan_req_id = secrets.token_urlsafe(12)
        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        payload = (
            loan_req_id,
            created_at,
            borrower_id,
            float(loan_amount),
            float(interest_rate),
            int(loan_period_months),
            text.strip(),
            lender_id,
            "PENDING",
            None,
            None,
            None,
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO loan_reqs (
                    loan_req_id, created_at, borrower_id, loan_amount, interest_rate,
                    loan_period, req_text, lender_id, req_status, approved_at, txn_id, loan_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            connection.commit()

        return {
            "loan_req_id": loan_req_id,
            "created_at": created_at,
            "borrower_id": borrower_id,
            "lender_id": lender_id,
            "loan_amount": float(loan_amount),
            "interest_rate": float(interest_rate),
            "loan_period_months": int(loan_period_months),
            "status": "PENDING",
        }

    def portfolio_summary(self) -> dict[str, Any]:
        loans = self.list_loans()
        requests = self.list_loan_requests()
        pending_requests = [request for request in requests if request["req_status"] == "PENDING"]
        approved_requests = [request for request in requests if request["req_status"] == "APPROVED"]
        active_principal = round(sum(float(loan["next_principal"]) for loan in loans), 2) if loans else 0.0
        next_due_date = min((loan["next_pay_date"] for loan in loans), default=None)
        return {
            "profiles": len(self.list_profiles()),
            "loan_requests_total": len(requests),
            "loan_requests_pending": len(pending_requests),
            "loan_requests_approved": len(approved_requests),
            "active_loans": len(loans),
            "active_principal_outstanding": active_principal,
            "next_due_date": next_due_date,
        }
