"""Core analytics for BharatFed built on the sample FIU payload format."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import json
from statistics import mean
from typing import Any, Iterable


DATE_FORMAT = "%Y-%m-%d"


@dataclass(frozen=True)
class Transaction:
    """Normalized transaction record."""

    date: date
    txn_type: str
    category: str
    amount: float
    current_balance: float
    narration: str

    @property
    def signed_amount(self) -> float:
        return self.amount if self.txn_type == "CREDIT" else -self.amount


def load_fiu_payload(path: str | Path) -> dict[str, Any]:
    """Load one of the BharatFed sample FIU response files."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_transactions(payload: dict[str, Any]) -> list[Transaction]:
    """Convert BharatFed FIU payloads into normalized transactions."""

    raw_transactions = payload["body"][0]["fiObjects"][0]["Transactions"]["Transaction"]
    transactions: list[Transaction] = []
    for item in raw_transactions:
        transactions.append(
            Transaction(
                date=datetime.strptime(item["valueDate"], DATE_FORMAT).date(),
                txn_type=item["type"],
                category=item["narration"],
                amount=float(item["amount"]),
                current_balance=float(item["currentBalance"]),
                narration=item["narration"],
            )
        )
    return sorted(transactions, key=lambda txn: txn.date)


def _aggregate_by_category(transactions: Iterable[Transaction], txn_type: str) -> list[dict[str, Any]]:
    totals: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for txn in transactions:
        if txn.txn_type != txn_type:
            continue
        totals[txn.category] += txn.amount
        counts[txn.category] += 1
    return [
        {
            "category": category,
            "total_amount": round(total_amount, 2),
            "transactions": counts[category],
        }
        for category, total_amount in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    ]


def _find_recurring_transactions(transactions: Iterable[Transaction]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Transaction]] = defaultdict(list)
    for txn in transactions:
        grouped[(txn.txn_type, txn.category)].append(txn)

    recurring: list[dict[str, Any]] = []
    for (txn_type, category), entries in grouped.items():
        if len(entries) < 2:
            continue
        ordered = sorted(entries, key=lambda txn: txn.date)
        intervals = [
            (ordered[index].date - ordered[index - 1].date).days
            for index in range(1, len(ordered))
        ]
        average_interval = mean(intervals)
        if average_interval > 45:
            continue
        recurring.append(
            {
                "type": txn_type,
                "category": category,
                "average_amount": round(mean(txn.amount for txn in ordered), 2),
                "average_interval_days": round(average_interval, 1),
                "last_seen": ordered[-1].date.isoformat(),
            }
        )
    return sorted(
        recurring,
        key=lambda item: (item["average_interval_days"], item["category"], item["type"]),
    )


def summarize_transactions(transactions: Iterable[Transaction]) -> dict[str, Any]:
    """Create an overview suitable for dashboards and API responses."""

    ordered = list(sorted(transactions, key=lambda txn: txn.date))
    if not ordered:
        return {
            "transaction_count": 0,
            "date_range": None,
            "credits_total": 0.0,
            "debits_total": 0.0,
            "net_cashflow": 0.0,
            "current_balance": 0.0,
            "average_credit": 0.0,
            "average_debit": 0.0,
            "top_income_categories": [],
            "top_expense_categories": [],
            "recurring_transactions": [],
        }

    credits = [txn.amount for txn in ordered if txn.txn_type == "CREDIT"]
    debits = [txn.amount for txn in ordered if txn.txn_type == "DEBIT"]
    credits_total = round(sum(credits), 2)
    debits_total = round(sum(debits), 2)

    return {
        "transaction_count": len(ordered),
        "date_range": {
            "start": ordered[0].date.isoformat(),
            "end": ordered[-1].date.isoformat(),
        },
        "credits_total": credits_total,
        "debits_total": debits_total,
        "net_cashflow": round(credits_total - debits_total, 2),
        "current_balance": round(ordered[-1].current_balance, 2),
        "average_credit": round(mean(credits), 2) if credits else 0.0,
        "average_debit": round(mean(debits), 2) if debits else 0.0,
        "top_income_categories": _aggregate_by_category(ordered, "CREDIT")[:5],
        "top_expense_categories": _aggregate_by_category(ordered, "DEBIT")[:5],
        "recurring_transactions": _find_recurring_transactions(ordered)[:10],
    }


def forecast_cashflow(transactions: Iterable[Transaction], days: int = 30) -> dict[str, Any]:
    """Produce a transparent rule-based cashflow forecast."""

    ordered = list(sorted(transactions, key=lambda txn: txn.date))
    if not ordered:
        return {"days": days, "forecast": [], "projected_end_balance": 0.0}

    recurring = _find_recurring_transactions(ordered)
    if not recurring:
        return {
            "days": days,
            "forecast": [],
            "projected_end_balance": round(ordered[-1].current_balance, 2),
        }

    current_balance = ordered[-1].current_balance
    last_date = ordered[-1].date
    forecast: list[dict[str, Any]] = []

    for recurring_item in recurring:
        interval = max(1, int(round(recurring_item["average_interval_days"])))
        avg_amount = float(recurring_item["average_amount"])
        next_date = datetime.strptime(recurring_item["last_seen"], DATE_FORMAT).date() + timedelta(
            days=interval
        )
        while next_date <= last_date + timedelta(days=days):
            signed_amount = avg_amount if recurring_item["type"] == "CREDIT" else -avg_amount
            current_balance += signed_amount
            forecast.append(
                {
                    "date": next_date.isoformat(),
                    "type": recurring_item["type"],
                    "category": recurring_item["category"],
                    "amount": round(avg_amount, 2),
                    "projected_balance": round(current_balance, 2),
                }
            )
            next_date += timedelta(days=interval)

    forecast.sort(key=lambda item: (item["date"], item["type"], item["category"]))
    return {
        "days": days,
        "forecast": forecast,
        "projected_end_balance": round(current_balance, 2),
    }
