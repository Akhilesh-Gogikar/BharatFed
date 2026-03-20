import unittest

from bharatfed.analytics import extract_transactions, forecast_cashflow, load_fiu_payload, summarize_transactions


class AnalyticsTests(unittest.TestCase):
    def test_summary_contains_expected_ranges(self) -> None:
        payload = load_fiu_payload("data_response_bharatfed99@finvu_1yr.txt")
        transactions = extract_transactions(payload)
        summary = summarize_transactions(transactions)

        self.assertEqual(summary["transaction_count"], len(transactions))
        self.assertEqual(summary["date_range"]["start"], "2019-07-20")
        self.assertEqual(summary["date_range"]["end"], "2020-07-06")
        self.assertGreater(summary["credits_total"], 0)
        self.assertGreater(summary["debits_total"], 0)
        self.assertGreaterEqual(summary["current_balance"], 0)
        self.assertTrue(summary["top_expense_categories"])

    def test_forecast_returns_sorted_projection(self) -> None:
        payload = load_fiu_payload("data_response_bharatfed99@finvu_1yr.txt")
        transactions = extract_transactions(payload)
        forecast = forecast_cashflow(transactions, days=14)

        dates = [entry["date"] for entry in forecast["forecast"]]
        self.assertEqual(dates, sorted(dates))
        self.assertGreaterEqual(forecast["projected_end_balance"], 0)


if __name__ == "__main__":
    unittest.main()
