import sqlite3
import tempfile
import unittest
from pathlib import Path

from bharatfed.lending import LendingRepository
from bharatfed.service import BharatFedService


def _copy_database(source: str, destination: Path) -> str:
    destination.write_bytes(Path(source).read_bytes())
    return str(destination)


class LendingTests(unittest.TestCase):
    def test_portfolio_summary_uses_existing_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = _copy_database("bharat_fed.db", Path(temp_dir) / "bharat_fed.db")
            service = BharatFedService(
                sample_data_path="data_response_bharatfed99@finvu_1yr.txt",
                database_path=database_path,
            )

            summary = service.portfolio_summary()

            self.assertGreaterEqual(summary["profiles"], 2)
            self.assertGreaterEqual(summary["active_loans"], 1)
            self.assertGreaterEqual(summary["loan_requests_total"], 1)

    def test_create_loan_request_inserts_pending_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = _copy_database("bharat_fed.db", Path(temp_dir) / "bharat_fed.db")
            repository = LendingRepository(database_path)

            created = repository.create_loan_request(
                borrower_id="000001",
                lender_id="000000",
                loan_amount=25000,
                interest_rate=12.5,
                loan_period_months=6,
                text="Need working capital for my shop",
            )

            self.assertEqual(created["status"], "PENDING")

            with sqlite3.connect(database_path) as connection:
                row = connection.execute(
                    "SELECT borrower_id, lender_id, loan_amount, req_status FROM loan_reqs WHERE loan_req_id = ?",
                    (created["loan_req_id"],),
                ).fetchone()

            self.assertEqual(row, ("000001", "000000", 25000.0, "PENDING"))


if __name__ == "__main__":
    unittest.main()
