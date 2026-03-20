import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from bharatfed.api import create_server
from bharatfed.service import BharatFedService


class APITests(unittest.TestCase):
    def test_http_endpoints_expose_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_copy = Path(temp_dir) / "bharat_fed.db"
            database_copy.write_bytes(Path("bharat_fed.db").read_bytes())
            service = BharatFedService(
                sample_data_path="data_response_bharatfed99@finvu_1yr.txt",
                database_path=str(database_copy),
            )
            server = create_server("127.0.0.1", 0, service)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]

            try:
                with urlopen(f"http://127.0.0.1:{port}/health") as response:
                    health = json.loads(response.read().decode("utf-8"))
                self.assertEqual(health["status"], "ok")

                payload = {
                    "borrower_id": "000001",
                    "lender_id": "000000",
                    "loan_amount": 15000,
                    "interest_rate": 9.5,
                    "loan_period_months": 4,
                    "text": "Inventory restocking",
                }
                request = Request(
                    f"http://127.0.0.1:{port}/loans/requests",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request) as response:
                    created = json.loads(response.read().decode("utf-8"))
                self.assertEqual(created["status"], "PENDING")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
