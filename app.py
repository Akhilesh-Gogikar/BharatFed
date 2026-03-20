"""Run the BharatFed API locally."""

from __future__ import annotations

import argparse

from bharatfed import BharatFedService, create_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the BharatFed API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--sample-data",
        default="data_response_bharatfed99@finvu_1yr.txt",
        help="Path to a BharatFed FIU response file.",
    )
    parser.add_argument(
        "--database",
        default="bharat_fed.db",
        help="Path to the BharatFed SQLite database.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = BharatFedService(sample_data_path=args.sample_data, database_path=args.database)
    server = create_server(args.host, args.port, service)
    print(f"BharatFed API listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
