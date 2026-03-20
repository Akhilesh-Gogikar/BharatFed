"""Dependency-light HTTP API for BharatFed."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any
from urllib.parse import parse_qs, urlparse

from .service import BharatFedService


class BharatFedAPIHandler(BaseHTTPRequestHandler):
    """JSON API exposing the modernized BharatFed service."""

    service: BharatFedService

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        self._dispatch("POST")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if method == "GET" and parsed.path == "/health":
                self._send_json(HTTPStatus.OK, self.service.health())
                return
            if method == "GET" and parsed.path == "/analytics/overview":
                self._send_json(HTTPStatus.OK, self.service.analytics_overview())
                return
            if method == "GET" and parsed.path == "/analytics/forecast":
                days = int(query.get("days", ["30"])[0])
                self._send_json(HTTPStatus.OK, self.service.analytics_forecast(days=days))
                return
            if method == "GET" and parsed.path == "/profiles":
                self._send_json(HTTPStatus.OK, {"profiles": self.service.list_profiles()})
                return
            if method == "GET" and parsed.path.startswith("/profiles/"):
                user_id = parsed.path.rsplit("/", 1)[-1]
                profile = self.service.get_profile(user_id)
                if not profile:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "profile not found"})
                    return
                self._send_json(HTTPStatus.OK, profile)
                return
            if method == "GET" and parsed.path == "/loans/requests":
                status = query.get("status", [None])[0]
                self._send_json(
                    HTTPStatus.OK,
                    {"loan_requests": self.service.list_loan_requests(status=status)},
                )
                return
            if method == "GET" and parsed.path == "/loans":
                self._send_json(HTTPStatus.OK, {"loans": self.service.list_loans()})
                return
            if method == "GET" and parsed.path == "/portfolio/summary":
                self._send_json(HTTPStatus.OK, self.service.portfolio_summary())
                return
            if method == "POST" and parsed.path == "/loans/requests":
                payload = self._read_json_body()
                created = self.service.create_loan_request(payload)
                self._send_json(HTTPStatus.CREATED, created)
                return
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "route not found"})

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(raw_body.decode("utf-8"))

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_server(host: str, port: int, service: BharatFedService) -> ThreadingHTTPServer:
    """Create a configured ThreadingHTTPServer instance."""

    handler_class = type("ConfiguredBharatFedHandler", (BharatFedAPIHandler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler_class)
