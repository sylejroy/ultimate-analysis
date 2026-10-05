"""The web server behind the phone labelling page.

It serves one page and a handful of JSON calls, with Python's own HTTP server. Every
request must carry the access key: the address is reachable by every device that can reach
this PC, and only the holder of the key may see frames or change the dataset.
"""

import hmac
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from ..utils.logger import get_logger
from .label_session import LabelSession

logger = get_logger("PHONE_LABELLING")

PAGE = Path(__file__).with_name("label_page.html")
MAX_REQUEST_BYTES = 10_000


def create_server(session: LabelSession, key: str, host: str, port: int) -> ThreadingHTTPServer:
    """A server for one labelling session; call serve_forever() on it."""
    # The session reads videos and runs a model: one request at a time
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002  (name given by the base class)
            logger.debug(format % args)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def _allowed(self, query: dict) -> bool:
            given = (query.get("key") or [""])[0]
            if hmac.compare_digest(given, key):
                return True
            self._send(HTTPStatus.FORBIDDEN, b"Wrong or missing key", "text/plain")
            return False

        def do_GET(self):  # noqa: N802  (name given by the base class)
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if not self._allowed(query):
                return
            if url.path == "/":
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/next":
                with lock:
                    self._json(session.next_task() or {})
            elif url.path == "/api/picture":
                try:
                    numbers = [float(query[name][0]) for name in ("x", "y", "w", "h", "out")]
                    task_id = query["task"][0]
                except (KeyError, ValueError):
                    self._send(HTTPStatus.BAD_REQUEST, b"Bad picture request", "text/plain")
                    return
                with lock:
                    picture = session.picture(task_id, *numbers[:4], int(min(numbers[4], 1920)))
                if picture is None:
                    self._send(HTTPStatus.NOT_FOUND, b"Unknown task", "text/plain")
                else:
                    self._send(HTTPStatus.OK, picture, "image/jpeg")
            else:
                self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")

        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            if not self._allowed(parse_qs(url.query)):
                return
            try:
                length = min(int(self.headers.get("Content-Length", 0)), MAX_REQUEST_BYTES)
                body = json.loads(self.rfile.read(length) or b"{}")
                task_id = str(body.get("task", ""))
                box: Optional[list] = body.get("box")
                if box is not None:
                    box = [float(value) for value in box]
                    if len(box) != 4:
                        raise ValueError("a box has four numbers")
            except (ValueError, TypeError, AttributeError):
                self._send(HTTPStatus.BAD_REQUEST, b"Bad request", "text/plain")
                return

            with lock:
                if url.path == "/api/save":
                    self._json({"ok": session.save(task_id, box)})
                elif url.path == "/api/skip":
                    self._json({"ok": session.skip(task_id)})
                elif url.path == "/api/undo":
                    self._json({"undone": session.undo()})
                else:
                    self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")

    return ThreadingHTTPServer((host, port), Handler)
