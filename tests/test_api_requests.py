import copy
import json
import logging
import os
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import anthropic
import openai

from agentless.util.api_requests import (
    create_chatgpt_config,
    request_anthropic_engine,
    request_chatgpt_engine,
)


@contextmanager
def completion_server(provider, delay):
    requests = []
    if provider == "openai":
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    else:
        response = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    body = json.dumps(response).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            requests.append(
                (
                    self.path,
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            time.sleep(delay)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # The client is allowed to close a timed-out request.

    class Server(ThreadingHTTPServer):
        daemon_threads = False

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class RecordErrors(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class RequestTimeoutTests(unittest.TestCase):
    def check_request(self, provider, *, delay, timeout, expect_timeout=False):
        handler = RecordErrors()
        logger = logging.Logger("request-timeout-test")
        logger.addHandler(handler)
        if provider == "openai":
            config = create_chatgpt_config("hello", max_tokens=8, model="gpt-4o")
            timeout_type = openai.APITimeoutError
        else:
            config = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hello"}],
            }
            timeout_type = anthropic.APITimeoutError
        before = copy.deepcopy(config)
        with completion_server(provider, delay) as (url, requests):
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "local-test-key",
                    "ANTHROPIC_API_KEY": "local-test-key",
                    "ANTHROPIC_BASE_URL": url,
                    "NO_PROXY": "127.0.0.1",
                },
            ):
                if provider == "openai":
                    result = request_chatgpt_engine(
                        config,
                        logger,
                        base_url=url + "/v1",
                        max_retries=1,
                        timeout=timeout,
                    )
                else:
                    result = request_anthropic_engine(
                        config, logger, max_retries=1, timeout=timeout
                    )
            self.assertTrue(requests)
            for path, request in requests:
                self.assertEqual(
                    path,
                    "/v1/chat/completions" if provider == "openai" else "/v1/messages",
                )
                self.assertEqual(request["messages"], before["messages"])
            self.assertEqual(config, before)
            if expect_timeout:
                self.assertIsNone(result)
                self.assertTrue(
                    any(
                        isinstance(record.msg, timeout_type)
                        or (
                            record.exc_info
                            and isinstance(record.exc_info[1], timeout_type)
                        )
                        for record in handler.records
                    )
                )
            else:
                text = (
                    result.choices[0].message.content
                    if provider == "openai"
                    else result.content[0].text
                )
                self.assertEqual(text, "hello")
                self.assertEqual(len(requests), 1)

    def test_openai_short_timeout(self):
        self.check_request("openai", delay=0.5, timeout=0.05, expect_timeout=True)

    def test_anthropic_short_timeout(self):
        self.check_request("anthropic", delay=0.5, timeout=0.05, expect_timeout=True)

    def test_openai_fast_response(self):
        self.check_request("openai", delay=0, timeout=2)

    def test_anthropic_fast_response(self):
        self.check_request("anthropic", delay=0, timeout=2)

    def test_openai_relaxed_timeout(self):
        self.check_request("openai", delay=0.15, timeout=2)

    def test_anthropic_relaxed_timeout(self):
        self.check_request("anthropic", delay=0.15, timeout=2)


if __name__ == "__main__":
    unittest.main()
