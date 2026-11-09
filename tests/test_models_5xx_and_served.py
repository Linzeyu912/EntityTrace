from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from entitytrace.cli import main as cli_main
from entitytrace.models import BudgetError, Models, RequestBudget
from entitytrace.util import dumps
from tests.test_bounded_identity_run import _make_fake_http_opener


class Models5xxAndServedModelTests(unittest.TestCase):
    def test_http_5xx_sequence_triggers_network_outage_while_4xx_does_not(self):
        def make_http_err(code: int):
            return urllib.error.HTTPError(
                "https://api.example.com/v1/chat/completions", code, "err", {}, io.BytesIO(b"{}")
            )

        net_4xx = Mock(side_effect=[make_http_err(400), make_http_err(429), make_http_err(404)])
        b_4xx = RequestBudget(10, 60, net_4xx, max_network_failures=2)
        for _ in range(3):
            with self.assertRaises(urllib.error.HTTPError):
                b_4xx(None, 30)
        self.assertEqual(b_4xx.consecutive_network_failures, 0)
        self.assertIsNone(b_4xx.stop_reason)

        net_5xx = Mock(side_effect=[make_http_err(502), make_http_err(503), make_http_err(504)])
        b_5xx = RequestBudget(10, 60, net_5xx, max_network_failures=3)
        for _ in range(3):
            with self.assertRaises(urllib.error.HTTPError):
                b_5xx(None, 30)
        self.assertEqual(b_5xx.consecutive_network_failures, 3)
        self.assertEqual(b_5xx.stop_reason, "network_outage")
        self.assertEqual(b_5xx.summary()["network_errors"], {"HTTP 502": 1, "HTTP 503": 1, "HTTP 504": 1})
        with self.assertRaises(BudgetError) as ctx:
            b_5xx(None, 30)
        self.assertEqual(ctx.exception.category, "network")
        self.assertEqual(ctx.exception.stop_reason, "network_outage")

    def test_pipeline_http_502_outage_and_served_model_logged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            chapters_text = "\n".join(f"第{i}章 标题\n林舟在第{i}章出场。" for i in range(1, 7)) + "\n"
            text = root / "novel.txt"
            text.write_text(chapters_text, encoding="utf-8")
            cfg = root / ".env"
            cfg.write_text(
                "TYPESAFE_API_KEY=test\nDEEPSEEK_API_KEY=test\nGENERATOR_PROVIDER=deepseek\n",
                encoding="utf-8",
            )
            db_path = root / "db.sqlite3"
            out_dir = root / "out"

            base_opener, _ = _make_fake_http_opener()

            def discovery_502_opener(request, timeout=None):
                if request.full_url.endswith("/chat/completions"):
                    req_payload = json.loads(request.data.decode("utf-8"))
                    req_state = json.loads(req_payload["messages"][1]["content"])
                    if "candidates" not in req_state:
                        raise urllib.error.HTTPError(
                            request.full_url, 502, "Bad Gateway", {}, io.BytesIO(b'{"error":"bad gateway"}')
                        )
                    resp_bytes = base_opener(request, timeout=timeout).read()
                    resp_obj = json.loads(resp_bytes.decode("utf-8"))
                    resp_obj["model"] = "served-actual-model-v1"
                    return io.BytesIO(dumps(resp_obj).encode("utf-8"))
                return base_opener(request, timeout=timeout)

            class Outage502Models(Models):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs, attempts=2, opener=discovery_502_opener)

            with patch("entitytrace.pipeline.Models", Outage502Models), patch("entitytrace.models.time.sleep"):
                code = cli_main([
                    "run", str(text), "--db", str(db_path), "--output", str(out_dir),
                    "--config", str(cfg), "--discovery-concurrency", "1", "--json",
                ])
            self.assertEqual(code, 2)
            subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
            manifest = json.loads((subdirs[0] / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["state"]["last_error"]["category"], "network")
            self.assertTrue(manifest["state"]["last_error"]["retryable"])
            self.assertEqual(manifest["state"]["last_error"]["stop_reason"], "network_outage")

            con = sqlite3.connect(db_path)
            try:
                rows = con.execute("SELECT stage, status, detail FROM calls ORDER BY rowid").fetchall()
            finally:
                con.close()
            self.assertTrue(any(
                r[1] == "success" and json.loads(r[2]).get("served_model") == "served-actual-model-v1"
                for r in rows
            ))
            self.assertTrue(any(
                r[1] == "failed" and "served_model" in json.loads(r[2]) and json.loads(r[2])["served_model"] is None
                for r in rows
            ))


if __name__ == "__main__":
    unittest.main()
