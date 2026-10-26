from datetime import datetime
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from pilar.web.server import create_app


class WebUploadLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "out" / datetime.now().strftime("%y-%m-%d")
        self.base.mkdir(parents=True)
        (self.base / "261026_01.jpg").write_bytes(b"local fixture")
        self.config = self.root / "config.ini"
        self.config.write_text("[global]\nkakao_url=https://example.invalid/posts\nheadless=true\nmanual_login=false\n")
        self.app = self.new_app()
        self.client = self.app.test_client()

    def new_app(self):
        app = create_app(str(self.base), start_scheduler=False, log_dir=self.root / "logs")
        app.config.update(TESTING=True, UPLOAD_CONFIG_FILE=str(self.config))
        return app

    def wait_finished(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = self.client.get("/status").get_json()["upload"]
            history = self.client.get("/jobs?stage=upload").get_json()
            if not state["running"] and history:
                return state, history[0]
            time.sleep(0.01)
        self.fail("Background upload did not complete")

    def test_skipped_upload_is_correlated_persisted_and_visible_after_restart(self):
        def fake_upload(**kwargs):
            kwargs["progress"]("duplicate_check", "Checking existing post", {})
            return {"status": "skipped", "image_count": 0, "title": "Daily date", "reason": "Upload skipped: already published"}
        with patch("pilar.web.server.upload_to_kakao", side_effect=fake_upload):
            response = self.client.post("/upload")
            state, record = self.wait_finished()
        self.assertEqual(response.status_code, 302)
        self.assertEqual(record["request_id"], response.headers["X-Request-ID"])
        self.assertEqual(record["status"], "skipped")
        self.assertEqual(record["trigger"], "web")
        self.assertEqual(state["run_id"], record["run_id"])
        self.assertEqual(state["upload_stage"], "duplicate_check")
        saved = json.loads((self.base.parent / "state.json").read_text())
        self.assertEqual(saved["upload"]["status"], "skipped")
        restarted = self.new_app().test_client()
        self.assertEqual(restarted.get("/status").get_json()["upload"]["status"], "skipped")
        self.assertIn(b"Upload skipped: already published", restarted.get("/results").data)

    def test_upload_failure_records_traceback_and_safe_ui_message(self):
        with self.assertLogs("pilar.web.server", level="ERROR") as captured:
            with patch("pilar.web.server.upload_to_kakao", side_effect=RuntimeError("Chrome failed <details>\nStacktrace: driver frames")):
                self.client.post("/upload")
                state, record = self.wait_finished()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(record["status"], "failed")
        self.assertIn("Chrome failed", record["error"])
        self.assertTrue(any(r.exc_info for r in captured.records))
        page = self.client.get("/results").data
        self.assertIn(b"&lt;details&gt;", page)
        self.assertNotIn(b"driver frames", page)

    def test_completed_upload_records_image_count(self):
        with patch("pilar.web.server.upload_to_kakao", return_value={"status": "completed", "image_count": 6, "title": "Daily date"}):
            self.client.post("/upload")
            state, record = self.wait_finished()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(record["image_count"], 6)
        self.assertIn("6 images", state["last_log"])

    def test_normal_event_stream_close_is_not_logged_as_an_error(self):
        with self.assertNoLogs("pilar.web.server", level="ERROR"):
            response = self.client.get("/events", buffered=False)
            next(response.response)
            response.close()

    def test_repeat_click_does_not_start_a_second_job(self):
        started, release = threading.Event(), threading.Event()
        def slow_upload(**kwargs):
            started.set()
            release.wait(timeout=3)
            return {"status": "completed", "image_count": 1}
        with patch("pilar.web.server.upload_to_kakao", side_effect=slow_upload) as upload:
            try:
                self.client.post("/upload")
                self.assertTrue(started.wait(timeout=2))
                self.client.post("/upload")
                self.assertEqual(upload.call_count, 1)
            finally:
                release.set()
            self.wait_finished()

    def test_interrupted_upload_is_not_restored_as_running(self):
        (self.base.parent / "state.json").write_text(json.dumps({"upload": {
            "running": True, "status": "running", "run_id": "old-run", "started_at": "2026-10-26T00:00:00+00:00"
        }}))
        client = self.new_app().test_client()
        state = client.get("/status").get_json()["upload"]
        self.assertFalse(state["running"])
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(client.get("/jobs").get_json()[0]["run_id"], "old-run")
        self.new_app()
        self.assertEqual(len(client.get("/jobs").get_json()), 1)


if __name__ == "__main__":
    unittest.main()
