from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from pilar_logging import JobHistory, JsonFormatter, background_thread, current_job, request_id


class LoggingTests(unittest.TestCase):
    def test_json_exception_preserves_traceback_and_correlation_on_one_line(self):
        token = request_id.set("request-1")
        try:
            with tempfile.TemporaryDirectory() as directory:
                history = JobHistory(directory)
                with self.assertLogs("pilar_logging", level="INFO") as captured:
                    with self.assertRaisesRegex(ValueError, "test failure"):
                        with history.run("upload"):
                            raise ValueError("test failure")
                error = next(r for r in captured.records if r.exc_info)
                # Formatting happens inside real handlers while the contexts are active;
                # the exception itself remains inspectable after the job exits.
                encoded = JsonFormatter().format(error)
                decoded = json.loads(encoded)
                self.assertIn("ValueError: test failure", decoded["exception"])
                self.assertNotIn("\n", encoded)
                result = history.recent()[0]
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["request_id"], "request-1")
                self.assertGreaterEqual(result["duration_seconds"], 0)
                self.assertIsNone(current_job.get())
        finally:
            request_id.reset(token)

    def test_background_job_inherits_request_and_separates_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            history = JobHistory(directory)
            token = request_id.set("web-request")
            try:
                @history.track("upload")
                def task():
                    self.assertEqual(request_id.get(), "web-request")
                    return {"status": "skipped", "reason": "already exists", "image_count": 0}
                worker = background_thread(target=task)
            finally:
                request_id.reset(token)
            worker.start()
            worker.join(timeout=3)
            self.assertFalse(worker.is_alive())
            record = JobHistory(directory).recent()[0]
            self.assertEqual(record["request_id"], "web-request")
            self.assertEqual(record["trigger"], "web")
            self.assertEqual(record["status"], "skipped")

    def test_retention_removes_only_expired_job_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            today = datetime.now().date()
            expired = root / f"jobs-{today - timedelta(days=3)}.jsonl"
            kept = root / f"jobs-{today - timedelta(days=1)}.jsonl"
            other = root / "keep.txt"
            for path in (expired, kept, other):
                path.write_text("")
            history = JobHistory(root, retention_days=2)
            history.append({"event": "job_finished", "stage": "upload", "status": "completed"})
            self.assertFalse(expired.exists())
            self.assertTrue(kept.exists())
            self.assertTrue(other.exists())
            self.assertEqual(len(history.recent(stage="upload")), 1)

    def test_common_configuration_emits_info_once_and_suppresses_ocr_debug(self):
        program = """
import logging
from pilar_logging import configure_logging
configure_logging()
configure_logging()
logging.getLogger('pilar.test').info('visible info')
logging.getLogger('ppocr').debug('hidden debug')
"""
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True)
        lines = result.stderr.splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["message"], "visible info")


if __name__ == "__main__":
    unittest.main()
