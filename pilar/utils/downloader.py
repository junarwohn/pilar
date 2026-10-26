import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
import os
import shutil
import subprocess
import logging
import sys
import importlib
from pathlib import Path


logger = logging.getLogger(__name__)


def _is_false_env(v):
    if not v:
        return False
    return v.strip().lower() in {"0", "false", "off", "no"}

class Downloader:
    def __init__(self, output_path):
        self.output_path = output_path
        self.url = None

    def _upgrade_yt_dlp(self):
        if _is_false_env(os.getenv("PILAR_YTDLP_AUTO_UPGRADE")):
            logger.info("Skipping yt-dlp pre-upgrade (PILAR_YTDLP_AUTO_UPGRADE disabled)")
            return
        cmd = os.getenv("PILAR_YTDLP_UPGRADE_CMD", 'uv pip install -U --pre "yt-dlp[default]"')
        logger.info("Running yt-dlp pre-upgrade: %s", cmd)
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "").strip()
                logger.warning(
                    "yt-dlp pre-upgrade failed (code=%s), continue with current version: %s",
                    proc.returncode,
                    msg,
                )
                return
            out = (proc.stdout or "").strip()
            if out:
                logger.info("yt-dlp pre-upgrade output: %s", out)
        except Exception as e:
            logger.warning("yt-dlp pre-upgrade error, continue with current version: %s", e)

    def _cleanup_partial_outputs(self):
        output = Path(self.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        patterns = [
            f"{output.name}.part",
            f"{output.stem}*.part",
            f"{output.stem}.f*.mp4",
            f"{output.stem}.f*.m4a",
            f"{output.stem}.f*.webm",
        ]
        candidates = set()
        for pattern in patterns:
            candidates.update(output.parent.glob(pattern))

        for candidate in candidates:
            if candidate == output or not candidate.is_file():
                continue
            try:
                candidate.unlink()
                logger.info("Removed stale partial download: %s", candidate)
            except OSError as e:
                logger.warning("Could not remove stale partial download %s: %s", candidate, e)

    def download_video(self, url=None, progress=None):
        if url is None:
            if self.url is None:
                self.url = self.get_yn_url()
            url = self.url

        if url is None:
            logger.error("No download URL retrieved; skipping download")
            raise ValueError("No download URL provided or found")
        url = self.normalize_youtube_url(url)

        now = time.localtime()
        now_formatted = time.strftime('%Y%m%d-%H%M%S', now)
        ret = False

        # Refresh package before each download and reload module in this process.
        self._upgrade_yt_dlp()
        yt = importlib.import_module("yt_dlp")
        yt = importlib.reload(yt)
        self._cleanup_partial_outputs()

        deno_path = shutil.which("deno")
        js_runtimes = {"deno": {"path": deno_path}} if deno_path else {"deno": {}}

        def _hook(d):
            if progress is None:
                return
            # Map yt-dlp dict to a compact payload
            status = d.get('status')
            payload = {
                'status': status,
                'downloaded': d.get('downloaded_bytes') or 0,
                'total': d.get('total_bytes') or d.get('total_bytes_estimate') or 0,
                'speed': d.get('speed') or 0,
                'eta': d.get('eta') or 0,
                'filename': d.get('filename') or self.output_path,
            }
            try:
                progress(payload)
            except Exception:
                pass

        # Prefer H.264 (AVC) video to avoid AV1 decode overhead/compat issues.
        # Falls back to best MP4 if AVC selection isn't available.
        ydl_opts = {
            'format': "bv*[vcodec*=avc1][height<=1080][fps<=60]+ba[ext=m4a]/best[ext=mp4]",
            'merge_output_format': 'mp4',
            'outtmpl': self.output_path,
            # Ensure we replace yesterday's file when a new day starts
            'overwrites': True,
            # YouTube signed media URLs can expire; do not resume stale .part files.
            'continuedl': False,
            'retries': 5,
            'fragment_retries': 5,
            'extractor_retries': 3,
            'file_access_retries': 3,
            'js_runtimes': js_runtimes,
            'progress_hooks': [_hook],
            'noprogress': not sys.stderr.isatty(),
        }

        with yt.YoutubeDL(ydl_opts) as ydl:
            ret = ydl.download([url])

        output = Path(self.output_path)
        if ret != 0:
            raise RuntimeError(f"yt-dlp failed with exit code {ret}")
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded file was not created: {output}")

        return ret, now_formatted
    

    @staticmethod
    def normalize_youtube_url(url):
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path_parts = [part for part in parsed.path.split('/') if part]
        youtube_hosts = {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtube-nocookie.com",
            "www.youtube-nocookie.com",
        }

        if host in youtube_hosts and len(path_parts) >= 2 and path_parts[0] == "embed":
            video_id = path_parts[1]
            query = parse_qs(parsed.query)
            params = {"v": video_id}
            for key in ("list", "start", "t"):
                values = query.get(key)
                if values and values[0]:
                    params[key] = values[0]
            return f"https://www.youtube.com/watch?{urlencode(params)}"

        return url

    @staticmethod
    def get_yn_url():
        list_url = "https://www.youngnak.net/rev_kws_bible_stroll/"
        print(f"Fetch list page {list_url}...")
        try:
            response = requests.get(list_url)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Request failed for {list_url}: {e}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        container = soup.find(id="after_section_1")
        link = None
        if container is not None:
            link_tag = container.select_one("section div div ul li:nth-of-type(1) h2 a")
            if link_tag is not None:
                link = link_tag.get("href")

        if not link:
            logger.error("No link found at expected XPath on list page")
            return None

        detail_url = urljoin(list_url, link)
        print(f"Try {detail_url}...")
        try:
            response = requests.get(detail_url)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Request failed for {detail_url}: {e}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        iframes = soup.find_all('iframe')

        for iframe in iframes:
            src = iframe.get('src', '')
            if 'youtube' in src:
                return Downloader.normalize_youtube_url(src)

        logger.error('No valid YouTube iframe found on detail page')
        return None
