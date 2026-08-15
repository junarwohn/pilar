import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import subprocess
import logging
import importlib


logger = logging.getLogger(__name__)


def _is_true_env(v):
    return bool(v and v.strip().lower() in {"1", "true", "on", "yes"})

class Downloader:
    def __init__(self, output_path):
        self.output_path = output_path
        self.url = None

    def _upgrade_yt_dlp(self):
        if not _is_true_env(os.getenv("PILAR_YTDLP_AUTO_UPGRADE")):
            logger.info("Skipping yt-dlp pre-upgrade (set PILAR_YTDLP_AUTO_UPGRADE=1 to enable)")
            return
        cmd = os.getenv("PILAR_YTDLP_UPGRADE_CMD", "uv pip install -U yt-dlp")
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

    def download_video(self, url=None, progress=None):
        if url is None:
            if self.url is None:
                self.url = self.get_yn_url()
            url = self.url

        if url is None:
            logger.error("No download URL retrieved; skipping download")
            raise ValueError("No download URL provided or found")

        now = time.localtime()
        now_formatted = time.strftime('%Y%m%d-%H%M%S', now)
        ret = False

        # Refresh package before each download and reload module in this process.
        self._upgrade_yt_dlp()
        yt = importlib.import_module("yt_dlp")
        yt = importlib.reload(yt)

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

        # Preserve 1080p subtitle detail, but avoid decoding 60 fps on low-power
        # devices when the processing pipeline samples only a few frames/second.
        ydl_opts = {
            'format': "bv*[vcodec*=avc1][height<=1080][fps<=30]+ba[ext=m4a]/best[ext=mp4][height<=1080]",
            'merge_output_format': 'mp4',
            'outtmpl': self.output_path,
            # Ensure we replace yesterday's file when a new day starts
            'overwrites': True,
            'progress_hooks': [_hook],
        }

        with yt.YoutubeDL(ydl_opts) as ydl:
            ret = ydl.download([url])

        return ret, now_formatted
    

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
                return src

        logger.error('No valid YouTube iframe found on detail page')
        return None
