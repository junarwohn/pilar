import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import yt_dlp as yt
import os
import shutil
import subprocess
import logging


logger = logging.getLogger(__name__)

class Downloader:
    def __init__(self, output_path):
        self.output_path = output_path
        self.url = None

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
