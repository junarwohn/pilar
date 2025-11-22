import time
import requests
from bs4 import BeautifulSoup
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

        ydl_opts = {
            'format': "bestvideo[ext=mp4][width=1920][height=1080][fps=60]",
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
        day_info = time.strftime('%Y-%m-%d', time.localtime())[2:]
        url = f'http://www.youngnak.net/portfolio-item/bible-stroll-{day_info}'
        #url = f'http://www.youngnak.net/portfolio-item/bible-stoll-{day_info}'
        # url = f'http://www.youngnak.net/portfolio-item/bible-storll-{day_info}'
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            iframes = soup.find_all('iframe')
            
            for iframe in iframes:
                src = iframe.get('src', '')
                if 'youtube' in src:
                    return src
                    
            raise ValueError('No YouTube iframe found on page')
            
        except requests.RequestException as e:
            logger.error(f'Request failed: {e}')
            return None
        except Exception as e:
            logger.error(f'Error: {e}')
            return None
