import time
import requests
from bs4 import BeautifulSoup
import yt_dlp as yt
import os
import shutil
import subprocess

class Downloader:
    def __init__(self, output_path):
        self.output_path = output_path
        self.url = None

    def download_video(self, url=None):
        if url is None and self.url is None:
            self.url = self.get_yn_url()
            url = self.url
            
        now = time.localtime() 
        now_formatted = time.strftime('%Y%m%d-%H%M%S', now)
        ret = False

        ydl_opts = {
            'format': "bestvideo[ext=mp4][width=1920][height=1080][fps=60]",
            'merge_output_format': 'mp4',
            'outtmpl': self.output_path
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
            print(f'Request failed: {e}')
            return None
        except Exception as e:
            print(f'Error: {e}')
            return None
