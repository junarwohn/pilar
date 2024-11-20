import time
import requests
from bs4 import BeautifulSoup
import yt_dlp as yt
import os
import shutil
import subprocess

if __name__ == '__main__':
    url = get_youtube_url()
    if url:
        print(url)

class Downloader:
    def __init__(self):
        self.day_info = time.strftime('%Y-%m-%d', time.localtime())[2:]
        self.extract_dir = f"./out/{self.day_info}/extract"
        self.thumbs_dir = f"./out/{self.day_info}/thumbs"
        pass

    def download_video(self, url):
        now = time.localtime() 
        now_formatted = time.strftime('%Y%m%d-%H%M%S', now)
        ret = False

        ydl_opts = {
            'format': "bestvideo[ext=mp4][width=1920][height=1080][fps=60]",
            'merge_output_format': 'mp4',
            'outtmpl': f'./out/{self.day_info}/src.mp4'
        }
        
        with yt.YoutubeDL(ydl_opts) as ydl:
            ret = ydl.download([url])
        
        return ret, now_formatted
    
    @staticmethod
    def get_yn_url():
        day_info = time.strftime('%Y-%m-%d', time.localtime())[2:]
        url = f'http://www.youngnak.net/portfolio-item/bible-stroll-{day_info}'
        
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

    def extract_frames(self, fps=2):
        """Extract frames from video using ffmpeg"""
        
        # Clean up existing directories
        for directory in [self.extract_dir, self.thumbs_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

        # Extract frames using ffmpeg
        ffmpeg_cmd = [
            'ffmpeg', 
            '-i', f'./out/{self.day_info}/src.mp4',
            '-vf', f'fps={fps}',
            f'{self.extract_dir}/img%04d.jpg'
        ]
        subprocess.run(ffmpeg_cmd)
