from pilar.utils import *

if __name__ == "__main__":

    downloader = Downloader()
    url = downloader.get_yn_url()
    downloader.download_video(url)
    downloader.extract_frames()

    processor = ImageProcessor()
    processor.select_thumbs()
    processor.get_bounds()
    processor.process_files()