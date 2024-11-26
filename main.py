from pilar.utils import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    args = parser.parse_args()

    downloader = Downloader()
    url = downloader.get_yn_url()
    downloader.download_video(url)
    downloader.extract_frames()

    processor = ImageProcessor(no_gui=args.no_gui)
    processor.select_thumbs()
    processor.get_bounds()
    processor.process_files()
