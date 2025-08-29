from pilar.utils import *
import argparse
import time
import os
import configparser
import logging
import sys
from pathlib import Path


def run_web(no_gui: bool, host: str = "0.0.0.0", port: int = 8000):
    # Lazy import to avoid dependency unless used
    from pilar.web.server import create_app

    day_info = time.strftime('%Y-%m-%d', time.localtime())[2:]
    # Anchor base_dir to the repository root (directory containing this file)
    repo_root = Path(__file__).resolve().parent
    base_dir = str((repo_root / "out" / day_info).resolve())

    app = create_app(base_dir=base_dir, no_gui=no_gui)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    parser.add_argument('--web', action='store_true', help='Start web UI (Flask)')
    parser.add_argument('--host', default='0.0.0.0', help='Web host (default 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Web port (default 8000)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.web:
        run_web(no_gui=args.no_gui, host=args.host, port=args.port)
        sys.exit(0)

    # Set up directories
    day_info = time.strftime('%Y-%m-%d', time.localtime())[2:]
    repo_root = Path(__file__).resolve().parent
    base_dir_path = (repo_root / "out" / day_info).resolve()
    base_dir = str(base_dir_path)
    video_path = str(base_dir_path / "src.mp4")
    extract_dir = str(base_dir_path / "extract")
    thumbs_dir = str(base_dir_path / "thumbs")
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
  
    # Download video
    downloader = Downloader(output_path=video_path)
    try:
        downloader.download_video()
    except ValueError as e:
        logging.error("Video download skipped: %s", e)
        sys.exit(1)

    AUTO_DETECTION = False

    # Process images
    processor = ImageProcessor(
        video_path=video_path,
        extract_dir=extract_dir,
        thumbs_dir=thumbs_dir,
        no_gui=args.no_gui,
        zoom=112,
        auto_detection_range=1/2
    )
    processor.select_thumbs()

    if not AUTO_DETECTION:
        processor.get_bounds()
        processor.process_files()
    else:
        processor.auto_process_files()
    
    # Load config and upload processed images using Selenium
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    uploader = ImageUploader(
        image_dir=base_dir,
        url=config['Credentials']['url'],
        id=config['Credentials']['id'], 
        password=config['Credentials']['password'],
        no_gui=args.no_gui,
        driver_path=config['Credentials']['driver_path'],
        user_data_dir=config['Credentials']['user_data_dir'],
        profile_directory=config['Credentials']['profile_directory']
    )
    uploader.upload_images()
