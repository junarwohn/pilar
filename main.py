from pilar.utils import *
import argparse
import time
import os
import configparser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    args = parser.parse_args()
    
    # Set up directories
    day_info = time.strftime('%Y-%m-%d', time.localtime())[2:]
    base_dir = f"./out/{day_info}"
    video_path = f"{base_dir}/src.mp4"
    extract_dir = f"{base_dir}/extract"
    thumbs_dir = f"{base_dir}/thumbs"
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
   
    # Download video
    downloader = Downloader(output_path=video_path)
    downloader.download_video()

    # # Process images
    processor = ImageProcessor(
        video_path=video_path,
        extract_dir=extract_dir,
        thumbs_dir=thumbs_dir,
        no_gui=args.no_gui
    )
    processor.select_thumbs()
    processor.get_bounds()
    processor.process_files()
    
    
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
