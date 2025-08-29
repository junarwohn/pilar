from tesserocr import PyTessBaseAPI, PSM, OEM
from PIL import Image
import cv2
import os
import re
import numpy as np
from rapidfuzz.distance import Levenshtein
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
from tqdm import tqdm
import time
import os
import cv2
import shutil
from pathlib import Path
import numpy as np
from datetime import datetime
import subprocess
import shutil
import platform
import hashlib

# 운영체제에 따른 화살표 키 코드 설정
if platform.system() == "Windows":
    RIGHT_ARROW = 2555904  # 오른쪽 화살표 (Windows)
    LEFT_ARROW = 2424832   # 왼쪽 화살표 (Windows)
elif platform.system() == "Linux":
    RIGHT_ARROW = 65363    # 오른쪽 화살표 (Linux, X11 환경)
    LEFT_ARROW = 65361     # 왼쪽 화살표 (Linux, X11 환경)
else:
    # 기본값(여기서는 Windows 값을 사용)
    RIGHT_ARROW = 2555904
    LEFT_ARROW = 2424832

TESSDATA_PATH=r"./res/"

class ImageProcessor:
    def __init__(self, video_path, extract_dir, thumbs_dir, no_gui=False, zoom=100, auto_detection_range=1/2, fresh=True, fps: int = 2, prompt_handler=None, progress_callback=None):
        self.video_path = video_path
        self.extract_dir = extract_dir
        self.thumbs_dir = thumbs_dir
        self.IS_DEBUG = False
        self.NO_GUI = no_gui
        self.zoom = zoom
        self.auto_detection_range = auto_detection_range
        self._init_fps = fps
        # Optional callable that resolves ambiguous comparisons.
        # Signature: prompt_handler(context: dict) -> bool (True means SAME/skip, False means DIFF/add)
        self.prompt_handler = prompt_handler
        # Optional progress reporter: progress_callback(cur: int, total: int, info: dict)
        self.progress_callback = progress_callback
        # Cache for ambiguous decisions to avoid repeated prompts on same word
        self._decision_cache = {}  # word -> (is_same: bool, idx: int)
        self._hash_decision_cache = {}  # content hash -> is_same
        self._idx = 0

        try:
            self.engine = PyTessBaseAPI(path=TESSDATA_PATH, lang='kor', oem=OEM.LSTM_ONLY, psm=PSM.SINGLE_COLUMN)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tesseract engine: {e}")

        # Clean and create directories (only when fresh)
        if fresh:
            for directory in [self.extract_dir, self.thumbs_dir]:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                os.makedirs(directory)
        else:
            # Ensure directories exist but do not delete contents
            os.makedirs(self.extract_dir, exist_ok=True)
            os.makedirs(self.thumbs_dir, exist_ok=True)

        # Extract frames only when fresh; otherwise reuse existing
        if fresh:
            self.extract_frames(fps=self._init_fps)
        self.load_models()
        self.init_parameters()
        self.load_file_lists()

    def extract_frames(self, fps=2):
        """Extract frames from video using ffmpeg"""
        ffmpeg_cmd = [
            'ffmpeg', 
            '-i', self.video_path,
            '-vf', f'fps={fps}',
            f'{self.extract_dir}/img%04d.jpg'
        ]
        subprocess.run(ffmpeg_cmd)

    def load_models(self):
        mod_path = 'svm_models.sav'
        self.svm_clfs = pickle.load(open(mod_path, 'rb'))

    def init_parameters(self):
        self.height_upper = 925
        self.height_lower = 1020
        self.kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.add_cnt = 0
        self.page_cnt = 0
        self.thumb_cnt = 0
        self.pre_word = [""] 

    def load_file_lists(self):
        self.file_list = [i for i in os.listdir(self.extract_dir) if not i.startswith('.') and not 'Zone.Identifier' in i]
        self.file_list.sort()
        self.thumb_list = [i for i in os.listdir(self.thumbs_dir) if not i.startswith('.') and not 'Zone.Identifier' in i]
        self.thumb_list.sort()

    @staticmethod
    def img_similarity(img1, img2):
        """Return similarity ratio between two binary images.

        The original implementation compared flattened arrays with
        ``np.isclose`` using an absolute tolerance of ``50``.  This
        rewrite uses OpenCV operations for better performance:

        1. ``cv2.absdiff`` computes the absolute per-pixel difference.
        2. ``cv2.countNonZero`` counts pixels whose difference is within
           the tolerance.

        The result is the fraction of pixels considered similar.
        """

        diff = cv2.absdiff(img1, img2)
        mask = (diff <= 50).astype(np.uint8)
        return cv2.countNonZero(mask) / diff.size

    def get_subtilte_bounds(self):
        pass

    def get_bounds(self):
        if self.NO_GUI:
            return
            
        sample_img = cv2.imread(f"{self.extract_dir}/" + self.file_list[50])
        
        # Show current bounds
        sliced = sample_img[self.height_upper:self.height_lower, :]
        cv2.namedWindow("current_bounds", cv2.WINDOW_NORMAL)
        cv2.imshow("current_bounds", sliced)
        cv2.resizeWindow("current_bounds", 600, 600)
        print("Are the current bounds acceptable? [enter/other]")
        ret = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if ret != 13:  # Not enter key
            # Set upper boundary
            bound_upper_complete = False
            while not bound_upper_complete:
                print(f"Current upper height: {self.height_upper}")
                print("Enter new upper height:")
                input_val = input()
                if input_val:
                    self.height_upper = int(input_val)
                sliced = sample_img[self.height_upper:, :]
                cv2.namedWindow("height_upper", cv2.WINDOW_NORMAL)
                cv2.imshow("height_upper", sliced)
                cv2.resizeWindow("height_upper", 600, 600)
                print("Is this okay? [enter/other]")
                ret = cv2.waitKey(0)
                if ret == 13:
                    bound_upper_complete = True
                cv2.destroyAllWindows()

            # Set lower boundary  
            bound_lower_complete = False
            while not bound_lower_complete:
                print(f"Current lower height: {self.height_lower}")
                print("Enter new lower height:")
                input_val = input()
                if input_val:
                    self.height_lower = int(input_val)
                sliced = sample_img[self.height_upper:self.height_lower, :]
                cv2.namedWindow("height_lower", cv2.WINDOW_NORMAL)
                cv2.imshow("height_lower", sliced)
                cv2.resizeWindow("height_lower", 600, 600)
                print("Is this okay? [enter/other]")
                ret = cv2.waitKey(0)
                if ret == 13:
                    bound_lower_complete = True
                cv2.destroyAllWindows()

    def process_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        bilateral_filter = cv2.bilateralFilter(inverted, 9, 16, 16)
        _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
        return bilateral_filter, binary

    def extract_text(self, img):
        start = time.time()
        try:
            self.engine.SetImage(Image.fromarray(img))
            text = self.engine.GetUTF8Text()
            conf = self.engine.MeanTextConf()
        except Exception as e:
            print(f"OCR failed: {e}")
            return ""
        if self.IS_DEBUG:
            elapsed = time.time() - start
            print(f"OCR time: {elapsed:.3f}s, confidence: {conf}")
        cleaned = re.sub(r"[^가-힣\n]", "", text)
        word_list = [w for w in cleaned.split('\n') if w]
        if not word_list or conf < 0:
            return ""
        return max(word_list, key=len)

    def process_files(self):
        result_img = self.get_new_result_img()
        original_img = cv2.imread(f"{self.extract_dir}/" + self.file_list[0])
        pre_img = original_img[self.height_upper:self.height_lower, :]
        # pre_img = cv2.resize(pre_img, dsize=(int(original_img.shape[1] * pre_img.shape[0] /  original_img.shape[0]), original_img.shape[0]))
        pre_img = pre_img[:, int(pre_img.shape[1] * (self.zoom - 100) / 100 / 2) : int(pre_img.shape[1] * (1 - (self.zoom - 100) / 100 / 2))]
        pre_img = cv2.resize(pre_img, dsize=(original_img.shape[1], int(original_img.shape[1] * pre_img.shape[0] /  original_img.shape[0])))
        pre_processed, pre_bin = self.process_image(pre_img)

        total = len(self.file_list)
        idx = 0
        iterable = self.file_list
        if not self.IS_DEBUG:
            iterable = tqdm(iterable)

        for file_name in iterable:
            idx += 1
            self._idx = idx
            if callable(self.progress_callback):
                try:
                    self.progress_callback(idx, total, {"pages": self.page_cnt, "added": self.add_cnt})
                except Exception:
                    pass
            original_img = cv2.imread(f"{self.extract_dir}/" + file_name)
            cur_img = original_img[self.height_upper:self.height_lower, :]
            cur_img = cur_img[:, int(cur_img.shape[1] * (self.zoom - 100) / 100 / 2) : int(cur_img.shape[1] * (1 - (self.zoom - 100) / 100 / 2))]
            # Resize image to same as original_img horizontal length
            # cur_img = cv2.resize(cur_img, None, fx=original_img.shape[1] / cur_img.shape[1], fy=original_img.shape[1] / cur_img.shape[1])
            # cur_img = cv2.resize(cur_img, dsize=(int(original_img.shape[1] * cur_img.shape[0] /  original_img.shape[0]), original_img.shape[0]))
            cur_img = cv2.resize(cur_img, dsize=(original_img.shape[1], int(original_img.shape[1] * cur_img.shape[0] /  original_img.shape[0])))
            processed_img, cur_bin = self.process_image(cur_img)
            cur_word = self.extract_text(processed_img)
            if len(cur_word) < 3:
                continue

            if not self.pre_word or cur_word != self.pre_word[0]:
                str_diff = (
                    max(
                        Levenshtein.normalized_similarity(cur_word, prev_word)
                        for prev_word in self.pre_word
                    )
                    if self.pre_word
                    else 0
                )
                img_sim = self.img_similarity(pre_bin, cur_bin)

                if self.handle_differences(processed_img, cur_bin, pre_img, cur_img, str_diff, img_sim, cur_word):
                    # When user marks as SAME (skip), advance baseline so we don't get stuck
                    self.pre_word = [cur_word]
                    pre_img = cur_img
                    pre_bin = cur_bin
                    print(f"\nSAME, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
                    continue

                print(f"\nDIFF, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
                try:
                    os.makedirs(f"out/{datetime.now().strftime('%y-%m-%d')}", exist_ok=True)
                    with open(f"out/{datetime.now().strftime('%y-%m-%d')}/words.txt", "a", encoding="utf-8") as f:
                        f.write(f"{cur_word}\n")
                except OSError as e:
                    print(f"Error writing to file: {e}")
                
                result_img = self.handle_multiline(original_img, cur_img, cur_word, result_img)
                
                self.add_cnt += 1
                if self.add_cnt % 20 == 0:
                    self.save_result(result_img)
                    result_img = self.get_new_result_img()

            # self.pre_word = cur_word
            self.pre_word = [cur_word]
            pre_img = cur_img
            pre_bin = cur_bin

        if self.add_cnt % 20 != 0:  
            self.save_result(result_img)

    def handle_differences(self, processed_img, cur_bin, pre_img, cur_img, str_diff, img_sim, cur_word):
        if self.IS_DEBUG and not self.NO_GUI:
            return self.handle_debug_mode(processed_img, cur_bin, pre_img, cur_img, str_diff, img_sim, cur_word)
        else:
            return self.handle_production_mode(str_diff, img_sim, processed_img, cur_bin, pre_img, cur_img, cur_word)

    def handle_debug_mode(self, processed_img, cur_bin, pre_img, cur_img, str_diff, img_sim, cur_word):
        cv2.imshow("dst", processed_img)
        cv2.imshow("cur_bin", cur_bin)
        add_img = cv2.addWeighted(pre_img, 0.5, cur_img, 0.5, 0)
        cv2.imshow("Okay to enter", add_img)
        ok = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if ok != 13:
            print(f"\nSAME, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
            return True
        return False

    # 기존 SVM 기반 코드는 handle_production_mode_backup으로 백업합니다.
    def handle_production_mode_backup(self, str_diff, img_sim, processed_img, cur_bin, pre_img, cur_img, cur_word):
        vote = [clf.predict([[str_diff, img_sim]])[0] for clf in self.svm_clfs]
        if sum(vote) == 4:
            return True
        elif sum(vote) == 0:
            return False
        elif self.NO_GUI:
            return False
        else:
            cv2.imshow("dst", processed_img)
            cv2.imshow("cur_bin", cur_bin)
            add_img = cv2.addWeighted(pre_img, 0.5, cur_img, 0.5, 0)
            cv2.imshow("Okay to enter", add_img)
            ok = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if ok != 13:
                self.pre_word = cur_word
                print(f"\nSAME, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
                return True
        return False

    # str_diff만을 기준으로 비교하는 코드입니다.
    def handle_production_mode(self, str_diff, img_sim, processed_img, cur_bin, pre_img, cur_img, cur_word):
        # Helper: content hash to identify near-identical frames regardless of OCR noise
        def _hash_img(img: np.ndarray) -> str:
            try:
                h = hashlib.sha1()
                h.update(img.shape.__repr__().encode('utf-8'))
                h.update(img.tobytes())
                return h.hexdigest()
            except Exception:
                return ""
        if img_sim > 0.95:
            # self.pre_word = cur_word
            self.pre_word.append(cur_word)
            return True
        if str_diff <= 0.35:
            return False
        elif str_diff > 0.8:
            # self.pre_word = cur_word   
            self.pre_word.append(cur_word)   
            return True
        # Check content-hash cache first
        hkey = _hash_img(cur_bin)
        if hkey and hkey in self._hash_decision_cache:
            return bool(self._hash_decision_cache[hkey])

        # Check cached decision for this word within a short horizon
        cached = self._decision_cache.get(cur_word)
        if cached is not None:
            is_same_cached, last_idx = cached
            if self._idx - last_idx <= 60:  # reuse decision within last ~60 frames
                return bool(is_same_cached)
        # Prefer prompt_handler when provided (web UI), otherwise fall back
        if callable(self.prompt_handler):
            overlay = cv2.addWeighted(pre_img, 0.5, cur_img, 0.5, 0)
            ctx = {
                "processed_img": processed_img,
                "cur_bin": cur_bin,
                "pre_img": pre_img,
                "cur_img": cur_img,
                "overlay": overlay,
                "str_diff": float(str_diff),
                "img_sim": float(img_sim),
                "cur_word": cur_word,
            }
            is_same = bool(self.prompt_handler(ctx))
            # remember decision for this word
            self._decision_cache[cur_word] = (is_same, self._idx)
            if hkey:
                self._hash_decision_cache[hkey] = is_same
            if is_same:
                self.pre_word.append(cur_word)
                print(f"\nSAME, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
                return True
            return False

        # NO_GUI 모드에서는 바로 False 반환
        if self.NO_GUI:
            return False
        # GUI 모드: OpenCV 창으로 확인
        cv2.imshow("dst", processed_img)
        cv2.imshow("cur_bin", cur_bin)
        add_img = cv2.addWeighted(pre_img, 0.5, cur_img, 0.5, 0)
        cv2.imshow("Okay to enter", add_img)
        ok = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if ok != 13:
            # self.pre_word = [cur_word]
            self.pre_word.append(cur_word)
            print(f"\nSAME, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
            return True
        return False

    def handle_multiline(self, original_img, cur_img, cur_word, result_img):
        if self.NO_GUI:
            return np.vstack((result_img, cur_img))
            
        cur_img2 = original_img[2 * self.height_upper - self.height_lower:self.height_upper, :]
        processed_img2, _ = self.process_image(cur_img2)
        cur_word2 = self.extract_text(processed_img2)
        
        if len(cur_word2) > len(cur_word):
            print("Check multiline")
            cv2.imshow("cur_img", cur_img)
            cv2.imshow("cur_img2", cur_img2)
            ok = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if ok == 13:
                return np.vstack((result_img, cur_img2))
        return np.vstack((result_img, cur_img))

    def save_result(self, result_img):
        out_dir = Path(self.video_path).parent
        img_path = f"{out_dir}/result-{str(self.page_cnt)}.jpg"
        cv2.imwrite(img_path, result_img)
        self.page_cnt += 1 
        self.thumb_cnt += 1

    def get_new_result_img(self):
        try:
            return cv2.imread(f"{self.thumbs_dir}/" + self.thumb_list[self.thumb_cnt])[:self.height_upper - 5, :]
        except:
            return cv2.imread(f"{self.thumbs_dir}/" + self.thumb_list[-1])[:self.height_upper - 5, :]

    def select_thumbs(self, step_size=150):
        # Get sorted list of jpg files from extract directory
        extract_dir = Path(self.extract_dir)
        thumbs_dir = Path(self.thumbs_dir)
        
        # Create thumbs directory if it doesn't exist
        thumbs_dir.mkdir(exist_ok=True)
        
        # Get and sort image files
        image_files = sorted([f for f in extract_dir.glob('*.jpg')])
        
        if not image_files:
            print("No jpg files found in ./src/extract")
            return
            
        if self.NO_GUI:
            # Save every step_size'th image
            for idx in range(step_size-1, len(image_files), step_size):
                dest_path = thumbs_dir / image_files[idx].name
                shutil.copy2(image_files[idx], dest_path)
                print(f"Copied {image_files[idx].name} to thumbs directory")
        else:
            current_idx = step_size - 1  # Start from step_size'th image (index step_size-1)
            while current_idx < len(image_files):
                # Show image number out of total
                print(f"Image {current_idx + 1} of {len(image_files)}")
                
                # Read and display image
                img = cv2.imread(str(image_files[current_idx]))
                cv2.imshow('Image', img)
                
                # 키 입력 대기
                key = cv2.waitKeyEx(0)
                # 디버깅: 입력된 키 코드 출력 (필요에 따라 주석처리)
                print("Key pressed:", key)

                if key == 13:  # Enter key
                    # Copy file to thumbs directory
                    dest_path = thumbs_dir / image_files[current_idx].name
                    shutil.copy2(image_files[current_idx], dest_path)
                    print(f"Copied {image_files[current_idx].name} to thumbs directory")
                    current_idx += step_size  # Move to next image
                elif key == RIGHT_ARROW:  # Right arrow or 'd' key
                    current_idx += 1  # Move to next image
                elif key == LEFT_ARROW:  # Left arrow key
                    current_idx = max(0, current_idx - 1)  # Move to previous image, but not below 0
                elif key == 27:  # ESC key
                    break
                    
                # Check if we've reached the end
                if current_idx >= len(image_files):
                    print("Reached end of images")
                    break
            cv2.destroyAllWindows()
            
        self.load_file_lists()

    def auto_process_files(self):
        
        pass
