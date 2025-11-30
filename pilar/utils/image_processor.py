from .ocr_engine import OCREngine
import unicodedata
from rapidfuzz import fuzz
from collections import Counter
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

TESSDATA_PATH=r"./res/"  # Kept for compatibility; no longer used by PaddleOCR

class ImageProcessor:
    def __init__(self, video_path, extract_dir, thumbs_dir, no_gui=False, zoom=100, auto_detection_range=1/2, fresh=True, fps: int = 2, prompt_handler=None, progress_callback=None, ffmpeg_q: int = 5, ffmpeg_hwaccel: bool = False, ffmpeg_threads: int = 0):
        self.video_path = video_path
        self.extract_dir = extract_dir
        self.thumbs_dir = thumbs_dir
        self.IS_DEBUG = False
        self.NO_GUI = no_gui
        self.zoom = zoom
        self.auto_detection_range = auto_detection_range
        self._init_fps = fps
        # ffmpeg extraction options
        self._ff_q = int(ffmpeg_q) if ffmpeg_q is not None else 5
        self._ff_hw = bool(ffmpeg_hwaccel)
        self._ff_threads = int(ffmpeg_threads) if ffmpeg_threads is not None else 0
        # Optional callable that resolves ambiguous comparisons.
        # Signature: prompt_handler(context: dict) -> bool (True means SAME/skip, False means DIFF/add)
        self.prompt_handler = prompt_handler
        # Optional progress reporter: progress_callback(cur: int, total: int, info: dict)
        self.progress_callback = progress_callback
        # Cache for ambiguous decisions to avoid repeated prompts on same word
        self._decision_cache = {}  # word -> (is_same: bool, idx: int)
        self._hash_decision_cache = {}  # content hash -> is_same
        self._idx = 0
        # Queue of ambiguous decisions to be reviewed later (processed before saving pages)
        self._pending_ctxs = []  # list[dict]
        # OCR result cache keyed by binary content hash
        self._ocr_cache = {}
        # Ambiguity hysteresis state
        self._ambiguous_state = {"word": None, "count": 0}
        # Tri-state decision codes
        self.DIFF = -1
        self.AMBIG = 0
        self.SAME = 1

        # Initialize PaddleOCR (CPU-only) for Korean text (optionally mixed with English)
        try:
            # On Intel N100, using mkldnn and a small thread count helps
            # Disable angle classifier for horizontal subtitles to speed up
            self.engine = OCREngine(lang="korean", enable_angle_cls=False, cpu_threads=4, use_mkldnn=True)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OCR engine: {e}")

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
        """Extract frames from video using ffmpeg with performance options"""
        ffmpeg_cmd = [
            'ffmpeg',
            '-hide_banner', '-loglevel', 'error', '-y',
        ]
        if self._ff_threads is not None:
            ffmpeg_cmd += ['-threads', str(self._ff_threads)]
        if self._ff_hw:
            ffmpeg_cmd += ['-hwaccel', 'auto']
        ffmpeg_cmd += [
            '-i', self.video_path,
            '-map', '0:v:0', '-an', '-sn', '-dn',
            '-vf', f'fps={fps}',
            '-q:v', str(self._ff_q),
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
        # Early skip threshold for image similarity to avoid unnecessary OCR
        self.sim_skip_threshold = 0.985
        # Max width for OCR input to limit compute
        self.ocr_max_width = 1280
        # Hysteresis: require consecutive ambiguous DIFF observations
        self.confirm_diff_frames = 2
        # Cache TTL in frames
        self.cache_ttl_frames = 120
        # Similarity thresholds (raised for higher OCR quality)
        # Short text (<6 chars)
        self.diff_lo_short = 0.50   # was 0.45
        self.same_hi_short = 0.92   # was 0.90
        # Long text (>=6 chars)
        self.diff_lo_long = 0.60    # was 0.45 (user request)
        self.same_hi_long = 0.88    # was 0.85

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

    def _normalize_text(self, s: str) -> str:
        s = unicodedata.normalize("NFKC", s or "")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _korean_only(self, s: str) -> str:
        return re.sub(r"[^가-힣\n ]", "", s)

    def _char_bigram_dice(self, a: str, b: str) -> float:
        a = (a or "").replace(" ", "")
        b = (b or "").replace(" ", "")
        if len(a) < 2 or len(b) < 2:
            # Fallback for very short strings
            try:
                return Levenshtein.normalized_similarity(a, b)
            except Exception:
                return 0.0
        def bigrams(s: str):
            return [s[i:i+2] for i in range(len(s)-1)]
        ca = Counter(bigrams(a))
        cb = Counter(bigrams(b))
        inter = sum((ca & cb).values())
        total = sum(ca.values()) + sum(cb.values())
        if total == 0:
            return 0.0
        return 2.0 * inter / total

    def _compute_similarity(self, cur: str, prev_list):
        # Keep last 3 prev words, most recent first
        prevs = list(prev_list)[-3:]
        prevs = prevs[::-1]  # most recent first
        weights = [1.0, 0.7, 0.5]
        best = 0.0
        for i, prev in enumerate(prevs):
            w = weights[i] if i < len(weights) else weights[-1]
            try:
                sim_lev = Levenshtein.normalized_similarity(cur, prev)
            except Exception:
                sim_lev = 0.0
            try:
                sim_partial = fuzz.partial_ratio(cur, prev) / 100.0
            except Exception:
                sim_partial = 0.0
            # Space-insensitive Levenshtein
            try:
                sim_space = Levenshtein.normalized_similarity(cur.replace(" ", ""), prev.replace(" ", ""))
            except Exception:
                sim_space = 0.0
            # Character bigram Dice coefficient (space-insensitive)
            sim_bigram = self._char_bigram_dice(cur, prev)
            fused = max(sim_lev, sim_partial, sim_space, sim_bigram)
            score = fused * w
            if score > best:
                best = score
        return best

    def extract_text(self, img):
        start = time.time()
        try:
            # Downscale to max width for speed
            h, w = img.shape[:2]
            if w > getattr(self, "ocr_max_width", 1280):
                scale = self.ocr_max_width / float(w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            # Get line-level results for score aggregation
            lines = self.engine.run_ocr_lines(img)
            text = "\n".join([t for t, _ in lines])
            self._last_ocr_score = float(np.mean([s for _, s in lines])) if lines else 0.0
        except Exception as e:
            print(f"OCR failed: {e}")
            return ""
        if self.IS_DEBUG:
            elapsed = time.time() - start
            print(f"OCR time: {elapsed:.3f}s, score: {getattr(self, '_last_ocr_score', 0.0):.3f}")
        # Normalizations
        text = self._normalize_text(text or "")
        cleaned = self._korean_only(text)
        word_list = [w for w in cleaned.split('\n') if w]
        if not word_list:
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

        # Page assembly buffers (used in web/headless path with deferred review)
        page_segments = []           # list of subtitle crop images for current page
        page_words = []              # words corresponding to segments (for optional logging)
        page_pending_indices = set() # indices within page_segments that are pending review

        def _finalize_and_save_page():
            """Finalize pending decisions, assemble and save a page image.

            - Prompts user for queued ambiguous items (if prompt_handler provided)
            - Removes segments marked as SAME from the page buffer
            - Composes final image and writes it
            - Resets page buffers
            """
            nonlocal result_img, page_segments, page_words, page_pending_indices

            # Resolve pending ambiguous items just before saving
            if callable(self.prompt_handler) and page_pending_indices:
                # We iterate in the order they were queued
                # Map each pending ctx to its then-current index; decisions to SAME will drop items
                remove_indices = set()
                for ctx in list(self._pending_ctxs):
                    # Only handle items that belong to the current page (identified via page_index)
                    pidx = ctx.get("page_index")
                    if pidx is None:
                        continue
                    # Ask decision now
                    is_same = bool(self.prompt_handler(ctx))
                    if is_same:
                        remove_indices.add(pidx)
                if remove_indices:
                    # Rebuild page buffers excluding removed segments
                    kept_segments = []
                    kept_words = []
                    for i, (seg, w) in enumerate(zip(page_segments, page_words)):
                        if i not in remove_indices:
                            kept_segments.append(seg)
                            kept_words.append(w)
                    # Adjust global counters: previously we counted all adds; decrement removed
                    removed_cnt = len(remove_indices)
                    self.add_cnt = max(0, self.add_cnt - removed_cnt)
                    page_segments = kept_segments
                    page_words = kept_words
                # Clear page-related pending marks
                page_pending_indices.clear()
                # Also clear consumed pending ctxs
                self._pending_ctxs.clear()

            if not page_segments:
                # Nothing to save for this page
                return

            # Compose final page image: start from top template and stack segments
            header = self.get_new_result_img()
            composed = header
            for seg in page_segments:
                composed = np.vstack((composed, seg))
            # Save page
            # Write accepted words for this page into out/date/words.txt (append)
            try:
                os.makedirs(f"out/{datetime.now().strftime('%y-%m-%d')}", exist_ok=True)
                with open(f"out/{datetime.now().strftime('%y-%m-%d')}/words.txt", "a", encoding="utf-8") as f:
                    for w in page_words:
                        f.write(f"{w}\n")
            except OSError as e:
                print(f"Error writing to file: {e}")

            cv2.imwrite(str(Path(self.video_path).parent / f"{datetime.now().strftime('%y%m%d')}_{self.page_cnt + 1:02d}.jpg"), composed)
            self.page_cnt += 1
            self.thumb_cnt += 1
            # Reset buffers for next page
            page_segments = []
            page_words = []
            page_pending_indices = set()
            # Reset result image base for next page
            result_img = self.get_new_result_img()

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
            # Early skip: if images are near-identical, skip OCR entirely
            img_sim_early = self.img_similarity(pre_bin, cur_bin)
            if img_sim_early >= getattr(self, "sim_skip_threshold", 0.985):
                # Advance baseline and continue
                pre_img = cur_img
                pre_bin = cur_bin
                if self.IS_DEBUG:
                    print(f"Skip OCR (img_sim={img_sim_early:.3f} >= {self.sim_skip_threshold})")
                continue

            # Cache lookup by content hash to avoid repeated OCR on same binarized content
            hkey = hashlib.sha1(cur_bin.tobytes()).hexdigest()
            cur_word = self._ocr_cache.get(hkey)
            if not cur_word:
                cur_word = self.extract_text(processed_img)
                if cur_word:
                    self._ocr_cache[hkey] = cur_word
            if len(cur_word) < 3:
                continue

            if not self.pre_word or cur_word != self.pre_word[0]:
                # Compute similarity signals
                str_diff = self._compute_similarity(cur_word, self.pre_word) if self.pre_word else 0.0
                img_sim = self.img_similarity(pre_bin, cur_bin)

                decision = self.handle_differences(
                    processed_img, cur_bin, pre_img, cur_img, str_diff, img_sim, cur_word, getattr(self, '_last_ocr_score', 0.0)
                )

                if decision == self.SAME:
                    # Keep baseline; advance frame baseline images
                    baseline = self.pre_word[0] if self.pre_word else cur_word
                    self.pre_word = ([baseline] + self.pre_word)[:3]
                    pre_img = cur_img
                    pre_bin = cur_bin
                    print(f"\nSAME, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
                    # Reset ambiguous state
                    self._ambiguous_state = {"word": None, "count": 0}
                    continue

                from_ambig = False
                if decision == self.AMBIG:
                    print(f"\nAMBIGUOUS, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
                    # In headless + prompt_handler mode, only enqueue after hysteresis threshold
                    if callable(self.prompt_handler) and self.NO_GUI:
                        amb = self._ambiguous_state
                        if amb.get("word") == cur_word:
                            amb["count"] = amb.get("count", 0) + 1
                        else:
                            amb["word"] = cur_word
                            amb["count"] = 1
                        if amb["count"] < getattr(self, 'confirm_diff_frames', 2):
                            # Do not enqueue yet; keep baseline
                            continue
                        # Threshold reached -> treat as DIFF and enqueue once
                        self._ambiguous_state = {"word": None, "count": 0}
                        decision = self.DIFF
                        from_ambig = True
                    else:
                        # GUI path: handled via prompt in handle_differences
                        pass

                if decision == self.DIFF:
                    print(f"\nDIFF, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
                    if callable(self.prompt_handler) and self.NO_GUI:
                        # Add segment to current page buffer; ambiguous ones reached threshold already
                        page_index = len(page_segments)
                        page_segments.append(cur_img)
                        page_words.append(cur_word)
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
                            "page_index": page_index,
                        }
                        self._pending_ctxs.append(ctx)
                        if from_ambig:
                            page_pending_indices.add(page_index)
                        # Update counters/progress
                        self.add_cnt += 1
                        if len(page_segments) >= 20:
                            _finalize_and_save_page()
                    else:
                        # Legacy behavior (GUI or no prompt handler): immediate stacking
                        result_img = self.handle_multiline(original_img, cur_img, cur_word, result_img)
                        self.add_cnt += 1
                        if self.add_cnt % 20 == 0:
                            self.save_result(result_img)
                            result_img = self.get_new_result_img()

            # self.pre_word = cur_word
            # New substring detected (DIFF path handled above): reset baseline group
            self.pre_word = [cur_word]
            pre_img = cur_img
            pre_bin = cur_bin

        # Flush remaining segments at the end
        if callable(self.prompt_handler) and self.NO_GUI:
            _finalize_and_save_page()
        else:
            if self.add_cnt % 20 != 0:
                self.save_result(result_img)

    def handle_differences(self, processed_img, cur_bin, pre_img, cur_img, str_diff, img_sim, cur_word, ocr_score: float = 0.0):
        if self.IS_DEBUG and not self.NO_GUI:
            return self.handle_debug_mode(processed_img, cur_bin, pre_img, cur_img, str_diff, img_sim, cur_word)
        else:
            return self.handle_production_mode(str_diff, img_sim, processed_img, cur_bin, pre_img, cur_img, cur_word, ocr_score)

    def handle_debug_mode(self, processed_img, cur_bin, pre_img, cur_img, str_diff, img_sim, cur_word):
        cv2.imshow("dst", processed_img)
        cv2.imshow("cur_bin", cur_bin)
        add_img = cv2.addWeighted(pre_img, 0.5, cur_img, 0.5, 0)
        cv2.imshow("Okay to enter", add_img)
        ok = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if ok != 13:
            print(f"\nSAME, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
            return self.SAME
        return self.DIFF

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
    def handle_production_mode(self, str_diff, img_sim, processed_img, cur_bin, pre_img, cur_img, cur_word, ocr_score: float = 0.0):
        # Helper: content hash to identify near-identical frames regardless of OCR noise
        def _hash_img(img: np.ndarray) -> str:
            try:
                h = hashlib.sha1()
                h.update(img.shape.__repr__().encode('utf-8'))
                h.update(img.tobytes())
                return h.hexdigest()
            except Exception:
                return ""
        # Early SAME: if space-removed strings are identical, consider them the same subtitle
        baseline_text = self.pre_word[0] if self.pre_word else ""
        if baseline_text and cur_word.replace(" ", "") == baseline_text.replace(" ", ""):
            baseline = baseline_text
            self.pre_word = ([baseline] + self.pre_word)[:3]
            return self.SAME

        if img_sim > 0.95:
            # SAME: keep consecutive baseline group (do not switch to a different string)
            baseline = self.pre_word[0] if self.pre_word else cur_word
            self.pre_word = ([baseline] + self.pre_word)[:3]
            return self.SAME
        # Dynamic thresholds based on text length (configurable)
        L = len(cur_word)
        diff_lo = self.diff_lo_short if L < 6 else self.diff_lo_long
        same_hi = self.same_hi_short if L < 6 else self.same_hi_long
        # Absolute edit distance as additional DIFF trigger for very short strings
        try:
            abs_dist = Levenshtein.distance(cur_word, self.pre_word[0] if self.pre_word else "")
        except Exception:
            abs_dist = 0

        if str_diff <= diff_lo or (L <= 4 and abs_dist >= 2):
            return self.DIFF
        elif str_diff > same_hi:
            # SAME: keep consecutive baseline group
            baseline = self.pre_word[0] if self.pre_word else cur_word
            self.pre_word = ([baseline] + self.pre_word)[:3]
            return self.SAME
        # Check content-hash cache first
        hkey = _hash_img(cur_bin)
        if hkey and hkey in self._hash_decision_cache:
            is_same_cached, last_idx = self._hash_decision_cache.get(hkey, (None, -10))
            if is_same_cached is not None and (self._idx - last_idx) <= getattr(self, 'cache_ttl_frames', 120):
                return self.SAME if is_same_cached else self.DIFF

        # Check cached decision for this word within a short horizon
        cached = self._decision_cache.get(cur_word)
        if cached is not None:
            is_same_cached, last_idx = cached
            if self._idx - last_idx <= getattr(self, 'cache_ttl_frames', 120):  # reuse decision within TTL
                return self.SAME if is_same_cached else self.DIFF

        # Hysteresis for ambiguous cases
        # Consider OCR score: if very low, be conservative (treat as SAME)
        if ocr_score < 0.5:
            # Low confidence -> mark ambiguous
            self._ambiguous_state = {"word": cur_word, "count": 0}
            return self.AMBIG

        # Prefer prompt_handler immediately in GUI/interactive paths
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
                self._hash_decision_cache[hkey] = (is_same, self._idx)
            if is_same:
                baseline = self.pre_word[0] if self.pre_word else cur_word
                self.pre_word = ([baseline] + self.pre_word)[:3]
                print(f"\nSAME, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
                return self.SAME
            return self.DIFF

        # NO_GUI 모드에서는 바로 False 반환
        if self.NO_GUI:
            # Caller applies hysteresis and queuing; mark ambiguous here
            return self.AMBIG
        # GUI 모드: OpenCV 창으로 확인
        cv2.imshow("dst", processed_img)
        cv2.imshow("cur_bin", cur_bin)
        add_img = cv2.addWeighted(pre_img, 0.5, cur_img, 0.5, 0)
        cv2.imshow("Okay to enter", add_img)
        ok = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if ok != 13:
            # self.pre_word = [cur_word]
            baseline = self.pre_word[0] if self.pre_word else cur_word
            self.pre_word = ([baseline] + self.pre_word)[:3]
            print(f"\nSAME, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
            return self.SAME
        return self.DIFF

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
        # Name results as YYMMDD_## (two-digit, 1-based index)
        date_str = datetime.now().strftime('%y%m%d')
        index = self.page_cnt + 1
        img_path = out_dir / f"{date_str}_{index:02d}.jpg"
        cv2.imwrite(str(img_path), result_img)
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
