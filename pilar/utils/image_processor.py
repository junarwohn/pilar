from .ocr_engine import OCREngine
import unicodedata
from rapidfuzz import fuzz
from collections import Counter
import cv2
import os
import re
import numpy as np
from rapidfuzz.distance import Levenshtein
import csv
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
    def __init__(self, video_path, extract_dir, thumbs_dir, no_gui=False, zoom=100, auto_detection_range=1/2, fresh=True, fps: int = 2, prompt_handler=None, progress_callback=None, ffmpeg_q: int = 5, ffmpeg_hwaccel: bool = False, ffmpeg_threads: int = 0, stop_fn=None):
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
        # Optional stop function to allow graceful cancellation
        self.stop_fn = (stop_fn if callable(stop_fn) else (lambda: False))
        # Optional per-frame log file inside date folder
        self._run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._log_file = Path(self.video_path).parent / f"process-{self._run_id}.csv"
        self._log_inited = False
        # Cache for ambiguous decisions to avoid repeated prompts on same word
        self._decision_cache = {}  # word -> (is_same: bool, idx: int)
        self._hash_decision_cache = {}  # content hash -> is_same
        self._idx = 0
        # Track page header picks to encourage diversity
        self._used_header_idxs: list[int] = []
        self._last_header_idx: int | None = None
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

        # Optional FaceMesh for smart header picking (best-effort)
        self._face_mesh = None
        try:
            import mediapipe as mp  # type: ignore
            self._mp = mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                refine_landmarks=False,
                max_num_faces=2,
                min_detection_confidence=0.5,
            )
        except Exception:
            self._mp = None

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

    def __del__(self):
        try:
            if getattr(self, "_face_mesh", None) is not None:
                self._face_mesh.close()
        except Exception:
            pass

    def _log_metrics(self, file_name: str, decision: str, img_sim: float | None, text_sim: float | None, prev_word: str, cur_word: str):
        try:
            out_dir = Path(self.video_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            write_header = (not self._log_inited) and (not self._log_file.exists())
            with open(self._log_file, 'a', encoding='utf-8', newline='') as fh:
                w = csv.writer(fh)
                if write_header:
                    w.writerow(["time", "frame", "decision", "img_sim", "text_sim", "prev", "cur"])
                w.writerow([
                    datetime.now().strftime('%H:%M:%S'),
                    file_name,
                    decision,
                    f"{img_sim:.3f}" if isinstance(img_sim, (int, float)) else "",
                    f"{text_sim:.3f}" if isinstance(text_sim, (int, float)) else "",
                    prev_word or "",
                    cur_word or "",
                ])
            self._log_inited = True
        except Exception as e:
            # Logging must never break processing
            try:
                print(f"log error: {e}")
            except Exception:
                pass

    def extract_frames(self, fps=2):
        """Extract frames from video using ffmpeg with performance options"""
        def build_cmd(use_hwaccel: bool):
            cmd = [
                'ffmpeg',
                '-hide_banner', '-loglevel', 'error', '-y',
            ]
            if self._ff_threads is not None:
                cmd += ['-threads', str(self._ff_threads)]
            if use_hwaccel:
                cmd += ['-hwaccel', 'auto']
            cmd += [
                '-i', self.video_path,
                '-map', '0:v:0', '-an', '-sn', '-dn',
                '-vf', f'fps={fps}',
                '-q:v', str(self._ff_q),
                f'{self.extract_dir}/img%04d.jpg'
            ]
            return cmd

        # Try with HW accel if requested; fall back to software decode on failure
        cmd = build_cmd(self._ff_hw)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 and self._ff_hw:
            # Common failure on AV1: platform lacks HW decode; retry without hwaccel
            try:
                # Best-effort: retry without hwaccel
                cmd_sw = build_cmd(False)
                subprocess.run(cmd_sw, check=True)
            except subprocess.CalledProcessError as e:
                # Re-raise with helpful context
                raise RuntimeError(f"ffmpeg failed (fallback as well). stderr: {e.stderr or res.stderr}")

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
        # Smart header selection params
        self.thumb_pick_window = 20
        self.thumb_sharp_weight = 0.2
        self.thumb_min_eye = 0.12
        self.header_recent_k = 8
        self.header_min_frame_gap = 120

    def load_file_lists(self):
        self.file_list = [i for i in os.listdir(self.extract_dir) if not i.startswith('.') and not 'Zone.Identifier' in i]
        self.file_list.sort()
        self.thumb_list = [i for i in os.listdir(self.thumbs_dir) if not i.startswith('.') and not 'Zone.Identifier' in i]
        self.thumb_list.sort()

    # --- Auto bounds detection utilities ---
    @staticmethod
    def _longest_run(mask: np.ndarray, min_len: int = 20) -> tuple[int, int] | None:
        """Return (start, end) indices of the longest contiguous True run in 1D mask.
        end is exclusive. Returns None if no run meets min_len.
        """
        best_s, best_e = -1, -1
        s = None
        for i, v in enumerate(mask.astype(np.uint8)):
            if v:
                if s is None:
                    s = i
            else:
                if s is not None:
                    if i - s > (best_e - best_s):
                        best_s, best_e = s, i
                    s = None
        if s is not None and (len(mask) - s) > (best_e - best_s):
            best_s, best_e = s, len(mask)
        if best_s >= 0 and (best_e - best_s) >= min_len:
            return best_s, best_e
        return None

    @staticmethod
    def _detect_bounds_one(gray: np.ndarray) -> tuple[int, int] | None:
        """Detect subtitle band bounds in a single grayscale frame.

        Heuristics tailored for white text over a dark rectangle near the bottom:
        - Work on bottom ~55% ROI
        - Enhance contrast (CLAHE)
        - Combine darkness ratio and edge density per row to score candidate band
        - Prefer long contiguous high-score run; fall back to edge-only
        - Constrain to plausible height and vertical position
        """
        h, w = gray.shape[:2]
        if h < 60 or w < 60:
            return None

        # ROI: bottom half+ (avoid mid-frame graphics)
        roi_top = int(h * 0.45)
        roi = gray[roi_top:, :]

        # Contrast-limited adaptive histogram equalization for robustness
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            roi_eq = clahe.apply(roi)
        except Exception:
            roi_eq = roi

        # Dark mask via Otsu on ROI
        try:
            _, dark = cv2.threshold(roi_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        except Exception:
            thr = int(np.quantile(roi_eq, 0.25))
            thr = max(30, min(110, thr))
            _, dark = cv2.threshold(roi_eq, thr, 255, cv2.THRESH_BINARY_INV)

        # Edges (white glyphs on dark background → strong edges)
        med = np.median(roi_eq)
        lo = int(max(0, 0.66 * med))
        hi = int(min(255, 1.33 * med))
        edges = cv2.Canny(roi_eq, lo, hi)

        # Row-wise metrics
        dark_ratio = dark.mean(axis=1) / 255.0              # 0..1
        edge_density = (edges.mean(axis=1) / 255.0)         # 0..1
        # Score: prefer edges within dark regions; still allow edge-only for transparent subs
        score = edge_density * (0.4 + 0.6 * dark_ratio)

        # Smooth and threshold by quantiles
        k = max(5, (h // 200) * 2 + 1)
        score_s = cv2.GaussianBlur(score.astype(np.float32).reshape(-1, 1), (1, k), 0).ravel()
        q = float(np.quantile(score_s, 0.75))
        mask = score_s >= q
        run = ImageProcessor._longest_run(mask, min_len=max(10, int(h * 0.03)))
        if not run:
            # Fallback: use edge density alone
            q2 = float(np.quantile(edge_density, 0.80))
            mask2 = edge_density >= q2
            run = ImageProcessor._longest_run(mask2, min_len=max(10, int(h * 0.02)))
            if not run:
                return None

        s, e = run
        up = roi_top + s
        lo = roi_top + e

        # Plausible band height constraints
        min_h = max(20, int(h * 0.06))
        max_h = int(h * 0.22)
        band_h = lo - up
        if band_h < min_h:
            # Expand around center
            c = (up + lo) // 2
            up = max(0, c - min_h // 2)
            lo = min(h, up + min_h)
        elif band_h > max_h:
            # Shrink to max_h around center
            c = (up + lo) // 2
            up = max(0, c - max_h // 2)
            lo = min(h, up + max_h)

        # Vertical position sanity: subtitles near bottom third
        center = 0.5 * (up + lo)
        if center < h * 0.55:
            # Too high → bias downwards same height
            shift = int(h * 0.60 - center)
            up = min(max(0, up + shift), h - 1)
            lo = min(h, lo + shift)

        # Clamp
        up = int(max(0, min(up, h - 2)))
        lo = int(max(up + 1, min(lo, h)))
        return up, lo

    @staticmethod
    def auto_detect_bounds_from_dir(extract_dir: str, sample_count: int = 20) -> tuple[int, int] | None:
        """Sample frames from extract_dir and estimate subtitle band bounds.
        Returns (upper, lower) as image row indices (int) or None if not found.
        """
        try:
            files = sorted([f for f in os.listdir(extract_dir) if f.lower().endswith('.jpg')])
            if not files:
                return None
            n = len(files)
            step = max(1, n // max(1, sample_count))
            picks = [files[i] for i in range(0, n, step)][:sample_count]
            ups, los = [], []
            for name in picks:
                p = os.path.join(extract_dir, name)
                img = cv2.imread(p)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                found = ImageProcessor._detect_bounds_one(gray)
                if found:
                    u, l = found
                    ups.append(u)
                    los.append(l)
            if ups and los:
                u = int(np.median(ups))
                l = int(np.median(los))
                # Basic sanity
                if l - u < 20:
                    l = u + 20
                return u, l
            return None
        except Exception:
            return None

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
                # De-emphasize partial match influence (x0.2)
                sim_partial = (fuzz.partial_ratio(cur, prev) / 100.0) * 0.2
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
        # Guard: ensure frames exist
        if not self.file_list:
            raise RuntimeError("No extracted frames found. Run extraction first.")
        result_img = self.get_new_result_img(center_idx=0)
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
        page_header_img = None       # header image picked when the first subtitle of the page appears

        def _finalize_and_save_page():
            """Finalize pending decisions, assemble and save a page image.

            - Prompts user for queued ambiguous items (if prompt_handler provided)
            - Removes segments marked as SAME from the page buffer
            - Composes final image and writes it
            - Resets page buffers
            """
            nonlocal result_img, page_segments, page_words, page_pending_indices, page_header_img

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

            # Compose final page image: use header chosen at page start if available
            header = page_header_img if page_header_img is not None else self.get_new_result_img(center_idx=self._idx)
            if page_header_img is None and getattr(self, '_last_header_idx', None) is not None:
                self._used_header_idxs.append(self._last_header_idx)
                if len(self._used_header_idxs) > 64:
                    self._used_header_idxs = self._used_header_idxs[-64:]
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
            page_header_img = None
            # Reset result image base for next page (GUI path usage)
            result_img = self.get_new_result_img(center_idx=self._idx)

        for file_name in iterable:
            # Allow external stop request
            try:
                if self.stop_fn():
                    break
            except Exception:
                pass
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
                # Log early skip (no OCR)
                prev_word = self.pre_word[0] if self.pre_word else ""
                self._log_metrics(file_name=file_name, decision="EARLY_SKIP", img_sim=img_sim_early, text_sim=None, prev_word=prev_word, cur_word="")
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

                # Reset manual flag for this comparison
                self._last_manual = False
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
                    # Log metrics
                    dec_label = "MANUAL_SAME" if getattr(self, "_last_manual", False) else "SAME"
                    prev_word = self.pre_word[0] if self.pre_word else ""
                    self._log_metrics(file_name=file_name, decision=dec_label, img_sim=img_sim, text_sim=str_diff, prev_word=prev_word, cur_word=cur_word)
                    # Reset ambiguous state
                    self._ambiguous_state = {"word": None, "count": 0}
                    continue

                from_ambig = False
                if decision == self.AMBIG:
                    print(f"\nAMBIGUOUS, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
                    # Log metrics (ambiguous observation)
                    prev_word = self.pre_word[0] if self.pre_word else ""
                    self._log_metrics(file_name=file_name, decision="AMBIG", img_sim=img_sim, text_sim=str_diff, prev_word=prev_word, cur_word=cur_word)
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
                    # Log metrics
                    dec_label = "MANUAL_DIFF" if getattr(self, "_last_manual", False) else "DIFF"
                    prev_word = self.pre_word[0] if self.pre_word else ""
                    self._log_metrics(file_name=file_name, decision=dec_label, img_sim=img_sim, text_sim=str_diff, prev_word=prev_word, cur_word=cur_word)
                    if callable(self.prompt_handler) and self.NO_GUI:
                        # Add segment to current page buffer; ambiguous ones reached threshold already
                        page_index = len(page_segments)
                        # On first segment of a new page, choose a header within [-10 .. current] frames
                        if page_index == 0 or page_header_img is None:
                            try:
                                cur_j = max(0, self._idx - 1)
                                pick_j = self._pick_header_in_range(max(0, cur_j - 10), cur_j)
                                img_h = cv2.imread(os.path.join(self.extract_dir, self.file_list[pick_j]))
                                if img_h is not None:
                                    page_header_img = img_h[:self.height_upper - 5, :]
                                else:
                                    page_header_img = None
                            except Exception:
                                page_header_img = None
                        page_segments.append(cur_img)
                        page_words.append(cur_word)
                        overlay = cv2.addWeighted(pre_img, 0.5, cur_img, 0.5, 0)
                        prev_word = (self.pre_word[0] if self.pre_word else "")
                        ctx = {
                            "processed_img": processed_img,
                            "cur_bin": cur_bin,
                            "pre_img": pre_img,
                            "cur_img": cur_img,
                            "overlay": overlay,
                            "str_diff": float(str_diff),
                            "img_sim": float(img_sim),
                            "cur_word": cur_word,
                            "prev_word": prev_word,
                            "text_sim": float(str_diff),
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
                            result_img = self.get_new_result_img(center_idx=self._idx)

            else:
                # Same recognized text; log with perfect text similarity
                img_sim_eq = self.img_similarity(pre_bin, cur_bin)
                prev_word = self.pre_word[0] if self.pre_word else cur_word
                self._log_metrics(file_name=file_name, decision="SAME_TEXT", img_sim=img_sim_eq, text_sim=1.0, prev_word=prev_word, cur_word=cur_word)

            # New substring baseline (or same) — advance baseline images
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
            prev_word = (self.pre_word[0] if self.pre_word else "")
            ctx = {
                "processed_img": processed_img,
                "cur_bin": cur_bin,
                "pre_img": pre_img,
                "cur_img": cur_img,
                "overlay": overlay,
                "str_diff": float(str_diff),
                "img_sim": float(img_sim),
                "cur_word": cur_word,
                "prev_word": prev_word,
                "text_sim": float(str_diff),
            }
            is_same = bool(self.prompt_handler(ctx))
            self._last_manual = True
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

    def _header_score(self, img) -> tuple[float, float]:
        try:
            from pilar.utils.smart_thumbs import _score_frame as _st_score
            eye, sc = _st_score(img, face_mesh=self._face_mesh, sharp_w=self.thumb_sharp_weight)
            return float(eye or 0.0), float(sc)
        except Exception:
            # Fallback: sharpness only
            try:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
                sc = min(sharp, 500.0) / 500.0
                return 0.0, sc
            except Exception:
                return 0.0, 0.0

    def _pick_header_index(self, center_idx: int) -> int:
        n = len(self.file_list)
        if n == 0:
            return 0
        c = max(0, min(center_idx, n - 1))
        lo = max(0, c - self.thumb_pick_window)
        hi = min(n - 1, c + self.thumb_pick_window)
        candidates = []  # (score, eye, j)
        for j in range(lo, hi + 1):
            try:
                img = cv2.imread(os.path.join(self.extract_dir, self.file_list[j]))
                if img is None:
                    continue
                eye, sc = self._header_score(img)
            except Exception:
                continue
            candidates.append((float(sc), float(eye or 0.0), int(j)))
        if not candidates:
            return c
        # Sort by score descending
        candidates.sort(key=lambda t: t[0], reverse=True)
        # Helper: far enough from recently used headers
        recent = self._used_header_idxs[-self.header_recent_k:]
        def far_enough(j: int) -> bool:
            if not recent:
                return True
            return min(abs(j - r) for r in recent) >= self.header_min_frame_gap
        # 1) eyes-open + far
        for sc, eye, j in candidates:
            if eye >= self.thumb_min_eye and far_enough(j):
                return j
        # 2) far only
        for sc, eye, j in candidates:
            if far_enough(j):
                return j
        # 3) fallback best overall
        return int(candidates[0][2])

    def get_new_result_img(self, center_idx: int | None = None):
        # Primary: dynamic pick from extract frames near the given center
        try:
            if center_idx is None:
                center_idx = max(0, min(self._idx, len(self.file_list)-1))
            j = self._pick_header_index(center_idx)
            self._last_header_idx = j
            img = cv2.imread(os.path.join(self.extract_dir, self.file_list[j]))
            if img is not None:
                return img[:self.height_upper - 5, :]
        except Exception:
            pass
        # Fallback: legacy thumbs list if available
        try:
            if self.thumb_list:
                return cv2.imread(f"{self.thumbs_dir}/" + self.thumb_list[min(self.thumb_cnt, len(self.thumb_list)-1)])[:self.height_upper - 5, :]
        except Exception:
            pass
        # Last resort: first frame top crop
        try:
            img = cv2.imread(os.path.join(self.extract_dir, self.file_list[0]))
            return img[:self.height_upper - 5, :]
        except Exception:
            return np.zeros((max(1, self.height_upper - 5), 1, 3), dtype=np.uint8)

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
