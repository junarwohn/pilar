import logging
from typing import List, Optional

import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
except Exception as e:  # pragma: no cover
    PaddleOCR = None  # type: ignore
    _import_error = e
else:
    _import_error = None


class OCREngine:
    """Thin wrapper around PaddleOCR for CPU-only Korean (optionally mixed English).

    - Initializes a CPU-only PaddleOCR instance optimized for low-power CPUs.
    - Provides single-image and batch OCR methods that accept numpy arrays (BGR or RGB).
    """

    def __init__(
        self,
        lang: str = "korean",  # Use "korean"; PaddleOCR model covers ko and basic latin.
        enable_angle_cls: bool = True,
        cpu_threads: Optional[int] = None,  # None: let Paddle decide; small CPUs often 2–4
        use_mkldnn: bool = True,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.6,
        det_db_unclip_ratio: float = 1.6,
    ) -> None:
        if PaddleOCR is None:
            raise RuntimeError(
                f"Failed to import PaddleOCR: {_import_error}. Install paddlepaddle (CPU) and paddleocr."
            )

        # Configure conservative settings for Intel N100 (small CPU, no GPU)
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Remember whether to run angle classifier at inference time
        self._enable_angle_cls = bool(enable_angle_cls)

        # Build kwargs with graceful fallback for older PaddleOCR versions
        base_kwargs = dict(
            use_gpu=False,
            lang=lang,
            use_angle_cls=enable_angle_cls,
        )
        opt_kwargs = {}
        if cpu_threads is not None:
            opt_kwargs["cpu_threads"] = cpu_threads
        # These may not exist on some releases; will be pruned on failure
        opt_kwargs["use_mkldnn"] = use_mkldnn
        opt_kwargs["det_db_thresh"] = det_db_thresh
        opt_kwargs["det_db_box_thresh"] = det_db_box_thresh
        opt_kwargs["det_db_unclip_ratio"] = det_db_unclip_ratio

        # Try initializing, removing unknown args iteratively
        import re

        kwargs = {**base_kwargs, **opt_kwargs}
        while True:
            try:
                self.ocr = PaddleOCR(**kwargs)
                break
            except TypeError as te:
                msg = str(te)
                m = re.search(r"Unknown argument: (\w+)", msg)
                if m:
                    bad = m.group(1)
                    if bad in kwargs:
                        kwargs.pop(bad, None)
                        continue
                # Some versions use different error text; try to detect 'got an unexpected keyword argument'
                m2 = re.search(r"unexpected keyword argument '([^']+)'", msg)
                if m2:
                    bad = m2.group(1)
                    if bad in kwargs:
                        kwargs.pop(bad, None)
                        continue
                raise RuntimeError(f"Failed to initialize PaddleOCR (CPU): {te}")
            except Exception as e:
                # Surface concise initialization errors
                raise RuntimeError(f"Failed to initialize PaddleOCR (CPU): {e}")

    @staticmethod
    def _ensure_color(img: np.ndarray) -> np.ndarray:
        """Ensure a 3-channel image (BGR)."""
        if img is None:
            return img
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    @staticmethod
    def _result_to_text(result) -> str:
        """Convert PaddleOCR result to a simple multiline string.

        result schema (per image):
        [ [ box(np.ndarray[4,2]), (text:str, score:float) ], ... ]
        """
        if not result:
            return ""
        lines = []
        for line in result:
            try:
                text, score = line[1]
            except Exception:
                # Unexpected structure; skip
                continue
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
        return "\n".join(lines)

    @staticmethod
    def _result_to_lines(result):
        """Return list of (text, score) pairs from PaddleOCR result."""
        lines = []
        if not result:
            return lines
        for line in result:
            try:
                text, score = line[1]
                if isinstance(text, str) and text.strip():
                    lines.append((text.strip(), float(score)))
            except Exception:
                continue
        return lines

    def run_ocr(self, image: np.ndarray) -> str:
        """Run OCR on a single BGR/RGB image and return recognized text.

        - Returns an empty string on failure or if no text is detected.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            self.logger.warning("run_ocr: empty or invalid image input")
            return ""
        try:
            img = self._ensure_color(image)
            result = self.ocr.ocr(img, cls=self._enable_angle_cls)
            # PaddleOCR returns list per image even for single input
            if isinstance(result, list) and result and isinstance(result[0], list):
                return self._result_to_text(result[0])
            # Some versions return inline list for single image
            return self._result_to_text(result)
        except Exception as e:
            self.logger.error(f"run_ocr failed: {e}")
            return ""

    def run_ocr_batch(self, images: List[np.ndarray]) -> List[str]:
        """Run OCR on a list of images; returns one text string per image."""
        if not images:
            return []
        safe_images = [self._ensure_color(img) for img in images if isinstance(img, np.ndarray) and img.size > 0]
        try:
            results = self.ocr.ocr(safe_images, cls=self._enable_angle_cls)
        except Exception as e:
            self.logger.error(f"run_ocr_batch failed: {e}")
            return ["" for _ in images]
        texts: List[str] = []
        # Results is a list aligned with input images
        for res in results:
            texts.append(self._result_to_text(res))
        # Pad in case of mismatch
        while len(texts) < len(images):
            texts.append("")
        return texts

    def run_ocr_lines(self, image: np.ndarray):
        """Run OCR and return list of (text, score) lines for a single image."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            self.logger.warning("run_ocr_lines: empty or invalid image input")
            return []
        try:
            img = self._ensure_color(image)
            result = self.ocr.ocr(img, cls=self._enable_angle_cls)
            if isinstance(result, list) and result and isinstance(result[0], list):
                return self._result_to_lines(result[0])
            return self._result_to_lines(result)
        except Exception as e:
            self.logger.error(f"run_ocr_lines failed: {e}")
            return []
