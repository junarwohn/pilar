from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import cv2
import shutil


def _laplacian_var(img) -> float:
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0.0
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _euclid(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _eye_open_ratio(img, face_landmarks) -> Optional[float]:
    """Compute eye openness ratio using MediaPipe Face Mesh landmarks.

    Uses vertical distance between eyelids divided by horizontal width
    for both eyes and returns the average. Returns None if landmarks are
    not available or invalid.
    """
    h, w = img.shape[:2]
    # Landmark indices from MediaPipe Face Mesh
    # Right eye (subject right): 33 (outer), 133 (inner), 159 (upper), 145 (lower)
    # Left eye (subject left): 362 (outer), 263 (inner), 386 (upper), 374 (lower)
    idx = {
        're_outer': 33, 're_inner': 133, 're_up': 159, 're_lo': 145,
        'le_outer': 362, 'le_inner': 263, 'le_up': 386, 'le_lo': 374,
    }
    lms = face_landmarks.landmark
    try:
        def pt(i):
            lm = lms[i]
            return (lm.x * w, lm.y * h)

        # Right eye
        re_w = _euclid(pt(idx['re_outer']), pt(idx['re_inner'])) + 1e-6
        re_h = _euclid(pt(idx['re_up']), pt(idx['re_lo']))
        # Left eye
        le_w = _euclid(pt(idx['le_outer']), pt(idx['le_inner'])) + 1e-6
        le_h = _euclid(pt(idx['le_up']), pt(idx['le_lo']))

        re_ratio = re_h / re_w
        le_ratio = le_h / le_w
        # Eye open ratios typically ~0.15–0.35 open, <0.08 closed (rough guide)
        return float((re_ratio + le_ratio) * 0.5)
    except Exception:
        return None


def _score_frame(img, face_mesh=None, sharp_w: float = 0.2) -> Tuple[float, float]:
    # Sharpness score (bounded)
    blur = _laplacian_var(img)
    blur_cap = 500.0
    sharp_score = min(blur, blur_cap) / blur_cap  # 0..1

    eye_score = 0.0
    if face_mesh is not None:
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                # Use best (max eye openness) across detected faces
                scores = []
                for fl in res.multi_face_landmarks:
                    r = _eye_open_ratio(img, fl)
                    if r is not None:
                        scores.append(r)
                if scores:
                    eye_score = max(scores)
        except Exception:
            pass

    # Combine: primary = eye openness, secondary = sharpness
    return eye_score, (eye_score + sharp_w * sharp_score)


def smart_auto_thumbs(
    extract_dir: str,
    thumbs_dir: str,
    step: int = 150,
    window: int = 20,
    sharp_w: float = 0.2,
    min_eye: float = 0.0,
    progress: Optional[Callable[[int, int, int], None]] = None,
) -> int:
    """Pick thumbnails automatically by maximizing eye openness within windows.

    - Iterates frames with stride `step`.
    - For each anchor, scans [i-window, i+window] and picks the highest score.
    - Score combines eye-open ratio and image sharpness.

    Returns number of thumbs selected/copied.
    """
    extract_p = Path(extract_dir)
    thumbs_p = Path(thumbs_dir)
    thumbs_p.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in extract_p.glob('*.jpg')])
    if not files:
        return 0

    # Try to import MediaPipe Face Mesh; fallback to sharpness-only if unavailable
    face_mesh = None
    try:
        import mediapipe as mp  # type: ignore
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=False,
            max_num_faces=2,
            min_detection_confidence=0.5,
        )
    except Exception:
        face_mesh = None

    selected = 0
    n = len(files)
    anchors = list(range(step - 1, n, step))
    total = len(anchors)
    for i, a in enumerate(anchors, start=1):
        lo = max(0, a - window)
        hi = min(n - 1, a + window)
        best_idx = a
        best_score = float('-inf')
        best_eye = -1.0
        # Also track best that meets the eye threshold, if any
        thr_idx = None
        thr_score = float('-inf')
        for j in range(lo, hi + 1):
            img = cv2.imread(str(files[j]))
            if img is None:
                continue
            eye, sc = _score_frame(img, face_mesh=face_mesh, sharp_w=sharp_w)
            if sc > best_score:
                best_score = sc
                best_idx = j
                best_eye = eye
            if eye is not None and eye >= min_eye and sc > thr_score:
                thr_score = sc
                thr_idx = j
        # Prefer candidate meeting eye threshold if available
        pick = thr_idx if (thr_idx is not None) else best_idx
        try:
            shutil.copy2(str(files[pick]), str(thumbs_p / files[pick].name))
            selected += 1
        except Exception:
            pass

        if progress is not None:
            try:
                progress(i, total, selected)
            except Exception:
                pass

    # Release mediapipe resources if used
    try:
        if face_mesh is not None:
            face_mesh.close()
    except Exception:
        pass

    return selected
