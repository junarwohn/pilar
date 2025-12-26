import os
import re
import shutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
import time

from flask import Flask, request, send_file, redirect, url_for, Response, jsonify, stream_with_context
import subprocess
import json

from pilar.utils.downloader import Downloader
from pilar.utils.image_uploader import ImageUploader
from pilar.utils.image_processor import ImageProcessor
from pilar.utils.smart_thumbs import smart_auto_thumbs


def create_app(base_dir: str, no_gui: bool = True, zoom: int = 112, fps: int = 2, step_size: int = 150):
    app = Flask(__name__)

    # Paths
    base_path = Path(base_dir)
    video_path = base_path / "src.mp4"
    extract_dir = base_path / "extract"
    thumbs_dir = base_path / "thumbs"
    # Persisted settings live at parent (stable across day rollover)
    bounds_file = (base_path.parent / "bounds.json").resolve()
    state_file = (base_path.parent / "state.json").resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(exist_ok=True)
    thumbs_dir.mkdir(exist_ok=True)

    # Ensure we rollover to a new day folder if the date changed while server is running.
    def ensure_today_base():
        nonlocal base_path, video_path, extract_dir, thumbs_dir
        try:
            today = datetime.now().strftime('%y-%m-%d')
            target = base_path.parent / today
            if target != base_path:
                base_path = target
                video_path = base_path / "src.mp4"
                extract_dir = base_path / "extract"
                thumbs_dir = base_path / "thumbs"
                base_path.mkdir(parents=True, exist_ok=True)
                extract_dir.mkdir(exist_ok=True)
                thumbs_dir.mkdir(exist_ok=True)
                # Reset thumbs navigation index for the new day
                state["idx"] = state.get("step_size", 150) - 1
        except Exception:
            # Fail open: if anything goes wrong, keep existing base paths
            pass

    # Simple in-memory state for thumbs navigation
    state = {
        "idx": step_size - 1,  # current index for thumbs selection
        "step_size": step_size,
        "processing": False,
        "stop": False,
        "bounds_nav": {"src": "extract", "idx": 0},
        "last_log": "",
        "source_url": None,
        "download": {
            "running": False,
            "last_log": "",
            "progress": {"downloaded": 0, "total": 0, "speed": 0, "eta": 0, "status": "idle"},
        },
        "smart_thumbs": {
            "running": False,
            "last_log": "",
            "progress": {"current": 0, "total": 0, "selected": 0},
        },
        "bounds": {"upper": 925, "lower": 1020},
        "review": {
            "pending": False,
            "seq": 0,
            "files": {},
            "metrics": {},
            "decision": None,
        },
        "progress": {"current": 0, "total": 0, "pages": 0, "added": 0},
        "extract": {
            "running": False,
            "last_log": "",
            "progress": {"current": 0, "total": 0, "fps": fps},
        },
    }
    cond = threading.Condition()

    # Persist/restore bounds across restarts
    def _save_bounds(upper: int, lower: int):
        try:
            data = {"upper": int(upper), "lower": int(lower)}
            bounds_file.parent.mkdir(parents=True, exist_ok=True)
            with open(bounds_file, 'w', encoding='utf-8') as fh:
                json.dump(data, fh)
        except Exception:
            pass

    def _load_bounds():
        try:
            if bounds_file.exists():
                with open(bounds_file, 'r', encoding='utf-8') as fh:
                    data = json.load(fh) or {}
                u = int(data.get('upper', state["bounds"]["upper"]))
                l = int(data.get('lower', state["bounds"]["lower"]))
                # basic sanity
                if u >= 0 and l > u:
                    state["bounds"] = {"upper": u, "lower": l}
        except Exception:
            pass

    # Load bounds on startup
    _load_bounds()

    # Persist/restore non-running state
    def _save_state_subset():
        try:
            data = {
                "idx": int(state.get("idx", 0)),
                "step_size": int(state.get("step_size", step_size)),
                "bounds": state.get("bounds", {}),
                "bounds_nav": state.get("bounds_nav", {"src": "thumbs", "idx": 0}),
                "last_log": state.get("last_log", ""),
                "progress": state.get("progress", {}),
                "source_url": state.get("source_url"),
            }
            with open(state_file, 'w', encoding='utf-8') as fh:
                json.dump(data, fh)
        except Exception:
            pass

    def _load_state_subset():
        try:
            if not state_file.exists():
                return
            with open(state_file, 'r', encoding='utf-8') as fh:
                data = json.load(fh) or {}
            state["idx"] = int(data.get("idx", state["idx"]))
            state["step_size"] = int(data.get("step_size", state["step_size"]))
            b = data.get("bounds")
            if isinstance(b, dict) and "upper" in b and "lower" in b:
                state["bounds"] = {"upper": int(b["upper"]), "lower": int(b["lower"])}
            bn = data.get("bounds_nav")
            if isinstance(bn, dict):
                state["bounds_nav"] = {"src": bn.get("src", "thumbs"), "idx": int(bn.get("idx", 0))}
            state["last_log"] = data.get("last_log", state["last_log"])
            pr = data.get("progress")
            if isinstance(pr, dict):
                # Do not mark running; just keep last numbers
                state["progress"] = {
                    "current": int(pr.get("current", 0)),
                    "total": int(pr.get("total", 0)),
                    "pages": int(pr.get("pages", 0)),
                    "added": int(pr.get("added", 0)),
                }
            srcu = data.get("source_url")
            if srcu:
                state["source_url"] = srcu
        except Exception:
            pass

    _load_state_subset()

    # Daily scheduler: auto switch to today's date and run prepare frames at 06:00 local time
    def _seconds_until_next_6am(now: datetime | None = None) -> float:
        try:
            now = now or datetime.now()
            target = now.replace(hour=6, minute=0, second=0, microsecond=0)
            if now >= target:
                target = target + timedelta(days=1)
            return max(1.0, (target - now).total_seconds())
        except Exception:
            return 3600.0  # fallback: 1 hour

    def _daily_scheduler():
        while True:
            try:
                to_sleep = _seconds_until_next_6am()
            except Exception:
                to_sleep = 3600.0
            time.sleep(to_sleep)
            try:
                ensure_today_base()
                with cond:
                    state["last_log"] = "Daily 06:00 auto refresh + prepare starting"
                    cond.notify_all()
                # Start prepare frames (download + extract) if not busy
                ok = start_download_extract(None, fps, 5, False)
                if not ok:
                    with cond:
                        state["last_log"] = "Daily run skipped: pipeline busy"
                        cond.notify_all()
                # Avoid immediate retrigger in case clock skew; short nap
                time.sleep(5)
            except Exception:
                # Ignore scheduler errors and continue
                time.sleep(5)

    # Build a snapshot of current progress for SSE/clients
    def _snapshot():
        snap = {}
        try:
            # Download
            dprog = state["download"].get("progress", {})
            downloaded = int(dprog.get("downloaded", 0) or 0)
            total = int(dprog.get("total", 0) or 0)
            dpercent = 0
            try:
                if total > 0:
                    dpercent = int(downloaded * 100 / total)
                elif dprog.get("status") == 'finished':
                    dpercent = 100
            except Exception:
                dpercent = 0
            snap["download"] = {
                "running": bool(state["download"].get("running", False)),
                "progress": {"downloaded": downloaded, "total": total, "speed": dprog.get("speed", 0), "eta": dprog.get("eta", 0), "status": dprog.get("status", "idle")},
                "percent": dpercent,
            }
            # Extract
            eprog = state["extract"].get("progress", {"current": 0, "total": 0, "fps": 0})
            try:
                epercent = int((int(eprog.get("current", 0)) * 100) / (int(eprog.get("total", 0)) or 1)) if (eprog.get("total", 0) or 0) > 0 else 0
            except Exception:
                epercent = 0
            snap["extract"] = {
                "running": bool(state["extract"].get("running", False)),
                "progress": {"current": int(eprog.get("current", 0) or 0), "total": int(eprog.get("total", 0) or 0), "fps": int(eprog.get("fps", 0) or 0)},
                "percent": epercent,
            }
            # Smart thumbs
            sprog = state["smart_thumbs"].get("progress", {"current": 0, "total": 0, "selected": 0})
            cur = int(sprog.get("current", 0) or 0)
            tot = int(sprog.get("total", 0) or 0)
            try:
                spercent = int(cur * 100 / (tot or 1))
            except Exception:
                spercent = 0
            snap["smart_thumbs"] = {
                "running": bool(state["smart_thumbs"].get("running", False)),
                "progress": {"current": cur, "total": tot, "selected": int(sprog.get("selected", 0) or 0)},
                "percent": spercent,
            }
            # Processing
            p = state.get("progress", {"current": 0, "total": 0, "pages": 0, "added": 0})
            try:
                ppercent = int((int(p.get("current", 0)) * 100) / (int(p.get("total", 0)) or 1)) if (p.get("total", 0) or 0) > 0 else 0
            except Exception:
                ppercent = 0
            snap["process"] = {
                "running": bool(state.get("processing", False)),
                "awaiting": bool(state.get("review", {}).get("pending", False)),
                "progress": {"current": int(p.get("current", 0) or 0), "total": int(p.get("total", 0) or 0), "pages": int(p.get("pages", 0) or 0), "added": int(p.get("added", 0) or 0)},
                "percent": ppercent,
            }
            snap["message"] = state.get("last_log", "")
            snap["results"] = len(_result_files()) if ' _result_files' in globals() or True else 0
        except Exception:
            pass
        return snap

    @app.get('/events')
    def events():
        def _stream():
            try:
                # advise retry interval
                yield 'retry: 2000\n\n'
                # initial snapshot
                with cond:
                    snap = _snapshot()
                yield 'data: ' + json.dumps(snap) + '\n\n'
                # updates or periodic heartbeats
                while True:
                    with cond:
                        cond.wait(timeout=1.0)
                        snap = _snapshot()
                    yield 'data: ' + json.dumps(snap) + '\n\n'
            except GeneratorExit:
                return
        headers = {'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        return Response(stream_with_context(_stream()), mimetype='text/event-stream', headers=headers)

    def img_files():
        return sorted([p for p in extract_dir.glob("*.jpg")])

    def thumb_files():
        return sorted([p for p in thumbs_dir.glob("*.jpg")])

    def copy_thumb(p: Path):
        dest = thumbs_dir / p.name
        shutil.copy2(p, dest)

    def youtube_embed_url(u: str | None) -> str | None:
        if not u:
            return None
        try:
            # already embed link
            if 'youtube.com/embed/' in u:
                return u
            # watch?v=
            m = re.search(r"youtube\.com/watch\?v=([\w-]{6,})", u)
            if m:
                return f"https://www.youtube.com/embed/{m.group(1)}"
            # youtu.be/
            m = re.search(r"youtu\.be/([\w-]{6,})", u)
            if m:
                return f"https://www.youtube.com/embed/{m.group(1)}"
            return None
        except Exception:
            return None

    def get_source_url() -> str | None:
        # Prefer explicitly set URL during download
        u = state.get("source_url")
        if u:
            return u
        # Cache autodetected URL once per server lifetime
        auto = state.get("auto_url")
        if auto is None:
            try:
                auto = Downloader(output_path=str(video_path)).get_yn_url()
            except Exception:
                auto = None
            state["auto_url"] = auto
        return auto

    def reset_workspace(hard: bool = False):
        # Stop flags
        with cond:
            state["download"]["running"] = False
            state["extract"]["running"] = False
            state["smart_thumbs"]["running"] = False
            state["processing"] = False
        # Delete video
        try:
            if video_path.exists():
                video_path.unlink()
        except Exception:
            pass
        # Remove extract and thumbs directories
        for d in (extract_dir, thumbs_dir):
            try:
                if d.exists():
                    shutil.rmtree(d)
            except Exception:
                pass
            d.mkdir(exist_ok=True)
        # Remove review temp files
        for p in base_path.glob("_review_*.jpg"):
            try:
                p.unlink()
            except Exception:
                pass
        if hard:
            # Remove result jpgs (e.g., 231205_01.jpg)
            for p in base_path.glob("*.jpg"):
                if re.match(r"^\d{6}_\d{2}\.jpg$", p.name):
                    try:
                        p.unlink()
                    except Exception:
                        pass
        # Reset state
        with cond:
            state["idx"] = state.get("step_size", 150) - 1
            state["last_log"] = "Workspace reset" + (" (hard)" if hard else "")
            state["download"]["progress"] = {"downloaded": 0, "total": 0, "speed": 0, "eta": 0, "status": "idle"}
            state["extract"]["progress"] = {"current": 0, "total": 0, "fps": fps}
            state["smart_thumbs"]["progress"] = {"current": 0, "total": 0, "selected": 0}
            state["progress"] = {"current": 0, "total": 0, "pages": 0, "added": 0}
            cond.notify_all()

    # Minimal HTML shell
    def page(body: str) -> str:
        return f"""
        <html>
          <head>
            <meta name='viewport' content='width=device-width, initial-scale=1'>
            <style>
              * {{ box-sizing: border-box; }}
              body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background:#fafafa; color:#222; }}
              .container {{ max-width: 960px; margin: 0 auto; padding: 12px; }}
              .nav {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom: 10px; }}
              .btn {{ display:inline-block; padding:12px 14px; border-radius:8px; text-decoration:none; background:#1f6feb; color:#fff; border:none; font-size:15px; }}
              .btn.secondary {{ background:#e9ecef; color:#111; }}
              .btn.warn {{ background:#d9534f; color:#fff; }}
              button {{ padding:12px 14px; border:none; border-radius:8px; background:#1f6feb; color:#fff; font-size:15px; }}
              label {{ font-size:14px; }}
              input[type='text'], input[type='number'] {{ width:100%; padding:10px; border:1px solid #ccc; border-radius:8px; font-size:16px; }}
              img.responsive {{ width:100%; height:auto; }}
              .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:10px; }}
              .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:6px; }}
              .muted {{ color:#666; font-size: 13px; }}
              /* Loading overlay */
              #loadingOverlay {{ position: fixed; inset: 0; background: rgba(0,0,0,0.35); display: none; align-items: center; justify-content: center; z-index: 9999; color:#fff; }}
              #loadingBox {{ background: rgba(0,0,0,0.85); padding: 14px 16px; border-radius: 10px; min-width: 200px; display:flex; align-items:center; gap:10px; }}
              .spinner {{ width: 18px; height: 18px; border: 3px solid rgba(255,255,255,0.3); border-top-color:#fff; border-radius: 50%; animation: spin 1s linear infinite; }}
              @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
              .mt-1 {{ margin-top:8px; }}
              .mt-2 {{ margin-top:12px; }}
              .section-title {{ margin: 4px 0 6px; font-weight:600; }}
              details summary {{ cursor: pointer; user-select:none; padding:6px 0; color:#333; }}
            </style>
            <script>
              function showOverlay(msg) {{
                try {{
                  const o = document.getElementById('loadingOverlay');
                  const t = document.getElementById('loadingText');
                  if (t && msg) t.textContent = msg; else if (t) t.textContent = '처리 중...';
                  if (o) o.style.display = 'flex';
                }} catch (e) {{}}
              }}
              function hideOverlay() {{
                try {{ const o = document.getElementById('loadingOverlay'); if (o) o.style.display = 'none'; }} catch (e) {{}}
              }}
              function submitForm(form, redirectTo, submitter) {{
                const fd = new FormData(form);
                try {{
                  if (submitter && submitter.name) {{
                    fd.append(submitter.name, submitter.value ?? '');
                  }} else if (document.activeElement && document.activeElement.name) {{
                    fd.append(document.activeElement.name, document.activeElement.value ?? '');
                  }}
                }} catch (e) {{}}
                try {{
                  const act = (form.action || '').toLowerCase();
                  if (act.endsWith('/today')) {{
                    showOverlay('오늘 날짜로 새로고침 중...');
                  }}
                }} catch (e) {{}}
                fetch(form.action, {{ method: 'POST', body: fd }})
                  .then(() => {{
                    if (redirectTo) {{ window.location.href = redirectTo; }}
                    else {{ window.location.reload(); }}
                  }})
                  .catch(() => {{ hideOverlay(); alert('Request failed'); }});
              }}
              document.addEventListener('DOMContentLoaded', () => {{
                document.querySelectorAll('form.js-post').forEach(f => {{
                  f.addEventListener('submit', (e) => {{
                    if (f.hasAttribute('data-confirm')) {{
                      if (!confirm(f.getAttribute('data-confirm'))) {{ e.preventDefault(); return; }}
                    }}
                    e.preventDefault();
                    submitForm(f, f.getAttribute('data-redirect'), e.submitter);
                  }});
                }});
              }});
            </script>
          </head>
          <body>
            <div id='loadingOverlay'>
              <div id='loadingBox'>
                <div class='spinner'></div>
                <div id='loadingText'>처리 중...</div>
              </div>
            </div>
            <div class='container'>
              <h3 style='margin:4px 0 10px;'>Pilar Web Controller</h3>
              <div class='nav'>
                <a class='btn secondary' href='{url_for('index')}'>Home</a>
                <a class='btn secondary' href='{url_for('prepare')}'>Prepare Frames</a>
                <a class='btn secondary' href='{url_for('preprocess')}'>Pre-Process</a>
                <a class='btn secondary' href='{url_for('process_run')}'>Process</a>
                <a class='btn secondary' href='{url_for('results')}'>Results</a>
              </div>
              {body}
            </div>
          </body>
        </html>
        """

    @app.get("/")
    def index():
        ensure_today_base()
        exists = video_path.exists()
        n_imgs = len(img_files())
        n_thumbs = len(list(thumbs_dir.glob("*.jpg")))
        # Busy flags
        downloading = state["download"]["running"]
        extracting = state["extract"]["running"]
        processing = state["processing"]
        disable_all = 'disabled' if (downloading or extracting or processing or state.get('smart_thumbs',{}).get('running')) else ''
        # Download progress snapshot
        dprog = state["download"]["progress"]
        try:
            dpercent = int((dprog.get("downloaded", 0) * 100) / (dprog.get("total", 0) or 1)) if downloading else (100 if dprog.get("status") == 'finished' else 0)
        except Exception:
            dpercent = 0
        dcounts = f"{dprog.get('downloaded', 0)//(1024*1024)}MB / {max(1, dprog.get('total', 0))//(1024*1024)}MB | eta: {dprog.get('eta', 0)}s"
        # Extract progress snapshot
        eprog = state["extract"].get("progress", {"current": 0, "total": 0, "fps": fps})
        try:
            epercent = int((eprog.get("current", 0) * 100) / (eprog.get("total", 0) or 1)) if extracting or (eprog.get("total", 0) > 0) else 0
        except Exception:
            epercent = 0
        ecounts = f"frames: {eprog.get('current', 0)}/{eprog.get('total', 0)} | fps: {eprog.get('fps', fps)}"
        # Build optional embed
        src_url = get_source_url()
        embed = youtube_embed_url(src_url)
        desc = (
            "<div class='card'><div class='section-title'>About</div>"
            "<div class='muted'>Use Prepare Frames to download/extract today's video."
            " Then set subtitle bounds in Pre-Process, and finally run Process and review Results.</div></div>"
        )
        # Today refresh control
        today_ctrl = (
            f"<div class='card mt-2'>"
            f"  <div class='section-title'>날짜</div>"
            f"  <div class='muted'>현재 작업 폴더: {base_path}</div>"
            f"  <form method='post' class='js-post mt-1' data-redirect='{url_for('index')}' action='{url_for('set_today')}'>"
            f"    <button type='submit'>오늘 날짜로 새로고침</button>"
            f"  </form>"
            f"</div>"
        )

        yt = (f"<div class='card mt-2'><div class='section-title'>Today's Video</div>"
              f"<div class='muted'>Source: {src_url or 'Not detected yet'}</div>"
              + (f"<div class='mt-1'><iframe style='width:100%; aspect-ratio:16/9;' src='{embed}' frameborder='0' allowfullscreen></iframe></div>" if embed else "")
              + "</div>")
        return page(desc + today_ctrl + yt)

    @app.post("/today")
    def set_today():
        # Force-refresh base paths to today's date
        ensure_today_base()
        try:
            # Clear cached/detected source so it re-detects for today
            with cond:
                state["source_url"] = None
                state["auto_url"] = None
                cond.notify_all()
        except Exception:
            pass
        try:
            with cond:
                state["last_log"] = "Switched to today's date"
                cond.notify_all()
        except Exception:
            pass
        return redirect(url_for('index'))

    # Helper to start the Download + Extract pipeline from code (route/scheduler)
    def start_download_extract(url: str | None, fps_val: int, q_val: int, hw: bool) -> bool:
        # Guard: do not start if any pipeline phase is running
        if state["download"]["running"] or state["processing"] or state["extract"]["running"]:
            with cond:
                state["last_log"] = "Busy: cannot start download+extract now."
            return False

        def _job(url_val, fps_v, q_v, hw_v):
            # Phase 1: Download
            dl = Downloader(output_path=str(video_path))
            used_url = url_val or dl.get_yn_url()
            with cond:
                state["source_url"] = used_url
            def hook(info: dict):
                with cond:
                    state["download"]["progress"] = info
                    cond.notify_all()
            try:
                with cond:
                    state["download"]["running"] = True
                    state["download"]["last_log"] = "Starting download"
                    state["download"]["progress"] = {"downloaded": 0, "total": 0, "speed": 0, "eta": 0, "status": "starting"}
                    cond.notify_all()
                dl.download_video(url=url_val, progress=hook)
                with cond:
                    p = state["download"]["progress"]
                    p["status"] = "finished"
                    state["download"]["progress"] = p
                    state["download"]["last_log"] = "Download completed"
            except Exception as e:
                with cond:
                    state["download"]["last_log"] = f"Download error: {e}"
            finally:
                with cond:
                    state["download"]["running"] = False
                    cond.notify_all()

            # Phase 2: Extract
            _extract_job(fps_v, q_v, hw_v)

        threading.Thread(target=_job, args=(url, fps_val, q_val, hw), daemon=True).start()
        return True

    @app.get("/prepare")
    def prepare():
        ensure_today_base()
        exists = video_path.exists()
        n_imgs = len(img_files())
        n_thumbs = len(list(thumbs_dir.glob("*.jpg")))
        downloading = state["download"]["running"]
        extracting = state["extract"]["running"]
        disable_all = 'disabled' if (downloading or extracting or state.get('smart_thumbs',{}).get('running') or state.get('processing')) else ''
        # Download progress
        dprog = state["download"]["progress"]
        try:
            dpercent = int((dprog.get("downloaded", 0) * 100) / (dprog.get("total", 0) or 1)) if downloading else (100 if dprog.get("status") == 'finished' else 0)
        except Exception:
            dpercent = 0
        dcounts = f"{dprog.get('downloaded', 0)//(1024*1024)}MB / {max(1, dprog.get('total', 0))//(1024*1024)}MB | eta: {dprog.get('eta', 0)}s"
        # Extract progress
        eprog = state["extract"].get("progress", {"current": 0, "total": 0, "fps": fps})
        try:
            epercent = int((eprog.get("current", 0) * 100) / (eprog.get("total", 0) or 1)) if extracting or (eprog.get("total", 0) > 0) else 0
        except Exception:
            epercent = 0
        ecounts = f"frames: {eprog.get('current', 0)}/{eprog.get('total', 0)} | fps: {eprog.get('fps', fps)}"
        src_url = get_source_url()
        embed = youtube_embed_url(src_url)
        body = f"""
        <div class='card'>
          <div><b>Base</b>: {base_path}</div>
          <div class='mt-1'>
            <span><b>Video</b>: {'OK' if exists else 'MISSING'}</span>
            <span class='muted' style='margin-left:10px;'><b>Extracted</b>: {n_imgs}</span>
            <span class='muted' style='margin-left:10px;'><b>Thumbs</b>: {n_thumbs}</span>
          </div>
        </div>
        <div class='card mt-2'>
          <div class='section-title'>Today's Video</div>
          <div class='muted'>Source: {src_url or 'Not detected yet'}</div>
          {(f"<div class='mt-1'><iframe style='width:100%; aspect-ratio:16/9;' src='{embed}' frameborder='0' allowfullscreen></iframe></div>" if embed else '')}
        </div>
        <div class='card mt-2'>
          <div class='muted'>Download</div>
          <div style='width:100%; background:#eee; border-radius:8px; overflow:hidden; height:14px;'>
            <div id='dlbar' style='width:{dpercent}%; height:14px; background:#1f6feb;'></div>
          </div>
          <div class='muted mt-1' id='dlcounts'>{dcounts}</div>
        </div>
        <div class='card mt-2'>
          <div class='muted'>Extract</div>
          <div style='width:100%; background:#eee; border-radius:8px; overflow:hidden; height:14px;'>
            <div id='exbar' style='width:{epercent}%; height:14px; background:#0aad3d;'></div>
          </div>
          <div class='muted mt-1' id='excounts'>{ecounts}</div>
        </div>
        <div id='prepare' class='card mt-2'>
          <div class='section-title'>Prepare Frames (Download + Extract)</div>
          <form method='post' class='js-post' data-redirect='{url_for('prepare')}' action='{url_for('download_extract')}'>
            <label>Source URL (optional)</label>
            <input name='url' placeholder='Leave empty to auto-detect from site'/>
            <details class='mt-1'>
              <summary>Advanced options</summary>
              <div class='mt-1'>
                <label>FPS:</label>
                <input name='fps' type='number' value='{fps}' min='1' max='60' style='width:90px;'/>
                <label style='margin-left:8px;'>JPEG q:</label>
                <input name='q' type='number' value='5' min='1' max='31' style='width:90px;'/>
                <label style='margin-left:8px;'><input type='checkbox' name='hw' value='1'/> HW Accel</label>
              </div>
            </details>
            <div class='muted mt-1' id='prepstatus'></div>
            <button id='pipelineBtn' type='submit' class='mt-1' {disable_all}>{'Running...' if (downloading or extracting) else 'Prepare Frames'}</button>
          </form>
        </div>
        <script>
          try {{
            const es = new EventSource('{url_for('events')}');
            es.onmessage = (ev) => {{
              try {{
                const s = JSON.parse(ev.data || '{{}}');
                // Download
                const d = s.download || {{}};
                const dbar = document.getElementById('dlbar');
                const dlc = document.getElementById('dlcounts');
                if (dbar && (d.percent !== undefined)) dbar.style.width = (d.percent||0) + '%';
                if (dlc && d.progress) {{
                  const p = d.progress; const mb = x => Math.floor((x||0)/(1024*1024));
                  dlc.textContent = d.running ? `${{mb(p.downloaded)}}MB / ${{mb(p.total)}}MB | eta: ${{p.eta||0}}s` : dlc.textContent;
                }}
                // Extract
                const e = s.extract || {{}};
                const ebar = document.getElementById('exbar');
                const exc = document.getElementById('excounts');
                if (ebar && (e.percent !== undefined)) ebar.style.width = (e.percent||0) + '%';
                if (exc && e.progress) exc.textContent = `frames: ${{e.progress.current}}/${{e.progress.total}} | fps: ${{e.progress.fps}}`;
                // Busy indicator + button text
                const busy = !!(d.running || e.running);
                const btn = document.getElementById('pipelineBtn');
                const pst = document.getElementById('prepstatus');
                if (btn) btn.disabled = busy;
                if (pst) pst.textContent = busy ? (d.running ? 'Starting download...' : 'Preparing...') : '';
                if (btn) btn.textContent = busy ? 'Preparing...' : 'Prepare Frames';
              }} catch {{}}
            }};
          }} catch {{}}
        </script>
        """
        return page(body)

    @app.post("/download_extract")
    def download_extract():
        url = request.form.get("url") or None
        try:
            fps_val = int(request.form.get('fps', fps))
        except Exception:
            fps_val = fps
        try:
            q_val = int(request.form.get('q', 5))
        except Exception:
            q_val = 5
        hw = request.form.get('hw') == '1'

        # Start background pipeline (guard inside helper)
        start_download_extract(url, fps_val, q_val, hw)
        # Redirect to index to show both download and extract progress
        return redirect(url_for("index"))

    @app.post("/download_extract_smart")
    def download_extract_smart():
        url = request.form.get("url") or None
        # Extract opts
        try:
            fps_val = int(request.form.get('fps', fps))
        except Exception:
            fps_val = fps
        try:
            q_val = int(request.form.get('q', 5))
        except Exception:
            q_val = 5
        hw = request.form.get('hw') == '1'
        # Smart thumbs opts
        try:
            step = int(request.form.get('step', state['step_size']))
            window = int(request.form.get('window', 20))
            sharpw = float(request.form.get('sharpw', 0.2))
            mineye = float(request.form.get('mineye', 0.12))
        except Exception:
            step, window, sharpw, mineye = state['step_size'], 20, 0.2, 0.12

        # Guard: do not start if any pipeline phase is running
        if state["download"]["running"] or state["processing"] or state["extract"]["running"] or state["smart_thumbs"]["running"]:
            with cond:
                state["last_log"] = "Busy: cannot start full pipeline now."
            return redirect(url_for("index"))

        def _st_progress(cur: int, total: int, selected: int):
            with cond:
                state["smart_thumbs"]["progress"] = {"current": int(cur), "total": int(total), "selected": int(selected)}
                cond.notify_all()

        def _job(url_val, fps_v, q_v, hw_v, step_v, window_v, sharpw_v, mineye_v):
            # Phase 1: Download
            dl = Downloader(output_path=str(video_path))
            used_url = url_val or dl.get_yn_url()
            with cond:
                state["source_url"] = used_url
            def hook(info: dict):
                with cond:
                    state["download"]["progress"] = info
                    cond.notify_all()
            try:
                with cond:
                    state["download"]["running"] = True
                    state["download"]["last_log"] = "Starting download"
                    state["download"]["progress"] = {"downloaded": 0, "total": 0, "speed": 0, "eta": 0, "status": "starting"}
                    cond.notify_all()
                dl.download_video(url=url_val, progress=hook)
                with cond:
                    p = state["download"]["progress"]
                    p["status"] = "finished"
                    state["download"]["progress"] = p
                    state["download"]["last_log"] = "Download completed"
            except Exception as e:
                with cond:
                    state["download"]["last_log"] = f"Download error: {e}"
            finally:
                with cond:
                    state["download"]["running"] = False
                    cond.notify_all()

            # Phase 2: Extract
            _extract_job(fps_v, q_v, hw_v)

            # Phase 3: Smart thumbs
            with cond:
                state["smart_thumbs"]["running"] = True
                state["smart_thumbs"]["last_log"] = "Starting smart thumbs"
                state["smart_thumbs"]["progress"] = {"current": 0, "total": 0, "selected": 0}
                cond.notify_all()
            try:
                smart_auto_thumbs(
                    str(extract_dir), str(thumbs_dir),
                    step=step_v, window=window_v, sharp_w=sharpw_v, min_eye=mineye_v,
                    progress=_st_progress,
                )
                with cond:
                    p = state["smart_thumbs"]["progress"]
                    state["smart_thumbs"]["last_log"] = "Smart thumbs completed"
                    total = p.get("total", p.get("current", 0))
                    cur = p.get("current", total)
                    state["smart_thumbs"]["progress"] = {"current": int(cur), "total": int(total), "selected": int(p.get("selected", 0))}
            except Exception as e:
                with cond:
                    state["smart_thumbs"]["last_log"] = f"Smart auto thumbs error: {e}"
            finally:
                with cond:
                    state["smart_thumbs"]["running"] = False
                    cond.notify_all()

        t = threading.Thread(target=_job, args=(url, fps_val, q_val, hw, step, window, sharpw, mineye), daemon=True)
        t.start()
        return redirect(url_for("index"))

    @app.post("/download")
    def download():
        url = request.form.get("url") or None
        # Guards: do not start if already running or pipeline is busy
        if state["download"]["running"] or state["processing"] or state["extract"]["running"]:
            with cond:
                state["last_log"] = "Busy: cannot start download now."
            return redirect(url_for("index"))

        def _download_job(url_val):
            dl = Downloader(output_path=str(video_path))
            used_url = url_val or dl.get_yn_url()
            with cond:
                state["source_url"] = used_url
            def hook(info: dict):
                with cond:
                    state["download"]["progress"] = info
                    cond.notify_all()
            try:
                with cond:
                    state["download"]["running"] = True
                    state["download"]["last_log"] = "Starting download"
                    state["download"]["progress"] = {"downloaded": 0, "total": 0, "speed": 0, "eta": 0, "status": "starting"}
                    cond.notify_all()
                dl.download_video(url=url_val, progress=hook)
                with cond:
                    p = state["download"]["progress"]
                    p["status"] = "finished"
                    state["download"]["progress"] = p
                    state["download"]["last_log"] = "Download completed"
            except Exception as e:
                with cond:
                    state["download"]["last_log"] = f"Download error: {e}"
            finally:
                with cond:
                    state["download"]["running"] = False
                    cond.notify_all()

        t = threading.Thread(target=_download_job, args=(url,), daemon=True)
        t.start()
        return redirect(url_for("index"))

    @app.get("/download/status")
    def download_status():
        prog = state["download"].get("progress", {})
        downloaded = int(prog.get("downloaded", 0) or 0)
        total = int(prog.get("total", 0) or 0)
        percent = 0
        try:
            if total > 0:
                percent = int(downloaded * 100 / total)
            elif prog.get("status") == 'finished':
                percent = 100
        except Exception:
            percent = 0
        return jsonify({
            "running": state["download"].get("running", False),
            "progress": {
                "downloaded": downloaded,
                "total": total,
                "speed": prog.get("speed", 0),
                "eta": prog.get("eta", 0),
                "status": prog.get("status", "idle"),
            },
            "percent": percent,
            "message": state["download"].get("last_log", ""),
        })

    def _estimate_total_frames(fps_out: int) -> int:
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return 0
            total_src = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 0
            cap.release()
            if fps_src and fps_src > 0:
                duration = float(total_src) / float(fps_src)
                return max(0, int(duration * fps_out))
            return 0
        except Exception:
            return 0

    def _extract_job(fps_val: int, q_val: int, hw: bool):
        try:
            with cond:
                state["extract"]["running"] = True
                state["extract"]["last_log"] = "Starting extraction"
                state["extract"]["progress"] = {"current": 0, "total": _estimate_total_frames(fps_val), "fps": fps_val}
                cond.notify_all()

            # Cleanup previous images and thumbnails
            for p in extract_dir.glob("*.jpg"):
                try:
                    p.unlink()
                except Exception:
                    pass
            for p in thumbs_dir.glob("*.jpg"):
                try:
                    p.unlink()
                except Exception:
                    pass

            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-threads', '0'
            ]
            if hw:
                cmd += ['-hwaccel', 'auto']
            cmd += [
                '-i', str(video_path),
                '-map', '0:v:0', '-an', '-sn', '-dn',
                '-vf', f'fps={fps_val}',
                '-q:v', str(q_val),
                str(extract_dir / 'img%04d.jpg')
            ]
            proc = subprocess.Popen(cmd)

            # Poll progress by counting files
            while True:
                ret = proc.poll()
                cur = len(list(extract_dir.glob('*.jpg')))
                with cond:
                    pr = state["extract"]["progress"]
                    total = pr.get("total", 0)
                    state["extract"]["progress"] = {"current": int(cur), "total": int(total), "fps": fps_val}
                    cond.notify_all()
                if ret is not None:
                    break
                time.sleep(0.5)

            # Final update
            cur = len(list(extract_dir.glob('*.jpg')))
            with cond:
                pr = state["extract"]["progress"]
                total = pr.get("total", 0)
                state["extract"]["progress"] = {"current": int(cur), "total": int(total), "fps": fps_val}
                state["extract"]["last_log"] = "Extraction completed"
                state["extract"]["running"] = False
                cond.notify_all()
        except Exception as e:
            with cond:
                state["extract"]["last_log"] = f"Extraction error: {e}"
                state["extract"]["running"] = False
                cond.notify_all()

    @app.route("/extract", methods=["GET", "POST"])
    def extract():
        ensure_today_base()
        if request.method == 'POST':
            # Guard: avoid starting extract while download is running
            if state["download"]["running"]:
                with cond:
                    state["last_log"] = "Busy: wait for download to finish."
                return redirect(url_for('extract'))
            try:
                fps_val = int(request.form.get('fps', fps))
            except Exception:
                fps_val = fps
            try:
                q_val = int(request.form.get('q', 5))
            except Exception:
                q_val = 5
            hw = request.form.get('hw') == '1'
            if not state["extract"]["running"]:
                t = threading.Thread(target=_extract_job, args=(fps_val, q_val, hw), daemon=True)
                t.start()
        # Render Extract page with progress bar
        prog = state["extract"]["progress"]
        running = state["extract"]["running"]
        try:
            percent = int(prog.get("current", 0) * 100 / prog.get("total", 0)) if prog.get("total", 0) > 0 else 0
        except Exception:
            percent = 0
        counts = f"frames: {prog.get('current', 0)}/{prog.get('total', 0)} | fps: {prog.get('fps', 0)}"
        body = f"""
        <div class='card'>
          <div class='muted'>Extract Progress</div>
          <div style='width:100%; background:#eee; border-radius:8px; overflow:hidden; height:14px;'>
            <div id='bar' style='width:{percent}%; height:14px; background:#1f6feb;'></div>
          </div>
          <div class='muted mt-1' id='counts'>{counts}</div>
        </div>
        <form method='post' class='js-post mt-2' data-redirect='{url_for('extract')}' action='{url_for('extract')}'>
          <label>FPS:</label>
          <input name='fps' type='number' value='{prog.get('fps', fps)}' min='1' max='60' style='width:80px;'/>
          <label style='margin-left:8px;'>JPEG q:</label>
          <input name='q' type='number' value='5' min='1' max='31' style='width:80px;'/>
          <label style='margin-left:8px;'><input type='checkbox' name='hw' value='1'/> HW Accel</label>
          <button type='submit' class='mt-1'>{'Restart' if running else 'Start'} Extract</button>
        </form>
        <div class='mt-2'>
          <a class='btn secondary' href='{url_for('preprocess')}#manual'>Open Manual Selection</a>
          <a class='btn secondary' href='{url_for('process_run')}' style='margin-left:6px;'>Process</a>
        </div>
        <script>
          async function pollExtract() {{
            try {{
              const r = await fetch('{url_for('extract_status')}');
              const s = await r.json();
              const bar = document.getElementById('bar');
              const counts = document.getElementById('counts');
              if (bar && s.percent !== undefined) {{
                bar.style.width = (s.percent||0) + '%';
              }}
              if (counts && s.progress) {{
                counts.textContent = `frames: ${{s.progress.current}}/${{s.progress.total}} | fps: ${{s.progress.fps}}`;
              }}
              if (s.running) {{
                setTimeout(pollExtract, 2000);
              }}
            }} catch (e) {{
              setTimeout(pollExtract, 3000);
            }}
          }}
          pollExtract();
        </script>
        """
        return page(body)

    @app.get("/extract/status")
    def extract_status():
        prog = state["extract"].get("progress", {"current": 0, "total": 0, "fps": 0})
        try:
            percent = int(prog.get("current", 0) * 100 / prog.get("total", 0)) if prog.get("total", 0) > 0 else 0
        except Exception:
            percent = 0
        return jsonify({
            "running": state["extract"].get("running", False),
            "message": state["extract"].get("last_log", ""),
            "progress": prog,
            "percent": percent,
        })

    # Bounds
    @app.route("/preprocess", methods=["GET", "POST"])
    def preprocess():
        ensure_today_base()
        files = img_files()
        if not files:
            return page("<p>No extracted frames yet. Use 'Prepare Frames' on Home to download and extract first.</p>")
        # Choose preview sample using navigation state
        tfiles = thumb_files()
        nav = state.get("bounds_nav", {"src": "extract", "idx": 0})
        src_choice = nav.get("src", "extract")
        # If thumbs selected but insufficient variety, fall back to extract
        if src_choice == 'thumbs' and len(tfiles) < 3:
            src_choice = 'extract'
        arr = tfiles if src_choice == 'thumbs' else files
        if not arr:
            arr = files
            src_choice = 'extract'
        idxv = max(0, min(int(nav.get("idx", 0)), len(arr)-1))
        sample = arr[idxv]
        sample_src = src_choice
        # Manual thumbs state
        t_idx = max(0, min(state["idx"], len(files)-1))
        t_cur = files[t_idx] if files else None
        hu = int(request.args.get("upper", state["bounds"]["upper"]))
        hl = int(request.args.get("lower", state["bounds"]["lower"]))
        if request.method == 'POST':
            # Current values from form
            try:
                hu = int(request.form.get('upper', hu))
                hl = int(request.form.get('lower', hl))
            except Exception:
                pass
            act = request.form.get('act') or ''
            run_smart = request.form.get('run_smart') == '1'
            # Navigation for bounds preview
            if act in ('b_prev', 'b_next', 'b_src_thumbs', 'b_src_extract'):
                nav = state.get("bounds_nav", {"src": sample_src, "idx": 0})
                nav_step = 50
                if act == 'b_src_thumbs':
                    nav['src'] = 'thumbs'
                    nav['idx'] = 0
                elif act == 'b_src_extract':
                    nav['src'] = 'extract'
                    nav['idx'] = 0
                else:
                    src_local = nav.get('src', 'extract')
                    arr_local = thumb_files() if src_local == 'thumbs' else files
                    # If thumbs selected but too few items, auto-switch to extract
                    if src_local == 'thumbs' and len(arr_local) < 3:
                        src_local = 'extract'
                        nav['src'] = 'extract'
                        arr_local = files
                    n = len(arr_local)
                    if n:
                        if act == 'b_prev':
                            nav['idx'] = max(0, min(n-1, int(nav.get('idx', 0)) - nav_step))
                        elif act == 'b_next':
                            nav['idx'] = max(0, min(n-1, int(nav.get('idx', 0)) + nav_step))
                state['bounds_nav'] = nav
                _save_state_subset()
                return redirect(url_for('preprocess'))
            # Smart options (used only when run_smart)
            try:
                st_step = int(request.form.get('step', state['step_size']))
                st_window = int(request.form.get('window', 20))
                st_sharpw = float(request.form.get('sharpw', 0.2))
                st_mineye = float(request.form.get('mineye', 0.12))
            except Exception:
                st_step, st_window, st_sharpw, st_mineye = state['step_size'], 20, 0.2, 0.12
            if act == 'upper_minus5':
                hu -= 5
            elif act == 'upper_plus5':
                hu += 5
            elif act == 'lower_minus5':
                hl -= 5
            elif act == 'lower_plus5':
                hl += 5
            elif act == 'save' or act == 'save_next':
                # Clamp before saving
                hu = max(0, int(hu))
                hl = max(hu + 1, int(hl))
                state["bounds"] = {"upper": hu, "lower": hl}
                # Persist bounds
                _save_bounds(hu, hl)
                _save_state_subset()
                # Optionally kick off smart thumbs after saving
                if run_smart:
                    def _job():
                        with cond:
                            state["smart_thumbs"]["running"] = True
                            state["smart_thumbs"]["last_log"] = "Starting smart thumbs"
                            state["smart_thumbs"]["progress"] = {"current": 0, "total": 0, "selected": 0}
                            cond.notify_all()
                        try:
                            smart_auto_thumbs(
                                str(extract_dir), str(thumbs_dir),
                                step=st_step, window=st_window, sharp_w=st_sharpw, min_eye=st_mineye,
                                progress=lambda cur, total, selected: (
                                    cond.acquire(),
                                    state["smart_thumbs"].__setitem__("progress", {"current": int(cur), "total": int(total), "selected": int(selected)}),
                                    cond.notify_all(),
                                    cond.release()
                                )
                            )
                            with cond:
                                p = state["smart_thumbs"]["progress"]
                                state["smart_thumbs"]["last_log"] = "Smart thumbs completed"
                                total = p.get("total", p.get("current", 0))
                                cur = p.get("current", total)
                                state["smart_thumbs"]["progress"] = {"current": int(cur), "total": int(total), "selected": int(p.get("selected", 0))}
                        except Exception as e:
                            with cond:
                                state["smart_thumbs"]["last_log"] = f"Smart auto thumbs error: {e}"
                        finally:
                            with cond:
                                state["smart_thumbs"]["running"] = False
                                cond.notify_all()
                    threading.Thread(target=_job, daemon=True).start()
                    return redirect(url_for('preprocess'))
            elif act == 'auto_bounds':
                try:
                    from pilar.utils.image_processor import ImageProcessor as _IP
                    auto = _IP.auto_detect_bounds_from_dir(str(extract_dir))
                except Exception:
                    auto = None
                if auto:
                    hu, hl = int(auto[0]), int(auto[1])
                    hl = max(hu + 1, hl)
                    state["bounds"] = {"upper": hu, "lower": hl}
                    _save_bounds(hu, hl)
                    _save_state_subset()
                else:
                    with cond:
                        state["last_log"] = "Auto bounds failed. Adjust manually."
            elif act == 'auto_bounds_cur':
                try:
                    import cv2 as _cv
                    from pilar.utils.image_processor import ImageProcessor as _IP
                    nav = state.get('bounds_nav', {"src": sample_src, "idx": 0})
                    src_local = nav.get('src', 'thumbs')
                    arr_local = thumb_files() if (src_local == 'thumbs' and thumb_files()) else files
                    i_local = max(0, min(int(nav.get('idx', 0)), len(arr_local)-1))
                    p = arr_local[i_local]
                    img = _cv.imread(str(p))
                    gray = _cv.cvtColor(img, _cv.COLOR_BGR2GRAY)
                    found = _IP._detect_bounds_one(gray)
                except Exception:
                    found = None
                if found:
                    hu, hl = int(found[0]), int(found[1])
                    hl = max(hu + 1, hl)
                    state["bounds"] = {"upper": hu, "lower": hl}
                    _save_bounds(hu, hl)
                    _save_state_subset()
                else:
                    with cond:
                        state["last_log"] = "Auto bounds (current) failed."
            # final clamp (safety for preview values)
            hu = max(0, hu)
            hl = max(hu+1, hl)
        body = f"""
        <div class='card'>
          <div class='section-title'>Subtitle Crop Bounds</div>
          <div class='muted'>Adjust the subtitle area before processing. Use +/- 5px controls.</div>
          <div class='mt-1'>
            <img class='responsive' src='{url_for('bounds_preview', path=sample.name, upper=hu, lower=hl, src=sample_src)}&cb={idxv}' alt='crop preview'/>
            <div class='nav mt-1'>
              <form method='post' style='display:inline;'>
                <button name='act' value='b_prev' type='submit' class='btn secondary'>← Prev 50</button>
                <button name='act' value='b_next' type='submit' class='btn secondary' style='margin-left:6px;'>Next 50 →</button>
                <button name='act' value='b_src_thumbs' type='submit' class='btn secondary' style='margin-left:6px;' {('' if thumb_files() else 'disabled')}>Use Thumbs</button>
                <button name='act' value='b_src_extract' type='submit' class='btn secondary' style='margin-left:6px;'>Use Extract</button>
                <button name='act' value='auto_bounds_cur' type='submit' class='btn secondary' style='margin-left:6px;'>Auto Detect (Current)</button>
                <button name='act' value='auto_bounds' type='submit' class='btn secondary' style='margin-left:6px;'>Auto Detect (Dataset)</button>
              </form>
            </div>
          </div>
          <form method='post' class='mt-2'>
            <div class='grid' style='grid-template-columns: 1fr 1fr; gap:12px;'>
              <div class='card'>
                <div class='muted'>Upper</div>
                <input name='upper' type='number' value='{hu}'/>
                <div class='mt-1'>
                  <button name='act' value='upper_minus5' type='submit' class='btn secondary'>-5</button>
                  <button name='act' value='upper_plus5' type='submit' class='btn secondary' style='margin-left:6px;'>+5</button>
                </div>
              </div>
              <div class='card'>
                <div class='muted'>Lower</div>
                <input name='lower' type='number' value='{hl}'/>
                <div class='mt-1'>
                  <button name='act' value='lower_minus5' type='submit' class='btn secondary'>-5</button>
                  <button name='act' value='lower_plus5' type='submit' class='btn secondary' style='margin-left:6px;'>+5</button>
                </div>
              </div>
            </div>
            <div class='mt-1'>
              <button name='act' value='auto_bounds' type='submit' class='btn secondary'>Auto Detect Bounds</button>
            </div>
            <div class='mt-2'>
              <button name='act' value='save' type='submit' class='btn'>Save</button>
            </div>
          </form>
        </div>
        
        """
        return page(body)

    @app.get("/bounds")
    def bounds():
        # preview cropped region from a sample image (50th if exists, else first)
        files = img_files()
        if not files:
            return page("<p>No extracted frames found. Please extract first.</p>")
        sample = files[min(49, len(files)-1)]
        hu = int(request.args.get("upper", state["bounds"]["upper"]))
        hl = int(request.args.get("lower", state["bounds"]["lower"]))
        # Save if requested
        if request.args.get("save") == "1":
            state["bounds"] = {"upper": hu, "lower": hl}
            _save_bounds(hu, hl)
        body = f"""
        <form method='get'>
          <label>Upper:</label> <input name='upper' type='number' value='{hu}'/>
          <label>Lower:</label> <input name='lower' type='number' value='{hl}'/>
          <button type='submit'>Preview</button>
          <button type='submit' name='save' value='1'>Save</button>
        </form>
        <div class='mt-1'>
           <img class='responsive' src='{url_for('bounds_preview', path=sample.name, upper=hu, lower=hl)}'/>
        </div>
        """
        return page(body)

    @app.get("/bounds/preview")
    def bounds_preview():
        import cv2
        path = request.args.get("path")
        hu = int(request.args.get("upper", 925))
        hl = int(request.args.get("lower", 1020))
        src_type = request.args.get('src', 'extract')
        if src_type == 'thumbs':
            src = str(thumbs_dir / path)
        else:
            src = str(extract_dir / path)
        img = cv2.imread(src)
        if img is None:
            return Response(status=404)
        hu = max(0, min(hu, img.shape[0]-1))
        hl = max(hu+1, min(hl, img.shape[0]))
        crop = img[hu:hl, :]
        tmp = base_path / "_preview.jpg"
        cv2.imwrite(str(tmp), crop)
        return send_file(str(tmp), mimetype='image/jpeg')

    # Thumbs navigation UI
    @app.get("/thumbs")
    def thumbs():
        ensure_today_base()
        files = img_files()
        if not files:
            return page("<p>No extracted frames found. Please extract first.</p>")
        idx = max(0, min(state["idx"], len(files)-1))
        cur = files[idx]
        body = f"""
        <div>
          <p style='font-size:16px;'>Image {idx+1} / {len(files)}</p>
          <img class='responsive' src='{url_for('image', name=cur.name)}' alt='frame {idx+1}'/>
        </div>
        <div class='nav'>
          <a class='btn' href='{url_for('thumb_action', action="keep")}'>Keep + Next (+{state['step_size']})</a>
          <a class='btn secondary' href='{url_for('thumb_action', action="next")}'>Skip 1 →</a>
          <a class='btn secondary' href='{url_for('thumb_action', action="prev")}'>← Back 1</a>
          <a class='btn secondary' href='{url_for('thumb_action', action="skip_step")}'>Skip +{state['step_size']}</a>
        </div>
        <div class='muted'>Hotkeys: K = Keep+Next, J = Next, H = Prev, S = Skip +{state['step_size']}</div>
        <script>
          document.addEventListener('keydown', (e) => {{
            const go = (url) => window.location.href = url;
            if (e.key === 'k' || e.key === 'K') {{ e.preventDefault(); go('{url_for('thumb_action', action="keep")}'); }}
            else if (e.key === 'j' || e.key === 'ArrowRight') {{ e.preventDefault(); go('{url_for('thumb_action', action="next")}'); }}
            else if (e.key === 'h' || e.key === 'ArrowLeft') {{ e.preventDefault(); go('{url_for('thumb_action', action="prev")}'); }}
            else if (e.key === 's' || e.key === 'S') {{ e.preventDefault(); go('{url_for('thumb_action', action="skip_step")}'); }}
          }});
        </script>
        """
        return page(body)

    @app.get("/thumbs/action/<action>")
    def thumb_action(action: str):
        ensure_today_base()
        files = img_files()
        if not files:
            return redirect(url_for("thumbs"))
        idx = max(0, min(state["idx"], len(files)-1))
        if action == "keep":
            copy_thumb(files[idx])
            idx = min(len(files)-1, idx + state["step_size"])
        elif action == "next":
            idx = min(len(files)-1, idx + 1)
        elif action == "prev":
            idx = max(0, idx - 1)
        elif action == "skip_step":
            idx = min(len(files)-1, idx + state["step_size"])
        state["idx"] = idx
        _save_state_subset()
        return redirect(url_for("thumbs"))

    @app.post("/thumbs/auto")
    def auto_thumbs():
        ensure_today_base()
        try:
            step = int(request.form.get("step", state["step_size"]))
            state["step_size"] = step
            _save_state_subset()
        except Exception:
            pass
        files = img_files()
        for i in range(state["step_size"]-1, len(files), state["step_size"]):
            copy_thumb(files[i])
        return redirect(url_for("index"))

    @app.post("/thumbs/auto_smart")
    def auto_thumbs_smart():
        ensure_today_base()
        try:
            step = int(request.form.get("step", state["step_size"]))
            window = int(request.form.get("window", 20))
            sharpw = float(request.form.get("sharpw", 0.2))
            mineye = float(request.form.get("mineye", 0.12))
            state["step_size"] = step
            _save_state_subset()
        except Exception:
            step, window, sharpw, mineye = state["step_size"], 20, 0.2, 0.12

        def _progress(cur: int, total: int, selected: int):
            with cond:
                state["smart_thumbs"]["progress"] = {"current": int(cur), "total": int(total), "selected": int(selected)}
                cond.notify_all()

        def _job():
            with cond:
                state["smart_thumbs"]["running"] = True
                state["smart_thumbs"]["last_log"] = "Starting smart thumbs"
                state["smart_thumbs"]["progress"] = {"current": 0, "total": 0, "selected": 0}
                cond.notify_all()
            try:
                smart_auto_thumbs(
                    str(extract_dir), str(thumbs_dir),
                    step=step, window=window, sharp_w=sharpw, min_eye=mineye,
                    progress=_progress,
                )
                with cond:
                    p = state["smart_thumbs"]["progress"]
                    state["smart_thumbs"]["last_log"] = "Smart thumbs completed"
                    # Ensure totals are set
                    total = p.get("total", p.get("current", 0))
                    cur = p.get("current", total)
                    state["smart_thumbs"]["progress"] = {"current": int(cur), "total": int(total), "selected": int(p.get("selected", 0))}
                # After thumbs are ready, auto-detect bounds (prefer thumbs as source)
                try:
                    from pilar.utils.image_processor import ImageProcessor as _IP
                    auto = _IP.auto_detect_bounds_from_dir(str(thumbs_dir))
                    if not auto:
                        auto = _IP.auto_detect_bounds_from_dir(str(extract_dir))
                except Exception:
                    auto = None
                if auto:
                    u, l = int(auto[0]), int(auto[1])
                    with cond:
                        state["bounds"] = {"upper": u, "lower": l}
                        _save_bounds(u, l)
                        state["last_log"] = f"Smart thumbs completed. Auto bounds: upper={u}, lower={l}"
                        cond.notify_all()
                    _save_state_subset()
            except Exception as e:
                with cond:
                    state["smart_thumbs"]["last_log"] = f"Smart auto thumbs error: {e}"
            finally:
                with cond:
                    state["smart_thumbs"]["running"] = False
                    cond.notify_all()

        threading.Thread(target=_job, daemon=True).start()
        # Redirect back to Pre-Process so user can fine-tune bounds on representative image
        return redirect(url_for("preprocess"))

    @app.get("/smart_thumbs/status")
    def smart_thumbs_status():
        prog = state["smart_thumbs"].get("progress", {"current": 0, "total": 0, "selected": 0})
        cur = int(prog.get("current", 0) or 0)
        tot = int(prog.get("total", 0) or 0)
        try:
            percent = int(cur * 100 / (tot or 1))
        except Exception:
            percent = 0
        return jsonify({
            "running": state["smart_thumbs"].get("running", False),
            "message": state["smart_thumbs"].get("last_log", ""),
            "progress": {"current": cur, "total": tot, "selected": int(prog.get("selected", 0) or 0)},
            "percent": percent,
        })

    @app.post("/workspace/reset")
    def workspace_reset():
        hard = request.form.get('hard') == '1'
        reset_workspace(hard=hard)
        return redirect(url_for('index'))

    @app.get("/image")
    def image():
        ensure_today_base()
        name = request.args.get("name")
        p = extract_dir / name
        if not p.exists():
            return Response(status=404)
        return send_file(str(p), mimetype='image/jpeg')

    # (postprocess page removed; manual + smart selection now lives in Pre-Process)

    # Processing and upload
    def _result_files():
        return sorted([p for p in base_path.glob('*.jpg') if re.match(r"^\d{6}_\d{2}\.jpg$", p.name)])

    def _process_job(fresh: bool = False, fps_val: int = fps, q_val: int = 5, hwaccel: bool = False):
        try:
            state["processing"] = True
            # reset progress
            with cond:
                state["progress"] = {"current": 0, "total": 0, "pages": 0, "added": 0}

            # Define web prompter to replace OpenCV key input
            def prompter(ctx: dict) -> bool:
                import cv2 as _cv
                with cond:
                    seq = state["review"]["seq"] + 1
                    state["review"]["seq"] = seq
                    # Save images for review
                    files = {}
                    files["processed"] = f"_review_processed_{seq}.jpg"
                    files["bin"] = f"_review_bin_{seq}.jpg"
                    files["pre"] = f"_review_pre_{seq}.jpg"
                    files["cur"] = f"_review_cur_{seq}.jpg"
                    files["overlay"] = f"_review_overlay_{seq}.jpg"
                    _cv.imwrite(str(base_path / files["processed"]), ctx["processed_img"])
                    _cv.imwrite(str(base_path / files["bin"]), ctx["cur_bin"])
                    _cv.imwrite(str(base_path / files["pre"]), ctx["pre_img"])
                    _cv.imwrite(str(base_path / files["cur"]), ctx["cur_img"])
                    _cv.imwrite(str(base_path / files["overlay"]), ctx["overlay"])
                    state["review"]["files"] = files
                    state["review"]["metrics"] = {
                        "str_diff": ctx.get("str_diff"),
                        "img_sim": ctx.get("img_sim"),
                        "text_sim": ctx.get("text_sim", ctx.get("str_diff")),
                        "cur_word": ctx.get("cur_word"),
                        "prev_word": ctx.get("prev_word"),
                    }
                    state["review"]["decision"] = None
                    state["review"]["pending"] = True
                    cond.notify_all()
                    # Wait for decision from UI; allow stop to break wait
                    while state["review"]["decision"] is None and not state.get("stop"):
                        cond.wait()
                    if state.get("stop") and state["review"]["decision"] is None:
                        # Auto-mark SAME to skip and resume so we can stop gracefully
                        state["review"]["decision"] = "same"
                    decision = state["review"]["decision"]
                    state["review"]["pending"] = False
                    return True if decision == "same" else False

            def progress(cur: int, total: int, info: dict):
                with cond:
                    state["progress"] = {
                        "current": int(cur),
                        "total": int(total),
                        "pages": int(info.get("pages", 0)),
                        "added": int(info.get("added", 0)),
                    }
                    cond.notify_all()

            # Ensure frames exist (extract if requested or missing)
            need_extract = fresh or (len(list(extract_dir.glob('*.jpg'))) == 0)
            if need_extract:
                _extract_job(fps_val, q_val, hwaccel)
            if len(list(extract_dir.glob('*.jpg'))) == 0:
                with cond:
                    state["last_log"] = "No frames to process. Prepare frames first."
                return

            proc = ImageProcessor(
                video_path=str(video_path),
                extract_dir=str(extract_dir),
                thumbs_dir=str(thumbs_dir),
                no_gui=True,  # Force headless during web processing
                zoom=zoom,
                auto_detection_range=0.5,
                fresh=fresh,
                fps=fps_val,
                prompt_handler=prompter,
                progress_callback=progress,
                ffmpeg_q=q_val,
                ffmpeg_hwaccel=hwaccel,
                ffmpeg_threads=0,
                stop_fn=lambda: bool(state.get("stop")),
            )
            # Apply saved bounds
            proc.height_upper = int(state["bounds"]["upper"])
            proc.height_lower = int(state["bounds"]["lower"])
            # Bounds come from defaults in ImageProcessor unless thumbs influence them.
            proc.process_files()
            if state.get("stop"):
                state["last_log"] = "Processing stopped by user"
            else:
                state["last_log"] = "Processing completed"
        except Exception as e:
            state["last_log"] = f"Processing error: {e}"
        finally:
            state["processing"] = False
            state["stop"] = False
            _save_state_subset()

    @app.route("/process", methods=["GET", "POST"])
    def process_run():
        ensure_today_base()
        # Parse UI options when POSTed
        fresh = False
        fps_val = fps
        q_val = 5
        hwaccel = False
        if request.method == 'POST':
            # Guard: avoid starting processing while download is running
            if state["download"]["running"]:
                with cond:
                    state["last_log"] = "Busy: wait for download to finish."
                return redirect(url_for('process_run'))
            fresh = request.form.get('reextract') == '1'
            try:
                fps_val = int(request.form.get('fps', fps))
            except Exception:
                fps_val = fps
            try:
                q_val = int(request.form.get('q', 5))
            except Exception:
                q_val = 5
            hwaccel = request.form.get('hw') == '1'

        should_start = (request.method == 'POST' and request.form.get('start') == '1')
        if should_start and not state["processing"] and not state["download"]["running"] and not state["extract"]["running"] and not state["smart_thumbs"]["running"]:
            t = threading.Thread(target=_process_job, args=(fresh, fps_val, q_val, hwaccel), daemon=True)
            t.start()
        msg = "Running..." if state["processing"] else state["last_log"]
        # Initialize progress UI based on current state to avoid flashing 0%
        prog = state["progress"]
        try:
            init_percent = int(prog.get("current", 0) * 100 / prog.get("total", 0)) if prog.get("total", 0) > 0 else 0
        except Exception:
            init_percent = 0
        init_counts = f"frames: {prog.get('current', 0)}/{prog.get('total', 0)} | pages: {prog.get('pages', 0)} | added: {prog.get('added', 0)}"
        has_results = len(_result_files()) > 0
        body = f"""
        <p id='pstatus'>Process status: {msg}</p>
        <div class='card'>
          <div class='muted'>Progress</div>
          <div style='width:100%; background:#eee; border-radius:8px; overflow:hidden; height:14px;'>
            <div id='bar' style='width:{init_percent}%; height:14px; background:#1f6feb;'></div>
          </div>
          <div class='muted mt-1' id='counts'>{init_counts}</div>
        </div>
        <div class='nav mt-1'>
          {(f"<a class='btn secondary' href='{url_for('results')}'>Open Results</a>") if has_results else ""}
          {(f"<form method='post' action='{url_for('process_stop')}' class='js-post' data-redirect='{url_for('process_run')}'><button type='submit' class='btn warn' style='margin-left:8px;'>Stop</button></form>") if state.get('processing') else (f"<form method='post' action='{url_for('process_run')}' class='js-post' data-redirect='{url_for('process_run')}'><input type='hidden' name='start' value='1'/><button type='submit' class='btn' style='margin-left:8px;'>Run</button></form>")}
        </div>
        <script>
          try {{
            const es = new EventSource('{url_for('events')}');
            es.onmessage = (ev) => {{
              try {{
                const s = JSON.parse(ev.data || '{{}}');
                const pr = s.process || {{}};
                const bar = document.getElementById('bar');
                const counts = document.getElementById('counts');
                if (bar && (pr.percent !== undefined)) bar.style.width = (pr.percent||0) + '%';
                if (counts && pr.progress) counts.textContent = `frames: ${{pr.progress.current}}/${{pr.progress.total}} | pages: ${{pr.progress.pages}} | added: ${{pr.progress.added}}`;
                const pst = document.getElementById('pstatus');
                if (pst) pst.textContent = (pr.running && (pr.percent||0) === 0) ? 'Process status: Starting...' : 'Process status: ' + (pr.running ? 'Running...' : (s.message||''));
                if (pr.awaiting) {{
                  window.location.href = '{url_for('review_page')}';
                }} else if (!pr.running && (s.results||0) > 0) {{
                  window.location.href = '{url_for('results')}';
                }}
              }} catch {{}}
            }};
          }} catch {{}}
        </script>
        """
        return page(body)

    @app.post("/process/stop")
    def process_stop():
        with cond:
            if state.get("processing"):
                state["stop"] = True
                state["last_log"] = "Stopping..."
                # If waiting for review, release the waiter
                if state["review"].get("pending") and state["review"].get("decision") is None:
                    state["review"]["decision"] = "same"  # auto-skip current prompt
                    state["review"]["pending"] = False
                cond.notify_all()
        return redirect(url_for('process_run'))

    @app.get("/upload")
    def upload_run():
        # Read config
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read('config.ini')
        creds = cfg['Credentials']
        uploader = ImageUploader(
            image_dir=str(base_path),
            url=creds['url'],
            id=creds['id'],
            password=creds['password'],
            no_gui=no_gui,
            driver_path=creds.get('driver_path', 'chromedriver'),
            user_data_dir=creds.get('user_data_dir'),
            profile_directory=creds.get('profile_directory', 'Default'),
        )
        try:
            uploader.upload_images()
            msg = "Upload requested."
        except Exception as e:
            msg = f"Upload failed: {e}"
        return page(f"<p>{msg}</p>")

    @app.get("/status")
    def status():
        prog = state["progress"]
        percent = 0
        try:
            if prog["total"] > 0:
                percent = int(prog["current"] * 100 / prog["total"])
        except Exception:
            percent = 0
        return jsonify({
            "processing": state["processing"],
            "awaiting": state["review"]["pending"],
            "message": state["last_log"],
            "results": len(_result_files()),
            "progress": prog,
            "percent": percent,
        })

    @app.get("/review")
    def review_page():
        rev = state["review"]
        # If nothing is pending, show a live-wait page that auto-navigates when ready
        if not rev["pending"]:
            body = f"""
            <div class='card'>
              <div class='section-title'>Waiting for a review item…</div>
              <div class='muted'>This page will open the comparison automatically.</div>
              <div class='mt-1'>
                <a class='btn secondary' href='{url_for('process_run')}'>Back to Process</a>
                <a class='btn secondary' href='{url_for('results')}' style='margin-left:6px;'>Results</a>
              </div>
            </div>
            <script>
              try {{
                const es = new EventSource('{url_for('events')}');
                es.onmessage = (ev) => {{
                  try {{
                    const s = JSON.parse(ev.data || '{{}}');
                    const pr = s.process || {{}};
                    if (pr.awaiting) {{
                      window.location.reload();
                      return;
                    }}
                    if (!pr.running && (s.results||0) > 0) {{
                      window.location.href = '{url_for('results')}';
                      return;
                    }}
                  }} catch {{}}
                }};
              }} catch {{}}
            </script>
            """
            return page(body)
        f = rev["files"]
        metrics = rev["metrics"]
        body = f"""
        <div class='card'>
          <div><b>Text Similarity:</b> {metrics.get('text_sim'):.3f} &nbsp;|&nbsp; <b>Image Similarity:</b> {metrics.get('img_sim'):.3f}</div>
          <div class='muted mt-1'><b>Prev OCR</b>: {metrics.get('prev_word')}</div>
          <div class='muted'><b>Cur OCR</b>: {metrics.get('cur_word')}</div>
        </div>
        <div class='grid mt-2'>
          <div class='card'><div class='muted'>Processed</div><img class='responsive' src='{url_for('review_image', name=f['processed'])}'/></div>
          <div class='card'><div class='muted'>Binary</div><img class='responsive' src='{url_for('review_image', name=f['bin'])}'/></div>
          <div class='card'><div class='muted'>Prev vs Cur</div><img class='responsive' src='{url_for('review_image', name=f['overlay'])}'/></div>
        </div>
        <form method='post' class='js-post mt-2' data-redirect='{url_for('process_run')}' action='{url_for('review_decide')}'>
          <button name='v' value='same'>Mark as SAME (skip)</button>
          <button name='v' value='diff' style='margin-left:8px;'>Mark as DIFF (add)</button>
        </form>
        """
        return page(body)

    @app.get("/review/img")
    def review_image():
        name = request.args.get('name')
        if not name:
            return Response(status=400)
        p = base_path / name
        if not p.exists():
            return Response(status=404)
        return send_file(str(p), mimetype='image/jpeg')

    @app.post("/review/decide")
    def review_decide():
        v = request.form.get('v')
        if v not in ("same", "diff"):
            return redirect(url_for('review_page'))
        with cond:
            state["review"]["decision"] = v
            cond.notify_all()
        return redirect(url_for('process_run'))

    @app.get("/results")
    def results():
        files = _result_files()
        if not files:
            return page("<p>No results yet. Run processing first.</p>")
        items = []
        names_js = []
        for p in files:
            name = p.name
            names_js.append(name)
            items.append(
                f"<div class='card'>"
                f"  <a href='{url_for('result_image', name=name)}'>"
                f"    <img class='responsive' src='{url_for('result_image', name=name)}' alt='{name}'/>"
                f"  </a>"
                f"  <div style='text-align:center;font-size:12px;color:#555;margin-top:4px;'>{name}</div>"
                f"</div>"
            )
        grid = "".join(items)
        body = f"""
        <div class='mt-2'>
          <a class='btn' href='{url_for('results_download')}'>Download All</a>
          <a class='btn secondary' href='#' onclick='downloadResultsSeq();return false;' style='margin-left:6px;'>Download Each</a>
          <a class='btn secondary' href='{url_for('results_viewer')}' style='margin-left:6px;'>iOS Viewer</a>
          <label style='margin-left:10px;font-size:12px;color:#555;'>
            Delay (ms):
            <input type='number' id='dlDelay' min='0' step='100' value='1200' style='width:90px;padding:4px;'>
          </label>
        </div>
        <script>
          const RESULT_FILES = {names_js};
          function getDownloadDelay() {{
            const el = document.getElementById('dlDelay');
            let v = parseInt((el && el.value) || '1200', 10);
            if (isNaN(v) || v < 0) v = 0;
            return v;
          }}

          // Restore saved delay from localStorage
          document.addEventListener('DOMContentLoaded', () => {{
            const el = document.getElementById('dlDelay');
            if (!el) return;
            const saved = localStorage.getItem('downloadDelayMs');
            if (saved !== null) el.value = saved;
            el.addEventListener('change', () => {{
              localStorage.setItem('downloadDelayMs', getDownloadDelay());
            }});
          }});

          async function downloadResultsSeq() {{
            const base = '{url_for('result_download')}';
            const delay = getDownloadDelay();
            for (const name of RESULT_FILES) {{
              const a = document.createElement('a');
              a.href = base + '?name=' + encodeURIComponent(name);
              a.download = name;
              document.body.appendChild(a);
              a.click();
              a.remove();
              await new Promise(r => setTimeout(r, delay));
            }}
          }}
        </script>
        <div class='grid'>
          {grid}
        </div>
        """
        return page(body)

    @app.get("/results/viewer")
    def results_viewer():
        files = _result_files()
        if not files:
            return page("<p>No results yet.</p>")
        names = [p.name for p in files]
        first = names[0]
        body = f"""
        <div class='card'>
          <div class='muted'>Tip for iOS: Tap and hold the image, then choose 'Save to Photos'.</div>
        </div>
        <div class='mt-2' style='text-align:center;'>
          <div class='muted' id='counter'></div>
          <img id='viewImg' class='responsive' src='{url_for('result_image', name=first)}' alt='{first}'/>
          <div class='nav mt-1' style='justify-content:center;'>
            <a class='btn secondary' href='#' id='prevBtn'>← Prev</a>
            <a class='btn secondary' href='#' id='nextBtn'>Next →</a>
          </div>
        </div>
        <script>
          const NAMES = {names};
          let idx = 0;
          const img = document.getElementById('viewImg');
          const counter = document.getElementById('counter');
          function update() {{
            img.src = '{url_for('result_image')}' + '?name=' + encodeURIComponent(NAMES[idx]);
            counter.textContent = (idx+1) + ' / ' + NAMES.length + ' — ' + NAMES[idx];
          }}
          document.getElementById('prevBtn').addEventListener('click', (e) => {{ e.preventDefault(); idx = (idx-1+NAMES.length)%NAMES.length; update(); }});
          document.getElementById('nextBtn').addEventListener('click', (e) => {{ e.preventDefault(); idx = (idx+1)%NAMES.length; update(); }});
          document.addEventListener('keydown', (e) => {{ if (e.key==='ArrowLeft') document.getElementById('prevBtn').click(); else if (e.key==='ArrowRight') document.getElementById('nextBtn').click(); }});
          update();
        </script>
        """
        return page(body)

    @app.get("/result")
    def result_image():
        name = request.args.get('name')
        if not name:
            return Response(status=400)
        p = base_path / name
        if not p.exists():
            return Response(status=404)
        return send_file(str(p), mimetype='image/jpeg')

    @app.get("/result/download")
    def result_download():
        name = request.args.get('name')
        if not name:
            return Response(status=400)
        p = base_path / name
        if not p.exists():
            return Response(status=404)
        return send_file(str(p), mimetype='image/jpeg', as_attachment=True, download_name=name)

    @app.get("/results/download")
    def results_download():
        import io, zipfile
        files = _result_files()
        if not files:
            return page("<p>No results to download.</p>")
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for p in files:
                # Store with filename only inside zip
                zf.write(p, arcname=p.name)
        mem.seek(0)
        return send_file(mem, mimetype='application/zip', as_attachment=True, download_name=f"results-{ts}.zip")

    @app.post("/shutdown")
    def shutdown():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is not None:
            func()
            return "Shutting down..."
        # Fallback if not running with Werkzeug (terminate after response)
        threading.Timer(0.5, lambda: os._exit(0)).start()
        return "Shutting down..."

    # Start daily scheduler thread once
    try:
        if not state.get('scheduler_started'):
            threading.Thread(target=_daily_scheduler, daemon=True).start()
            state['scheduler_started'] = True
    except Exception:
        pass

    return app
