import os
import shutil
import threading
from pathlib import Path
from datetime import datetime
import time

from flask import Flask, request, send_file, redirect, url_for, Response, jsonify
import subprocess

from pilar.utils.downloader import Downloader
from pilar.utils.image_uploader import ImageUploader
from pilar.utils.image_processor import ImageProcessor


def create_app(base_dir: str, no_gui: bool = True, zoom: int = 112, fps: int = 2, step_size: int = 150):
    app = Flask(__name__)

    # Paths
    base_path = Path(base_dir)
    video_path = base_path / "src.mp4"
    extract_dir = base_path / "extract"
    thumbs_dir = base_path / "thumbs"
    base_path.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(exist_ok=True)
    thumbs_dir.mkdir(exist_ok=True)

    # Simple in-memory state for thumbs navigation
    state = {
        "idx": step_size - 1,  # current index for thumbs selection
        "step_size": step_size,
        "processing": False,
        "last_log": "",
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

    def img_files():
        return sorted([p for p in extract_dir.glob("*.jpg")])

    def copy_thumb(p: Path):
        dest = thumbs_dir / p.name
        shutil.copy2(p, dest)

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
              .btn {{ display:inline-block; padding:10px 12px; border-radius:8px; text-decoration:none; background:#1f6feb; color:#fff; border:none; font-size:14px; }}
              .btn.secondary {{ background:#e9ecef; color:#111; }}
              .btn.warn {{ background:#d9534f; color:#fff; }}
              button {{ padding:10px 12px; border:none; border-radius:8px; background:#1f6feb; color:#fff; font-size:14px; }}
              label {{ font-size:14px; }}
              input[type='text'], input[type='number'] {{ width:100%; padding:10px; border:1px solid #ccc; border-radius:8px; font-size:16px; }}
              img.responsive {{ width:100%; height:auto; }}
              .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:10px; }}
              .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:6px; }}
              .muted {{ color:#666; font-size: 13px; }}
              .mt-1 {{ margin-top:8px; }}
              .mt-2 {{ margin-top:12px; }}
            </style>
            <script>
              function submitForm(form, redirectTo, submitter) {{
                const fd = new FormData(form);
                try {{
                  if (submitter && submitter.name) {{
                    fd.append(submitter.name, submitter.value ?? '');
                  }} else if (document.activeElement && document.activeElement.name) {{
                    fd.append(document.activeElement.name, document.activeElement.value ?? '');
                  }}
                }} catch (e) {{}}
                fetch(form.action, {{ method: 'POST', body: fd }})
                  .then(() => {{
                    if (redirectTo) {{ window.location.href = redirectTo; }}
                    else {{ window.location.reload(); }}
                  }})
                  .catch(() => {{ alert('Request failed'); }});
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
            <div class='container'>
              <h3 style='margin:4px 0 10px;'>Pilar Web Controller</h3>
              <div class='nav'>
                <a class='btn secondary' href='{url_for('index')}'>Home</a>
                <a class='btn secondary' href='{url_for('extract')}'>Extract</a>
                <a class='btn secondary' href='{url_for('bounds')}'>Bounds</a>
                <a class='btn secondary' href='{url_for('thumbs')}'>Thumbs</a>
                <a class='btn secondary' href='{url_for('process_run')}'>Process</a>
                <a class='btn secondary' href='{url_for('results')}'>Results</a>
                <form method='post' class='js-post' action='{url_for('shutdown')}' data-confirm='Shutdown the web server?'>
                  <button type='submit' class='btn warn'>Shutdown</button>
                </form>
              </div>
              {body}
            </div>
          </body>
        </html>
        """

    @app.get("/")
    def index():
        exists = video_path.exists()
        n_imgs = len(img_files())
        n_thumbs = len(list(thumbs_dir.glob("*.jpg")))
        return page(
            f"""
            <div class='card'>
              <div><b>Base</b>: {base_path}</div>
              <div class='mt-1'>
                <span><b>Video</b>: {'OK' if exists else 'MISSING'}</span>
                <span class='muted' style='margin-left:10px;'><b>Extracted</b>: {n_imgs}</span>
                <span class='muted' style='margin-left:10px;'><b>Thumbs</b>: {n_thumbs}</span>
              </div>
            </div>
            <hr/>
            <form method='post' class='js-post' data-redirect='{url_for('index')}' action='{url_for('download')}'>
              <label>YouTube URL (optional):</label><br/>
              <input name='url' placeholder='Leave empty to auto-detect from site'/>
              <button type='submit' class='mt-1'>Download Video</button>
            </form>
            <form method='post' class='js-post' data-redirect='{url_for('extract')}' action='{url_for('extract')}'>
              <label>Extract FPS:</label>
              <input name='fps' type='number' value='{fps}' min='1' max='60' style='width:100px;'/>
              <label style='margin-left:8px;'>JPEG q (1-31):</label>
              <input name='q' type='number' value='5' min='1' max='31' style='width:100px;'/>
              <label style='margin-left:8px;'><input type='checkbox' name='hw' value='1'/> HW Accel</label>
              <button type='submit' class='mt-1'>Extract Frames</button>
            </form>
            <form method='post' class='js-post' data-redirect='{url_for('index')}' action='{url_for('auto_thumbs')}'>
              <label>Auto Thumbs step:</label>
              <input name='step' type='number' value='{state['step_size']}' min='1' max='1000'/>
              <button type='submit' class='mt-1'>Auto Select</button>
            </form>
            <div class='mt-2'>
              <a class='btn secondary' href='{url_for('thumbs')}'>Manual thumbs UI</a>
              <a class='btn secondary' href='{url_for('bounds')}' style='margin-left:6px;'>Bounds UI</a>
            </div>
            <form method='post' class='js-post' data-redirect='{url_for('process_run')}' action='{url_for('process_run')}'>
              <label><input type='checkbox' name='reextract' value='1'/> Re-extract frames</label>
              <label style='margin-left:8px;'>FPS:</label>
              <input name='fps' type='number' value='{fps}' min='1' max='60' style='width:80px;'/>
              <label style='margin-left:8px;'>JPEG q:</label>
              <input name='q' type='number' value='5' min='1' max='31' style='width:80px;'/>
              <label style='margin-left:8px;'><input type='checkbox' name='hw' value='1'/> HW Accel</label>
              <button type='submit' class='mt-1'>Run Processing</button>
            </form>
            <p><a href='{url_for('results')}'>View Results</a></p>
            """
        )

    @app.post("/download")
    def download():
        url = request.form.get("url") or None
        dl = Downloader(output_path=str(video_path))
        try:
            dl.download_video(url=url)
            msg = "Download requested."
        except Exception as e:
            msg = f"Download failed: {e}"
        return redirect(url_for("index"))

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

            # Cleanup previous images
            for p in extract_dir.glob("*.jpg"):
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
        if request.method == 'POST':
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
          <a class='btn secondary' href='{url_for('thumbs')}'>Open Thumbs</a>
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
        files = img_files()
        if not files:
            return page("<p>No extracted frames found. Please extract first.</p>")
        idx = max(0, min(state["idx"], len(files)-1))
        cur = files[idx]
        body = f"""
        <div>
          <p>Image {idx+1} / {len(files)}</p>
          <img class='responsive' src='{url_for('image', name=cur.name)}'/>
        </div>
        <div class='nav'>
          <a class='btn secondary' href='{url_for('thumb_action', action="keep")}'>Keep</a>
          <a class='btn secondary' href='{url_for('thumb_action', action="next")}'>Next</a>
          <a class='btn secondary' href='{url_for('thumb_action', action="prev")}'>Prev</a>
          <a class='btn secondary' href='{url_for('thumb_action', action="skip_step")}'>Skip +{state['step_size']}</a>
        </div>
        """
        return page(body)

    @app.get("/thumbs/action/<action>")
    def thumb_action(action: str):
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
        return redirect(url_for("thumbs"))

    @app.post("/thumbs/auto")
    def auto_thumbs():
        try:
            step = int(request.form.get("step", state["step_size"]))
            state["step_size"] = step
        except Exception:
            pass
        files = img_files()
        for i in range(state["step_size"]-1, len(files), state["step_size"]):
            copy_thumb(files[i])
        return redirect(url_for("index"))

    @app.get("/image")
    def image():
        name = request.args.get("name")
        p = extract_dir / name
        if not p.exists():
            return Response(status=404)
        return send_file(str(p), mimetype='image/jpeg')

    # Processing and upload
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
                        "cur_word": ctx.get("cur_word"),
                    }
                    state["review"]["decision"] = None
                    state["review"]["pending"] = True
                    cond.notify_all()
                    # Wait for decision from UI
                    while state["review"]["decision"] is None:
                        cond.wait()
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
            )
            # Apply saved bounds
            proc.height_upper = int(state["bounds"]["upper"])
            proc.height_lower = int(state["bounds"]["lower"])
            # Bounds come from defaults in ImageProcessor unless thumbs influence them.
            proc.process_files()
            state["last_log"] = "Processing completed"
        except Exception as e:
            state["last_log"] = f"Processing error: {e}"
        finally:
            state["processing"] = False

    @app.route("/process", methods=["GET", "POST"])
    def process_run():
        # Parse UI options when POSTed
        fresh = False
        fps_val = fps
        q_val = 5
        hwaccel = False
        if request.method == 'POST':
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

        if not state["processing"]:
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
        body = f"""
        <p>Process status: {msg}</p>
        <div class='card'>
          <div class='muted'>Progress</div>
          <div style='width:100%; background:#eee; border-radius:8px; overflow:hidden; height:14px;'>
            <div id='bar' style='width:{init_percent}%; height:14px; background:#1f6feb;'></div>
          </div>
          <div class='muted mt-1' id='counts'>{init_counts}</div>
        </div>
        <p class='mt-1'><a class='btn secondary' href='{url_for('results')}'>Open Results</a></p>
        <script>
          async function poll() {{
            try {{
              const r = await fetch('{url_for('status')}');
              const s = await r.json();
              const bar = document.getElementById('bar');
              const counts = document.getElementById('counts');
              if (bar && s.percent !== undefined) {{
                bar.style.width = (s.percent||0) + '%';
              }}
              if (counts && s.progress) {{
                counts.textContent = `frames: ${{s.progress.current}}/${{s.progress.total}} | pages: ${{s.progress.pages}} | added: ${{s.progress.added}}`;
              }}
              if (s.awaiting) {{
                window.location.href = '{url_for('review_page')}';
              }} else if (!s.processing) {{
                window.location.href = '{url_for('results')}';
              }} else {{
                setTimeout(poll, 2000);
              }}
            }} catch(e) {{
              setTimeout(poll, 3000);
            }}
          }}
          setTimeout(poll, 1500);
        </script>
        """
        return page(body)

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
            "results": len(list(base_path.glob('result-*.jpg'))),
            "progress": prog,
            "percent": percent,
        })

    @app.get("/review")
    def review_page():
        rev = state["review"]
        if not rev["pending"]:
            return page("<p>No review pending.</p><p><a href='" + url_for('results') + "'>Results</a></p>")
        f = rev["files"]
        metrics = rev["metrics"]
        body = f"""
        <div class='card'>
          <div><b>str_diff:</b> {metrics.get('str_diff'):.3f} &nbsp;|&nbsp; <b>img_sim:</b> {metrics.get('img_sim'):.3f}</div>
          <div class='muted mt-1'><b>OCR</b>: {metrics.get('cur_word')}</div>
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
        files = sorted(base_path.glob('result-*.jpg'))
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
        </div>
        <script>
          const RESULT_FILES = {names_js};
          async function downloadResultsSeq() {{
            const base = '{url_for('result_download')}';
            for (const name of RESULT_FILES) {{
              const a = document.createElement('a');
              a.href = base + '?name=' + encodeURIComponent(name);
              a.download = name;
              document.body.appendChild(a);
              a.click();
              a.remove();
              await new Promise(r => setTimeout(r, 200));
            }}
          }}
        </script>
        <div class='grid'>
          {grid}
        </div>
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
        files = sorted(base_path.glob('result-*.jpg'))
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

    return app
