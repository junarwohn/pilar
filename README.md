# Pilar

영상 프레임의 한글 자막을 OCR로 읽고, 자막 변화를 기준으로 대표 이미지와 자막 영역을 세로로 합쳐 요약 이미지를 만듭니다. Flask Web UI에서 다운로드·처리·검토·카카오 소식 발행을 수행합니다. 이전 프로젝트 이름은 `yoyakbot_hangul`입니다.

이 문서는 **현재 Python Web 서비스의 실행·구조·장애 분석 기준**입니다. 구현을 분석할 때 아래 진입점과 실제 코드를 함께 읽으세요. SRT/VTT 자막 다운로드·파싱은 현재 파이프라인에 포함되지 않습니다.

## 1. 실행 구조 요약

| 항목 | 현재 동작 |
|---|---|
| 운영 서비스 | systemd `pilar-web.service` → `run-web.sh` → Gunicorn → `pilar.web.wsgi:app` |
| 개발 서버 | `python main.py --web --no-gui` → Flask 개발 서버 |
| 기본 포트 | `8000`, Web UI 주소 `http://<서버IP>:8000` |
| 동시 실행 | Gunicorn worker **1개**, gthread 8개. 상태와 예약 스레드가 프로세스 내부에 있으므로 worker 수를 늘리면 안 됩니다. |
| 자동 실행 | 서버 로컬 시간 매일 **06:00**, 다운로드·추출 → 이미지 처리 → 업로드. 자동 업로드는 `manual_login=false` 필요 |
| 업로드 | **블로그형** 선택 → 이미지 첨부 → 제목·내용에 `YYYY-MM-DD-요일` 입력 → 등록 → 발행 성공 확인 |
| 중복 방지 | 소식 목록에 오늘 날짜 제목이 있으면 `skipped` 처리 |
| 시간 기준 | 결과 폴더·예약·이력 파일명은 서버 로컬 날짜, JSON 로그의 시각은 UTC |
| 외부 알림 | 웹훅·메신저 전송은 미구현. 현재는 로그·작업 이력·Web UI로 확인 |

```text
영상 URL → Downloader → src.mp4 → ffmpeg → extract/*.jpg
         → 자막 영역/썸네일 선택 → PaddleOCR → YYMMDD_##.jpg
         → Results → POST /upload → Selenium → 카카오 블로그형 소식
```

## 2. 코드 탐색 지도

| 파일 | 역할 / 확인할 내용 |
|---|---|
| [pilar/web/server.py](pilar/web/server.py) | `create_app()`, Web 라우트, 작업 스레드, 예약 실행, 상태 저장, 업로드 UI |
| [pilar/web/wsgi.py](pilar/web/wsgi.py) | 운영 WSGI 진입점, 로깅 초기화, 기본 결과 폴더 |
| [scripts/kakao_admin_uploader.py](scripts/kakao_admin_uploader.py) | `upload_to_kakao()`, 로그인 세션, Shadow DOM 탐색, 블로그형 선택, 첨부·중복·발행 확인 |
| [pilar_logging.py](pilar_logging.py) | JSON 로그, 요청·실행 ID 전파, `JobHistory`, 이력 보관 정책 |
| [gunicorn.conf.py](gunicorn.conf.py) | HTTP 접근 로그, 응답의 요청 ID, worker 로깅 초기화 |
| [pilar/utils/downloader.py](pilar/utils/downloader.py) | 자동 영상 URL 감지, yt-dlp 업데이트·다운로드, 부분 파일 정리 |
| [pilar/utils/image_processor.py](pilar/utils/image_processor.py) | 자막 영역·유사도 판단·이미지 조합·처리 CSV |
| [pilar/utils/ocr_engine.py](pilar/utils/ocr_engine.py) | PaddleOCR 초기화·실행·OCR 오류 기록 |
| [pilar/utils/smart_thumbs.py](pilar/utils/smart_thumbs.py) | 대표 썸네일 선택. MediaPipe가 없으면 선명도 기반으로 동작 |
| [upload_pics.py](upload_pics.py) | 기존 결과 이미지만 독립적으로 업로드하는 CLI |
| [run-web.sh](run-web.sh) | 운영 실행, 가상환경 활성화, 시작 시 오래된 결과 정리 |
| [deploy/systemd/pilar-web.service.example](deploy/systemd/pilar-web.service.example) | 설치 템플릿. 실제 설정은 `systemctl cat pilar-web`로 확인 |

별도 실행 경로를 구분하세요. `python main.py`의 기존 GUI 파이프라인은 `config.ini`의 `[Credentials]`를 사용합니다. 현재 Web/독립 업로더는 `user_config.ini`의 `[global]`을 사용합니다. `run-cli.sh`는 이름과 달리 Python Web 개발 서버를 실행하며, `run.sh`·`src/`·`c-porting/`은 별도의 C++ 실행 경로입니다.

## 3. 환경 준비와 실행

명령은 저장소 루트에서 실행합니다. Python 패키지는 [requirements.txt](requirements.txt)를 기준으로 설치합니다. 외부 실행 파일로 `ffmpeg`와 Chrome 및 호환 ChromeDriver가 필요합니다. 기본 OCR 엔진은 PaddleOCR입니다.

```bash
# 새 환경에서만 가상환경 생성·패키지 설치
python3 -m venv pilar-venv
source pilar-venv/bin/activate
python -m pip install -r requirements.txt

# 기존 설정을 덮어쓰지 않고 템플릿 복사
cp -n res/user_config.ini user_config.ini
cp -n .env.example .env
```

운영 방식으로 직접 실행할 때:

```bash
# .env는 run-web.sh가 직접 읽지 않습니다. 직접 실행 시 값을 export합니다.
set -a
source .env
set +a
./run-web.sh
# 포트를 인자로 지정할 수도 있습니다: ./run-web.sh 8001
```

`run-web.sh`는 시작할 때 기본적으로 오래된 `out/` 파일을 정리합니다. 개발 서버만 실행하려면 다음을 사용합니다.

```bash
source pilar-venv/bin/activate
python main.py --web --no-gui --host 0.0.0.0 --port 8000
```

Windows의 `run-web.bat [포트]`도 Flask 개발 서버를 실행합니다. Web 모드는 OpenCV GUI를 사용하지 않습니다.

### systemd 설치·운영

```bash
sudo cp deploy/systemd/pilar-web.service.example /etc/systemd/system/pilar-web.service
sudoedit /etc/systemd/system/pilar-web.service
```

템플릿의 `YOUR_USER`, 프로젝트 경로, 가상환경 `PATH`를 실제 환경에 맞춥니다. `WorkingDirectory`는 저장소 루트여야 합니다. `.env`는 서비스의 `EnvironmentFile`로 주입됩니다.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now pilar-web
systemctl status pilar-web --no-pager
```

운영 변경 시 진행 중인 작업을 먼저 확인합니다. 서비스 재시작은 작업을 중단할 수 있으며, 업로드가 중단됐다면 카카오 소식 목록을 확인한 후 재시도합니다.

```bash
curl --fail --max-time 5 http://127.0.0.1:8000/status
systemctl cat pilar-web

# .env 변경 후 적용
sudo systemctl restart pilar-web

# 서비스 유닛 변경 후 적용
sudo systemctl daemon-reload
sudo systemctl restart pilar-web
```

현재 유닛에는 `ExecReload`가 없습니다. `systemctl reload`를 가정하지 마세요. `status=203/EXEC`이면 `ExecStart` 경로·`YOUR_USER` 치환·`run-web.sh` 실행 권한부터 확인합니다.

## 4. 설정과 데이터 위치

### 카카오 설정: `user_config.ini`

```ini
[global]
kakao_url = https://business.kakao.com/<채널 식별자>/posts
headless = true
manual_login = false
```

GUI가 없는 systemd 환경에서는 `headless=true`를 사용합니다. 로그인 세션이 없는 경우 `email`, `password` 또는 GUI에서 최초 로그인·인증이 필요합니다. 기본 Chrome 프로필은 실행 디렉터리의 `chrome_temp/`이며, `user_data_dir`, `profile_directory`, `driver_path`, `title`도 선택적으로 지정할 수 있습니다. `title`은 블로그형 제목과 내용에 사용할 날짜 문자열을 대체합니다.

Web의 `headless` 기본값은 `true`, 독립 업로더의 기본값은 `false`입니다. 두 경로 모두 `manual_login` 기본값은 `true`이므로 서버 운영 시 값을 명시하세요. 독립 CLI는 자동 로그인 모드에서 `email/password` 설정을 요구합니다.

기존 이미지의 독립 업로드 명령은 다음과 같습니다. **실제 외부 게시물을 발행하는 명령**입니다.

```bash
source pilar-venv/bin/activate
python upload_pics.py --headless --image-dir out/YY-MM-DD
```

### 환경 변수: `.env` 또는 export

| 변수 | 기본값 | 용도 |
|---|---|---|
| `PORT` | `8000` | 운영 Web 포트. `run-web.sh` 인자가 우선 |
| `PILAR_NO_GUI` | `true` | WSGI가 읽어 전달하는 값. 현재 Web 처리 코드는 `no_gui=True`를 강제하며 카카오 `headless`와 별개 |
| `RESULT_RETENTION_DAYS` | `7` | 실행 스크립트 시작 시 파일 수정 시각 기준 오래된 결과 정리 |
| `PILAR_CLEANUP_OLD_RESULTS` | `1` | `0`이면 위 정리 비활성화 |
| `PYTHONUNBUFFERED` | 실행 스크립트가 `1` 지정 | Python 출력 버퍼링 방지 |
| `PILAR_LOG_LEVEL` | `INFO` | 공통 애플리케이션 로그 레벨 |
| `PILAR_LOG_DIR` | 저장소의 `logs/` | 작업 이력 디렉터리. 운영 시 절대 경로 권장 |
| `PILAR_LOG_RETENTION_DAYS` | `30` | 새 날짜의 이력을 쓸 때 오래된 이력 파일 정리 |
| `PILAR_OCR_DEBUG` | `0` | `1`이면 PaddleOCR 상세 로그 출력 |
| `PILAR_YTDLP_AUTO_UPGRADE` | 활성화 | `0`이면 다운로드 전 yt-dlp 업데이트 생략 |
| `PILAR_YTDLP_UPGRADE_CMD` | `uv pip install -U --pre "yt-dlp[default]"` | 다운로드 전 패키지 업데이트 명령 |

`PILAR_BASE_DIR`은 WSGI의 시작 결과 경로를 대체하지만 Web 서버는 날짜 변경 시 그 상위 디렉터리 아래 오늘 날짜 폴더로 전환합니다. 고정 폴더로 계속 유지하는 설정은 아닙니다.

```text
out/
  state.json                       # UI 설정·마지막 상태·업로드 결과 일부
  bounds.json                      # 자막 영역 설정
  YY-MM-DD/
    src.mp4                        # 원본 영상
    extract/img*.jpg               # 추출 프레임
    thumbs/*.jpg                   # 선택 썸네일
    YYMMDD_01.jpg ...               # 최종 업로드 이미지
    process-YYYYMMDD-HHMMSS.csv     # OCR/유사도 판단 기록
    words.txt                      # 채택한 자막
logs/
  jobs-YYYY-MM-DD.jsonl             # 작업 시작·진행·종료 이력
chrome_temp/                       # 로그인 세션이 포함된 Chrome 프로필
```

`out/` 결과 정리와 작업 이력 30일 보관은 별개입니다. journal의 보관 기간·용량은 호스트의 journald 정책을 따릅니다.

## 5. 상태 API와 로그 스키마

| 경로 | 의미 |
|---|---|
| `GET /status` | 처리 상태·결과 개수·업로드 상태 스냅샷 |
| `GET /events` | SSE 진행 상태. 연결이 유지되므로 수집 시 시간 제한 필요 |
| `GET /jobs?stage=upload&limit=20` | 최근 **종료된** 작업 이력. `stage` 생략 시 전체, `limit` 최대 100 |
| `GET /results` | 결과 이미지, Upload 버튼, 현재 업로드 상태·최근 이력 |
| `POST /upload` | 업로드 작업 시작. 응답 `302`는 발행 성공이 아니라 요청 처리·이동을 의미 |

운영 journal에는 애플리케이션 JSON, Gunicorn 접근·기동 로그, 외부 라이브러리의 일반 텍스트가 섞입니다. 전체 로그를 JSON이라고 가정하면 안 됩니다. 직접 실행한 개발 서버·CLI의 출력은 별도로 수집하지 않는 한 `pilar-web` journal에 들어가지 않습니다.

애플리케이션 로그 예시이며 ID와 내용은 설명용입니다.

```json
{"time":"2026-10-26T13:00:00.000+00:00","level":"ERROR","logger":"pilar.web.server","message":"Kakao upload failed","pid":1234,"request_id":"example-request","run_id":"example-run","stage":"upload","event":"job_error","exception":"Traceback ...\nRuntimeError: example failure"}
```

| 필드 | 분석 시 의미 |
|---|---|
| `time`, `level`, `message`, `logger`, `pid` | UTC 발생 시각, 논리 로그 레벨, 내용, 코드 모듈, 프로세스 |
| `request_id` | HTTP 요청과 백그라운드 작업 연결. 응답 헤더 `X-Request-ID`와 동일. 요청이 없으면 `-` |
| `run_id`, `parent_run_id` | 작업 실행 식별자와 상위 작업. `parent_run_id`는 작업 이력에서 확인 |
| `stage` | 작업 단계. `prepare`, `download`, `extract`, `smart_thumbs`, `process`, `upload`, `daily` |
| `event` | `http_request`, `job_started`, `job_error`, `job_finished` 등 |
| `status` | 최종 결과: `completed`, `failed`, `skipped`, `stopped`, `interrupted`. 진행 중은 `running` |
| `exception` | journal의 JSON 로그에 들어가는 전체 Python 예외 호출 경로 |
| `error`, `reason` | 작업 이력의 오류 요약 또는 건너뛰기 사유. 전체 traceback을 대체하지 않음 |
| `started_at`, `finished_at`, `duration_seconds` | 작업 이력의 시작·종료 시각·소요 시간 |

업로드의 세부 단계는 `browser`, `login`, `duplicate_check`, `editor`, `attachments`, `content`, `publishing`, `completed`, `skipped`입니다. 애플리케이션 `event=upload_stage` 로그에서는 `stage`, 작업 이력의 `event=job_progress`에서는 `upload_stage`에 기록합니다.

## 6. 에러 로그 조회와 수집

### 바로 확인

```bash
# 실시간 원문 / 오늘 전체 기록
journalctl -u pilar-web -f -o cat
journalctl -u pilar-web --since today --no-pager --all -o short-iso

# 실패 시각 주변 구간 지정. 시각에는 실제 서버 시간대 또는 오프셋 사용
journalctl -u pilar-web --since '2026-10-26 06:00:00 +0900' --until '2026-10-26 06:10:00 +0900' --no-pager --all -o short-iso
```

journal 조회 권한이 없으면 `sudo journalctl`을 사용합니다. **`journalctl -p err`만으로 애플리케이션 오류를 추리지 마세요.** journal 우선순위와 JSON 내부의 `level`은 다를 수 있습니다. 우선 해당 시간대 전체를 수집한 뒤 `level=ERROR/CRITICAL`을 분석합니다.

### AI 분석용 진단 묶음 만들기

아래 명령은 상태 조회와 로그 복사만 수행합니다. 먼저 시간 범위·포트·처리 날짜를 장애 상황에 맞춥니다. 예시는 Linux/bash이며 Python 표준 라이브러리만 사용합니다.

```bash
source pilar-venv/bin/activate
pilar_diag_dir="/tmp/pilar-diagnostics-$(date +%Y%m%d-%H%M%S)"
pilar_diag_since="30 minutes ago"
pilar_diag_until="$(date --iso-8601=seconds)"
pilar_diag_url="http://127.0.0.1:${PORT:-8000}"
pilar_diag_day="$(date +%y-%m-%d)"
mkdir -p "$pilar_diag_dir"

# journal 원문과 journal 자체의 JSON 봉투를 모두 보존
journalctl -u pilar-web --since "$pilar_diag_since" --until "$pilar_diag_until" --all --no-pager -o short-iso > "$pilar_diag_dir/journal.txt"
journalctl -u pilar-web --since "$pilar_diag_since" --until "$pilar_diag_until" --all --no-pager -o json > "$pilar_diag_dir/journal.jsonl"
systemctl status pilar-web --no-pager --full > "$pilar_diag_dir/service-status.txt" 2>&1 || true

# 서버가 내려갔다면 조회 실패 자체를 기록하고 journal 분석을 계속합니다.
curl --fail --silent --show-error --connect-timeout 2 --max-time 5 "$pilar_diag_url/status" > "$pilar_diag_dir/status.json" 2> "$pilar_diag_dir/status-request.stderr" || true
curl --fail --silent --show-error --connect-timeout 2 --max-time 5 "$pilar_diag_url/jobs?limit=100" > "$pilar_diag_dir/recent-jobs.json" 2> "$pilar_diag_dir/jobs-request.stderr" || true

# JSON 내부 MESSAGE를 파싱하고, 작업 이력과 해당 날짜의 처리 CSV를 복사
# 서비스에 별도 PILAR_LOG_DIR을 지정했다면 두 번째 인자에 실제 경로를 사용합니다.
python - "$pilar_diag_dir" "${PILAR_LOG_DIR:-logs}" "out/$pilar_diag_day" <<'PY'
import json
from pathlib import Path
import shutil
import sys

dest, history_dir, day_dir = map(Path, sys.argv[1:])
records = []
for line in (dest / "journal.jsonl").read_text(encoding="utf-8").splitlines():
    try:
        envelope = json.loads(line)
        if not isinstance(envelope, dict):
            continue
        record = json.loads(envelope.get("MESSAGE", ""))
    except (ValueError, TypeError):
        continue
    if isinstance(record, dict) and "level" in record:
        record["journal_time_us"] = envelope.get("__REALTIME_TIMESTAMP")
        records.append(record)

def save_jsonl(name, items):
    with (dest / name).open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl("application.jsonl", records)
save_jsonl("errors.jsonl", [r for r in records if r.get("level") in ("ERROR", "CRITICAL")])
failed = []
for source in sorted(history_dir.glob("jobs-*.jsonl")):
    shutil.copy2(source, dest / source.name)
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("event") == "job_finished" and item.get("status") in ("failed", "interrupted"):
            failed.append(item)
save_jsonl("failed-jobs.jsonl", failed)
for source in day_dir.glob("process-*.csv"):
    shutil.copy2(source, dest / source.name)
state_file = day_dir.parent / "state.json"
if state_file.exists():
    shutil.copy2(state_file, dest / "state.json")
print(dest)
PY
```

수집 결과:

| 파일 | 용도 |
|---|---|
| `journal.txt`, `journal.jsonl` | 기동 실패·native stderr·Gunicorn을 포함한 원본. JSON이 아닌 줄도 보존 |
| `application.jsonl`, `errors.jsonl` | 앱 JSON 전체와 ERROR/CRITICAL 추출. `exception`에 traceback |
| `jobs-*.jsonl`, `failed-jobs.jsonl` | 보관 중인 작업 이력과 실패·중단의 최종 결과. journal과 수집 기간이 다를 수 있음 |
| `status.json`, `recent-jobs.json`, `state.json` | 현재·최근·저장된 상태. API 응답이 비어 있으면 해당 `.stderr` 확인 |
| `process-*.csv`, `service-status.txt` | 자막 선택 상세와 서비스 상태 |

공유할 때는 발생 시각·시간대, 누른 버튼 또는 자동 실행 여부, 기대 결과·실제 결과, `run_id`를 진단 묶음과 함께 전달합니다. `.env`, `user_config.ini`, Chrome 프로필·쿠키는 묶음에 포함하지 않습니다. 로그에 포함된 URL·자막 등은 공유 전에 확인하세요.

### 분석 순서와 오판 방지

1. `failed-jobs.jsonl` 또는 `/jobs`에서 실패·중단된 실행의 `run_id`, `stage`, 시각을 찾습니다. 실행 중인 작업은 `/status`나 원본의 `job_started`를 확인합니다.
2. `application.jsonl`에서 같은 `run_id`를 따라 `job_started` → 진행 단계 → `job_error` → `job_finished`를 읽습니다. `request_id`로 요청을, `parent_run_id`로 상위 예약/준비 작업을 연결합니다.
3. 전체 원인은 `errors.jsonl`의 `exception`과 `journal.txt`에서 확인합니다. 같은 실행의 `job_error`와 `job_finished`를 별도 장애 두 건으로 세지 않습니다.
4. `HTTP 200/302`는 백그라운드 작업 성공을 뜻하지 않습니다. `skipped`는 중복 건너뛰기이며 실패가 아닙니다. `interrupted`는 카카오에서 발행 여부를 확인한 후 재시도합니다.
5. 계획된 서비스 갱신 직후 Paddle의 `SIGTERM`/`FatalError`가 보이면 Gunicorn 기동 기록과 이후 상태 응답을 함께 확인합니다. 이 경우와 예기치 않은 종료를 구분해야 합니다.
6. 로그가 없으면 시간대·수집 구간·서비스 이름·journal 권한·별도 `PILAR_LOG_DIR`을 확인합니다. 원본을 남긴 채 분석하며, 원인 수집 전에 재시작·재처리·재업로드하지 않습니다.

## 7. 검증 명령

```bash
source pilar-venv/bin/activate
python -m unittest discover -s tests -v
python -m py_compile pilar_logging.py gunicorn.conf.py pilar/web/server.py scripts/kakao_admin_uploader.py
bash -n run-web.sh
```

`test_kakao_admin_uploader.py`는 로컬 HTML과 Chrome/ChromeDriver로 새 레이아웃·Shadow DOM·블로그형 흐름을 검증하며 실제 카카오에 발행하지 않습니다. Chrome/ChromeDriver가 없으면 해당 브라우저 검사는 건너뜁니다. `test_logging.py`, `test_web_upload_logging.py`는 예외·요청 ID·보관 기간·업로드 결과·재시작 복원을 검증합니다. 테스트가 의도적으로 만든 오류 출력은 최종 테스트 성공/실패 결과와 구분하세요.
