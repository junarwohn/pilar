# yoyakbot_hangul

한글 자막이 있는 영상을 자막에 따라 캡쳐하여 요약할 수 있는 프로그램입니다.

결과물은 한 영상 이미지에 자막이 20줄이 아래로 붙여진 모습이며 줄 개수는 커스텀이 가능합니다.

## Web 모드(모바일 지원)

`main.py --web`로 간단한 Flask 웹 UI가 실행됩니다. 같은 네트워크의 모바일에서도 접속 가능합니다.

- 실행: `python main.py --web --no-gui`
- 스크립트:
  - Windows: `run-web.bat [포트]` (기본 8000, 브라우저 자동 오픈 없음)
  - macOS/Linux: `./run-web.sh [포트]` (기본 8000, 브라우저 자동 오픈 없음)
- 접속: 브라우저에서 `http://<서버IP>:8000`
- 기능:
  - Download: YouTube URL 입력 또는 자동 URL 탐지 후 다운로드
  - Extract: `ffmpeg`로 프레임 추출(FPS 지정)
  - Bounds: 자막 영역 상/하 한계 미리보기로 조정
  - Thumbs: 프레임을 보며 Keep/Prev/Next/Skip 조작 (Auto Select도 지원)
  - Process: OCR 기반 자막 변화 감지로 `YYMMDD_##.jpg` 생성
  - Upload: `config.ini` 설정으로 자동 업로드

주의: Web 모드에서는 OpenCV GUI를 사용하지 않습니다(`--no-gui`). `ffmpeg`, `tesseract(ko)`, Chrome/ChromeDriver, Selenium 등의 환경이 필요합니다.

## Linux 부팅 시 Web UI 자동 실행(systemd)

tmux에서 직접 가상환경을 활성화하고 `run-web.sh`를 실행하는 대신, Linux에서는 `systemd` 서비스로 등록하여 부팅 시 자동 실행할 수 있습니다.

GitHub에는 실제 서버 계정명이나 경로가 들어간 서비스 파일을 올리지 않고, 예시 파일만 관리합니다. 이 저장소에는 `deploy/systemd/pilar-web.service.example` 파일이 포함되어 있습니다.

아래 예시는 프로젝트가 `/home/YOUR_USER/workspace/pilar`에 있고, 가상환경이 프로젝트 내부의 `pilar-venv`에 있는 경우입니다. `YOUR_USER`는 실제 Linux 계정명으로 바꿔서 사용하세요.

환경 파일을 준비합니다.

```bash
cp .env.example .env
nano .env
```

서비스 예시 파일을 시스템 경로로 복사한 뒤 서버 환경에 맞게 수정합니다.

```bash
sudo cp deploy/systemd/pilar-web.service.example /etc/systemd/system/pilar-web.service
sudo sed -i "s#YOUR_USER#$(whoami)#g" /etc/systemd/system/pilar-web.service
sudo nano /etc/systemd/system/pilar-web.service
```

서비스 파일 안의 `YOUR_USER`가 모두 실제 Linux 계정명으로 바뀌었는지 확인하고, 프로젝트 경로가 다르면 함께 수정합니다. `.env` 파일에는 기본적으로 아래 값들이 들어가며, 서비스 실행 시 `run-web.sh`가 이 값을 읽습니다.

```bash
PORT=8000
RESULT_RETENTION_DAYS=7
PILAR_CLEANUP_OLD_RESULTS=1
```

`PILAR_CLEANUP_OLD_RESULTS=1`이면 Web UI 실행 시 `out/` 아래에서 `RESULT_RETENTION_DAYS`보다 오래된 결과 파일을 삭제하고, 비어 있는 결과 폴더도 정리합니다. 자동 삭제를 끄고 싶으면 `PILAR_CLEANUP_OLD_RESULTS=0`으로 변경하세요.

서비스를 등록하고 실행합니다.

```bash
sudo systemctl daemon-reload
sudo systemctl enable pilar-web
sudo systemctl start pilar-web
```

상태와 로그는 아래 명령으로 확인할 수 있습니다.

```bash
systemctl status pilar-web
journalctl -u pilar-web -f
```

`status=203/EXEC` 오류가 나오면 `ExecStart`에 적힌 파일을 systemd가 실행하지 못한 상태입니다. 보통 서비스 파일에 `YOUR_USER`가 남아 있거나, 경로가 틀렸거나, `run-web.sh`에 실행 권한이 없을 때 발생합니다.

```bash
systemctl cat pilar-web
ls -l /home/$(whoami)/workspace/pilar/run-web.sh
chmod +x /home/$(whoami)/workspace/pilar/run-web.sh
sudo systemctl daemon-reload
sudo systemctl restart pilar-web
```

중지 또는 재시작이 필요할 때는 아래 명령을 사용합니다.

```bash
sudo systemctl stop pilar-web
sudo systemctl restart pilar-web
```

서비스 실행 후 같은 네트워크의 브라우저에서 `http://<서버IP>:8000`으로 접속합니다.

포트를 변경하려면 `.env`의 `PORT` 값을 수정한 뒤 서비스를 재시작합니다.

```bash
sudo systemctl restart pilar-web
```

## 날짜 기반 폴더 생성 도구

비디오 다운로드 등 날짜별로 정리할 수 있도록, 날짜(오늘 기준) 폴더를 만들어 경로를 출력하는 간단한 CLI를 추가했습니다.

- 실행: `python3 scripts/mkdaily.py` → `out/video/YYYY/MM/DD` 폴더 생성 후 경로 출력
- 옵션:
  - `--base /path/to/videos`: 기준 폴더 지정(기본 `out/video`)
  - `--style yyyy/mm/dd` 또는 `yyyy-mm-dd`: 날짜 폴더 스타일 선택
  - `--date 2025-01-02`: 오늘 대신 특정 날짜로 생성
  - `--scaffold`: 내부에 `raw`, `clips`, `thumbs`, `meta` 하위 폴더도 함께 생성
  - `--print-env`: `export VIDEO_DIR=...` 라인도 함께 출력(쉘에서 바로 `eval` 가능)

예시:

```
# 기본 사용(오늘 날짜로 out/video/YYYY/MM/DD 생성)
python3 scripts/mkdaily.py

# 평면 스타일과 사용자 비디오 폴더 사용
python3 scripts/mkdaily.py --base ~/Videos --style yyyy-mm-dd

# yt-dlp와 함께 사용하여 저장 경로 지정
yt-dlp -P "$(python3 scripts/mkdaily.py --base ~/Videos)" <URL>

# 쉘 환경변수로 내보내기
eval "$(python3 scripts/mkdaily.py --base ~/Videos --print-env)"
```
