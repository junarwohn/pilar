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
