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
  - Process: OCR 기반 자막 변화 감지로 `result-*.jpg` 생성
  - Upload: `config.ini` 설정으로 자동 업로드

주의: Web 모드에서는 OpenCV GUI를 사용하지 않습니다(`--no-gui`). `ffmpeg`, `tesseract(ko)`, Chrome/ChromeDriver, Selenium 등의 환경이 필요합니다.
