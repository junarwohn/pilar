# Pilar 기능 정리

**Quickstart Flow**
1. `python3 main.py --web --no-gui` 또는 `./run-web.sh`로 웹 UI를 시작합니다.
2. 웹 UI에서 YouTube URL을 입력(또는 자동 감지 URL 사용)하고 Download를 실행합니다.
3. Extract를 실행해 `ffmpeg`로 프레임을 날짜 폴더에 생성합니다.
4. Bounds/Pre-Process 단계에서 자막 영역(upper/lower)을 조정합니다.
5. Thumbs에서 수동 선택 또는 Smart Thumbs로 자동 선택을 진행합니다.
6. Process를 실행해 OCR 기반 자막 변화 감지 후 페이지 이미지를 생성합니다.
7. 결과를 리뷰하거나 확정하고, 날짜 폴더 내 최종 stitched 이미지를 확인합니다.
8. (옵션) `config.ini` 자격 증명을 사용해 Selenium 업로드를 실행합니다.

**영상 획득 (Python 다운로드 + 자동 URL)**
What it does: YouTube 영상을 다운로드하고, URL이 없으면 특정 사이트에서 YouTube iframe URL을 자동 감지합니다.
Where it lives: `pilar/utils/downloader.py`, `main.py`, `pilar/web/server.py`.
How to run it: `python3 main.py`(자동 URL) 또는 `python3 main.py --web --no-gui` 후 UI에서 Download 실행.
Inputs/Outputs: 입력 URL(옵션); 출력은 `out/YY-MM-DD/src.mp4`.
Notes: 자동 URL 감지는 `youngnak.net`에 하드코딩되어 있으며, 페이지 구조가 바뀌면 실패합니다. URL을 찾지 못하면 다운로드를 건너뜁니다.

**자막 획득 (다운로드/파싱)**
What it does: SRT/VTT 등 자막 파일의 다운로드나 파싱은 구현되어 있지 않습니다.
Where it lives: N/A. 자막 크롭/OCR 관련 코드만 존재합니다.
How to run it: N/A.
Inputs/Outputs: N/A.
Notes: 자막 텍스트는 외부 자막 파일이 아니라 프레임 OCR로부터 얻습니다.

**프레임 추출 / 샘플링 (Python ffmpeg)**
What it does: `ffmpeg`로 지정 FPS에 맞춰 프레임을 추출합니다. HW 가속, 품질/스레드 옵션을 지원합니다.
Where it lives: `pilar/utils/image_processor.py` (`extract_frames`).
How to run it: `python3 main.py` 또는 웹 UI의 Extract 단계.
Inputs/Outputs: 입력 `out/YY-MM-DD/src.mp4`; 출력 `out/YY-MM-DD/extract/img%04d.jpg`.
Notes: HW 가속 실패 시 소프트웨어 디코딩으로 폴백합니다. `fresh=True`인 경우에만 새로 추출합니다.

**OCR / 텍스트 처리 (PaddleOCR + 휴리스틱 + SVM)**
What it does: 자막 크롭 영역에 OCR을 수행하고, 정규화/한글 필터링 후 유사도+SVM 투표로 자막 변경 여부를 판정합니다.
Where it lives: `pilar/utils/ocr_engine.py`, `pilar/utils/image_processor.py`, `svm_models.sav`, `classifier.py`(학습 스크립트).
How to run it: `python3 main.py`(Process) 또는 웹 UI의 Process 단계.
Inputs/Outputs: 입력 `out/YY-MM-DD/extract/*.jpg`; 출력 `out/YY-MM-DD/YYMMDD_##.jpg` 및 `out/YY-MM-DD/words.txt`.
Notes: `paddleocr`와 `paddlepaddle`(CPU) 설치가 필요합니다. 동일 이미지 해시 기준 캐시를 사용해 OCR 중복을 줄입니다.

**자막 영역 감지 / 크롭 (Python)**
What it does: 샘플 프레임에서 자막 밴드 영역을 자동 추정할 수 있으며, GUI로 상/하 bounds를 수동 조정합니다. 처리 시 해당 영역으로 크롭합니다.
Where it lives: `pilar/utils/image_processor.py` (자동 감지 유틸, `get_bounds`).
How to run it: `python3 main.py`(GUI 모드) 또는 웹 UI Bounds 단계(수동 조정, `bounds.json`에 저장).
Inputs/Outputs: 입력 `out/YY-MM-DD/extract/*.jpg`; bounds는 상위 폴더의 `out/bounds.json`에 저장됩니다.
Notes: `--no-gui`에서는 `get_bounds`가 비활성화됩니다. 자동 감지는 존재하지만 별도 CLI로 노출되어 있지는 않습니다.

**썸네일 선택 / 페이지(스티치) 생성**
What it does: 대표 썸네일을 선택한 뒤, 헤더 + 자막 크롭을 세로로 합쳐 `YYMMDD_##.jpg` 페이지 이미지를 생성합니다.
Where it lives: `pilar/utils/image_processor.py`(선택/조합), `pilar/utils/smart_thumbs.py`(자동 선택).
How to run it: 웹 UI Thumbs/Process 단계 또는 `python3 main.py`(GUI 경로).
Inputs/Outputs: 입력 `out/YY-MM-DD/extract/*.jpg` 및 `out/YY-MM-DD/thumbs/*.jpg`; 출력 `out/YY-MM-DD/YYMMDD_##.jpg`.
Notes: Smart Thumbs는 MediaPipe Face Mesh가 있으면 눈뜸 점수를 활용하고, 없으면 선명도 기반 점수로 폴백합니다.

**웹 UI / API (Flask + Gunicorn)**
What it does: 다운로드, 추출, bounds 조정, 썸네일 선택(수동/자동), 처리, 업로드까지 브라우저에서 수행하는 UI를 제공합니다. 진행 상태 스트리밍을 포함합니다.
Where it lives: `pilar/web/server.py`, `pilar/web/wsgi.py`.
How to run it: `python3 main.py --web --no-gui` 또는 `./run-web.sh [PORT]`.
Inputs/Outputs: 날짜 폴더 `out/YY-MM-DD`를 사용하며, 상위에 `out/state.json`, `out/bounds.json`을 저장합니다.
Notes: 서버는 날짜 변경 시 자동 롤오버되며, 매일 06:00에 자동 파이프라인을 시도합니다. 상태가 메모리에 있으므로 워커는 1개만 사용해야 합니다.

**자동화 / 스크립트 (CLI 헬퍼, 업로더, 빌드)**
What it does: 날짜 폴더 생성, 카카오 업로드, C++ 빌드/실행을 위한 보조 스크립트를 제공합니다.
Where it lives: `scripts/mkdaily.py`, `scripts/kakao_admin_uploader.py`, `scripts/kakao_upload_button.py`, `build.sh`, `run.sh`, `run-web.sh`, `run-cli.sh`.
How to run it: `python3 scripts/mkdaily.py`, `python3 scripts/kakao_admin_uploader.py ...`, `./build.sh`, `./run.sh ...`, `./run-web.sh [PORT]`.
Inputs/Outputs: `mkdaily.py`는 `out/video/YYYY/MM/DD` 또는 `yyyy-mm-dd` 폴더를 생성(옵션으로 하위 폴더)합니다. 카카오 스크립트는 이미지 디렉터리에서 `YYMMDD_##.jpg` 패턴 파일을 업로드합니다.
Notes: `run-cli.sh`는 이름과 달리 웹 UI를 실행합니다. 업로드 스크립트는 Selenium + Chrome/ChromeDriver에 의존하며, `main.py`에서 참조하는 `config.ini`는 리포에 없습니다.

**독립 텍스트 감지 (EAST 모델, 미통합)**
What it does: OpenCV EAST 텍스트 디텍터로 이미지 하단 영역의 텍스트 박스를 찾습니다.
Where it lives: `opencv_text_detection_image.py`, `frozen_east_text_detection.pb`.
How to run it: `python3 opencv_text_detection_image.py --image_path <dir> --east frozen_east_text_detection.pb`.
Inputs/Outputs: 입력 폴더의 이미지; 결과는 화면에 박스가 표시되며 파일 저장은 하지 않습니다.
Notes: 메인 파이프라인과 연결되어 있지 않은 실험용 스크립트입니다.
