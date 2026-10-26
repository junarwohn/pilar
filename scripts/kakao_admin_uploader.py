import argparse
import glob
import os
import re
import sys
import time
import logging
from datetime import datetime
from pilar_logging import configure_logging

logger = logging.getLogger(__name__)

from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchFrameException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

try:
    # Optional: falls back to local driver if available
    from webdriver_manager.chrome import ChromeDriverManager  # type: ignore
    _HAS_WDM = True
except Exception:
    _HAS_WDM = False


_KOREAN_WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
_EDITOR_FIELD_SELECTOR = (
    "textarea#description, textarea[name='description'], "
    "div.tit_tf input.tf_g, input[name='title'], input[placeholder*='제목']"
)


def default_daily_title(now: datetime | None = None) -> str:
    cur = now or datetime.now()
    return f"{cur.strftime('%Y-%m-%d')}-{_KOREAN_WEEKDAYS[cur.weekday()]}"


def _build_driver(headless: bool = False, user_data_dir: str | None = None, profile_directory: str | None = None, driver_path: str | None = None) -> webdriver.Chrome:
    opts = Options()

    # User data dir: prefer explicit path; else create a temp profile directory in CWD to avoid clobbering user's main profile
    if user_data_dir:
        opts.add_argument(f"--user-data-dir={user_data_dir}")
    else:
        temp_profile = os.path.join(os.getcwd(), "chrome_temp")
        opts.add_argument(f"--user-data-dir={temp_profile}")

    if profile_directory:
        opts.add_argument(f"--profile-directory={profile_directory}")

    # Stability flags
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-extensions")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])  # reduce automation banner
    opts.add_experimental_option("useAutomationExtension", False)

    if headless:
        opts.add_argument("--headless=new")

    # Driver resolution priority: explicit path -> system -> webdriver-manager
    service: Service | None = None
    if driver_path and os.path.exists(driver_path):
        service = Service(driver_path)
    else:
        # Try plain constructor first (Selenium Manager/system driver)
        try:
            drv = webdriver.Chrome(options=opts)
            if headless:
                drv.set_window_size(1600, 1100)
            else:
                drv.maximize_window()
            return drv
        except Exception:
            logger.warning("Chrome startup through Selenium Manager failed; trying fallback driver", exc_info=True)
            if _HAS_WDM:
                service = Service(ChromeDriverManager().install())
            else:
                # Re-raise with a clearer message
                raise RuntimeError(
                    "No ChromeDriver found. Provide --driver-path, install a system driver, or install webdriver-manager."
                )

    drv = webdriver.Chrome(service=service, options=opts)
    if headless:
        drv.set_window_size(1600, 1100)
    else:
        drv.maximize_window()
    return drv


def _find_elements(drv, selector: str, text: str | None = None):
    """Find CSS matches across Kakao Business microfrontends' open shadow roots."""
    return drv.execute_script(r"""
        const [selector, text] = arguments;
        const matches = [];
        function visit(root) {
            for (const element of root.querySelectorAll(selector)) {
                if (text === null || element.textContent.replace(/\s+/g, ' ').trim() === text) {
                    matches.push(element);
                }
            }
            for (const element of root.querySelectorAll('*')) {
                if (element.shadowRoot) visit(element.shadowRoot);
            }
        }
        visit(document);
        return matches;
    """, selector, text)


def _wait_visible(wait: WebDriverWait, selectors: list[tuple[str, str]]):
    """Wait for the first matching element, including hidden file inputs."""
    def located(drv):
        for by, selector in selectors:
            elements = (
                _find_elements(drv, selector)
                if by == By.CSS_SELECTOR else drv.find_elements(by, selector)
            )
            if elements:
                return elements[0]
        return False

    return wait.until(located)


def _open_editor(wait: WebDriverWait):
    """Open the composer from the post list, or reuse an already open editor.

    Match the create button by its label, including the nested inner_g span,
    and keep the driver in the frame containing the visible content field.
    """
    create_clicked = False

    def find_in_context(drv):
        nonlocal create_clicked
        for field in _find_elements(drv, _EDITOR_FIELD_SELECTOR):
            if field.is_displayed() and field.is_enabled():
                return field
        if not create_clicked:
            for selector in ("button, a, [role='button']", "span.inner_g"):
                for button in _find_elements(drv, selector, "새 소식 만들기"):
                    if selector == "span.inner_g" and drv.execute_script(
                        "return arguments[0].closest('button, a, [role=button]') !== null", button
                    ):
                        continue
                    if button.is_displayed() and button.is_enabled():
                        button.click()
                        create_clicked = True
                        logger.info("Opened '새 소식 만들기'; waiting for the editor.")
                        return False
        return False

    def editor_ready(drv):
        drv.switch_to.default_content()
        try:
            content_input = find_in_context(drv)
            if content_input:
                return content_input
            for frame in _find_elements(drv, "iframe"):
                try:
                    drv.switch_to.frame(frame)
                    content_input = find_in_context(drv)
                    if content_input:
                        return content_input
                finally:
                    if not content_input:
                        drv.switch_to.default_content()
        except (StaleElementReferenceException, NoSuchFrameException, ElementClickInterceptedException):
            drv.switch_to.default_content()
        return False

    try:
        return wait.until(editor_ready)
    except TimeoutException as exc:
        raise TimeoutException(
            "Kakao post editor did not open. Could not find a visible content field "
            "after looking for '새 소식 만들기'. Check the login session and kakao_url."
        ) from exc


def _select_blog_type(wait: WebDriverWait) -> None:
    """Select 블로그형 before attachments and require its selected state."""
    clicked = False

    def selected(drv):
        nonlocal clicked
        try:
            for tab in _find_elements(drv, "[role='tab']", "블로그형"):
                if not tab.is_displayed():
                    continue
                if tab.get_attribute("aria-selected") == "true":
                    return True
                if not clicked and tab.is_enabled() and tab.get_attribute("aria-disabled") != "true":
                    drv.execute_script("arguments[0].scrollIntoView({block:'center', inline:'center'});", tab)
                    tab.click()
                    clicked = True
        except (StaleElementReferenceException, ElementClickInterceptedException):
            pass
        return False

    try:
        wait.until(selected)
    except TimeoutException as exc:
        raise TimeoutException("Could not select '블로그형'; upload stopped before attaching images.") from exc


def _fill_post_content(wait: WebDriverWait, content: str) -> None:
    """Fill the daily date in blog title and content, or the card content field."""
    _open_editor(wait)
    for field in _find_elements(wait._driver, _EDITOR_FIELD_SELECTOR):
        if field.is_displayed() and field.is_enabled():
            field.clear()
            field.send_keys(content)


def _maybe_login(wait: WebDriverWait, email: str, password: str) -> None:
    """If redirected to Kakao Account login, perform login.

    Tries to be resilient to markup changes by using name-based selectors.
    """
    drv = wait._driver  # type: ignore[attr-defined]
    cur = drv.current_url
    if "accounts.kakao.com" not in cur and "account.kakao.com" not in cur:
        return

    # Inputs
    email_input = _wait_visible(
        wait,
        [
            (By.CSS_SELECTOR, "input[name='loginId']"),
            (By.ID, "loginId--1"),
        ],
    )
    pwd_input = _wait_visible(
        wait,
        [
            (By.CSS_SELECTOR, "input[name='password']"),
            (By.ID, "password--2"),
        ],
    )
    email_input.clear()
    email_input.send_keys(email)
    pwd_input.clear()
    pwd_input.send_keys(password)

    # Submit button
    try:
        btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit'], .btn_g.highlight.submit"))
        )
        btn.click()
    except Exception:
        # Fallback: hit Enter in password field
        from selenium.webdriver.common.keys import Keys

        pwd_input.send_keys(Keys.ENTER)

    # Handle potential 2FA or consent screens: give time to complete
    # If a 2FA challenge is detected, wait up to 90s for redirect away from accounts domain
    WebDriverWait(drv, 90).until(lambda d: "accounts.kakao" not in d.current_url)


def _attach_images(wait: WebDriverWait, image_dir: str) -> int:
    drv = wait._driver  # type: ignore[attr-defined]
    cwd = os.getcwd()

    # The project uses a 6-digit date prefix like YYMMDD_XX.jpg
    pattern = os.path.join(cwd, image_dir, "[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9].jpg")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No images found matching pattern: {pattern}")

    # Locate file input
    file_input = _wait_visible(
        wait,
        [
            (By.CSS_SELECTOR, ".ibox_upload input[type='file'][accept*='image/']"),
            (By.CSS_SELECTOR, "input[type='file'][accept*='image/']"),
            (By.CSS_SELECTOR, "input.custom.uploadInput"),
            (By.CSS_SELECTOR, "input[type='file']"),
        ],
    )

    # Send newline-joined absolute paths to upload multiple files
    file_input.send_keys("\n".join(files))

    # Heuristic: wait until uploads finish. Try to detect thumbnails or no progress spinners.
    try:
        WebDriverWait(drv, 60).until(
            lambda d: not any(
                element.is_displayed()
                for element in _find_elements(d, ".loading, .spinner, .progress")
            )
        )
    except Exception:
        # Best-effort fallback
        logger.warning("Could not confirm attachment progress; waiting before checking registration", exc_info=True)
        time.sleep(8)

    return len(files)


def _already_uploaded_today(drv: webdriver.Chrome, today_title: str) -> bool:
    """Check both the new Business table and legacy Channel post cards."""
    def list_loaded(d):
        if any(field.is_displayed() for field in _find_elements(d, _EDITOR_FIELD_SELECTOR)):
            return True
        # A table skeleton can exist before the rows have arrived. Do not decide
        # that today's post is absent until dated post content is available.
        return any(
            re.search(r"\d{4}-\d{2}-\d{2}", row.text)
            for row in _find_elements(d, ".cont_posts tbody tr, .card_list .box_card, #mArticle")
        )

    try:
        WebDriverWait(drv, 10).until(list_loaded)
    except TimeoutException:
        return False
    return any(
        element.is_displayed()
        for element in _find_elements(drv, "a, strong, .tit_post, .tit_g, .txt_g", today_title)
    )


def _wait_submit_enabled(drv: webdriver.Chrome, timeout: int = 180):
    """Wait until submit button becomes enabled after image upload."""
    selectors = [
        ("button, a, [role='button']", "등록"),
        ("button.btn_g.btn_g2[type='submit']", None),
        ("button[aria-label*='등록'], button[aria-label*='저장']", None),
        ("button[type='submit']", None),
    ]

    end = time.monotonic() + max(1, timeout)
    while time.monotonic() < end:
        for selector, text in selectors:
            try:
                for btn in _find_elements(drv, selector, text):
                    disabled_attr = btn.get_attribute("disabled")
                    class_name = (btn.get_attribute("class") or "").lower()
                    is_disabled = (
                        disabled_attr is not None
                        or "disabled" in class_name
                        or btn.get_attribute("aria-disabled") == "true"
                    )
                    if btn.is_displayed() and btn.is_enabled() and not is_disabled:
                        return btn
            except StaleElementReferenceException:
                continue
        time.sleep(0.5)

    raise TimeoutError("Submit button did not become enabled within timeout") from None


def _wait_upload_complete(drv: webdriver.Chrome, content: str, timeout: int = 60):
    """Require Kakao's publication acknowledgement before reporting success."""
    def published(d):
        if any(el.is_displayed() for el in _find_elements(d, "*", "발행 성공")):
            return True
        # Legacy pages return directly to the post list instead of showing a dialog.
        if _find_elements(d, ".cont_posts tbody tr, .card_list .box_card"):
            return any(
                el.is_displayed()
                for el in _find_elements(d, "a, strong, .tit_post, .tit_g, .txt_g", content)
            )
        return False

    try:
        WebDriverWait(drv, timeout).until(published)
    except TimeoutException as exc:
        raise TimeoutException("Kakao did not confirm publication after clicking '등록'.") from exc


def upload_to_kakao(url: str, email: str, password: str, image_dir: str, headless: bool = False, user_data_dir: str | None = None, profile_directory: str | None = None, driver_path: str | None = None, title: str | None = None, manual_login: bool = False, progress=None) -> dict:
    def report(stage, message, **details):
        logger.info(message, extra={"event": "upload_stage", "stage": stage, **details})
        if progress:
            progress(stage, message, details)

    report("browser", "Starting Chrome for Kakao upload")
    drv = _build_driver(
        headless=headless,
        user_data_dir=user_data_dir,
        profile_directory=profile_directory,
        driver_path=driver_path,
    )
    wait = WebDriverWait(drv, 15)

    try:
        report("login", "Opening Kakao and checking login session")
        drv.get(url)

        today_title = default_daily_title()
        # If we land on Kakao login
        if manual_login:
            # Give the user time to complete Kakao verification/2FA
            try:
                WebDriverWait(drv, 300).until(lambda d: "accounts.kakao" not in d.current_url)
            except TimeoutException:
                logger.exception("Timed out waiting for manual Kakao login")
                raise
        else:
            _maybe_login(wait, email, password)

        report("duplicate_check", "Checking whether today's post already exists")
        if _already_uploaded_today(drv, today_title):
            message = f"Upload skipped: today's post already exists ({today_title})."
            report("skipped", message, title=today_title)
            return {"status": "skipped", "title": today_title, "image_count": 0, "reason": message}

        # Open the composer, attach images, then put the daily date in 내용.
        ttl = title or today_title
        report("editor", "Opening the blog post editor")
        _open_editor(wait)
        _select_blog_type(wait)
        report("attachments", "Attaching processed images")
        count = _attach_images(wait, image_dir)
        report("content", f"Attached {count} images; filling title and content", image_count=count, title=ttl)
        _fill_post_content(wait, ttl)

        # Give UI a short settle time, then wait until submit button is truly enabled.
        time.sleep(5)
        submit_btn = _wait_submit_enabled(drv, timeout=180)
        report("publishing", "Registering blog post and waiting for publication confirmation")
        submit_btn.click()

        _wait_upload_complete(drv, ttl)

        report("completed", f"Uploaded {count} images as blog with content '{ttl}'.", image_count=count, title=ttl)
        return {"status": "completed", "title": ttl, "image_count": count}
    finally:
        try:
            time.sleep(2)
            drv.quit()
        except Exception:
            logger.warning("Could not close Chrome after Kakao upload", exc_info=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload images to KakaoTalk Admin page (Channel).")
    p.add_argument("--url", required=True, help="Kakao admin editor URL to create a post")
    p.add_argument("--email", help="Kakao Account email (login). Omit with --manual-login")
    p.add_argument("--password", help="Kakao Account password. Omit with --manual-login")
    p.add_argument("--image-dir", required=True, help="Directory containing YYMMDD_XX.jpg files")
    p.add_argument("--title", default=None, help="Optional post title (default: YYYY-MM-DD-요일)")

    # Browser options
    p.add_argument("--headless", action="store_true", help="Run Chrome in headless mode")
    p.add_argument("--user-data-dir", default=None, help="Chrome user data dir to reuse session")
    p.add_argument("--profile-directory", default=None, help="Chrome profile directory name (e.g., 'Default')")
    p.add_argument("--driver-path", default=None, help="Path to chromedriver binary (if not using Selenium Manager)")
    p.add_argument("--manual-login", action="store_true", help="Do not auto-fill credentials; wait for manual Kakao login")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    configure_logging()
    args = parse_args(argv)
    if not args.manual_login:
        if not args.email or not args.password:
            print("Error: --email and --password are required unless you use --manual-login", file=sys.stderr)
            return 2
    upload_to_kakao(
        url=args.url,
        email=args.email or "",
        password=args.password or "",
        image_dir=args.image_dir,
        headless=args.headless,
        user_data_dir=args.user_data_dir,
        profile_directory=args.profile_directory,
        driver_path=args.driver_path,
        title=args.title,
        manual_login=args.manual_login,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
