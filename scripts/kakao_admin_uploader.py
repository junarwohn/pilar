import argparse
import glob
import os
import shutil
import sys
import time
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

_KOREAN_WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]


def default_daily_title(now: datetime | None = None) -> str:
    cur = now or datetime.now()
    return f"{cur.strftime('%Y-%m-%d')}-{_KOREAN_WEEKDAYS[cur.weekday()]}"


def _find_browser_binary() -> str | None:
    for name in ("chromium-browser", "chromium", "google-chrome"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _build_driver(headless: bool = False, user_data_dir: str | None = None, profile_directory: str | None = None, driver_path: str | None = None) -> webdriver.Chrome:
    opts = Options()

    # Keep Pilar's browser session separate from the user's normal Chromium profile.
    if user_data_dir:
        profile_path = os.path.abspath(os.path.expanduser(user_data_dir))
    else:
        profile_path = os.path.join(os.path.expanduser("~"), ".config", "pilar-chromium")
    os.makedirs(profile_path, exist_ok=True)
    opts.add_argument(f"--user-data-dir={profile_path}")
    opts.add_argument("--password-store=basic")

    if profile_directory:
        opts.add_argument(f"--profile-directory={profile_directory}")

    # Stability flags
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-extensions")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])  # reduce automation banner
    opts.add_experimental_option("useAutomationExtension", False)

    if headless:
        opts.add_argument("--headless=new")

    browser_binary = _find_browser_binary()
    if browser_binary:
        opts.binary_location = browser_binary

    if driver_path:
        resolved_driver = os.path.abspath(os.path.expanduser(driver_path))
        if not os.path.isfile(resolved_driver):
            raise FileNotFoundError(f"ChromeDriver not found: {resolved_driver}")
    else:
        resolved_driver = shutil.which("chromedriver")

    # Raspberry Pi packages ship a matching Chromium/ChromeDriver pair. Prefer
    # that pair and let Selenium Manager run only when no system driver exists.
    print(f"- browser: {browser_binary or 'Selenium Manager default'}")
    print(f"- driver: {resolved_driver or 'Selenium Manager'}")
    print(f"- browser_profile: {profile_path}")
    service = Service(resolved_driver) if resolved_driver else None
    drv = webdriver.Chrome(service=service, options=opts) if service else webdriver.Chrome(options=opts)
    try:
        drv.maximize_window()
    except Exception:
        # Some Raspberry Pi window managers do not implement maximize.
        try:
            drv.set_window_size(1280, 900)
        except Exception:
            pass
    return drv


def _is_kakao_login_url(url: str) -> bool:
    return "accounts.kakao.com" in url or "account.kakao.com" in url


def _wait_visible(wait: WebDriverWait, selectors: list[tuple[str, str]]):
    """Try multiple selectors, return the first located WebElement."""
    last_exc = None
    for by, sel in selectors:
        try:
            el = wait.until(EC.presence_of_element_located((by, sel)))
            return el
        except Exception as e:
            last_exc = e
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("No selector provided")


def _maybe_login(wait: WebDriverWait, email: str, password: str) -> None:
    """If redirected to Kakao Account login, perform login.

    Tries to be resilient to markup changes by using name-based selectors.
    """
    drv = wait._driver  # type: ignore[attr-defined]
    cur = drv.current_url
    if not _is_kakao_login_url(cur):
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
    WebDriverWait(drv, 90).until(lambda d: not _is_kakao_login_url(d.current_url))


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
            (By.CSS_SELECTOR, "input.custom.uploadInput"),
            (By.CSS_SELECTOR, "input[type='file']"),
        ],
    )

    # Send newline-joined absolute paths to upload multiple files
    file_input.send_keys("\n".join(files))

    # Heuristic: wait until uploads finish. Try to detect thumbnails or no progress spinners.
    try:
        WebDriverWait(drv, 60).until_not(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".loading, .spinner, .progress"))
        )
    except Exception:
        # Best-effort fallback
        time.sleep(8)

    return len(files)


def _already_uploaded_today(drv: webdriver.Chrome, today_title: str) -> bool:
    """Return True if card list already contains today's title."""
    container_xpath = "//*[@id='mArticle']/div/div/div[2]/div/div"
    try:
        WebDriverWait(drv, 10).until(
            EC.presence_of_element_located((By.XPATH, container_xpath))
        )
    except Exception:
        # If list area is not present (e.g. directly on editor), do not block upload.
        return False

    # Primary selector from provided structure
    title_xpath = f"{container_xpath}/div/div[2]/div[2]/div[1]/strong"
    try:
        titles = drv.find_elements(By.XPATH, title_xpath)
        for el in titles:
            txt = (el.text or "").strip()
            if txt == today_title:
                return True
    except Exception:
        pass

    # Fallback selector by classes under card_list/box_card
    try:
        cards = drv.find_elements(By.CSS_SELECTOR, "div.card_list div.box_card")
        for card in cards:
            try:
                strong = card.find_element(By.CSS_SELECTOR, "strong")
                txt = (strong.text or "").strip()
                if txt == today_title:
                    return True
            except Exception:
                continue
    except Exception:
        pass

    return False


def _wait_submit_enabled(drv: webdriver.Chrome, timeout: int = 180):
    """Wait until submit button becomes enabled after image upload."""
    selectors = [
        "button.btn_g.btn_g2[type='submit']",
        "button[type='submit']",
        "button[aria-label*='등록'], button[aria-label*='저장']",
    ]

    end = time.time() + max(1, timeout)
    last_btn = None
    while time.time() < end:
        for sel in selectors:
            try:
                btn = drv.find_element(By.CSS_SELECTOR, sel)
                last_btn = btn
                disabled_attr = btn.get_attribute("disabled")
                class_name = (btn.get_attribute("class") or "").lower()
                is_disabled = (disabled_attr is not None) or ("disabled" in class_name)
                if btn.is_enabled() and not is_disabled:
                    return btn
            except Exception:
                continue
        time.sleep(0.5)

    raise TimeoutError("Submit button did not become enabled within timeout") from None


def upload_to_kakao(url: str, email: str, password: str, image_dir: str, headless: bool = False, user_data_dir: str | None = None, profile_directory: str | None = None, driver_path: str | None = None, title: str | None = None, manual_login: bool = False) -> None:
    drv = _build_driver(
        headless=headless,
        user_data_dir=user_data_dir,
        profile_directory=profile_directory,
        driver_path=driver_path,
    )
    wait = WebDriverWait(drv, 25)

    try:
        drv.get(url)
        time.sleep(10)

        today_title = default_daily_title()
        if _already_uploaded_today(drv, today_title):
            print(f"Skip upload: today's post already exists ({today_title}).")
            return

        # If we land on Kakao login
        if manual_login:
            # Give the user time to complete Kakao verification/2FA
            try:
                WebDriverWait(drv, 300).until(lambda d: not _is_kakao_login_url(d.current_url))
            except Exception:
                pass
        else:
            _maybe_login(wait, email, password)

        # Ensure we are at the editor page
        # If the site uses an iframe-based editor, switch if needed (best-effort)
        try:
            iframes = drv.find_elements(By.TAG_NAME, "iframe")
            for f in iframes:
                try:
                    drv.switch_to.frame(f)
                    # If title field exists in this frame, stay; else revert
                    _ = drv.find_elements(By.CSS_SELECTOR, "div.tit_tf input.tf_g, input[name='title']")
                    if _:
                        break
                    drv.switch_to.default_content()
                except Exception:
                    drv.switch_to.default_content()
        except Exception:
            pass

        # Title input
        ttl = title or today_title
        title_input = _wait_visible(
            wait,
            [
                (By.CSS_SELECTOR, "div.tit_tf input.tf_g"),
                (By.CSS_SELECTOR, "input[name='title']"),
                (By.CSS_SELECTOR, "input[placeholder*='제목']"),
            ],
        )
        try:
            title_input.clear()
        except Exception:
            pass
        title_input.send_keys(ttl)

        # Attach images
        count = _attach_images(wait, image_dir)

        # Give UI a short settle time, then wait until submit button is truly enabled.
        time.sleep(20)
        submit_btn = _wait_submit_enabled(drv, timeout=180)
        submit_btn.click()

        # User requested immediate close behavior after click.
        time.sleep(20)

        print(f"Uploaded {count} images with title '{ttl}'.")
    finally:
        try:
            time.sleep(2)
            drv.quit()
        except Exception:
            pass


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
