from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
from datetime import datetime
from scripts.kakao_admin_uploader import (
    _attach_images, _fill_post_content, _open_editor, _select_blog_type,
    _wait_submit_enabled, _wait_upload_complete,
)

class ImageUploader:
    def __init__(self, image_dir, url, id, password, no_gui=False, driver_path="chromedriver", user_data_dir=None, profile_directory="Default"):
        self.image_dir = image_dir
        self.url = url
        self.id = id
        self.password = password
        self.no_gui = no_gui
        self.driver_path = driver_path
        
        # Setup Chrome options
        self.chrome_options = Options()
        
        # Use existing Chrome profile
        if user_data_dir is None:
            if os.name == 'posix':  # Linux
                user_data_dir = os.path.expanduser('~/.config/google-chrome')
            else:  # Windows
                user_data_dir = os.path.expanduser('~') + r'\AppData\Local\Google\Chrome\User Data'
            
        # self.chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
        temp_profile = os.path.join(os.getcwd(), "chrome_temp")
        self.chrome_options.add_argument(f'--user-data-dir={temp_profile}')

        self.chrome_options.add_argument(f'--profile-directory={profile_directory}')
        
        # Add arguments for stable automation
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--disable-extensions')
        
        # Prevent automation detection
        self.chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        self.chrome_options.add_experimental_option('useAutomationExtension', False)
        
        if self.no_gui:
            self.chrome_options.add_argument('--headless=new')
        
        # Setup Chrome service
        self.service = Service(ChromeDriverManager().install())
        
    def setup_driver(self):
        """Setup Chrome WebDriver with configured options"""
        driver = webdriver.Chrome(
            service=self.service,
            options=self.chrome_options
        )
        if self.no_gui:
            driver.set_window_size(1600, 1100)
        else:
            driver.maximize_window()
        return driver
    
    # Current not working
    # Just use the login button
    # TODO: Fix this
    def login(self, driver):
        # Wait for login form elements
        wait = WebDriverWait(driver, 10)
        
        # Find and fill email field
        email_input = wait.until(
            EC.presence_of_element_located((By.ID, "loginId--1"))
        )
        email_input.send_keys(self.id)
        
        # Find and fill password field
        password_input = wait.until(
            EC.presence_of_element_located((By.ID, "password--2"))
        )
        password_input.send_keys(self.password)
        
        # Click login button
        login_button = wait.until(
            EC.element_to_be_clickable((By.CLASS_NAME, "btn_g.highlight.submit"))
        )
        login_button.click()
        
    def upload_images(self):
        # get current YYYY-MM-DD-<Korean weekday>
        now = datetime.now()
        weekday_kor = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"][now.weekday()]
        current_date = f"{now.strftime('%Y-%m-%d')}-{weekday_kor}"
        # Read config file
        url = self.url
        driver = self.setup_driver()
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 15)
            _open_editor(wait)
            _select_blog_type(wait)
            _attach_images(wait, self.image_dir)
            _fill_post_content(wait, current_date)
            _wait_submit_enabled(driver).click()
            _wait_upload_complete(driver, current_date)
        finally:
            driver.quit()
