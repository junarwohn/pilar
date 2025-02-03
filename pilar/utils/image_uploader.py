from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
import re
import tempfile
import configparser
import glob
import time
from datetime import datetime

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
            
        self.chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
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
        # get current YY-MM-DD
        current_date = datetime.now().strftime("%Y-%m-%d")  
        # Read config file
        url = self.url
        driver = self.setup_driver()
        driver.get(url)  # Use the URL from config

        wait = WebDriverWait(driver, 10)

        # 2. [제목] 입력
        # 'div.tit_tf' 내부의 input.tf_g 요소를 찾아 제목 입력
        title_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.tit_tf input.tf_g"))
        )
        title_input.clear()
        title_input.send_keys(f"{current_date}")

        # 3. [사진 첨부]
        # 파일 첨부 input 요소는 'input.custom.uploadInput' 클래스를 사용하고 있음
        photo_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input.custom.uploadInput"))
        )
        # find all jpg in out/current_day/result-#.jpg
        cwd = os.getcwd()
        jpg_list = glob.glob(os.path.join(cwd, self.image_dir, "result-*.jpg"))

        # 첨부할 사진의 전체 경로를 지정 (예시: Windows 경로)
        photo_input.send_keys("\n".join(jpg_list))

        # 4. [등록] 버튼 클릭
        # 제출 버튼은 'button' 요소 중 type="submit"을 가지며, 클래스 'btn_g btn_g2'를 사용
        submit_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn_g.btn_g2[type='submit']"))
        )

        # 버튼에 disabled 속성이 있다면, 제거하여 클릭 가능하게 만듦
        if submit_button.get_attribute("disabled"):
            driver.execute_script("arguments[0].removeAttribute('disabled');", submit_button)

        submit_button.click()

        time.sleep(10)
        driver.quit()
