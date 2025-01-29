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

class ImageUploader:
    def __init__(self, image_dir, url, id, password, no_gui=False):
        self.image_dir = image_dir
        self.url = url
        self.id = id
        self.password = password
        self.no_gui = no_gui
        
        # Setup Chrome options
        self.chrome_options = Options()
        
        # Use existing Chrome profile with Pilar profile
        user_data_dir = os.path.expanduser('~/.config/google-chrome')
        self.chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
        self.chrome_options.add_argument('--profile-directory=Default')  # 프로필 디렉토리 이름 수정
        
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
        # Initialize Chrome driver with service
        driver = webdriver.Chrome(
            service=self.service,
            options=self.chrome_options
        )
        
        try:
            # Navigate to login page
            driver.get(self.url)
            a = input()
            # Perform login
            self.login(driver)
            a = input()
            # Wait for successful login
            wait = WebDriverWait(driver, 10)
            wait.until(
                EC.url_contains("center-pf.kakao.com")
            )
            print("Login successful")
            a = input()
            
            # TODO: Implement upload logic
            
        finally:
            driver.quit() 