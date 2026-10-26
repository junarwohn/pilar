import html
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from scripts import kakao_admin_uploader as uploader


@unittest.skipUnless(
    shutil.which("google-chrome") and shutil.which("chromedriver"),
    "Chrome and ChromeDriver are required for the local browser regression tests",
)
class KakaoEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        cls.driver = webdriver.Chrome(
            service=Service(shutil.which("chromedriver")), options=options
        )

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()

    def load_page(self, markup):
        self.driver.get("data:text/html;charset=utf-8," + quote(markup))

    def open_editor(self):
        return uploader._open_editor(WebDriverWait(self.driver, 3, poll_frequency=0.05))

    def composer_page(self, tag="button", delay=0, editor_markup=None, shadow=False):
        editor_markup = editor_markup or '<div class="tit_tf"><input class="tf_g"></div>'
        return f"""
        <{tag} id="create">
          <span class="inner_g"><span class="ico_comm ico_plus_s"></span>새 소식 만들기</span>
        </{tag}>
        <script>
          window.createClicks = 0;
          const root = (() => {{
            if (!{json.dumps(shadow)}) return document.body;
            const button = document.getElementById('create');
            const host = document.createElement('div');
            document.body.appendChild(host);
            const outer = host.attachShadow({{mode: 'open'}});
            const inner = document.createElement('div');
            outer.appendChild(inner);
            const root = inner.attachShadow({{mode: 'open'}});
            root.appendChild(button);
            return root;
          }})();
          root.querySelector('#create').onclick = function(event) {{
            window.createClicks++;
            window.clickedTag = event.currentTarget.tagName;
            setTimeout(() => {{
              root.innerHTML = {json.dumps(editor_markup)};
            }}, {delay});
          }};
        </script>
        """

    def test_new_nested_button_opens_editor_once(self):
        self.load_page(self.composer_page(delay=150))
        field = self.open_editor()
        field.send_keys("2026-10-26-월요일")
        self.assertEqual(field.get_attribute("value"), "2026-10-26-월요일")
        self.assertEqual(self.driver.execute_script("return window.createClicks"), 1)
        self.assertEqual(self.driver.execute_script("return window.clickedTag"), "BUTTON")

    def test_link_and_standalone_span_are_supported(self):
        for tag in ("a", "span"):
            with self.subTest(tag=tag):
                self.load_page(self.composer_page(tag=tag))
                self.assertTrue(self.open_editor().is_displayed())

    def test_role_button_is_supported(self):
        self.load_page(self.composer_page(tag="div").replace('<div id="create">', '<div id="create" role="button">'))
        self.assertTrue(self.open_editor().is_displayed())
        self.assertEqual(self.driver.execute_script("return window.clickedTag"), "DIV")

    def test_already_open_editor_does_not_click_create(self):
        for markup in (
            '<div class="tit_tf"><input class="tf_g"></div>',
            '<input name="title">',
            '<input placeholder="제목을 입력하세요">',
            '<textarea id="description" name="description"></textarea>',
        ):
            with self.subTest(markup=markup):
                self.load_page(self.composer_page() + markup)
                self.assertTrue(self.open_editor().is_displayed())
                self.assertEqual(self.driver.execute_script("return window.createClicks"), 0)

    def test_hidden_title_and_hidden_duplicate_button_are_ignored(self):
        self.load_page(
            '<input name="title" style="display:none">'
            '<button style="display:none">새 소식 만들기</button>'
            + self.composer_page()
        )
        self.assertTrue(self.open_editor().is_displayed())
        self.assertEqual(self.driver.execute_script("return window.createClicks"), 1)

    def test_disabled_create_button_is_not_clicked_until_enabled(self):
        self.load_page(
            self.composer_page().replace('id="create"', 'id="create" disabled', 1)
            + "<script>setTimeout(() => document.getElementById('create').disabled = false, 200)</script>"
        )
        self.assertTrue(self.open_editor().is_displayed())
        self.assertEqual(self.driver.execute_script("return window.createClicks"), 1)

    def test_editor_in_iframe_keeps_frame_context(self):
        self.load_page(
            '<iframe srcdoc="unrelated content"></iframe>'
            '<iframe id="editor" srcdoc="'
            + html.escape('<input name="title">', quote=True)
            + '"></iframe>'
        )
        self.open_editor().send_keys("iframe title")
        self.assertEqual(self.driver.find_element(By.NAME, "title").get_attribute("value"), "iframe title")

    def test_create_button_inside_iframe(self):
        self.load_page('<iframe srcdoc="' + html.escape(self.composer_page(), quote=True) + '"></iframe>')
        self.assertTrue(self.open_editor().is_displayed())
        self.assertEqual(self.driver.execute_script("return window.createClicks"), 1)

    def test_async_editor_iframe_after_create(self):
        self.load_page("""
        <button onclick="setTimeout(() => {
          const frame = document.createElement('iframe');
          frame.srcdoc = '<input name=title>';
          document.body.replaceChildren(frame);
        }, 150)"><span class="inner_g"><span class="ico_comm ico_plus_s"></span>새 소식 만들기</span></button>
        """)
        self.assertTrue(self.open_editor().is_displayed())

    def test_missing_editor_has_actionable_error(self):
        self.load_page("<p>No editor or create button</p>")
        with self.assertRaisesRegex(TimeoutException, "새 소식 만들기.*kakao_url"):
            uploader._open_editor(WebDriverWait(self.driver, 0.1, poll_frequency=0.05))

    def test_upload_flow_attaches_images_after_opening_editor(self):
        self.assert_upload_flow(shadow=False)

    def test_nested_shadow_upload_flow(self):
        self.assert_upload_flow(shadow=True)

    def test_blog_type_already_selected_does_not_click_again(self):
        self.load_page('<a role="tab" aria-selected="true" onclick="throw Error(\'unexpected click\')">블로그형</a>')
        uploader._select_blog_type(WebDriverWait(self.driver, 0.2))

    def test_missing_blog_type_stops_upload(self):
        self.load_page('<a role="tab" aria-selected="true">카드형</a><textarea name="description"></textarea>')
        with self.assertRaisesRegex(TimeoutException, "블로그형.*before attaching images"):
            uploader._select_blog_type(WebDriverWait(self.driver, 0.1, poll_frequency=0.05))

    def test_duplicate_post_in_shadow_table(self):
        today = "2026-10-26-월요일"
        self.load_page("""
        <div id="app"></div><script>
        document.getElementById('app').attachShadow({mode: 'open'}).innerHTML =
          '<main class="cont_posts"><table><tbody><tr><td><a>2026-10-26-월요일</a></td></tr></tbody></table></main>';
        </script>
        """)
        self.assertTrue(uploader._already_uploaded_today(self.driver, today))
        self.assertFalse(uploader._already_uploaded_today(self.driver, "2026-10-25-일요일"))

    def test_duplicate_detection_waits_for_loaded_table_rows(self):
        self.load_page("""
        <main class="cont_posts"><table><tbody><tr><td></td></tr></tbody></table></main>
        <script>setTimeout(() => {
          document.querySelector('td').innerHTML = '<a>2026-10-26-월요일</a>';
        }, 200);</script>
        """)
        self.assertTrue(uploader._already_uploaded_today(self.driver, "2026-10-26-월요일"))

    def test_no_success_report_without_publication_acknowledgement(self):
        self.load_page('<textarea name="description">2026-10-26-월요일</textarea><button>登録</button>')
        with self.assertRaisesRegex(TimeoutException, "did not confirm publication"):
            uploader._wait_upload_complete(self.driver, "2026-10-26-월요일", timeout=0.1)

    def test_publication_confirmed_by_post_list(self):
        self.load_page('<main class="cont_posts"><table><tbody><tr><td><a>2026-10-26-월요일</a></td></tr></tbody></table></main>')
        uploader._wait_upload_complete(self.driver, "2026-10-26-월요일", timeout=0.1)

    def assert_upload_flow(self, shadow):
        markup = self.composer_page(editor_markup=(
            '<a role="tab" aria-selected="true">카드형</a>'
            '<a role="tab" aria-selected="false" onclick="'
            'for (const tab of this.getRootNode().querySelectorAll(\'[role=tab]\')) '
            'tab.setAttribute(\'aria-selected\', tab === this ? \'true\' : \'false\'); '
            'window.actionOrder = [\'blog\']">블로그형</a>'
            '<section class="section_comm"><div class="ibox_upload">'
            '<input accept="image/gif, image/jpeg, image/png, image/bmp" multiple hidden type="file" '
            'onchange="window.actionOrder.push(\'images\')">'
            '<button type="button"><span class="inner_g">이미지 추가</span></button></div>'
            '<input id="title" name="title" type="text">'
            '<textarea id="description" name="description" class="tf_comm" '
            'oninput="if (!window.actionOrder.includes(\'content\')) window.actionOrder.push(\'content\')"></textarea>'
            '</section><button type="button" class="btn_gm" '
            'onclick="window.actionOrder.push(\'register\'); window.submitted = true; '
            'const success = document.createElement(\'strong\'); '
            'success.textContent = \'발행 성공\'; this.parentNode.appendChild(success)">'
            '<span class="inner_g">등록</span></button>'
        ), shadow=shadow)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, "261026_01.jpg")
            second = Path(directory, "261026_02.jpg")
            first.write_bytes(b"local upload fixture")
            second.write_bytes(b"local upload fixture")
            with (
                patch.object(uploader, "_build_driver", return_value=self.driver),
                patch.object(uploader, "_maybe_login") as login,
                patch.object(uploader, "_already_uploaded_today", return_value=False) as duplicate,
                patch.object(uploader.time, "sleep"),
                patch.object(self.driver, "quit") as quit_driver,
            ):
                # Checking duplicates must happen after authentication.
                duplicate.side_effect = lambda *_: (self.assertTrue(login.called) or False)
                uploader.upload_to_kakao(
                    url="data:text/html;charset=utf-8," + quote(markup),
                    email="", password="", image_dir=directory, title="2026-10-26-월요일",
                )
                quit_driver.assert_called_once()
            self.assertTrue(self.driver.execute_script("return window.submitted"))
            self.assertEqual(uploader._find_elements(self.driver, "#description")[0].get_attribute("value"), "2026-10-26-월요일")
            self.assertEqual(uploader._find_elements(self.driver, "#title")[0].get_attribute("value"), "2026-10-26-월요일")
            self.assertEqual(self.driver.execute_script("return window.actionOrder"), ["blog", "images", "content", "register"])
            self.assertEqual(
                self.driver.execute_script("return Array.from(arguments[0].files, f => f.name)", uploader._find_elements(self.driver, "input[type=file]")[0]),
                [first.name, second.name],
            )


if __name__ == "__main__":
    unittest.main()
