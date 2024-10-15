import requests
from bs4 import BeautifulSoup
import time

day_info = time.strftime('%Y-%m-%d', time.localtime())[2:]
url = 'http://www.youngnak.net/portfolio-item/bible-stroll-' + day_info
response = requests.get(url)

if response.status_code == 200:
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    
    iframes = soup.find_all('iframe')
    for i in iframes:
        try:
            target_url = i.get_attribute_list('src')[0]
            if 'youtube' in target_url:
                break
        except:
            pass
    # info = soup.select_one("#after_grid_row_1 > div > div > div > div > div.flex_column.av-1f9dq2x3-92d8fbb7476554abe42d49745cc17e69.av_three_fifth.avia-builder-el-7.el_after_av_one_fifth.el_before_av_one_fifth.flex_column_div.av-zero-column-padding > section.avia_codeblock_section.avia_code_block_0 > div > div > iframe")
    # # info = soup.select_one("#after_grid_row_1 > div > div > div > div > div.flex_column.av-2mxui32-1f1281242c3b9ca49b4990949f00d4d4.av_three_fifth.avia-builder-el-7.el_after_av_one_fifth.el_before_av_one_fifth.flex_column_div.av-zero-column-padding > section.avia_codeblock_section.avia_code_block_0 > div > div > iframe")
    # # info = soup.select_one("#after_submenu_1 > div > div > div > div > div.flex_column.av_three_fifth.flex_column_div.av-zero-column-padding.avia-builder-el-7.el_after_av_one_fifth.el_before_av_one_fifth > section.avia_codeblock_section.avia_code_block_0 > div > div > iframe")
    # target_url = info.get_attribute_list('src')[0]
    print(target_url)
else : 
    print(response.status_code)
