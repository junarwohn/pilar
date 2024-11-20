from pytesseract import *
import cv2
import os
import re
import numpy as np
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
from tqdm import tqdm
import time

class ImageProcessor:
    def __init__(self):
        self.day_info = time.strftime('%Y-%m-%d', time.localtime())[2:]
        self.IS_DEBUG = False
        self.load_models()
        self.init_parameters()
        self.load_file_lists()
        out_dir = f"./out/{self.day_info}"
        os.makedirs(out_dir, exist_ok=True)

    def load_models(self):
        mod_path = 'svm_models.sav'
        self.svm_clfs = pickle.load(open(mod_path, 'rb'))

    def init_parameters(self):
        self.height_upper = 925
        self.height_lower = 1020
        self.kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.add_cnt = 0
        self.page_cnt = 0
        self.thumb_cnt = 0
        self.pre_word = ""

    def load_file_lists(self):
        self.file_list = [i for i in os.listdir("src/extract/") if not i.startswith('.') and not 'Zone.Identifier' in i]
        self.file_list.sort()
        self.thumb_list = [i for i in os.listdir("src/thumbs/") if not i.startswith('.') and not 'Zone.Identifier' in i]
        self.thumb_list.sort()

    @staticmethod
    def img_similarity(img1, img2):
        data1 = img1.flatten()
        data2 = img2.flatten()
        return sum(np.isclose(data1, data2, atol=50)) / len(data1)

    def get_bounds(self):
        sample_img = cv2.imread("src/extract/" + self.file_list[50])
        bound_upper_complete = bound_lower_complete = False

        yn = 'y'
        if yn == 'y':
            return True, True

        while not bound_upper_complete:
            print("input_upper")
            self.height_upper = int(input())
            sliced = sample_img[self.height_upper:, :]
            print(sliced.shape)
            cv2.namedWindow("height_upper", cv2.WINDOW_NORMAL)
            cv2.imshow("height_upper", sliced)
            cv2.resizeWindow("height_upper", 600, 600)
            print("is it ok?[enter/other]")
            ret = cv2.waitKey(0)
            if ret == 13:
                bound_upper_complete = True
            cv2.destroyAllWindows()

        while not bound_lower_complete:
            print("input_lower")
            self.height_lower = int(input())
            sliced = sample_img[self.height_upper:self.height_lower - 20, :]
            cv2.namedWindow("height_lower", cv2.WINDOW_NORMAL)
            cv2.imshow("height_lower", sliced)
            cv2.resizeWindow("height_lower", 600, 600)
            print("is it ok?[enter/other]")
            ret = cv2.waitKey(0)
            if ret == 13:
                bound_lower_complete = True
            cv2.destroyAllWindows()

    def process_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        bilateral_filter = cv2.bilateralFilter(inverted, 9, 16, 16)
        _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
        return bilateral_filter, binary

    def extract_text(self, img):
        text = image_to_string(img, lang="Hangul", config="--psm 4 --oem 1")
        word_list = re.sub("|[ ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅕㅓㅕㅗㅛㅜㅠㅡㅣ\{\}\[\]\/?.,;:|\)「＊ㆍ：\"…*~`!^\-_+<>@\#$%&》\\\=\(\'\"\f]|[A-Za-z]™", "", text).split('\n')
        return max(word_list, key=len)

    def process_files(self):
        result_img = cv2.imread("src/thumbs/" + self.thumb_list[self.thumb_cnt])[:self.height_upper - 20, :]
        
        original_img = cv2.imread("src/extract/" + self.file_list[0])
        pre_img = original_img[self.height_upper:self.height_lower, :]
        pre_processed, pre_bin = self.process_image(pre_img)

        if not self.IS_DEBUG:
            self.file_list = tqdm(self.file_list)

        for file_name in self.file_list:
            original_img = cv2.imread("src/extract/" + file_name)
            cur_img = original_img[self.height_upper:self.height_lower, :]
            processed_img, cur_bin = self.process_image(cur_img)
            
            cur_word = self.extract_text(processed_img)
            if len(cur_word) < 2:
                continue

            if cur_word != self.pre_word:
                str_diff = SequenceMatcher(None, cur_word, self.pre_word).ratio()
                img_sim = self.img_similarity(pre_bin, cur_bin)

                if self.handle_differences(processed_img, cur_bin, pre_img, cur_img, str_diff, img_sim, cur_word):
                    continue

                print(f"\nDIFF, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
                
                result_img = self.handle_multiline(original_img, cur_img, cur_word, result_img)
                
                self.add_cnt += 1
                if self.add_cnt % 25 == 0:
                    self.save_result(result_img)
                    result_img = self.get_new_result_img()

            self.pre_word = cur_word
            pre_img = cur_img
            pre_bin = cur_bin

        if self.add_cnt % 25 != 0:
            self.save_result(result_img)

    def handle_differences(self, processed_img, cur_bin, pre_img, cur_img, str_diff, img_sim, cur_word):
        if self.IS_DEBUG:
            return self.handle_debug_mode(processed_img, cur_bin, pre_img, cur_img, str_diff, img_sim, cur_word)
        else:
            return self.handle_production_mode(str_diff, img_sim, processed_img, cur_bin, pre_img, cur_img, cur_word)

    def handle_debug_mode(self, processed_img, cur_bin, pre_img, cur_img, str_diff, img_sim, cur_word):
        cv2.imshow("dst", processed_img)
        cv2.imshow("cur_bin", cur_bin)
        add_img = cv2.addWeighted(pre_img, 0.5, cur_img, 0.5, 0)
        cv2.imshow("Okay to enter", add_img)
        ok = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if ok != 13:
            print(f"\nSAME, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
            return True
        return False

    def handle_production_mode(self, str_diff, img_sim, processed_img, cur_bin, pre_img, cur_img, cur_word):
        vote = [clf.predict([[str_diff, img_sim]])[0] for clf in self.svm_clfs]
        if sum(vote) == 4:
            return True
        elif sum(vote) == 0:
            return False
        else:
            cv2.imshow("dst", processed_img)
            cv2.imshow("cur_bin", cur_bin)
            add_img = cv2.addWeighted(pre_img, 0.5, cur_img, 0.5, 0)
            cv2.imshow("Okay to enter", add_img)
            ok = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if ok != 13:
                self.pre_word = cur_word
                print(f"\nSAME, str_diff : {str_diff:.03f}, img_sim : {img_sim:.03f}, pre : [{self.pre_word}], cur : [{cur_word}]")
                return True
        return False

    def handle_multiline(self, original_img, cur_img, cur_word, result_img):
        cur_img2 = original_img[2 * self.height_upper - self.height_lower:self.height_upper, :]
        processed_img2, _ = self.process_image(cur_img2)
        cur_word2 = self.extract_text(processed_img2)

        if len(cur_word2) > len(cur_word):
            print("Check multiline")
            cv2.imshow("cur_img", cur_img)
            cv2.imshow("cur_img2", cur_img2)
            ok = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if ok == 13:
                return np.vstack((result_img, cur_img2))
        return np.vstack((result_img, cur_img))

    def save_result(self, result_img):
        out_dir = f"./out/{self.day_info}"
        img_path = f"{out_dir}/result-{str(self.page_cnt)}.jpg"
        cv2.imwrite(img_path, result_img)
        self.page_cnt += 1 
        self.thumb_cnt += 1

    def get_new_result_img(self):
        try:
            return cv2.imread("src/thumbs/" + self.thumb_list[self.thumb_cnt])[:self.height_upper - 5, :]
        except:
            return cv2.imread("src/thumbs/" + self.thumb_list[-1])[:self.height_upper - 5, :]
