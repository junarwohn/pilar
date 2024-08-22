from pytesseract import *
import cv2
import os
import re
import numpy as np
import difflib
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
from tqdm import tqdm
import time

day_info = time.strftime('%Y-%m-%d', time.localtime())[2:]

IS_DEDUG=False
# IS_DEDUG=True

mod_path = 'svm_models.sav'
svm_clfs = pickle.load(open(mod_path, 'rb'))
def img_similarity(img1, img2):
    data1 = img1.flatten()
    data2 = img2.flatten()
    return sum(np.isclose(data1, data2, atol=50)) / len(data1)


file_list = [i for i in os.listdir("src/extract/") if not i.startswith('.') and not 'Zone.Identifier' in i]
file_list.sort()
thumb_list = [i for i in os.listdir("src/thumbs/") if not i.startswith('.') and not 'Zone.Identifier' in i]
thumb_list.sort()
bound_upper_complete = False
bound_lower_complete = False
# height_upper = 925
height_upper = 960
# height_lower = 1020
height_lower = 1050
pre_word = ""
diff = difflib.Differ()
sample_img = cv2.imread("src/extract/" + file_list[50])
kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
add_cnt = 0
page_cnt = 0
thumb_cnt = 0
str_diff = 0
img_sim = 0
ok = 13

yn = 'y'
if yn == 'y':
    bound_upper_complete = True
    bound_lower_complete = True

while not bound_upper_complete:
    print("input_upper")
    height_upper = int(input())
    sliced =  sample_img[height_upper:, :]
    print(sliced.shape)
    cv2.namedWindow("height_upper",cv2.WINDOW_NORMAL)
    cv2.imshow("height_upper", sliced) 
    cv2.resizeWindow("height_upper", 600,600)
    print("is it ok?[enter/other]")
    ret = cv2.waitKey(0)
    if ret == 13:
        bound_upper_complete = True
    cv2.destroyAllWindows()

while not bound_lower_complete:
    print("input_lower")
    height_lower = int(input())
    sliced = sample_img[height_upper:height_lower - 20, :]
    cv2.namedWindow("height_lower",cv2.WINDOW_NORMAL)
    cv2.imshow("height_lower", sliced) 
    cv2.resizeWindow("height_lower", 600,600)
    print("is it ok?[enter/other]")
    ret = cv2.waitKey(0)
    if ret == 13:
        bound_lower_complete = True
    cv2.destroyAllWindows()

result_img = cv2.imread("src/thumbs/" + thumb_list[thumb_cnt])[:height_upper - 20, :]
# result_img = cv2.imread("src/extract/" + file_list[7])[:height_upper - 5, :]

pre_img = cv2.imread("src/extract/" + file_list[0])[height_upper:height_lower, :]
cur_img = cv2.imread("src/extract/" + file_list[0])[height_upper:height_lower, :]
gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
inverted = cv2.bitwise_not(gray)
bilateral_filter = cv2.bilateralFilter(inverted, 9, 16, 16)
r, pre_bin = cv2.threshold(bilateral_filter, 200, 255, cv2.THRESH_BINARY)
if not IS_DEDUG:
    file_list = tqdm(file_list)
for file_name in file_list:
    # original_img = cv2.imread("src/extract/" + file_name)
    original_img = cv2.imread("src/extract/" + file_name)
    #cv2.imshow("original_img", original_img)
    #cv2.waitKey(0)

    cur_img = original_img[height_upper:height_lower, :]
    gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    bilateral_filter = cv2.bilateralFilter(inverted, 9, 16, 16)
    r, cur_bin = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
    # cur_bin = cv2.bilateralFilter(cur_bin, 9, 16, 16)
    # dst = cv2.filter2D(bilateral_filter, -1, kernel_sharpen)
    # dst = cur_bin
    dst = bilateral_filter
    new_img = dst
    #
    # gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    # inverted = cv2.bitwise_not(gray)
    # dst = cv2.filter2D(inverted, -1, kernel_sharpen)
    # new_img = dst
    text = image_to_string(dst, lang="Hangul", config="--psm 4 --oem 1")
    #text = image_to_string(dst, lang="kor", config="--psm 4 --oem 1")
    word_list = re.sub("\d+|[ ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅕㅓㅕㅗㅛㅜㅠㅡㅣ\{\}\[\]\/?.,;:|\)「＊ㆍ：”…*~`!^\-_+<>@\#$%&》\\\=\(\'\"\f]|[A-Za-z]", "", text).split('\n')
    cur_word = max(word_list, key=len)
    # Filter trash recognition
    if len(cur_word) < 2:
        continue
    if cur_word != pre_word:
        str_diff = SequenceMatcher(None, cur_word, pre_word).ratio()
        img_sim = img_similarity(pre_bin, cur_bin)


        if IS_DEDUG:
            cv2.imshow("dst", dst)
            cv2.imshow("cur_bin", cur_bin)
            add_img = cv2.addWeighted(pre_img, 0.5, cur_img, 0.5, 0)
            cv2.imshow("Okay to enter", add_img)
            ok = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if ok != 13:
                print()
                print("SAME, str_diff : {:.03f}, img_sim : {:.03f}, pre : [{}], cur : [{}]".format(str_diff, img_sim, pre_word, cur_word))
                continue

        else:
            # diff sim order
            vote = [clf.predict([[str_diff, img_sim]])[0] for clf in svm_clfs]
            if sum(vote) == 4:
                continue
            elif sum(vote) == 0:
                pass
            else:
                cv2.imshow("dst", dst)
                cv2.imshow("cur_bin", cur_bin)
                add_img = cv2.addWeighted(pre_img, 0.5, cur_img, 0.5, 0)
                cv2.imshow("Okay to enter", add_img)
                ok = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if ok != 13:
                    pre_word = cur_word
                    pre_img = cur_img
                    pre_bin = cur_bin
                    print()
                    print("SAME, str_diff : {:.03f}, img_sim : {:.03f}, pre : [{}], cur : [{}]".format(str_diff, img_sim, pre_word, cur_word))
                    continue

        print()
        print("DIFF, str_diff : {:.03f}, img_sim : {:.03f}, pre : [{}], cur : [{}]".format(str_diff, img_sim, pre_word, cur_word))
        cur_img2 = original_img[2 * height_upper - height_lower:height_upper, :]
        gray2 = cv2.cvtColor(cur_img2, cv2.COLOR_BGR2GRAY)
        inverted2 = cv2.bitwise_not(gray2)
        bilateral_filter2 = cv2.bilateralFilter(inverted2, 9, 16, 16)
        r, cur_bin2 = cv2.threshold(bilateral_filter2, 127, 255, cv2.THRESH_BINARY)
        dst2 = bilateral_filter2
        text = image_to_string(dst2, lang="Hangul", config="--psm 4 --oem 1")
        word_list = re.sub("\d+|[ ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ\{\}\[\]\/?.,;:|\)「＊ㆍ：”…*~`!^\-_+<>@\#$%&》\\\=\(\'\"\f]|[A-Za-z]", "",
                           text).split('\n')
        cur_word2 = max(word_list, key=len)
        """ MULTILINE CHECK """
        if len(cur_word2) > len(cur_word):
            print("Check multiline")
            cv2.imshow("cur_img", cur_img)
            cv2.imshow("cur_img2", cur_img2)
            ok = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if ok == 13:
                result_img = np.vstack((result_img, cur_img2))
            else:
                result_img = np.vstack((result_img, cur_img))
        else:
            result_img = np.vstack((result_img, cur_img))

        add_cnt += 1
        if add_cnt % 25 == 0:
            cv2.imwrite("result-{}.jpg".format(str(page_cnt)), result_img)
            page_cnt += 1
            thumb_cnt += 1
            try:
                result_img = cv2.imread("src/thumbs/" + thumb_list[thumb_cnt])[:height_upper - 5, :]
            except:
                result_img = cv2.imread("src/thumbs/" + thumb_list[-1])[:height_upper - 5, :]

    pre_word = cur_word
    pre_img = cur_img
    pre_bin = cur_bin
if add_cnt % 25 != 0:
    cv2.imwrite("result-{}.jpg".format(str(page_cnt)), result_img)
