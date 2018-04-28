# -*- coding: utf-8 -*-

import numpy as np
import random
import cv2

def add_guassian_noise(img, means, stddev):
    gauss_noise = cv2.randn(img.copy(), means, stddev)
    res_img = cv2.add(img, gauss_noise)
    return res_img


def add_sp_noise(img, prob):
    res_img = img.copy()
    thres = 1 - prob
    for i in range(res_img.shape[0]):
        for j in range(res_img.shape[1]):
            tmp = random.random()
            # 利用tmp大小，随机地产生黑白噪点
            if tmp < prob: 
                res_img[i][j] = 0
            elif tmp > thres:
                res_img[i][j] = 255
    return res_img


def add_impulse_noise(img, prob):
    res_img = img.copy()
    for i in range(res_img.shape[0]):
        for j in range(res_img.shape[1]):
            tmp = random.random()
            if tmp < prob: 
                res_img[i][j] = 255
    return res_img

def press_key_2_destory_all_windows():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 添加噪声部分
    # 图片仅需要展示，就不调用namedWindow了
    src_img = cv2.imread('lena.bmp', 0)
    cv2.imshow('original image', src_img)

    g_img = add_guassian_noise(src_img, 0, 80)
    cv2.imshow('add Gaussian noise', g_img)

    sp_img = add_sp_noise(src_img, 0.05)
    cv2.imshow('add salt and pepper noise', sp_img)

    i_img = add_impulse_noise(src_img, 0.05)
    cv2.imshow('add impulse noise', i_img)

    press_key_2_destory_all_windows()

    # 噪声处理部分
    median_flt_img = cv2.medianBlur(sp_img, 3)
    cv2.imshow('median filter', median_flt_img)

    mean_flt_img = cv2.blur(sp_img, (2, 2))
    cv2.imshow('mean filter', mean_flt_img)

    press_key_2_destory_all_windows()




    

