# -*- coding: utf-8 -*-

import numpy as np
import cv2
# 自写文件
import noise

def img_hist(img):
    res_img = img.copy()
    row, col = res_img.shape
    # 亮度在0-255
    h = [0] * 256
    hc = [0] * 256
    t = [0] * 256

    # 构建亮度直方图
    for i in range(row):
        for j in range(col):
            h[res_img[i][j]] += 1
    
    # 得到累计直方图
    hc[0] = h[0]
    for i in range(1, 256):
        hc[i] = hc[i-1] + h[i]
    
    # 打表
    for i in range(256):
        t[i] = round(255.0*hc[i] / (row*col))

    # 查表
    for i in range(row):
        for j in range(col):
            res_img[i][j] = t[res_img[i][j]]
            
    return res_img

if __name__ == "__main__":
    src_img = cv2.imread('landscape.jpg', 0)
    cv2.imshow('original image', src_img)

    hist_img = img_hist(src_img)
    cv2.imshow('histogram equalization', hist_img)

    noise.press_key_2_destory_all_windows()
