# -*- coding: utf-8 -*-

import cv2
import numpy as np

def get_otsu_threshold(src_pic):
    res_th = 0
    max_v = 0.0
    hist = [0]*257

    for i in range(src_pic.shape[0]):
        for j in range(src_pic.shape[1]):
            hist[src_pic[i][j]] += 1

    for threshold in range(256):
        n0 = np.sum(hist[:threshold+1])     
        n1 = src_pic.size - n0
        
        if n0 == 0:
            continue
        if n1 == 0:
            break

        w0 = float(n0) / src_pic.size
        w1 = 1 - w0

        u0 = 0.0
        u1 = 0.0
        for i in range(threshold+1):
            u0 += i * float(hist[i])
        u0 = u0 / n0
        for i in range(threshold+1, 256):
            u1 += i * float(hist[i])
        u1 = u1 / n1

        tmp_v = w0 * w1 * (u1-u0) * (u1-u0)
        if tmp_v > max_v:
            res_th = threshold
            max_v = tmp_v
    
    return res_th


def bin_scale(img, th):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = 255 if img[i][j] > th else 0 

    return img


if __name__ == "__main__":
    img = cv2.imread('cells.bmp')
    cv2.imshow('source', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th = get_otsu_threshold(img)

    img = bin_scale(img, th)
    
    cv2.imshow('result', img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()


 