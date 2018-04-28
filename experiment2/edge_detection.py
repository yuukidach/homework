# -*- coding: utf-8 -*-

import cv2
import numpy as np

def conv_2d(img, filter):
    img_h, img_w = img.shape
    fil_h, fil_w = filter.shape
    # 补0的图像
    pad_h = img_h + fil_h - 1
    pad_w = img_w + fil_w - 1
    pad_img = np.zeros((pad_h, pad_w))
    pad_img[((fil_h-1)>>1):pad_h-((fil_h-1)>>1), ((fil_w-1)>>1):pad_w-((fil_w-1)>>1)] = img[:img_h, :img_w]

    res_img = np.zeros((img_h, img_w))
 
    for i in range(img_h):
        for j in range(img_w):
            res_img[i][j] = np.sum(pad_img[i:i+fil_h, j:j+fil_w] * filter)

    return res_img


# 取绝对值并将数值类型转化为uint8
def img_abs_clip(img):
    img = np.abs(img)
    img = img.clip(0, 255)
    img = np.uint8(img)

    return img 


def canny_detect(img, th_low, th_high, sigma = 0.8):
    # 高斯滤波
    img = cv2.GaussianBlur(img, (5, 5), sigma)

    # 生成梯度和方向
    dx = conv_2d(img, SOBEL_X)
    dy = conv_2d(img, SOBEL_Y)
    grad_map = np.sqrt(np.square(dx) + np.square(dy))  
    
    # 非极大值抑制
    img_nms = non_maximum_suppression(grad_map, dx, dy)

    # 标记所有的高于阈值的点
    canny_res = np.zeros(img_nms.shape)
    for i in range(1, img_nms.shape[0]-1):
        for j in range(1, img_nms.shape[1]-1):
            if (img_nms[i, j] > th_high):
                canny_res[i, j] = 255
    
    # 对阈值间的值进行联通拓展
    for i in range(0, img_nms.shape[0]-1):
        for j in range(0, img_nms.shape[1]-1):
            if canny_res[i, j] == 255:
                if img_nms[i+1, j] >= th_low:
                    canny_res[i+1, j] = 255
                if img_nms[i+1, j+1] >= th_low:
                    canny_res[i+1, j+1] = 255 
                if img_nms[i, j+1] >= th_low:
                    canny_res[i, j+1] = 255 
    
    for i in range(img_nms.shape[0]-1, 0, -1):
        for j in range(img_nms.shape[1]-1, 0, -1):
            if canny_res[i, j] == 255:
                if img_nms[i-1, j] >= th_low:
                    canny_res[i-1, j] = 255
                if img_nms[i-1, j-1] >= th_low:
                    canny_res[i-1, j-1] = 255 
                if img_nms[i, j-1] >= th_low:
                    canny_res[i, j-1] = 255 

    for i in range(img_nms.shape[0]-1, 0, -1):
        for j in range(0, img_nms.shape[1]-1):
            if canny_res[i, j] == 255:
                if img_nms[i-1, j+1] >= th_low:
                    canny_res[i-1, j+1] = 255
    
    for i in range(0, img_nms.shape[0]-1):
        for j in range(img_nms.shape[1]-1, 0, -1):
            if canny_res[i, j] == 255:
                if img_nms[i+1, j-1] >= th_low:
                    canny_res[i+1, j-1] = 255
    
    return canny_res


def non_maximum_suppression(grad, dx, dy):
    res = np.zeros(grad.shape)

    for i in range (1, grad.shape[0]-1):
        for j in range (2, grad.shape[1]-1):
            if grad[i, j] == 0:
                continue
            
            gradX = dx[i, j]
            gradY = dy[i, j]
            
            # 如果Y方向幅度值较大
            if np.abs(gradY) > np.abs(gradX):
                weight = np.abs(gradX) / np.abs(gradY)
                grad2 = grad[i-1, j]
                grad4 = grad[i+1, j]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = grad[i-1, j-1]
                    grad3 = grad[i+1, j+1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = grad[i-1, j+1]
                    grad3 = grad[i+1, j-1]
                    
            # 如果x方向幅度值较大
            else:
                weight = np.abs(gradY) / np.abs(gradX)
                grad2 = grad[i, j-1]
                grad4 = grad[i, j+1]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = grad[i+1, j-1]
                    grad3 = grad[i-1, j+1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = grad[i-1, j-1]
                    grad3 = grad[i+1, j+1]
        
            dot1 = weight * grad1 + (1-weight) * grad2
            dot2 = weight * grad3 + (1-weight) * grad4
            if grad[i, j] >= dot1 and grad[i, j] >= dot2:
                res[i, j] = grad[i, j]
            else:
                res[i, j] = 0

    res = res.clip(0, 255).round().astype(np.uint8)
    return res


# 各个算子
LAPLACIAN = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]))  
SOBEL_X = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))  
SOBEL_Y = np.array(([1, 2, 1], [0, 0, 0], [-1, -2, -1])) 


if __name__ == "__main__":
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('source', img)

    # 使用拉普拉斯算子处理图像   
    lap_img = conv_2d(img, LAPLACIAN)
    lap_img = lap_img.clip(0, 255)
    _, lap_img = cv2.threshold(lap_img, 0, 255, cv2.THRESH_BINARY)

    # 使用sobel算子处理图像
    sobx_img = conv_2d(img, SOBEL_X)
    sobx_img = img_abs_clip(sobx_img)
    soby_img = conv_2d(img, SOBEL_Y)
    soby_img = img_abs_clip(soby_img)
    sob_img = cv2.addWeighted(sobx_img, 0.5, soby_img, 0.5, 0)
    _, sobx_img = cv2.threshold(sobx_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, soby_img = cv2.threshold(soby_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, sob_img = cv2.threshold(sob_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    

    canny_img = canny_detect(img, th_low=30, th_high=80)
    
    cv2.imshow('laplace', lap_img)
    cv2.imshow('sobel x operator', sobx_img)
    cv2.imshow('sobel y operator', soby_img)
    cv2.imshow('sobel', sob_img)
    cv2.imshow('canny', canny_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


