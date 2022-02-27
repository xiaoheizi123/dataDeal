import cv2
import numpy as np
import  matplotlib as mp
mp.use('TkAgg')
import matplotlib.pyplot as plt
'''
author:zr
'''

def image_analysis(image):
    #B
    b = image[:,:,0]
    cv2.namedWindow('b sole', cv2.WINDOW_NORMAL)
    cv2.imshow('b sole', b)
    # cv2.waitKey()
    g_zero = np.zeros((height, width), dtype=np.uint8)
    r_zero = np.zeros((height, width), dtype=np.uint8)
    new_b = cv2.merge([b, g_zero, r_zero])
    cv2.namedWindow('b merge', cv2.WINDOW_NORMAL)
    cv2.imshow('b merge', new_b)
    # cv2.waitKey(0)
    b[:,:] = 255
    new_b_255 = cv2.merge([b, g_zero, r_zero])
    cv2.namedWindow('b 255 merge', cv2.WINDOW_NORMAL)
    cv2.imshow('b 255 merge', new_b_255)
    # cv2.waitKey(0)
    b[:,:] = 30
    new_b_30 = cv2.merge([b, g_zero, r_zero])
    cv2.namedWindow('b 30 merge', cv2.WINDOW_NORMAL)
    cv2.imshow('b 30 merge', new_b_30)
    # cv2.waitKey(0)

    #G
    g = image[:,:,1]
    cv2.namedWindow('g sole', cv2.WINDOW_NORMAL)
    cv2.imshow('g sole', g)
    # cv2.waitKey(0)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    cv2.imshow('gray', img_gray)
    cv2.waitKey()

    #灰度图
    cv2.imwrite('g.jpg', g)
    g_img = cv2.imread('g.jpg')
    print(g_img.shape)
    cv2.imshow('g_img_read', g_img)
    cv2.waitKey()

def img_light(hsvimg, v):
    ii = 0.1
    tvimg = hsvimg[:,:,2]
    while(ii < v):
        print('ii is:', ii)
        vimg = tvimg*ii
        new_hsvimg = cv2.merge([hsvimg[:,:,0],hsvimg[:,:,1], np.asarray(vimg, dtype= np.uint8)])
        new_hsvimg_show = cv2.cvtColor(new_hsvimg, cv2.COLOR_HSV2BGR)
        cv2.imshow('v %s change'%(str(ii)), new_hsvimg_show)
        cv2.imwrite('v %s change.jpg'%(str(ii)), new_hsvimg_show)
        cv2.waitKey(500)
        ii += 0.2

def img_h(hsvimg, h):
    ii = 0.1
    thimg = hsvimg[:,:,0]
    while(ii <  h):
        print('ii is:', ii)
        himg = thimg * ii
        new_hsvimg = cv2.merge([np.asarray(himg, dtype= np.uint8), hsvimg[:, :, 1], hsvimg[:, :, 2]])
        new_hsvimg_show = cv2.cvtColor(new_hsvimg, cv2.COLOR_HSV2BGR)
        cv2.imshow('h %s change'%(str(ii)), new_hsvimg_show)
        cv2.imwrite('h %s change.jpg'%(str(ii)), new_hsvimg_show)
        cv2.waitKey(500)
        ii += 0.2

def img_contrast(img, cs):
    ii = 0.1
    while(ii < cs):
        newhsvimg = np.clip(img*ii, 0, 255).astype(np.uint8)
        # new_hsvimg_show = cv2.cvtColor(newhsvimg, cv2.COLOR_HSV2BGR)
        cv2.imshow('contrast %s change'%(str(ii)), newhsvimg)
        cv2.imwrite('contrast %s change.jpg'%(str(ii)), newhsvimg)
        cv2.waitKey(500)
        ii += 0.2

if __name__ == "__main__":
    image = cv2.imread('Cartoon-house.jpg')
    print(image.shape)
    height, width, channel = image.shape
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_light(img_hsv, 1)
    img_h(img_hsv, 0.9)
    img_contrast(image, 1)


