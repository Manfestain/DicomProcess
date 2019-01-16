# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 18/12/29
'''
    对图像进行Gabor滤波，清晰图像中的各个部分
'''
import os
import cv2
import scipy.misc
import numpy as np
import SimpleITK as sitk
import skimage.morphology as sm
import matplotlib.pyplot as plt
from skimage import exposure
from scipy.stats import tmax, tmin

def build_filters():
    filters = []
    ksize = [7,9,11,13,15,17] #gabor尺度 6个
    lamda = np.pi/2.0 # 波长
 
    for theta in np.arange(0,np.pi,np.pi/4): #gabor方向 0 45 90 135
        for k in range(6):
            params = {'ksize':(ksize[k], ksize[k]), 'sigma':3.3, 'theta':theta, 'lambd':27.3,    
                  'gamma':10.5, 'psi':6, 'ktype':cv2.CV_32F}
            # kern = cv2.getGaborKernel((ksize[k],ksize[k]),1.0,theta,lamda,0.5,0,ktype=cv2.CV_32F)
            kern = cv2.getGaborKernel(**params)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

# 生成多个方向的滤波器
# def build_filters():
#     filters = []
#     ksize = 31     #gaborl尺度 这里是一个
#     for theta in np.arange(0, np.pi, np.pi / 4):    #gaborl方向 0 45 90 135 角度尺度的不同会导致滤波后图像不同
#         params = {'ksize':(ksize, ksize), 'sigma':3.3, 'theta':theta, 'lambd':18.3,    
#                   'gamma':4.5, 'psi':0.89, 'ktype':cv2.CV_32F}
#             #gamma越大核函数图像越小，条纹数不变，sigma越大 条纹和图像都越大
#             #psi这里接近0度以白条纹为中心，180度时以黑条纹为中心
#             #theta代表条纹旋转角度
#             #lambd为波长 波长越大 条纹越大
#         kern = cv2.getGaborKernel(**params)     #创建内核
#         kern /= 1.5*kern.sum()
#         filters.append((kern,params))
#     return filters   

# Gabor滤波处理函数
def gabor_process(img, filters):
    accum = np.zeros_like(img)      #初始化img一样大小的矩阵
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)      #2D滤波函数  kern为其滤波模板
        np.maximum(accum, fimg, accum)       #参数1与参数2逐位比较  取大者存入参数3  这里就是将纹理特征显化更加明显
    return accum

def convertImagePixel(image):
    x_min = tmin(image, axis=None)
    x_max = tmax(image, axis=None)
    result = (image-x_min)/(x_max-x_min)*255 
    return result

if __name__ == '__main__':
    # filters = build_filters()
    # print(len(filters))
    path = './TempData/brain_1142709300012.dcm'
    img_array = sitk.GetArrayFromImage(sitk.ReadImage(path))[0, :, :]
    converted = convertImagePixel(img_array)
    scipy.misc.toimage(converted, cmin=0, cmax=255).save('./TempData/filters.jpg')
    print(img_array.shape)

    img = cv2.imread('./TempData/filters.jpg')
    filters = build_filters()
    result = gabor_process(img, filters)
    result = exposure.equalize_hist(result)
    converted = convertImagePixel(result)

    thresh = np.array(converted > int(255*0.4), dtype=np.int8)     # 变为二值图像， 可调整参数
    
    dst_close = sm.closing(thresh, sm.disk(3.5))   #闭操作滤波，填充空洞， 可调整参数
    dst_open = sm.opening(dst_close, sm.disk(3.5))   # 开操作滤波，消除小斑点， 可调整参数
    plt.imshow(dst_open)
    plt.show()