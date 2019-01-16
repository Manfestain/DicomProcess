# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 18/12/6
'''
    检测增强后的图像中的肿瘤大概区域
    此处有几个参数可以调整
'''
import os
import cv2
import scipy.misc
import numpy as np
import SimpleITK as sitk
import skimage.morphology as sm
import matplotlib.pyplot as plt
from skimage import measure
from scipy.stats import tmax, tmin

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 将图像像素值转化为0~255
def convertImagePixel(image):
    x_min = tmin(image, axis=None)
    x_max = tmax(image, axis=None)
    result = (image-x_min)/(x_max-x_min)*255 
    return result


# 分割切片中的肿瘤区域，并保存对应的mask为jpg格式
def segmentTumor(path, save_path=None, return_contour=True):
    name = path[path.rfind('/')+1:path.rfind('.')]

    image_array = sitk.GetArrayFromImage(sitk.ReadImage(path))
    converted = convertImagePixel(image_array[0, :, :]).astype(np.int16)   # 转化为0~255的图像
    
    # scipy.misc.toimage(converted, cmin=0, cmax=255).save(name)   # 暂存灰度图
    # img = cv2.imread(name)
    # gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    thresh = np.array(converted > int(255*0.4), dtype=np.int8)     # 变为二值图像， 可调整参数
    
    dst_close = sm.binary_closing(thresh, sm.disk(3.5))   #闭操作滤波，填充空洞， 可调整参数
    dst_open = sm.binary_opening(dst_close, sm.disk(3.5))   # 开操作滤波，消除小斑点， 可调整参数
    plt.imshow(dst_open)
    plt.show()
    # segmented = dst_open.astype(np.int8)

    # if return_contour:
    #     contours = measure.find_contours(segmented, level=0)
    #     contours = sorted(contours, reverse=True, key=lambda c: len(c))   # 保证按照轮廓长度逆序，第二个为肿瘤轮廓
    #     if save_path != None:
    #         np.save(save_path + '/' + name + '.npy', contours[1])
    #     return contours[1]
    # else:
    #     plt.figure('morphology',figsize=(10, 10))
    #     plt.subplot(121)
    #     plt.imshow(converted)
    #     plt.title('原始图片')
    #     plt.axis('off')
    #     plt.subplot(122)
    #     plt.imshow(segmented)
    #     plt.title('定位结果')
    #     plt.axis('off')
    #     plt.show()
    #     if save_path != None:
    #         scipy.misc.toimage(segmented).save(save_path + '/' + name + '.jpg')
    #     return 


if __name__ == '__main__':
    # detectLine('histEq_1115892100016.dcm')
    # detectLine('./tempImage.jpg')
    # cv2New()
    # another_fun()
    s = segmentTumor('./TempData/histEq_brain_1142709300012.dcm', './TempData/', False)
    