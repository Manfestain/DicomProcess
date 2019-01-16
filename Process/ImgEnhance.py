# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 18/12/3
'''
    对原始dcm图像进行增强
'''
import os
import SimpleITK as sitk
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from math import ceil
from skimage import exposure
from scipy.stats import tmax, tmin

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 直方图均值化主函数，可选择窗口大小进行均值化
def doHistEqualiztion(array, height=0, width=0):
    if height == 0 & width == 0:
        equalized = exposure.equalize_hist(array)
    else:
        w,h = array.shape
        equalized = np.ndarray((w, h))
        for i in range(ceil(w/width)):
            for j in range(ceil(h/height)):
                equalized[i*width:i*width + width, j*height:h*height + height] = exposure.equalize_hist(array[i*width:i*width + width, j*height:h*height + height])

    return equalized

# 直方图均值化，增强脑区的对比度
def histEqualiztion(path, save_path, plot_hist=False):
    file_name = path[path.rfind('/')+1:]

    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)
    array = image_array[0, :, :]

    # equalized = exposure.equalize_hist(array)   # 直方图均衡化
    equalized = doHistEqualiztion(array)

    if plot_hist:
        plt.subplot(121)
        plt.hist(array)
        plt.title('原始图像直方图')
        plt.subplot(122)
        plt.hist(equalized)
        plt.title('直方图均衡化')
        plt.show()

    x_min = tmin(equalized, axis=None)
    x_max = tmax(equalized, axis=None)
    result = (equalized-x_min)/(x_max-x_min)*(1000 - -2048) + -2048   # 修改像素值的数据范围到(-2014, 1000)之间

    result[result < -500] = result[0, 0]
    result = result.astype(np.int16)
    result = result.reshape(image_array.shape)
    img = sitk.GetImageFromArray(result)
    sitk.WriteImage(img, save_path + '/' + '2_histEq_' + file_name)


if __name__ == '__main__':
    # path = 'E:/new_data(18-12)/Diseased/LGG/Qiu_xiu_xia/1121704600009.dcm'
    path_1 = './1121704600009.dcm'
    path_2 = './1115892100016.dcm'
    path_3 = './TempData/brain_1142709300012.dcm'
    
    histEqualiztion(path_3, './TempData/')