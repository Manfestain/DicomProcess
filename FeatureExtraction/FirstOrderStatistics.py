# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/1/3
'''
    计算二维矩阵的一阶统计特征
'''

import os
import math
import numpy as np

# 判断该矩阵是否是二维矩阵，返回True表示是二维矩阵
def is2DArray(array):
    demension = np.array(array).shape
    return True if demension == 2 else False

# 判断该路径是否存在，不存在则创建并返回True，存在则直接返回False
def isPathExists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

# 一阶统计特征
class FirstOrderStatistics(object):
    def __init__(array):
        if not is2DArray(array):
            print('数据维度不符合！')
            exit()

        self.array = array
        self.width = array.shape[0]
        self.height = array.shape[1]
        self.N = self.width * self.height

    def getFirstOrderStatistics():
        pass

    # 能量
    def getEnergy(array):
        square = np.multiply(array, array)
        return square.sum()

    # 熵
    def getEntropy(array):
        pass

    # 峰度
    def getKurtosis(array):
        N = self.N
        numerator = np.power(array - array.mean(), 4).sum() / N
        denominator = np.power(np.sqrt(np.power(array - array.mean(), 2).sum() / N), 2)
        return numerator / denominator

    # 最大值
    def getMaximum(array):
        return array.max()

    # 最小值
    def getMinimum(array):
        return array.min()

    # 均值
    def getMean(array):
        return array.mean()

    def getAbsoluteDeviation(array):
        pass

    # 中值
    def getMedian(array):
        return array.median()

    # 波动范围
    def getRange(array):
        max_pixel = np.max(array)
        min_pixel = np.min(array)
        return max_pixel - min_pixel

    # 均方根
    def getRootMeanSquare(array):
        N = self.N
        under_sqrt = (np.multiply(array, array).sum()) / N
        return np.sqrt(under_sqrt)

    # 偏度
    def getSkewness(array):
        N = self.N
        numerator = np.power(array - array.mean(), 3).sum() / N
        denominator = np.power(np.sqrt(np.power(array - array.mean(), 2).sum() / N), 3)
        return numerator / denominator

    # 标准偏差
    def getStandardDeviation(array):
        N = self.N
        numerator = np.power(array - array.mean(), 2).sum()
        denominator = N - 1
        return np.power(numerator / denominator, 0.5)

    # 联合度
    def getUniformity(array):
        pass

    # 方差
    def getVariance(array):
        N = self.N
        numerator = np.power(array - array.mean(), 2).sum()
        denominator = N - 1
        return numerator / denominator


if __name__ == '__main__':
    if 1 == 1:
        print('程序正常退出！')
        exit()
    else:
        print('异常退出')