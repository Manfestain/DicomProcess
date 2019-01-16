# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/1/4
'''
    提取二维矩阵的纹理特征信息，包括GLCM（灰度共生矩阵）、GLRLM（灰度游程矩阵）、LBP（局部二值模式）和HOG（方向梯度直方图）
'''
import os
import math
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from itertools import groupby

class GLCMFeatures(object):
    '''
    Gray-Level Co-Occurrence Matrix based features
    '''
    def __init__():
        pass

    def getGLCMFeatures():
        pass

    def calcuteIJ(comatrix):
        (num_level, num_level2, num_dist, num_angle) = comatrix.shape
        assert num_level == num_level2
        assert num_dist > 0
        assert num_angle > 0
        I, J = np.ogrid[0:num_level, 0:num_level]
        return I, J

    def getGrayLevelComatrix(array, distance, theta, levels):
        return greycomatrix(array, distance, theta, levels)

    # 1.
    def getContrast(comatrix):
        return greycoprops(comatrix, 'contrast')

    # 2.
    def getDissimilarity(comatrix):
        return greycoprops(comatrix, 'dissimilarity')

    # 3.
    def getHomogeneity1(comatrix):
        I, J = self.calcuteIJ(comatrix)
        weights = 1. / (1. + np.abs(I - J))
        return np.apply_over_axes(np.sum, (comatrix * weights), axes=(0, 1))[0, 0]

    # 4.
    def getHomogeneity2(comatrix):
        return greycoprops(comatrix, 'homogeneity')

    # 5.
    def getASM(comatrix):
        return greycoprops(comatrix, 'ASM')

    # 6.
    def getEnergy(comatrix):
        return greycoprops(comatrix, 'energy')

    # 7.
    def getCorrelation(comatrix):
        return greycoprops(comatrix, 'correlation')

    # 8.
    def getAutocorrelation(comatrix):
        I, J = self.calcuteIJ(comatrix)
        weights = I * J
        return np.apply_over_axes(np.sum, (comatrix * weights), axex=(0, 1))[0, 0]

    # 9.
    def getEntropy(comatrix):
        log = np.log2(comatrix)
        return - np.apply_over_axes(np.sum, (comatrix * log), axes=(0, 1))[0, 0]

    # 10.
    def getInverseVariance(comatrix):
        I, J = self.calcuteIJ(comatrix)
        equals = np.array(I == J, dtype=np.float)
        weights = 1. / ((I - J + equals) ** 2) - equals   # 计算i != j，此处先将i == j置为1，计算除法，完成之后将其再置为0
        return np.apply_over_axes(np.sum, (comatrix * weights), axes=(0, 1))[0, 0]

    # 11.
    def getSUMAverage(comatrix):
        pass

    # 12.
    def getSUMEntropy(comatrix):
        pass

    # 13.
    def getDifferenceEntropy(comatrix):
        pass

    # 14.
    def getSUMVariance(comatrix):
        pass


class GLRLMFeatures(object):
    '''
    Gray-Level Run-Length matrix based features
    '''
    def __init__(self, flag=1):
        self.flag = flag

    def getGLRLMFeatures(self):
        pass

    def getGrayLevelRumatrix(self, array, theta):
        '''
        计算给定图像的灰度游程矩阵
        参数：
        array: 输入，需要计算的图像
        theta: 输入，计算灰度游程矩阵时采用的角度，list类型，可包含字段:['deg0', 'deg45', 'deg90', 'deg135']
        glrlm: 输出，灰度游程矩阵的计算结果
        '''
        P = array
        x, y = P.shape
        min_pixels = np.min(P)   # 图像中最小的像素值
        run_length = max(x, y)   # 像素的最大游行长度
        num_level = np.max(P) - np.min(P) + 1   # 图像的灰度级数

        deg0 = [val.tolist() for sublist in np.vsplit(P, x) for val in sublist]   # 0度矩阵统计
        deg90 = [val.tolist() for sublist in np.split(np.transpose(P), y) for val in sublist]   # 90度矩阵统计
        diags = [P[::-1, :].diagonal(i) for i in range(-P.shape[0]+1, P.shape[1])]   #45度矩阵统计
        deg45 = [n.tolist() for n in diags]
        Pt = np.rot90(P, 3)   # 135度矩阵统计
        diags = [Pt[::-1, :].diagonal(i) for i in range(-Pt.shape[0]+1, Pt.shape[1])]
        deg135 = [n.tolist() for n in diags]

        def length(l):
            if hasattr(l, '__len__'):
                return np.size(l)
            else:
                i = 0
                for _ in l:
                    i += 1
                return i

        glrlm = np.zeros((num_level, run_length, len(theta)))   # 按照统计矩阵记录所有的数据， 第三维度表示计算角度
        for angle in theta:
            for splitvec in range(0, len(eval(angle))):
                flattened = eval(angle)[splitvec]
                answer = []
                for key, iter in groupby(flattened):   # 计算单个矩阵的像素统计信息
                    answer.append((key, length(iter)))   
                for ansIndex in range(0, len(answer)):
                    glrlm[int(answer[ansIndex][0]-min_pixels), int(answer[ansIndex][1]-1), theta.index(angle)] += 1   # 每次将统计像素值减去最小值就可以填入GLRLM矩阵中
        return glrlm


    def apply_over_degree(self, function, x1, x2):
        rows, cols, nums = x1.shape
        result = np.ndarray((rows, cols, nums))
        for i in range(nums):
            print(x1[:, :, i])
            result[:, :, i] = function(x1[:, :, i], x2)
            print(result[:, :, i])
        result[result == np.inf] = 0
        result[np.isnan(result)] = 0
        return result

    def calcuteIJ(self, rlmatrix):
        gray_level, run_length, _ = rlmatrix.shape
        I, J = np.ogrid[0:gray_level, 0:run_length]
        return I, J+1

    def calcuteS(self, rlmatrix):
        return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]

        
    # 1. SRE
    def getShortRunEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
        
    # 2. LRE
    def getLongRunEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 3. GLN
    def getGrayLevelNonUniformity(self, rlmatrix):
        G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
        numerator = np.apply_over_axes(np.sum, (G*G), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 4. RLN
    def getRunLengthNonUniformity(self, rlmatrix):
        R = np.apply_over_axes(np.sum, rlmatrix, axes=0)
        numerator = np.apply_over_axes(np.sum, (R*R), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 5. RP
    def getRunPercentage(self, rlmatrix):
        gray_level, run_length, _ = rlmatrix.shape
        num_voxels = gray_level * run_length
        return self.calcuteS(rlmatrix) / num_voxels

    # 6. LGLRE
    def getLowGrayLevelRunEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (I*I)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 7. HGLRE
    def getHighGrayLevelRunEmphais(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (I*I)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 8. SRLGLE
    def getShortRunLowGrayLevelEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 9. SRHGLE
    def getShortRunHighGrayLevelEmphasis(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        temp = self.apply_over_degree(np.multiply, rlmatrix, (I*I))
        print('-----------------------')
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 10. LRLGLE
    def getLongRunLow(self, rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        temp = self.apply_over_degree(np.multiply, rlmatrix, (J*J), axes=(0, 1))
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 11. LRHGLE
    def getLongRunHighGrayLevelEmphais(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
    

class LBPFeatures(object):
    def __init__():
        pass


class HOGFeatures(object):
    def __init__(self):
        pass

    def getHOGFeatures(self):
        pass

    # 计算矩阵水平和垂直方向的梯度，在边界区域有两种计算方法：1.使用0填充；2.使用单个值计算梯度
    def getGradient(self, array, boundary):
        g_row = np.zeros(array.shape)
        g_row[0, :] = 0
        g_row[-1, :] = 0
        g_row[1:-1, :] = array[2:, :] - array[:-2, :]

        g_col = np.zeros(array.shape)
        g_col[:, 0] = 0
        g_col[:, -1] = 0
        g_col[:, 1:-1] = array[:, 2:] - array[:, :-2]

        if boundary:
            g_row[0, :] = array[1, :] - array[0, :]
            g_row[-1, :] = g_row[-1, :] - g_row[-2, :]

            g_col[:, 0] = array[:, 1] - array[:, 0]
            g_col[:, -1] = array[:, -1] - array[:, -2]

        return g_row, g_col

    # 计算梯度的大小和方向
    def getMagnitudeOrientation(self, gradient_x, gradient_y):
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        orientation = (arc)

    def getHOGMatrix(self, array):
        gradient_x, gradient_y = self.getGradient(array, False)


if __name__ == '__main__':
    Img = np.array([[5, 2, 5, 4, 4],
                    [3, 3, 3, 1, 3],
                    [2, 1, 1, 1, 3],
                    [4, 2, 2, 2, 3],
                    [3, 5, 3, 3, 2]])
    Img2 = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 2, 2],
                     [0, 0, 2, 2]])
    
    glrlm = GLRLMFeatures(1)
    result = glrlm.getGrayLevelRumatrix(Img2, ['deg45'])
    print('\nresult:')
    print(result[:, :, :])
    print('---------------------')
    # glrlm.getShortRunLowGrayLevelEmphasis(result)
    data = glrlm.getShortRunHighGrayLevelEmphasis(result)
    print('-----------------------')
    print(data)
    # print(glrlm.getShortRunEmphasis(result))
    # print(glrlm.getGrayLevelNonUniformity(result))
    # temp = np.arange(4)
    # dat = np.true_divide(result[:, :, 0], (temp*temp))
    # print(dat)

    # Img3 = np.array([[2, 4, 6],
    #                  [4, 6, 8],
    #                  [6, 8, 10]])
    # print(np.divide(Img3, [1, 2, 3]))



    # image = np.array([[0, 0, 1, 1],
    #                   [0, 0, 1, 1],
    #                   [0, 2, 2, 2],
    #                   [2, 2, 3, 3]], dtype=np.uint8)
    # result = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=4)
    # print(result.shape)
    # print(result)
    # print('--------------------------------------')
    # (num_level, num_level2, num_dist, num_angle) = result.shape
    # # assert num_level == num_level2
    # # assert num_dist > 0
    # # assert num_angle > 0
    # I, J = np.ogrid[0:num_level, 0:num_level]
    # print(I, '\n')
    # print(J)
    # # print('--------------------------------------')
    # # print(I * J, '\n')
    # print(I - J )
    # print('--------------------------------------')
    # I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
    # J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
    # print(I, '\n')
    # print(J)
    # print('--------------------------------------')
    # diff_i = I - np.apply_over_axes(np.sum, (I * result), axes=(0, 1))[0, 0]
    # diff_j = J - np.apply_over_axes(np.sum, (J * result), axes=(0, 1))[0, 0]
    # print(diff_i, '\n')
    # print(diff_j)
    # print('--------------------------------------')
    # print(I)
    # print('>>>>>>>>>>>>>>>>>>')
    # print(result)
    # print('>>>>>>>>>>>>>>>>>>')
    # print(I * result)
    # print(np.apply_over_axes(np.sum, (I * result), axes=(0, 1)))
    # a = np.array([2, 4])
    # aa = np.array([[3],
    #                [2]])
    # print(a * aa)
    # print(-a * np.log2(a))
    