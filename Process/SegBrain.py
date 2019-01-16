# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 18/12/3
'''
    将脑实质分割出来
'''
import numpy as np
import skimage
import pydicom
import skimage.morphology
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure
import cv2

# 标注脑部最大体积
def largestLabelVolume(image, fill_brain_structures=True):
    binary_image_1 = np.array(image < 150 , dtype=np.int8) + 1   # 可调整参数
    binary_image_2 = np.array(image >=-10, dtype=np.int8) + 1   # 可调整参数
    binary_image = binary_image_2 + binary_image_1
    labels = measure.label(binary_image)

    bg = labels[0, 0]   # 去掉背景，选取最大面积的组织（也就是脑组织）
    vals, counts = np.unique(labels, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]

    l_max = vals[np.argmax(counts)]
    binary_image[labels != l_max] = 1   # 将非最大组织的像素置为1

    if fill_brain_structures:
        binary_image[binary_image != 1] = 0 # 将脑区置为0，然后反转变为1，与原始图像相乘得到脑区像素
        binary_image = 1 - binary_image
        brain_image = binary_image * image
        brain_image[brain_image == 0] = 0
        return brain_image

    # plt.imshow(binary_image)
    # plt.show()
    # print(binary_image_2)
    # print(binary_image_2[350, :])
    return binary_image

# 根据路径读取图像，处理完保存脑实质图像
def segmentBrainMask(path, save_path, fill_brain_structures=True):
    file_name = path[path.rfind('/')+1:]
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)
    print(image_array.shape)
    brain_image = largest_label_volume(image_array[0, :, :], fill_brain_structures).reshape(image_array.shape)
    # # image_array[:, :, :] = brain_image[:, :, :]
    
    brain_image = brain_image.astype(np.int16)
    brain = sitk.GetImageFromArray(brain_image)
    # pydicom.filewriter.dcmwrite('./pydicomtest.dcm', image, write_like_original=True)
    sitk.WriteImage(brain, save_path + '/' + 'brain_' + file_name)




if __name__ == '__main__':
    path = 'E:/new_data(18-12)/Diseased/LGG/Qiu_xiu_xia/1121704600009.dcm'
    path_1 = 'E:/new_data(18-12)/Diseased/LGG/Xu_rong_you/1115892100016.dcm'
    path_2 = 'E:/new_data(18-12)/Diseased/HGG/Chen_wei_zhong/1142709300012.dcm'
    segment_brain_mask(path_2, './TempData/')
    # root_dir = '/new_data(18-12)/Diseased/HGG/'  #根目录
    # names = [root_dir + '/' f for f in os.listdir(root_dir)]   # 取得该目录下的所有病人文件夹
    # for name in names:
    #     files = [name + '/' + f for f in os.listdir(name)]   #取得该病人文件夹下的所有切片
    #     for file in files:
    #         segment_brain_mask(file, '对应的保存目录')


    