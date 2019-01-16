# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 18/12/1
'''
    将整理的切片信息导入excel中
    格式：编号，病人编号，姓名，性别，年龄，病症，文件名
    其中，性别按1（男），2（女）进行编号；病症按1（正常），2（良性）和3（恶性）进行编号。
'''
import os
import pydicom
import numpy as np
import pandas as pd

root_dir = 'E:/new_data(18-12)/'
folder = ['Diseased/LGG', 'Diseased/HGG', 'Undiseased']
save_name = ['LGG.csv', 'HGG.csv', 'Undiseased.csv']
disease_dict = {'Undiseased': 1, 'LGG': 2, 'HGG': 3}
sex_dict = {'M': 1, 'F': 2}

def creat_dataIndex(path, save_path):
    num = 0
    columns = ['编号', '病人ID', '姓名', '性别', '年龄', '病症', '文件名']
    df = pd.DataFrame(columns=columns)
    patients = [path + '/' + s for s in os.listdir(path)]
    for p in patients:
        slices = [p + '/' + s for s in os.listdir(p)]
        for s in slices:
            ds = pydicom.dcmread(s)
            file_name = s[s.rfind('/')+1:]
            disease = disease_dict[path[path.rfind('/')+1:]]
            row = [num, ds.PatientID, ds.PatientName, sex_dict[str(ds.PatientSex)], ds.PatientAge[1:3], disease, file_name]
            df.loc[num] = row
            num += 1
    df.to_csv(save_path, encoding='gb2312', index=False)
    
if __name__ == '__main__':
    for f in folder:
        creat_dataIndex(root_dir+f, root_dir+save_name[folder.index(f)])