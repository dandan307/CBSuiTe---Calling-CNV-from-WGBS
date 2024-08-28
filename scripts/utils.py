import os
import torch
import numpy as np
import datetime


'''
Helper function to print informative messages to the user.
'''
def message(message):
    print("[",datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"]\t", message)


'''
Helper function to convert string chromosome to integer.
'''
def convert_chr_to_integer(chr_list):
    for i in range(len(chr_list)):
            crs = chr_list[i]
            if(len(crs) == 4):
                if(crs[3] == "Y"):
                    crs = 23
                elif (crs[3] == "X"):
                    crs = 24
                else:
                    crs = int(crs[3])
            elif(len(crs) == 5):
                crs = int(crs[3:5])
            chr_list[i] = crs

def calculate_mean_std(directory_path):
    # 获取目录中的所有.npy文件
    npy_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]
    print('file list: ', npy_files)
    # 用于存储所有倒数第二列数据的列表
    depth_all = []

    # 逐个读取.npy文件并提取倒数第二列数据
    for npy_file in npy_files:
        file_path = os.path.join(directory_path, npy_file)
        data = np.load(file_path, allow_pickle=True)
        
        depth = data[:, 4]
        depth_all.extend(depth)

    # 合并所有数据列
    depth_all = np.concatenate(depth)
    depth_all = np.array(depth_all)

    # 计算平均值和标准差
    mean_value = np.mean(depth_all)
    std_deviation = np.std(depth_all)
    
    return mean_value, std_deviation


def calculate_mean_std_single(sampledata):
    # 用于存储所有倒数第二列数据的列表
    depth_all = []


    data = sampledata
    
    depth = data[:, 4]
    depth_all.extend(depth)

    # 合并所有数据列
    depth_all = np.concatenate(depth)
    depth_all = np.array(depth_all)

    # 计算平均值和标准差
    mean_value = np.mean(depth_all)
    std_deviation = np.std(depth_all)
    
    return mean_value, std_deviation