import numpy as np
import os
from os import listdir
import pandas as pd
from tqdm import tqdm
import argparse

from utils import message, label_encode, chr_encode, label_encode_str, filter_lowqua_data

description = "hi"

parser = argparse.ArgumentParser(description=description)

required_args = parser.add_argument_group('Required Arguments')
opt_args = parser.add_argument_group("Optional Arguments")

required_args.add_argument("-dt", "--depth_tumor", help="Please provide readdepth path of tumor sample.", required=True)
required_args.add_argument("-dn", "--depth_normal", help="Please provide readdepth path of normal sample .", required=True)

required_args.add_argument("-o", "--output", help="Please provide the output path for the preprocessing.", required=True)
required_args.add_argument("-methy", "--methy", help="Please provide the methylation path.", required=True)
required_args.add_argument("-m", "--menu", help="Menu of paired WGS label and WGBS data.", required=True)
opt_args.add_argument("-down", "--downsample", help="Downsample Test.")

parser.add_argument("-V", "--version",   help="show program version", action="store_true")
args = parser.parse_args()
print(args.gc)

readdepths_path_t = args.depth_tumor
readdepths_path_n = args.depth_normal
output_path = args.output
os.makedirs(output_path, exist_ok=True)
if args.downsample:
    down = args.downsample
    readdepth_files = [file for file in listdir(readdepths_path_t) if not file.startswith('.') and file.endswith(f'.{down}.bam.txt')]
    normal_readdepth_files = [file for file in listdir(readdepths_path_n) if not file.startswith('.') and file.endswith(f'.{down}.bam.txt')]
else:
    readdepth_files = [file for file in listdir(readdepths_path_t) if not file.startswith('.')]
    normal_readdepth_files = [file for file in listdir(readdepths_path_n) if not file.startswith('.')]

#message(f"readdepth_files: {readdepth_files}")
readdepth_files.sort()
if args.menu:
    menu = pd.read_csv(args.menu, sep='\s+', names=['WGBS_tumor', 'WGBS_normal'])
    print(menu)
binsize = 100000
depthsize = 100
seq_length = int(binsize / depthsize)

# 注意！！target_data，depth_list，label_list的染色体顺序必须一致！
chromosome_lengths = {
    1: 248956422, 10: 133797422, 11: 135086622, 12: 133275309, 13: 114364328, 
    14: 107043718, 15: 101991189, 16: 90338345, 17: 83257441, 18: 80373285, 
    19: 58617616, 2: 242193529, 20: 64444167, 21: 46709983, 22: 50818468, 
    3: 198295559, 4: 190214555, 5: 181538259, 6: 170805979, 7: 159345973, 
    8: 145138636, 9: 138394717, #23: 156040895, 24: 57227415,  #X：23， Y：24
}

# 创建target bed
num_rows = sum(length // binsize + 1 for length in chromosome_lengths.values())
target_data = np.zeros((num_rows, 3), dtype=int)
index = 0
for chromosome, length in chromosome_lengths.items():
    for start in (range(2781479, length, binsize) if chromosome == 24 else range(0, length, binsize)):
        end = length if start + binsize > length else start + binsize
        target_data[index] = [chromosome, start, end]
        index += 1
target_data = target_data[~np.all(target_data == 0, axis=1)]
#print(np.shape(target_data))

gc_data = pd.read_csv('GC_100K_noxy.bed', sep="\t")
gc_values = gc_data.iloc[:, 4]
gc_values = np.array(gc_values).reshape(-1, 1)
print('gc_values: ',np.shape(gc_values))
    
for wgbs_t_file in readdepth_files:
    sample_name = wgbs_t_file.split(".")[0]
    if "_" in sample_name:
        sample_name = sample_name.split("_")[0]

    if args.menu:
        if sample_name in menu['WGBS_tumor'].values.astype(str):  # 检查 sample_name 是否在 'WGBS' 列中
            wgbs_n_name = menu.loc[menu['WGBS_tumor'] == sample_name, 'WGBS_normal'].values[0]   # 选择对应行的 'WGBS' 列数据
            wgbs_n_file = wgbs_n_name + "_" + wgbs_t_file.split("_", 1)[1]
            #print(wgbs_n_file)
            if wgbs_n_file not in normal_readdepth_files:
                continue
            message(f"WGBS normal files: {wgbs_n_file}")
        else:
            continue
    else:
        print("no menu")
        wgbs_n_file = normal_readdepth_files[i]
        message(f"WGBS normal files: {wgbs_n_file}")

    name_list = [sample_name] * len(target_data)
    print('name_list: ',np.shape(name_list))
    
    methy_path = os.path.join(args.methy, sample_name + '_methy.bed')
    if not os.path.exists(methy_path):          # 检查文件是否存在
        print('no methy information, continue')
        continue
    for i in range(1, 23):
        array_name = f'methy_{i}'
        array_length = chromosome_lengths[i] // depthsize + 1
        #print(f"chr{i} length: {array_length}")
        locals()[array_name] = np.zeros(array_length, dtype=np.float)
    methy_dict = {
        1: methy_1, 10: methy_10, 11: methy_11, 12: methy_12, 13: methy_13,
        14: methy_14, 15: methy_15, 16: methy_16, 17: methy_17, 18: methy_18,
        19: methy_19, 2: methy_2, 20: methy_20, 21: methy_21, 22: methy_22,
        3: methy_3, 4: methy_4, 5: methy_5, 6: methy_6, 7: methy_7,
        8: methy_8, 9: methy_9, #23: methy_23, 24: methy_24  # X: 23, Y: 24
    }
    # 读取read depth 文件，并写入对应染色体的对应位置
    message(f"methy_files: {methy_path}")
    with open(methy_path, 'r') as f:
        header = next(f)
        for line in f:
            value = line.strip().split('\t')
            chr, pos = chr_encode(value[0]), int(int(value[1])//depthsize)
            #print(chr, pos)
            met = 0 if value[3] == 'NA' else float(value[3])
            if int(chr) > 22:
                break
            methy_cur = methy_dict.get(int(chr))
            methy_cur[pos] = met
    # 划分区域
    methy_list = []
    for key, data in methy_dict.items():
        for i in range(0, len(data), seq_length):
            methy_arr = data[i : i+seq_length]
            methy_list.append(methy_arr)
    print('methy_list: ', np.shape(methy_list))

    # 创建24个数组用于depth匹配:depth_1, depth_2, ..., depth_24
    for i in range(1, 23):
        array_name = f'depth_t_{i}'
        array_length = chromosome_lengths[i] // depthsize + 1
        locals()[array_name] = np.zeros(array_length, dtype=np.float)
        array_name = f'depth_n_{i}'
        array_length = chromosome_lengths[i] // depthsize + 1
        locals()[array_name] = np.zeros(array_length, dtype=np.float)
    depth_dict_t = {
        1: depth_t_1, 10: depth_t_10, 11: depth_t_11, 12: depth_t_12, 13: depth_t_13,
        14: depth_t_14, 15: depth_t_15, 16: depth_t_16, 17: depth_t_17, 18: depth_t_18,
        19: depth_t_19, 2: depth_t_2, 20: depth_t_20, 21: depth_t_21, 22: depth_t_22,
        3: depth_t_3, 4: depth_t_4, 5: depth_t_5, 6: depth_t_6, 7: depth_t_7,
        8: depth_t_8, 9: depth_t_9, #23: depth_t_23, 24: depth_t_24  # X: 23, Y: 24
    }
    depth_dict_n = {
        1: depth_n_1, 10: depth_n_10, 11: depth_n_11, 12: depth_n_12, 13: depth_n_13,
        14: depth_n_14, 15: depth_n_15, 16: depth_n_16, 17: depth_n_17, 18: depth_n_18,
        19: depth_n_19, 2: depth_n_2, 20: depth_n_20, 21: depth_n_21, 22: depth_n_22,
        3: depth_n_3, 4: depth_n_4, 5: depth_n_5, 6: depth_n_6, 7: depth_n_7,
        8: depth_n_8, 9: depth_n_9, #23: depth_n_23, 24: depth_n_24  # X: 23, Y: 24
    }

    # 读取read depth 文件，并写入对应染色体的对应位置
    file_path1 = os.path.join(readdepths_path_t, wgbs_t_file)
    file_path2 = os.path.join(readdepths_path_n, wgbs_n_file)
    message(f"readdepth files tumor: {wgbs_t_file}")
    message(f"readdepth files normal: {wgbs_n_file}")
    with open(file_path1, 'r') as f1, open(file_path2, 'r') as f2:
        header1 = next(f1)
        header2 = next(f2)
        for l1,l2 in zip(f1,f2):
            value = l1.strip().split('\t')[:5]
            depth_n = float(l2.strip().split('\t')[4])     #第5个元素，depth
            chr, pos, depth_t = chr_encode(value[0]), int(int(value[1])//depthsize), float(value[4])
            if int(chr) > 22:
                break
            depth_cur_t = depth_dict_t.get(int(chr))
            depth_cur_t[pos] = depth_t
            depth_cur_n = depth_dict_n.get(int(chr))
            depth_cur_n[pos] = depth_n


    depth_list = []
    
    for key_t, data_t in depth_dict_t.items():
        data_n = depth_dict_n.get(key_t)
        for i in range(0, len(data_t), seq_length):
            depth_arr_t = data_t[i : i+seq_length]
            depth_arr_n = data_n[i : i+seq_length]
            #depth_arr_t = filter_lowqua_data(depth_arr_t)
            #depth_arr_n = filter_lowqua_data(depth_arr_n)
            depth_arr = []
            for dt, dn in zip(depth_arr_t, depth_arr_n):
                if dt == 0 or dn == 0:
                    dep = 0
                else:
                    dep = np.log2(dt/dn)
                depth_arr.append(dep)
            depth_list.append(depth_arr)


    flattened_depth = np.concatenate(depth_list)
    non_zero_elements = flattened_depth[flattened_depth != 0]
    total_mean = np.mean(non_zero_elements)
    total_std = np.std(non_zero_elements)
    print("mean: ", total_mean, "std: ", total_std)

    z_normalized_list = []
    count = 0 
    for elem in depth_list:
    # Check if sum of element is not 0
        if np.sum(elem) != 0:
            z_normalized_elem = (elem - total_mean) / total_std     # Z-score normalize
            z_normalized_list.append(z_normalized_elem)
        else:
            # If sum of element is 0, append the element as it is
            z_normalized_list.append(elem)
            count += 1
    print('low qua array: ', count)
    print('depth_list: ', np.shape(z_normalized_list))
    
    name_list = np.array(name_list).reshape(-1, 1)
    z_normalized_list = np.array(z_normalized_list).reshape(-1, 1)
    methy_list = np.array(methy_list).reshape(-1, 1)
    assert name_list.shape[0] == target_data.shape[0] == z_normalized_list.shape[0]
    final_data = np.hstack((name_list, target_data, z_normalized_list, gc_values, methy_list,))
    #print('final_data: ',np.shape(final_data))

    # 筛选掉depth为全0的行
    filtered_data = []

    for row in final_data:
        sum_of_array = np.sum(row[4])
        #if sum_of_array != 0 and row[0]!= 23 and row[0]!= 24:
        if sum_of_array != 0:
            filtered_data.append(row)

    # 将筛选后的数据转换回NumPy数组
    final_data = np.array(filtered_data)
    print('final_data: ',np.shape(final_data))
        
    if args.downsample:
        down = args.downsample
        np.save(os.path.join(output_path, f"{sample_name}_{down}_final_data.npy"), final_data)
    else:
        np.save(os.path.join(output_path, f"{sample_name}_final_data.npy"), final_data)
    print(f'{sample_name} preprocess complete!')
    print("____________________")
