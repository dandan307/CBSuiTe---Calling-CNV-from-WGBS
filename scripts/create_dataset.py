'''
CBSuiTe samples preprocessing source code for training.
'''

import numpy as np
from sklearn.metrics import confusion_matrix as cm
from tensorflow.keras.preprocessing import sequence
import os
from itertools import groupby
from tqdm import tqdm
import argparse
import datetime

description = "hi"

parser = argparse.ArgumentParser(description=description)

required_args = parser.add_argument_group('Required Arguments')

required_args.add_argument("-i", "--input", help="Please provide the path of the input to data.", required=True)
required_args.add_argument("-o", "--output", help="Please provide the path of the output to data.", required=True)
required_args.add_argument("-m", "--menu", help="Please provide the menu of data.")

parser.add_argument("-V", "--version", help="show program version", action="store_true")
args = parser.parse_args()

data_path = args.input
out_path = args.output

sample_files = os.listdir(data_path)
all_samples_names = [item.split("_")[0] for item in sample_files if item.endswith('.npy')]


DELETION_FOLDER = os.path.join(f'{out_path}_dataset',"deletion")
DUPLICATION_FOLDER = os.path.join(f'{out_path}_dataset',"duplication")
NOCALL_FOLDER = os.path.join(f'{out_path}_dataset',"nocall")

os.makedirs(f'{out_path}_dataset', exist_ok=True)
os.makedirs(DELETION_FOLDER, exist_ok=True)
os.makedirs(DUPLICATION_FOLDER, exist_ok=True)
os.makedirs(NOCALL_FOLDER, exist_ok=True)
print(f"creating the training dataset")

if args.menu:
    menu = args.menu
    with open(menu, 'r') as file:
        # 读取文件内容并去除每行末尾的换行符
        menu = [line.strip() for line in file.readlines()]
        print("menu: ", menu)


for item in tqdm(all_samples_names):
    file_name = item+"_labeled_data.npy"
    if args.menu:
        if file_name not in menu:
            continue
    gtcalls = np.load(os.path.join(data_path, item+"_labeled_data.npy"), allow_pickle=True)
    print(f"processing sample: {item}")

    sampnames_data = []
    chrs_data = []
    readdepths_data = []
    start_inds_data = []
    end_inds_data = []
    gc_data = []
    methy_data = []
    wgscalls_data = []

    temp_sampnames = gtcalls[:,0]
    temp_chrs = gtcalls[:,1]
    temp_start_inds = gtcalls[:,2]
    temp_end_inds = gtcalls[:,3]
    temp_readdepths = gtcalls[:,4]
    temp_gc = gtcalls[:,5]
    temp_methy = gtcalls[:,6]
    temp_wgscalls = gtcalls[:,7]

    
    for i in range(len(temp_chrs)):
            
        temp_readdepths[i] = list(temp_readdepths[i])
        temp_methy[i] = list(temp_methy[i])
        combined_list = temp_readdepths[i] + temp_methy[i]
        combined_list.extend([temp_gc[i], temp_end_inds[i], temp_start_inds[i], temp_chrs[i]])   #在序列末尾插入个0，end，start和chr
        temp_readdepths[i] = combined_list

    readdepths_data.extend(temp_readdepths)
    wgscalls_data.extend(temp_wgscalls)
    print("readdepths_data: ",np.shape(readdepths_data))

    lens = [len(v) for v in readdepths_data]

    lengthfilter = [True if v < 2005  else False for v in lens]

    readdepths_data = np.array(readdepths_data)[lengthfilter]
    wgscalls_data = np.array(wgscalls_data)[lengthfilter]

    wgscalls_data= wgscalls_data.astype(int)

    readdepths_data = np.array([np.array(k) for k in readdepths_data])
    readdepths_data = sequence.pad_sequences(readdepths_data, maxlen=2004,dtype=np.float32,value=-1)
    readdepths_data = readdepths_data[ :, None, :]
    tot_nc = 0
    tot_del = 0
    tot_dp = 0
    print("creating exon samples:")
    #for i in tqdm(range(len(wgscalls_data))):
    for i in range(len(wgscalls_data)):
        call_made =  wgscalls_data[i]
        exon_sample = readdepths_data[i]

        if call_made == 0:
            np.save(os.path.join(NOCALL_FOLDER,f"{item}_datapoint_{i}.npy"), exon_sample)
            tot_nc += 1
        elif call_made == 1:
            np.save(os.path.join(DELETION_FOLDER,f"{item}_datapoint_{i}.npy"), exon_sample)
            tot_del += 1
        elif call_made == 2:
            np.save(os.path.join(DUPLICATION_FOLDER,f"{item}_datapoint_{i}.npy"), exon_sample)
            tot_dp += 1
    print(tot_nc, tot_del, tot_dp)