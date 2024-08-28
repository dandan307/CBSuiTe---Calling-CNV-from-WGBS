'''
ECOLÉ samples preprocessing source code for training.
This script generates and processes the samples to perform CNV call training.
'''
from decimal import DecimalException
import numpy as np
from sklearn.metrics import confusion_matrix as cm
from tensorflow.keras.preprocessing import sequence
import torch
from performer_pytorch import Performer
from einops import rearrange, repeat
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader,Dataset
import pandas as pd
import os
from itertools import groupby
from tqdm import tqdm
import argparse
import datetime




description = "ECOLÉ is a deep learning based WES CNV caller tool. \
            For academic research the software is completely free, although for \
            commercial usage the software is licensed. \n \
            please see ciceklab.cs.bilkent.edu.tr/ECOLÉlicenceblablabla."

parser = argparse.ArgumentParser(description=description)

required_args = parser.add_argument_group('Required Arguments')

required_args.add_argument("-i", "--input", help="Please provide the path of the input to data.\n This path should contain .npy files for each human sample.", required=True)
required_args.add_argument("-o", "--output", help="Please provide the path of the output to data.\n This path should contain .npy files for each human sample.", required=True)
required_args.add_argument("-m", "--menu", help="Please provide the path of the output to data.\n This path should contain .npy files for each human sample.")

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
            
        crs = temp_chrs[i]
        '''
        if(len(crs) == 4):
            if(crs[3] == "Y"):
                crs = 23
            elif (crs[3] == "X"):
                crs = 24
            else:
                crs = int(crs[3])
        elif(len(crs) == 5):
            crs = int(crs[3:5])
        '''
        temp_chrs[i] = crs
        temp_readdepths[i] = list(temp_readdepths[i])
        temp_methy[i] = list(temp_methy[i])
        combined_list = temp_readdepths[i] + temp_methy[i]
        combined_list.extend([temp_gc[i], temp_end_inds[i], temp_start_inds[i], crs])   #在序列末尾插入个0，end，start和chr
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