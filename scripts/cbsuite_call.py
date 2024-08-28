'''
ECOLÉ source code.
ECOLÉ is a deep learning based WES CNV caller tool.
This script, ECOLÉ_call.py, is only used to load the weights of pre-trained models
and use them to perform CNV calls.
'''
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
import gc

from net_my_gm_v2 import CNVcaller
from utils import message, convert_chr_to_integer, calculate_mean_std_single


cur_dirname = os.path.dirname(__file__)
try:
    os.mkdir(os.path.join(cur_dirname,"../tmp/"))
    os.mkdir(os.path.join(cur_dirname,"../tmp2/"))
except OSError:
    print ("Creation of the directory failed")
else:
    print ("Successfully created the directory")



''' 
Perform I/O operations.
'''

description = "ECOLÉ is a deep learning based WES CNV caller tool. \
            For academic research the software is completely free, although for \
            commercial usage the software is licensed. \n \
            please see ciceklab.cs.bilkent.edu.tr/ECOLÉlicenceblablabla."

parser = argparse.ArgumentParser(description=description)

'''
Required arguments group:
(i) -m, pretrained models of the paper, one of the following: (1) ecole, (2) ecole-ft-expert, (3) ecole-ft-somatic. 
(ii) -i, input data path comprised of WES samples with read depth data.
(iii) -o, relative or direct output directory path to write ECOLÉ output file.
(v) -c, Depending on the level of resolution you desire, choose one of the options: (1) exonlevel, (2) merged
(vii) -n, The path for mean&std stats of read depth values.
'''

required_args = parser.add_argument_group('Required Arguments')
opt_args = parser.add_argument_group("Optional Arguments")

required_args.add_argument("-m", "--model", help="If you want to use pretrained ECOLÉ weights choose one of the options: \n \
                   (i) ecole \n (ii) ecole-ft-expert \n (iii) ecole-ft-somatic.", required=True)

required_args.add_argument("-bs", "--batch_size", help="Batch size to be used in the finetuning.", required=True)

required_args.add_argument("-i", "--input", help="Relative or direct path to input files for ECOLÉ CNV caller, these are the processed samples.", required=True)

required_args.add_argument("-o", "--output", help="Relative or direct output directory path to write ECOLÉ output file.", required=True)

required_args.add_argument("-c", "--cnv", help="Depending on the level of resolution you desire, choose one of the options: \n \
                                                (i) exonlevel, (ii) merged", required=True)

required_args.add_argument("-n", "--normalize", help="Please provide the path for mean&std stats of read depth values to normalize. \n \
                                                    These values are obtained precalculated from the training dataset before the pretraining.", required=True)
#required_args.add_argument("-menu", "--menu", help="Relative or direct path to input files for ECOLÉ CNV caller, these are the processed samples.", required=True)


opt_args.add_argument("-g", "--gpu", help="Specify gpu", required=False)

'''
Optional arguments group:
-v or --version, version check
-h or --help, help
-g or --gpu, specify gpu
-
'''

parser.add_argument("-V", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("ECOLÉ version 0.1")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    message("Using GPU!")
else:
    message("Using CPU!")

os.makedirs(args.output, exist_ok=True)
print("Output Dir: ",args.output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bs = int(args.batch_size)

model = CNVcaller(1000, 1, 3, 192, 3)


#model.load_state_dict(torch.load(os.path.join(cur_dirname,"../ecole_model_weights/ecole_ft192_depth3_exonsize1000_patchsize1_lr5e-05.pt")))
print("load model: ", args.model)
model.load_state_dict(torch.load(args.model, map_location = device))


model.eval()
model = model.to(device)

input_files = os.listdir(args.input)
all_samples_names = [file.split("_labeled_data.npy")[0] for file in input_files]

message("Calling for CNV regions...")
'''
menu = args.menu
with open(menu, 'r') as file:
    # 读取文件内容并去除每行末尾的换行符
    samplelist = [line.strip() for line in file.readlines()]
'''
for sample_name in tqdm(all_samples_names):
    
    #file_name = sample_name+"_labeled_data.npy"
    #if file_name not in samplelist:
    #    continue
    message(f"calling sample: {sample_name}")

    sampledata = np.load(os.path.join(args.input, sample_name+"_labeled_data.npy"), allow_pickle=True)
    


    sampnames_data = []
    chrs_data = []
    readdepths_data = []
    start_inds_data = []
    end_inds_data = []
    wgscalls_data = []

    temp_sampnames = sampledata[:,0]
    temp_chrs = sampledata[:,1]
    temp_start_inds = sampledata[:,2]
    temp_end_inds = sampledata[:,3]
    temp_readdepths = sampledata[:,4]
    temp_gc = sampledata[:,5]
    temp_methy = sampledata[:,6]

    if args.normalize:
        file1 = open(args.normalize, 'r')
        line = file1.readline()
        means_ = float(line.split(",")[0])
        stds_ = float(line.split(",")[1])
    else:
        means_, stds_ = calculate_mean_std_single(sampledata)
    print(means_, stds_)

    for i in range(len(temp_chrs)):
        
        crs = temp_chrs[i] 

        temp_chrs[i] = crs
        temp_readdepths[i] = list(temp_readdepths[i])
        temp_methy[i] = list(temp_methy[i])
        combined_list = temp_readdepths[i] + temp_methy[i]
        combined_list.extend([temp_gc[i], temp_end_inds[i], temp_start_inds[i], crs])   #在序列末尾插入个0，end，start和chr
        temp_readdepths[i] = combined_list

    sampnames_data.extend(temp_sampnames)
    chrs_data.extend(temp_chrs)
    readdepths_data.extend(temp_readdepths)
    start_inds_data.extend(temp_start_inds)
    end_inds_data.extend(temp_end_inds)

    lens = [len(v) for v in readdepths_data]

    lengthfilter = [True if v < 2005  else False for v in lens]

    sampnames_data = np.array(sampnames_data)[lengthfilter]
    chrs_data = np.array(chrs_data)[lengthfilter]
    readdepths_data = np.array(readdepths_data)[lengthfilter]
    start_inds_data = np.array(start_inds_data)[lengthfilter]
    end_inds_data = np.array(end_inds_data)[lengthfilter]

    readdepths_data = np.array([np.array(k) for k in readdepths_data])
    readdepths_data = sequence.pad_sequences(readdepths_data, maxlen=2004,dtype=np.float32,value=-1)
    readdepths_data = readdepths_data[ :, None, :]

    test_x = torch.FloatTensor(readdepths_data)
    test_x = TensorDataset(test_x)
    x_test = DataLoader(test_x, batch_size=bs)

    allpreds = []
    #for exons in tqdm(x_test):
    print("len(x_test): ", len(x_test))
    with torch.no_grad():
        for exons in x_test:

            exons = exons[0].to(device)
            exons[:,:,:1000] -= means_
            exons[:,:,:1000] /=  stds_

            mask = torch.logical_and(exons != -1, exons != 0)

            mask = torch.squeeze(mask)
            real_mask = torch.ones(exons.size(0),2005, dtype=torch.bool).to(device)
            real_mask[:,1:] = mask

            output1 = model(exons,real_mask)

            _, predicted = torch.max(output1.data, 1)
            
            preds = list(predicted.cpu().numpy().astype(np.int64))
            allpreds.extend(preds)        

    chrs_data = chrs_data.astype(int)
    allpreds = np.array(allpreds)
    print("allpreds: ", np.shape(allpreds))

    result = pd.DataFrame(columns=["chr", "start", "end", "prediction"])

    for j in range(1,25):
        indices = chrs_data == j

        predictions = allpreds[indices]
        start_inds = start_inds_data[indices]
        end_inds = end_inds_data[indices]

        sorted_ind = np.argsort(start_inds)
        predictions = predictions[sorted_ind]
     
        end_inds = end_inds[sorted_ind]
        start_inds = start_inds[sorted_ind]
        chr_ = "chr"
        if j < 23:
            chr_ = str(j)
        elif j == 23:
            chr_ = "X"
        elif j == 24:
            chr_ = "Y"

        for k_ in range(len(end_inds)):
            #os.makedirs(os.path.join(os.path.dirname(os.path.join(cur_dirname, "../", args.output)), sample_name + ".csv"), exist_ok=True)
            f = open(os.path.join(cur_dirname, "../", args.output, sample_name + ".csv"), "a")
            f.write(chr_ + "," + str(start_inds[k_]) + "," + str(end_inds[k_]) + ","+ str(predictions[k_]) + "\n")
            f.close()

    del sampledata, temp_sampnames, temp_chrs, temp_start_inds, temp_end_inds, temp_readdepths

    gc.collect()

message("call CNV complete!")
'''
message("Calling for regions without read depth information..")

binsize = 1000
chromosome_lengths = {
    1: 248956422, 10: 133797422, 11: 135086622, 12: 133275309, 13: 114364328, 
    14: 107043718, 15: 101991189, 16: 90338345, 17: 83257441, 18: 80373285, 
    19: 58617616, 2: 242193529, 20: 64444167, 21: 46709983, 22: 50818468, 
    3: 198295559, 4: 190214555, 5: 181538259, 6: 170805979, 7: 159345973, 
    8: 145138636, 9: 138394717, 23: 156040895, 24: 57227415,  #X：23， Y：24
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

os.makedirs(os.path.join('ecole_calls_output', args.output), exist_ok=True)

for sample_name in tqdm(all_samples_names):
 
    message(f"Processing sample: {sample_name}")

    out_folder = args.output
    if args.cnv == "exonlevel":
        f = open(os.path.join(args.output, sample_name + ".csv"), "a")
        f.write("Chromosome" + "\t" + "CNV Start Index" + "\t" + "CNV End Index" + "\t" + "ECOLÉ Prediction" + "\n")
        f.close()


    calls_ecole = pd.read_csv("./tmp/"+ sample_name + ".csv", sep=",", header=None).values
    #target_data = pd.read_csv("./hglft_genome_64dc_dcbaa0.bed", sep="\t", header=None).values

    #convert_chr_to_integer(target_data[:,0])
    convert_chr_to_integer(calls_ecole[:,0])
 
    chrs_data = calls_ecole[:,0].astype(int)
    allpreds = calls_ecole[:,3].astype(int)
    start_inds_data = calls_ecole[:,1].astype(int)
    end_inds_data = calls_ecole[:,2].astype(int)

    chrs_data_target = target_data[:,0].astype(int)
    start_inds_data_target = target_data[:,1].astype(int)
    end_inds_data_target = target_data[:,2].astype(int)

    for l in range(1,25):
        indices = chrs_data == l
        indices_target = chrs_data_target == l

        if not any(indices):
            continue
      
        start_inds_target = start_inds_data_target[indices_target]
        end_inds_target = end_inds_data_target[indices_target]
        chrs_data_target_ = chrs_data_target[indices_target]
        sorted_ind_target = np.argsort(start_inds_target)
        start_inds_target = start_inds_target[sorted_ind_target]
        end_inds_target = end_inds_target[sorted_ind_target]
        chrs_data_target_ = chrs_data_target_[sorted_ind_target]
        
        predictions = allpreds[indices]
        start_inds = start_inds_data[indices]
        end_inds = end_inds_data[indices]
        sorted_ind = np.argsort(start_inds)
        predictions = predictions[sorted_ind]
        end_inds = end_inds[sorted_ind]
        start_inds = start_inds[sorted_ind]

        np_last_preds = np.zeros(len(sorted_ind_target))

        i = j = k = 0

        while i < len(start_inds_target) and j < len(start_inds):
        
            if start_inds[j] <= end_inds_target[i] and end_inds[j] >= start_inds_target[i]:
                np_last_preds[k] = predictions[j]
                i += 1
                j += 1
            else:
                
                np_last_preds[k] = 3
                i += 1
                
            k +=1
        
        while i < len(start_inds_target):
            np_last_preds[k] = 3
            i += 1
            k += 1        
        
        wind = 1
        a = np_last_preds
        np_last_preds_copy = np.zeros(len(np_last_preds))
        for idx in range(len(np_last_preds_copy)):
            if np_last_preds[idx] == 3:
                left_counter = 0
                right_counter = 0
                left_pointer = idx
                right_pointer = idx
                list_found = [0,0,0]
                while left_counter < wind  and left_pointer > 0:
                    left_pointer -= 1
                    if np_last_preds[left_pointer] == 3:
                        continue
                    else:
                        left_counter += 1
                        if np.abs(left_counter) == 0:
                            print(left_counter, idx)
                        dist = float(np.abs(left_counter))**-2
                        list_found[int(np_last_preds[left_pointer])] += dist
                
                while right_counter < wind + 1 and right_pointer < len(np_last_preds) - 1:
                    right_pointer += 1
                    if np_last_preds[right_pointer] == 3:
                        continue
                    else:
                        right_counter += 1
                        if np.abs(right_counter) == 0:
                            print(right_counter, idx)
                        dist = float(np.abs(right_counter))**-2
                        list_found[int(np_last_preds[right_pointer])] += dist

                
                np_last_preds_copy[idx] = np.argmax(np.array(list_found))
            else:
                np_last_preds_copy[idx] = np_last_preds[idx]
            
            chr_ = "chr"
            if chrs_data_target_[idx] < 23:
                chr_ += str(chrs_data_target_[idx])
            elif chrs_data_target_[idx] == 23:
                chr_ += "X"
            elif chrs_data_target_[idx] == 24:
                chr_ += "Y"

            
            if args.cnv == "exonlevel":
                call_made = np_last_preds_copy[idx]
                cnv_call_string = "<NO-CALL>"
                if call_made == 1:
                    cnv_call_string = "<DEL>"
                elif call_made == 2:
                    cnv_call_string = "<DUP>"

                f = open(os.path.join('ecole_calls_output', args.output, sample_name + ".csv"), "a")
                f.write(chr_ + "\t" + str(start_inds_target[idx]) + "\t" + str(end_inds_target[idx]) + "\t"+ cnv_call_string+ "\n")
                f.close()                
            elif args.cnv == "merged":
                f = open(os.path.join(cur_dirname,"../tmp2/")  + sample_name + ".csv", "a")
                f.write(chr_ + "," + str(start_inds_target[idx]) + "," + str(end_inds_target[idx]) + ","+ str(np_last_preds_copy[idx]) + "\n")
                f.close()

def grouped_preds(preds):
    idx = 0
    result = []
    ele = -1
    for key, sub in groupby(preds):
        ele = len(list(sub))
        result.append((idx,idx + ele-1))
        idx += ele

    return result

if args.cnv == "merged":
    for sample_name in tqdm(all_samples_names):

        f = open(os.path.join(args.output, sample_name + ".csv"), "a")
        f.write("Sample Name" + "\t" +"Chromosome" + "\t" + "CNV Start Index" + "\t" + "CNV End Index" + "\t" + "ECOLÉ Prediction" + "\n")
        f.close()

        calls_ecole = pd.read_csv(os.path.join(cur_dirname,"../tmp2/")+ sample_name + ".csv", sep=",", header=None).values
        convert_chr_to_integer(calls_ecole[:,0])

        chrs_data = calls_ecole[:,0].astype(int)
        allpreds = calls_ecole[:,3].astype(int)
        start_inds_data = calls_ecole[:,1].astype(int)
        end_inds_data = calls_ecole[:,2].astype(int)

        for l in range(1,25):
            indices = chrs_data == l

            predictions = allpreds[indices]
            start_inds = start_inds_data[indices]
            end_inds = end_inds_data[indices]
            sorted_ind = np.argsort(start_inds)
            predictions = predictions[sorted_ind]
            end_inds = end_inds[sorted_ind]
            start_inds = start_inds[sorted_ind]

            S = grouped_preds(predictions)

            for i in range(0,len(S)):
        
                call_ecole = np.bincount(predictions[S[i][0]:S[i][1]+1],minlength=3).argmax()
                j = l
                chr_ = "chr"
                if j < 23:
                    chr_ += str(j)
                elif j == 23:
                    chr_ += "Y"
                elif j == 24:
                    chr_ += "X"
                
                call_made = np.bincount(predictions[S[i][0]:S[i][1]+1],minlength=3).argmax()
                cnv_call_string = "NO-CALL"
                if call_made == 1:
                    cnv_call_string = "DEL"
                elif call_made == 2:
                    cnv_call_string = "DUP"

                f = open(os.path.join(args.output, sample_name + ".csv"), "a")
                f.write(sample_name + "\t" + chr_ + "\t" + str(start_inds[S[i][0]]) + "\t" + str(end_inds[S[i][1]]) + "\t"+ cnv_call_string + "\n")
                f.close()


'''
#filelisttmp1 = os.listdir(os.path.join(cur_dirname,"../tmp/"))
#filelisttmp2 = os.listdir(os.path.join(cur_dirname,"../tmp2/"))
#
#for f in filelisttmp1:
#    os.remove(os.path.join(os.path.join(cur_dirname,"../tmp/"), f))
#for f in filelisttmp2:
#    os.remove(os.path.join(os.path.join(cur_dirname,"../tmp2/"), f))
#
#os.rmdir(os.path.join(cur_dirname,"../tmp/")) 
#os.rmdir(os.path.join(cur_dirname,"../tmp2/"))
