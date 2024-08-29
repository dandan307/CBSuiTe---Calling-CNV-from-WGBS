import os
import numpy as np
import datetime

def message(message):
    print("[",datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"]\t", message)


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

def calculate_accuracy(preds, labels):
    correct = sum(pred == label for pred, label in zip(preds, labels))
    accuracy = correct / len(labels)
    return accuracy

def chr_encode(a):
    if a == 'X':
        a = 23
    elif a == 'Y':
        a = 24
    return a

def label_encode_str(a):
    if a == "nocall":
        b = 0
    elif a == "deletion" or a == "DEL":
        b = 1
    elif a == "duplication" or a == "DUP":
        b = 2
    return b

def label_encode(data):
    for row in data:
        if row[-1] == "nocall":
            row[-1] = 0
        elif row[-1] == "deletion" or row[-1] == "DEL":
            row[-1] = 1
        elif row[-1] == "duplication" or row[-1] == "DUP":
            row[-1] = 2
    return data


def label_decode(data):
    for row in data:
        if row[-1] == 0:
            row[-1] = "nocall"
        elif row[-1] == 1:
            row[-1] = "deletion"
        elif row[-1] == 2:
            row[-1] = "duplication"
    return data

def label_decode_array(data):
    data = [int(x) for x in data]
    for i in range(len(data)):
        if data[i] == 0:
            data[i] = "nocall"
        elif data[i] == 1:
            data[i] = "deletion"
        elif data[i] == 2:
            data[i] = "duplication"
    return data

def calculate_metrics(tp_nocall, tp_plus_fp_nocall, tp_duplication, tp_plus_fp_duplication, tp_deletion, tp_plus_fp_deletion, tp_plus_fn_nocall, tp_plus_fn_duplication, tp_plus_fn_deletion):
    nocall_prec = tp_nocall / (tp_plus_fp_nocall+1e-15) 
    dup_prec = tp_duplication / (tp_plus_fp_duplication+1e-15) 
    del_prec = tp_deletion / (tp_plus_fp_deletion+1e-15) 
    nocall_recall = tp_nocall / (tp_plus_fn_nocall+1e-15) 
    dup_recall = tp_duplication / (tp_plus_fn_duplication+1e-15) 
    del_recall = tp_deletion / (tp_plus_fn_deletion+1e-15) 

    return nocall_prec, dup_prec, del_prec, nocall_recall, dup_recall, del_recall

def filter_lowqua_data(depth_arr):
    # 筛选元素数量小于5的情况
    if np.sum(depth_arr < 5) / len(depth_arr) > 0.5:
        depth_arr = np.zeros_like(depth_arr)  # 将所有元素赋值为0
    return depth_arr