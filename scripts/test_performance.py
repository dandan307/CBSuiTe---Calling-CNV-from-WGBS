import numpy as np
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import argparse
import os
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import label_binarize

from utils import message

description = "hi"

parser = argparse.ArgumentParser(description=description)

required_args = parser.add_argument_group('Required Arguments')
opt_args = parser.add_argument_group("Optional Arguments")

#required_args.add_argument("-g", "--gold", help="Batch size to be used in the training.", required=True)

required_args.add_argument("-i", "--input", help="Relative or direct path to input dataset for ECOLÉ model training, these are the processed samples for training.", required=True)
required_args.add_argument("-g", "--gold", help="Relative or direct path to input dataset for ECOLÉ model training, these are the processed samples for training.", required=True)
required_args.add_argument("-o", "--output", help="Relative or direct path to input dataset for ECOLÉ model training, these are the processed samples for training.")
required_args.add_argument("-m", "--menu", help="Relative or direct output directory path to write ECOLÉ output model weights.")

args = parser.parse_args()
os.makedirs('performance', exist_ok=True)

call_path = args.input
gold_path = args.gold
test_set = [file for file in os.listdir(call_path) if not file.startswith('cnv_')]
gold = os.listdir(gold_path)
test_set.sort()
gold.sort()

samples = [sample.split("_")[1].split(".")[0] if "fixed_" in sample 
           else sample.split("_")[0].split(".")[0]for sample in test_set]
print("input dir: ", call_path)
samples.sort()
preds_ = np.array([])
wgs_preds_ = np.array([])

if args.menu:
    menu = pd.read_csv(args.menu, sep='\s+', names=['WGS', 'WGBS'], header=None)
excel = pd.DataFrame(columns=['samplename', 'auc', 'F1', 'precision', 'recall', '1_precision', '1_recall', '2_precision', '2_recall'])

print(samples)
for i, sample_name in enumerate(samples):
    if args.menu:
        if sample_name in menu['WGBS'].values.astype(str):  # 检查 sample_name 是否在 'WGBS' 列中
            wgs_name = menu.loc[menu['WGBS'] == sample_name, 'WGS'].values[0]   # 选择对应行的 'WGS' 列数据
            wgs_file = wgs_name + '.gold.bed'
            #if wgs_file not in label_files:
            #    continue
        else:
            print("not in menu, continue")
            continue
    else:
        if len(gold) == 1:
            wgs_file = gold[0]
        else:
            wgs_file = gold[i]
    
    print(f"sample file: {sample_name}")
    print(f"labelfile: {wgs_file}")
    
    ecole_calls_data = pd.read_csv(os.path.join(call_path, test_set[i]), sep=",", header=None)
    ecole_calls_data.columns = ['0','1','2','3']
    ecole_calls_data['0'] = ecole_calls_data['0'].astype(str)
    #print("ecole_calls_data1: ", ecole_calls_data.shape)
    #ecole_calls_data = ecole_calls_data[(ecole_calls_data['0'] != 'X') & (ecole_calls_data['0'] != 'Y')]
    print("ecole_calls_data2: ", ecole_calls_data.shape)
    wgs_calls_data = pd.read_csv(os.path.join(gold_path, wgs_file), sep="\t",header=None)
    wgs_calls_data.columns = ['0','1','2','3']
    wgs_calls_data['0'] = wgs_calls_data['0'].replace({23: 'X', 24: 'Y'}).astype(str)
    print("wgs_calls_data: ", wgs_calls_data.shape)
    wgs_calls_data[['0', '1', '2']] = wgs_calls_data[['0', '1', '2']].astype(str)
    ecole_calls_data[['0', '1', '2']] = ecole_calls_data[['0', '1', '2']].astype(str)

    new_df = pd.merge(wgs_calls_data, ecole_calls_data,  how='left', left_on=['0','1','2'], right_on = ['0','1','2'])
    
    duplicate_rows = new_df[new_df.duplicated(subset=['0','1','2'], keep=False)]
    print("Duplicate Rows:", duplicate_rows.shape)
    missing_values = new_df[new_df.isnull().any(axis=1)]
    print("Missing Values:", missing_values.shape)

    new_df = new_df.drop_duplicates(subset=['0','1','2'], keep='first').dropna()

    print("new_df: ", new_df.shape)
    wgs_preds_n = np.array(new_df.iloc[:,3])
    wgs_preds_ = np.append(wgs_preds_,wgs_preds_n)
    preds_n= np.array(new_df.iloc[:,4])
    preds_ = np.append(preds_,preds_n)
    #wgs_preds_n = wgs_preds_n.reshape(-1,1)
    #preds_n = preds_n.reshape(-1,1)
        
    wgs_preds_n[wgs_preds_n == "duplication"] = 2
    wgs_preds_n[wgs_preds_n == "deletion"] = 1
    wgs_preds_n[wgs_preds_n == "nocall"] = 0
    preds_n[preds_n == "duplication"] = 2
    preds_n[preds_n == "deletion"] = 1
    preds_n[preds_n == "nocall"] = 0
    #preds_[preds_ == "<DEL>"] = 2
    #preds_[preds_ == "<DUP>"] = 1
    #preds_[preds_ == "<NO-CALL>"] = 0
    preds_n = preds_n.astype(int)
    wgs_preds_n = wgs_preds_n.astype(int)

    val_precision = precision_score(wgs_preds_n, preds_n, average='macro')
    val_recall = recall_score(wgs_preds_n, preds_n, average='macro')
    val_f1 = f1_score(wgs_preds_n, preds_n, average='macro')
    class_1_precision = precision_score(wgs_preds_n, preds_n, labels=[1], average=None)[0]
    class_1_recall = recall_score(wgs_preds_n, preds_n, labels=[1], average=None)[0]
    class_2_precision = precision_score(wgs_preds_n, preds_n, labels=[2], average=None)[0]
    class_2_recall = recall_score(wgs_preds_n, preds_n, labels=[2], average=None)[0]

    wgs_preds_n_binary = label_binarize(wgs_preds_n, classes=[0, 1, 2])  # 将标签进行二值化编码
    preds_n_binary = label_binarize(preds_n, classes=[0, 1, 2])  # 将标签进行二值化编码
    val_auc2 = roc_auc_score(wgs_preds_n_binary, preds_n_binary, average='weighted', multi_class='ovr')
    print(f"Predict Precision: {val_precision:.4f}")
    print(f"Predict Recall: {val_recall:.4f}")
    print(f"Predict F1-score: {val_f1:.4f}")
    print(f"AUC: {val_auc2:.4f}")
    print(f"Class 1 - Precision: {class_1_precision:.4f}")
    print(f"Class 1 - Recall: {class_1_recall:.4f}")
    print(f"Class 2 - Precision: {class_2_precision:.4f}")
    print(f"Class 2 - Recall: {class_2_recall:.4f}")

    print("confusion matrix: \n",cm(wgs_preds_n, preds_n))
    print("------------------------------------------")
    excel = excel.append({'samplename': sample_name,
                                    'auc': val_auc2,
                                    'F1': val_f1,
                                    'precision': val_precision,
                                    'recall': val_recall,
                                    '1_precision': class_1_precision,
                                    '1_recall': class_1_recall,
                                    '2_precision': class_2_precision,
                                    '2_recall': class_2_recall}, ignore_index=True)

print("------------------------------------------")
print("------------------------------------------")
if args.output:
    excel.to_excel(f"performance/p-{args.output}.xlsx", index=False, encoding='utf-8')
else:
    excel.to_excel(f"performance/p-{call_path}.xlsx", index=False, encoding='utf-8')

wgs_preds_[wgs_preds_ == "duplication"] = 2
wgs_preds_[wgs_preds_ == "deletion"] = 1
wgs_preds_[wgs_preds_ == "nocall"] = 0
preds_[preds_ == "duplication"] = 2
preds_[preds_ == "deletion"] = 1
preds_[preds_ == "nocall"] = 0
#preds_[preds_ == "<DEL>"] = 2
#preds_[preds_ == "<DUP>"] = 1
#preds_[preds_ == "<NO-CALL>"] = 0
wgs_preds_ = wgs_preds_.astype(int)
preds_ = preds_.astype(int)
print('wgs_preds_: ', wgs_preds_, np.shape(wgs_preds_))
print('preds_: ', preds_, np.shape(preds_))
unique_values = np.unique(wgs_preds_)
print("Unique values in wgs_preds_:", unique_values)
print("all data: ")
val_precision = precision_score(wgs_preds_, preds_, average='macro')
val_recall = recall_score(wgs_preds_, preds_, average='macro')
val_f1 = f1_score(wgs_preds_, preds_, average='macro')
class_1_precision = precision_score(wgs_preds_, preds_, labels=[1], average=None)[0]
class_1_recall = recall_score(wgs_preds_, preds_, labels=[1], average=None)[0]
class_2_precision = precision_score(wgs_preds_, preds_, labels=[2], average=None)[0]
class_2_recall = recall_score(wgs_preds_, preds_, labels=[2], average=None)[0]
wgs_preds_binary = label_binarize(wgs_preds_, classes=[0, 1, 2])  # 将标签进行二值化编码
preds_binary = label_binarize(preds_, classes=[0, 1, 2])  # 将标签进行二值化编码
val_auc2 = roc_auc_score(wgs_preds_binary, preds_binary, average='weighted', multi_class='ovr')
print(f"Predict Precision: {val_precision:.4f}")
print(f"Predict Recall: {val_recall:.4f}")
print(f"Predict F1-score: {val_f1:.4f}")
print(f"AUC: {val_auc2:.4f}")
print(f"Class 1 - Precision: {class_1_precision:.4f}")
print(f"Class 1 - Recall: {class_1_recall:.4f}")
print(f"Class 2 - Precision: {class_2_precision:.4f}")
print(f"Class 2 - Recall: {class_2_recall:.4f}")


print("confusion matrix: \n",cm(wgs_preds_, preds_))

