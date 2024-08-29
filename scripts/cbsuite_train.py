'''
CBSuiTe source code.
Training
'''

import os
import torch
torch.cuda.empty_cache()
import numpy as np
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torchvision.datasets import DatasetFolder
import time
from torch.utils.data.dataset import random_split
import argparse
import datetime
from net_my_gm_v2 import CNVcaller
from utils import message, calculate_metrics





description = "hi"

parser = argparse.ArgumentParser(description=description)

'''
Required arguments group:
(i) -bs, batch size to be used in the training. 
(ii) -i, input dataset path comprised of WGBS samples with read depth data.
(iii) -o, relative or direct output directory path to save CBSuiTe output model weights.
(iv) -n, The path for mean&std stats of read depth values.
(v) -e, epochs of training
(vi) -lr, learning rate of training
(vii) -nl, number of layers of transformer
(viii) -w, weight of CNV class in loss function

'''

required_args = parser.add_argument_group('Required Arguments')
opt_args = parser.add_argument_group("Optional Arguments")

required_args.add_argument("-bs", "--batch_size", help="Batch size to be used in the training.", required=True)

required_args.add_argument("-i", "--input", help="Relative or direct input directory path which stores input files(npy files) for CBSuiTe.", required=True)

required_args.add_argument("-o", "--output", help="Relative or direct output directory path to write CBSuiTe output model weights.", required=True)

required_args.add_argument("-n", "--normalize", help="Please provide the path for mean&std stats of read depth values to normalize. \n \
                                                    These values are obtained precalculated from the training dataset.", required=True)

required_args.add_argument("-e", "--epochs", help="Please provide the number of epochs the training will be performed.", required=True)

required_args.add_argument("-lr", "--learning_rate", help="Please provide the learning rate to be used in training.", required=True)
required_args.add_argument("-nl", "--n_layer", help="Please provide the number of Transformer layers to be used in training.", required=True)
required_args.add_argument("-w", "--weight", help="Please provide the CNV class weight to be used in training.", default=2)

'''
Optional arguments group:
-v or --version, version check
-h or --help, help
-g or --gpu, specify gpu
-
'''

opt_args.add_argument("-m", "--model", help="Model path if you want to finetune existed model")
opt_args.add_argument("-g", "--gpu", help="Specify gpu", required=False)
opt_args.add_argument("-V", "--version", help="show program version", action="store_true")

args = parser.parse_args()

if args.version:
    print("CBSuiTe version 0.1")


if args.gpu:
    message("Using GPU!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    message("Using CPU!")
    device = torch.device("cpu")
print(f"Using device: {device}")

os.makedirs(os.path.join('ckpt',args.output), exist_ok=True)
os.makedirs(os.path.join('model',args.output), exist_ok=True)



bs = int(args.batch_size)
n_epoch = int(args.epochs)
weight = float(args.weight)
lr = float(args.learning_rate)


segment_size = 1000
patch_size = 1
n_layer = int(args.n_layer)
feature_size = 192
n_class = 3

class cbsuiteDataset(DatasetFolder):
    def _find_classes(self, directory: str):

        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {"nocall":0,"deletion":1,"duplication":2}
      
        return classes, class_to_idx

print("input: ", args.input)
cbsuite_dataset = cbsuiteDataset(args.input, loader=np.load, extensions = (".npy",".npz"))
print("Dataset Length:", len(cbsuite_dataset))

file1 = open(args.normalize, 'r')
line = file1.readline()
means_ = float(line.split(",")[0])
stds_ = float(line.split(",")[1])
sub_train_ = DataLoader(cbsuite_dataset, batch_size = bs, shuffle=True, pin_memory=True, num_workers=8) # dataloader

model = CNVcaller(segment_size, patch_size, n_layer, feature_size, n_class).to(device)
if args.model:
    model.load_state_dict(torch.load(args.model))
    print(f"load model: {args.model}")

print("number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_func(epoch,trainloader):

    data = trainloader
    
    train_loss, train_acc = 0, 0

    tp_tn_fp_fn = 0
    tp_nocall = 0
    tp_duplication = 0
    tp_plus_fp_nocall = 0
    tp_plus_fp_duplication = 0 
    tp_plus_fp_deletion = 0

    tp_plus_fn_nocall = 0
    tp_plus_fn_duplication = 0
    tp_plus_fn_deletion = 0
    tp_deletion = 0

    model.train()
    for i, (segment, labels) in enumerate(data):
        ind = torch.arange(labels.size(0))
        optimizer.zero_grad()
        
        labels = labels.long()
        segment, labels = segment.to(device),  labels.to(device)
        
        segment = segment.float()
        mask = torch.logical_and(segment != -1, segment != 0)

        mask = torch.squeeze(mask)
        real_mask = torch.ones(labels.size(0),2005, dtype=torch.bool).to(device)
        
        real_mask[:,1:] = mask
       
        output1 = model(segment, real_mask)       
     
        loss1 = CEloss(output1, labels) 
  
        train_loss += loss1.item()
        loss1.backward()
       
        optimizer.step()
        
        _, predicted = torch.max(output1.data, 1)
     
        tp_nocall += (torch.logical_and(predicted == labels,predicted == 0)).sum().item() 
        tp_deletion += (torch.logical_and(predicted == labels,predicted == 1)).sum().item()
        tp_duplication += (torch.logical_and(predicted == labels,predicted == 2)).sum().item() 

        train_acc += (predicted == labels).sum().item()
        tp_plus_fp_nocall += (predicted == 0).sum().item()
        tp_plus_fp_deletion += (predicted == 1).sum().item()
        tp_plus_fp_duplication += (predicted == 2).sum().item()
        
        tp_plus_fn_nocall += (labels == 0).sum().item()
        tp_plus_fn_deletion += (labels == 1).sum().item()
        tp_plus_fn_duplication += (labels == 2).sum().item()
        tp_tn_fp_fn += labels.size(0)

        if i % 5000 == 0 and i > 0:
            nocall_prec, dup_prec, del_prec, nocall_recall, dup_recall, del_recall = calculate_metrics(tp_nocall, tp_plus_fp_nocall, tp_duplication, tp_plus_fp_duplication, tp_deletion, tp_plus_fp_deletion, tp_plus_fn_nocall, tp_plus_fn_duplication, tp_plus_fn_deletion)    
            message(f'Epoch {int(epoch)+1}: Batch no: {i}\tLoss: {train_loss / (i+1):.4f}(train)\t|\tNocall_prec: {nocall_prec * 100:.1f}%(train)|\tDup_prec: {dup_prec * 100:.1f}%(train)|\tDel_prec: {del_prec * 100:.1f}%(train)|\tNocall_recall: {nocall_recall * 100:.1f}%(train)|\tDup_recall: {dup_recall * 100:.1f}%(train)|\tDel_recall: {del_recall * 100:.1f}%(train)')
    
    nocall_prec, dup_prec, del_prec, nocall_recall, dup_recall, del_recall = calculate_metrics(tp_nocall, tp_plus_fp_nocall, tp_duplication, tp_plus_fp_duplication, tp_deletion, tp_plus_fp_deletion, tp_plus_fn_nocall, tp_plus_fn_duplication, tp_plus_fn_deletion)
    
    return train_loss / len(data), ( train_acc / tp_tn_fp_fn), nocall_prec,dup_prec,del_prec, nocall_recall, dup_recall,del_recall


print('learning rate: ',lr)
print('epochs : ',n_epoch)
print('batch size : ',bs)

min_valid_loss = float('inf')

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,n_epoch)

weight_class0, weight_class1, weight_class2 = 1.0, weight, weight
class_weights = [weight_class0, weight_class1, weight_class2]
CEloss = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
print("weight: ", weight_class0, weight_class1, weight_class2)
#CEloss = nn.CrossEntropyLoss()

message("Starting training...")


for epoch in range(n_epoch):
    
    start_time = time.time()

    train_loss, train_acc,nocall_prec,dup_prec,del_prec, nocall_recall, dup_recall,del_recall = train_func(epoch, sub_train_)
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    message(f"Model weights are saved for Epoch: {epoch}")
    torch.save(model.state_dict(), os.path.join("ckpt", args.output, f"cbsuite_depth{str(NO_LAYERS)}_lr{str(lr)}_epoch{epoch}.pt"))
    scheduler.step()
    message('Epoch: %d | time in %d minutes, %d seconds' % (epoch + 1, mins, secs))
    #message('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    message(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)|\tNocall_prec: {nocall_prec * 100:.1f}%(train)|\tDup_prec: {dup_prec * 100:.1f}%(train)|\tDel_prec: {del_prec * 100:.1f}%(train)|\tNocall_recall: {nocall_recall * 100:.1f}%(train)|\tDup_recall: {dup_recall * 100:.1f}%(train)|\tDel_recall: {del_recall * 100:.1f}%(train)')
torch.save(model.state_dict(), os.path.join('model',args.output, f"cbsuite_depth{str(NO_LAYERS)}_lr{str(lr)}.pt"))
   
